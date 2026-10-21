"""Scikit-learn meta-estimators for vaccine/cocktail formulation design, LOO stability, and model comparison."""

import copy
import io
import itertools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import numpy as np
import polars as pl
from scipy import stats
from shiny_kit.progress import current_progress
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import MetaEstimatorMixin, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, LeaveOneGroupOut

from seroepi.aggregate import Aggregator
from seroepi.estimators.base import BaseEstimator
from seroepi.variables import Criterion, Target


class Formulation(SklearnBaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Scikit-learn meta-estimator for target formulation design and Leave-One-Stratum-Out stability.

    Wraps a single prevalence estimator to:
    1. Rank target categories globally (1..K) using pooled or equal-stratum weighting.
    2. Evaluate burden threshold exceedance ($H_{0,j}: p_j \\le \\tau_{\\min}$) with Bayesian or BH FDR control.
    3. Select the top-``valency`` target formulation and export it as a binary ``Target``.
    4. Quantify sampling/geographic sensitivity via Leave-One-Stratum-Out (LOSO) or group cross-validation.

    Args:
        estimator: Base prevalence estimator instance to wrap.
        valency: Number of top-ranked target categories to include in the formulation.
        weighting: Stratum aggregation scheme ('pooled' for sample-size weighted or 'equal' for macro-averaged).
        targets: Optional explicit sequence of target names overriding top-valency selection.
        cv: Cross-validation strategy ('loso', False/None to disable, or a scikit-learn BaseCrossValidator).
        min_prevalence: Minimum prevalence threshold for burden exceedance hypothesis testing.
        alpha: Significance level for credible/confidence intervals and FDR control.
        name: Optional custom display name for the formulation.

    Attributes:
        is_fitted_: Boolean flag indicating whether ``fit()`` has been called.
        aggregator_: Fitted ``Aggregator`` bound during ``fit()``.
        estimator_: Fitted ``BaseEstimator`` instance used to derive target rankings.
        rankings_: Polars ``DataFrame`` of ranked targets with prevalence, cumulative coverage, and FDR metrics.
        selected_targets_: List of selected target category names in the formulation.
        cv_rankings_: Polars ``DataFrame`` of per-fold target rankings across LOSO cross-validation splits.
        stability_metrics_: Polars ``DataFrame`` summarizing per-target CV rank stability and selection frequency.
        fold_summary_: Polars ``DataFrame`` summarizing Jaccard similarity and held-out coverage per CV fold.
        warnings_: Tuple of warning messages captured from the underlying estimator.

    Examples:
        >>> import polars as pl
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.estimators.unpooled import UnpooledEstimator
        >>> from seroepi.estimators.formulation import Formulation
        >>> from seroepi.variables import Identifier, Spatial, Trait
        >>> ds = Dataset(
        ...     pl.DataFrame({"id": ["s1", "s2", "s3", "s4"], "site": ["A", "A", "B", "B"], "k": ["K1", "K2", "K1", "K3"]}),
        ...     variables={"id": Identifier("id"), "site": Spatial("site"), "k": Trait("k", compositional=True)},
        ... )
        >>> agg = Aggregator(target=ds.var("k"), strata=[ds.var("site")]).fit(ds)
        >>> form = Formulation(UnpooledEstimator(method="wilson"), valency=2, cv="loso").fit(agg)
        >>> form.selected_targets_
        ['K1', 'K2']
    """

    is_fitted_: bool = False
    aggregator_: Aggregator
    estimator_: BaseEstimator
    rankings_: pl.DataFrame
    selected_targets_: list[str]
    cv_rankings_: pl.DataFrame
    stability_metrics_: pl.DataFrame
    fold_summary_: pl.DataFrame
    warnings_: tuple[str, ...] = ()

    def __init__(
        self,
        estimator: BaseEstimator,
        valency: int = 6,
        weighting: Literal["pooled", "equal"] = "pooled",
        targets: Sequence[str] | None = None,
        cv: Literal["loso"] | bool | BaseCrossValidator | None = "loso",
        min_prevalence: float = 0.0,
        alpha: float = 0.05,
        name: str | None = None,
    ) -> None:
        """Initializes the Formulation meta-estimator.

        Args:
            estimator: Base prevalence estimator instance to wrap.
            valency: Number of top-ranked target categories to include in the formulation.
            weighting: Stratum aggregation scheme ('pooled' for sample-size weighted or 'equal' for macro-averaged).
            targets: Optional explicit sequence of target names overriding top-valency selection.
            cv: Cross-validation strategy ('loso', False/None to disable, or a scikit-learn BaseCrossValidator).
            min_prevalence: Minimum prevalence threshold for burden exceedance hypothesis testing.
            alpha: Significance level for credible/confidence intervals and FDR control.
            name: Optional custom display name for the formulation.

        Raises:
            TypeError: If ``estimator`` is not a ``BaseEstimator`` instance.
            ValueError: If ``valency < 1``, ``weighting`` is invalid, or ``min_prevalence`` / ``alpha`` are out of bounds.
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                f"Formulation requires a single BaseEstimator instance, got {type(estimator).__name__}. "
                "To compare multiple estimators, use FormulationComparison."
            )
        if valency < 1:
            raise ValueError(f"valency must be >= 1, got {valency}.")
        if weighting not in ("pooled", "equal"):
            raise ValueError(f"weighting must be 'pooled' or 'equal', got {weighting!r}.")
        if not (0.0 <= min_prevalence < 1.0):
            raise ValueError(f"min_prevalence must be in [0.0, 1.0), got {min_prevalence}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}.")
        self.estimator = estimator
        self.valency = valency
        self.weighting = weighting
        self.targets = targets
        self.cv = cv
        self.min_prevalence = min_prevalence
        self.alpha = alpha
        self.name = name

    @property
    def display_name(self) -> str:
        """Returns a human-readable label for this formulation configuration.

        Returns:
            Custom ``name`` if provided, otherwise ``"<estimator> (<valency>v, <weighting>)"``.
        """
        if self.name:
            return self.name
        return f"{self.estimator.display_name} ({self.valency}v, {self.weighting})"

    @property
    def summary_label(self) -> str:
        """Returns a concise target and valency summary for UI registries.

        Returns:
            Formatted summary string describing the target title, valency, and weighting.
        """
        if getattr(self, "is_fitted_", False) and getattr(self, "aggregator_", None) is not None:
            return f"{self.aggregator_.target_title} · {self.valency}v ({self.weighting})"
        return f"{self.valency}v ({self.weighting})"

    def check_is_fitted(self) -> None:
        """Validates that the formulation has been fitted.

        Raises:
            RuntimeError: If ``fit()`` has not been called on this instance.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with an Aggregator before using it."
            )

    def save(self, path: str | Path | None = None, compress: int = 3) -> bytes | Path:
        """Serializes the fitted ``Formulation`` into a compressed Joblib file or in-memory bytes.

        Args:
            path: Optional file path to write the Joblib archive to. If ``None``, returns serialized bytes.
            compress: Joblib compression level from ``0`` to ``9``.

        Returns:
            The destination ``Path`` if ``path`` was provided, or raw ``bytes`` otherwise.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
        """
        self.check_is_fitted()
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, dest, compress=compress)
            return dest
        buffer = io.BytesIO()
        joblib.dump(self, buffer, compress=compress)
        return buffer.getvalue()

    @classmethod
    def load(cls, source: str | Path | bytes | io.BytesIO) -> Self:
        """Loads a fitted ``Formulation`` from a Joblib file, path, or bytes buffer.

        Args:
            source: File path, ``bytes``, or ``io.BytesIO`` buffer containing a serialized ``Formulation``.

        Returns:
            The deserialized ``Formulation`` instance.

        Raises:
            FileNotFoundError: If a path is given and does not exist.
            TypeError: If ``source`` is an unsupported type or does not deserialize to ``cls``.
        """
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Joblib file not found: {src_path}")
            obj = joblib.load(src_path)
        elif isinstance(source, bytes):
            obj = joblib.load(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            obj = joblib.load(source)
        else:
            raise TypeError(f"Unsupported source type for load: {type(source).__name__}")

        if not isinstance(obj, cls):
            raise TypeError(f"Expected deserialized object to be {cls.__name__}, got {type(obj).__name__}")
        return obj

    @staticmethod
    def _validate_aggregator(X: Aggregator) -> None:
        if not isinstance(X, Aggregator):
            raise TypeError(f"Formulation requires a fitted Aggregator instance, got {type(X).__name__}.")
        X.check_is_fitted()

    @staticmethod
    def _cumulative_bayesian_fdr(error_probs: np.ndarray) -> np.ndarray:
        """Computes cumulative Bayesian FDR (q-values) from local posterior error probabilities."""
        if len(error_probs) == 0:
            return np.array([], dtype=float)
        clipped = np.clip(np.asarray(error_probs, dtype=float), 0.0, 1.0)
        order = np.argsort(clipped, kind="stable")
        sorted_err = clipped[order]
        cum_mean = np.cumsum(sorted_err) / np.arange(1, len(sorted_err) + 1, dtype=float)
        q_vals = np.empty_like(cum_mean)
        q_vals[order] = np.clip(cum_mean, 0.0, 1.0)
        return q_vals

    @staticmethod
    def _compute_bayesian_threshold_fdr(
        draws: np.ndarray,
        min_prevalence: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes posterior error probability P(p_j <= min_prevalence), Bayesian FDR q_value, and significance."""
        if draws.shape[1] == 0:
            empty = np.array([], dtype=float)
            return empty, empty, np.array([], dtype=bool)
        pep = np.mean(draws <= min_prevalence, axis=0).astype(float)
        q_vals = Formulation._cumulative_bayesian_fdr(pep)
        return pep, q_vals, q_vals < alpha

    @staticmethod
    def _compute_frequentist_threshold_fdr(
        k: np.ndarray,
        n: np.ndarray,
        min_prevalence: float,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes one-sided exact binomial p-values for H0: p_j <= min_prevalence and Benjamini-Hochberg q-values."""
        if len(k) == 0:
            empty = np.array([], dtype=float)
            return empty, empty, np.array([], dtype=bool)
        p_vals = np.ones(len(k), dtype=float)
        for idx in range(len(k)):
            n_i = int(round(max(0.0, float(n[idx]))))
            k_i = int(round(np.clip(float(k[idx]), 0.0, float(n_i))))
            if n_i <= 0:
                p_vals[idx] = 1.0
            elif min_prevalence <= 0.0:
                p_vals[idx] = 0.0 if k_i > 0 else 1.0
            else:
                p_vals[idx] = float(stats.binomtest(k_i, n_i, p=min_prevalence, alternative="greater").pvalue)
        q_vals = np.asarray(stats.false_discovery_control(np.clip(p_vals, 0.0, 1.0), method="bh"), dtype=float)
        return p_vals, q_vals, q_vals < alpha

    def _build_rankings(self, est: BaseEstimator) -> tuple[pl.DataFrame, list[str]]:
        """Computes 1D target rankings (1..K), threshold FDR metrics, and selected top-valency targets."""
        target_col = est.target_col
        est_col = est.estimate.column
        summary = est.summarize_by_target(weighting=self.weighting)

        ranks = summary.with_columns(
            rank=pl.int_range(1, pl.len() + 1),
            cum_estimate=pl.col(est_col).cum_sum().clip(0.0, 1.0),
        )
        if self.targets is not None:
            selected = [str(t) for t in self.targets]
        else:
            selected = [str(t) for t in ranks.head(self.valency)[target_col].to_list()]

        rank_targets = ranks[target_col].cast(pl.String).to_list()
        draw_res = est.target_posterior_draws(weighting=self.weighting)
        if draw_res is not None and len(rank_targets) > 0:
            draw_targets, draws_mat = draw_res
            idx_map = {t: i for i, t in enumerate(draw_targets)}
            aligned = np.zeros((draws_mat.shape[0], len(rank_targets)), dtype=float)
            for col_i, t_name in enumerate(rank_targets):
                src_i = idx_map.get(t_name)
                if src_i is not None:
                    aligned[:, col_i] = draws_mat[:, src_i]
            p_or_pep, q_vals, sig = self._compute_bayesian_threshold_fdr(
                aligned,
                self.min_prevalence,
                self.alpha,
            )
        else:
            p_or_pep, q_vals, sig = self._compute_frequentist_threshold_fdr(
                ranks["event"].to_numpy().astype(float),
                ranks["n"].to_numpy().astype(float),
                self.min_prevalence,
                self.alpha,
            )

        ranks = ranks.with_columns(
            in_formulation=pl.col(target_col).cast(pl.String).is_in(selected),
            p_or_pep=pl.Series(p_or_pep, dtype=pl.Float64),
            q_value=pl.Series(q_vals, dtype=pl.Float64),
            significant=pl.Series(sig, dtype=pl.Boolean),
        )
        return ranks, selected

    def _coverage_by_stratum(
        self,
        X: Aggregator,
        estimator: BaseEstimator,
        selected_targets: Sequence[str],
    ) -> pl.DataFrame:
        """Computes observed and predicted formulation coverage for each stratum in X."""
        df = X.data_
        target_col = estimator.target_col
        preds = np.asarray(estimator.predict(X), dtype=float)
        active_set = [str(t) for t in selected_targets]
        in_form = pl.col(target_col).cast(pl.String).is_in(active_set)
        scored = df.with_columns(
            _pred=pl.Series(preds),
            _in_form=in_form,
        ).with_columns(
            _form_event=pl.when(pl.col("_in_form")).then(pl.col("event")).otherwise(0.0),
            _form_pred=pl.when(pl.col("_in_form")).then(pl.col("_pred")).otherwise(0.0),
        )
        strata_cols = list(X.strata_cols)
        if strata_cols:
            grouped = scored.group_by(strata_cols, maintain_order=True).agg(
                pl.col("n").max().alias("n"),
                pl.col("_form_event").sum().alias("_w_event"),
                pl.col("_form_pred").sum().alias("_w_pred"),
            )
        else:
            grouped = scored.select(
                pl.col("n").max().alias("n"),
                pl.col("_form_event").sum().alias("_w_event"),
                pl.col("_form_pred").sum().alias("_w_pred"),
            )

        return grouped.with_columns(
            observed_coverage=pl.when(pl.col("n") > 0)
            .then((pl.col("_w_event") / pl.col("n")).clip(0.0, 1.0))
            .otherwise(0.0),
            predicted_coverage=pl.col("_w_pred").clip(0.0, 1.0),
        ).drop(["_w_event", "_w_pred"])

    @classmethod
    def _from_fitted_estimator(
        cls,
        estimator: BaseEstimator,
        *,
        valency: int = 6,
        weighting: Literal["pooled", "equal"] = "pooled",
        targets: Sequence[str] | None = None,
        cv: Literal["loso"] | bool | BaseCrossValidator | None = None,
        min_prevalence: float = 0.0,
        alpha: float = 0.05,
        name: str | None = None,
    ) -> Self:
        """Wraps an already-fitted BaseEstimator into a fitted Formulation without re-running model inference."""
        estimator.check_is_fitted()
        if estimator.aggregator_ is None:
            raise ValueError("Fitted BaseEstimator must have a non-None `aggregator_` attribute.")
        instance = cls(
            estimator=estimator,
            valency=valency,
            weighting=weighting,
            targets=targets,
            cv=cv,
            min_prevalence=min_prevalence,
            alpha=alpha,
            name=name,
        )
        instance.aggregator_ = estimator.aggregator_
        instance.estimator_ = estimator
        instance.warnings_ = tuple(getattr(estimator, "warnings_", ()))
        instance.rankings_, instance.selected_targets_ = instance._build_rankings(estimator)
        if cv is not None and cv is not False and estimator.aggregator_.n_strata >= 1:
            instance._run_cv(estimator.aggregator_)
        else:
            instance.cv_rankings_ = pl.DataFrame()
            instance.stability_metrics_ = pl.DataFrame()
            instance.fold_summary_ = pl.DataFrame()
        instance.is_fitted_ = True
        return instance

    def fit(self, X: Aggregator | None = None, y: Any = None, sample_weight: Any = None) -> Self:
        """Fits the underlying estimator on ``X`` (or reuses a pre-fitted estimator), ranks targets, and runs LOSO CV.

        Args:
            X: Optional fitted ``Aggregator`` instance. May be ``None`` if ``self.estimator`` is already fitted.
            y: Ignored; present for scikit-learn compatibility.
            sample_weight: Optional sample weights forwarded to ``estimator.fit()``.

        Returns:
            The fitted ``Formulation`` instance (``self``).

        Raises:
            ValueError: If ``X`` is ``None`` and ``self.estimator`` is not already fitted.
            TypeError: If ``X`` is not a fitted ``Aggregator``.
        """
        prog = current_progress()
        est_is_fitted = (
            getattr(self.estimator, "is_fitted_", False) and getattr(self.estimator, "aggregator_", None) is not None
        )
        if X is None:
            if not est_is_fitted:
                raise ValueError("Formulation.fit(X=None) requires `self.estimator` to be already fitted.")
            eff_X = self.estimator.aggregator_
            assert eff_X is not None
        else:
            self._validate_aggregator(X)
            eff_X = X

        self.aggregator_ = eff_X
        will_run_cv = self.cv is not None and self.cv is not False and eff_X.n_strata >= 1
        fit_end = 0.15 if will_run_cv else 0.90

        if est_is_fitted and self.estimator.aggregator_ is eff_X and y is None and sample_weight is None:
            self.estimator_ = self.estimator
            prog.report(fit_end, message=f"Designing {self.display_name}", detail="Ranking targets...")
        else:
            with prog.subrange(0.0, fit_end, message=f"Designing {self.display_name}", detail_prefix="Full dataset"):
                self.estimator_ = clone(self.estimator).fit(eff_X, y=y, sample_weight=sample_weight)

        self.warnings_ = tuple(getattr(self.estimator_, "warnings_", ()))
        self.rankings_, self.selected_targets_ = self._build_rankings(self.estimator_)

        if will_run_cv:
            with prog.subrange(fit_end, 0.98, message=f"LOSO Cross-Validation ({self.display_name})"):
                self._run_cv(eff_X)
        else:
            self.cv_rankings_ = pl.DataFrame()
            self.stability_metrics_ = pl.DataFrame()
            self.fold_summary_ = pl.DataFrame()

        self.is_fitted_ = True
        prog.report(1.0, message=f"Fitted {self.display_name}", detail="Formulation ready")
        return self

    def _run_cv(self, X: Aggregator) -> None:
        """Executes Leave-One-Stratum-Out or group cross-validation across strata."""
        prog = current_progress()
        target_col = self.estimator_.target_col
        est_col = self.estimator_.estimate.column
        strata_cols = list(X.strata_cols)

        if len(strata_cols) == 1:
            groups = X.data_[strata_cols[0]].cast(pl.String).to_numpy()
        else:
            groups = (
                X.data_.select(pl.concat_str([pl.col(c).cast(pl.String) for c in strata_cols], separator="__"))
                .to_series()
                .to_numpy()
            )

        unique_groups = list(dict.fromkeys(groups))
        if len(unique_groups) < 2:
            self.cv_rankings_ = pl.DataFrame()
            self.stability_metrics_ = pl.DataFrame()
            self.fold_summary_ = pl.DataFrame()
            return

        splitter: BaseCrossValidator = self.cv if isinstance(self.cv, BaseCrossValidator) else LeaveOneGroupOut()
        baseline_map = self.rankings_.select(
            [target_col, pl.col("rank").alias("baseline_rank"), pl.col(est_col).alias("baseline_estimate")]
        )
        base_set = set(self.selected_targets_)

        splits = list(splitter.split(X.data_, groups=groups))
        n_folds = max(1, len(splits))

        fold_rank_frames: list[pl.DataFrame] = []
        fold_summaries: list[dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = copy.copy(X)
            X_train.data_ = X.data_[train_idx]
            X_test = copy.copy(X)
            X_test.data_ = X.data_[test_idx]

            holdout_groups = list(dict.fromkeys(str(g) for g in groups[test_idx]))
            holdout_label = ", ".join(holdout_groups) if holdout_groups else str(fold_idx)

            f_start = fold_idx / n_folds
            f_end = (fold_idx + 1) / n_folds
            with prog.subrange(
                f_start,
                f_end,
                detail_prefix=f"Fold {fold_idx + 1}/{n_folds} ({holdout_label})",
            ) as fold_prog:
                fold_est = clone(self.estimator).fit(X_train)
                fold_ranks, fold_selected = self._build_rankings(fold_est)

                fold_df = (
                    fold_ranks.rename({"rank": "fold_rank", "in_formulation": "in_fold_formulation"})
                    .join(baseline_map, on=target_col, how="left")
                    .with_columns(
                        holdout_stratum=pl.lit(holdout_label),
                        rank_shift=pl.col("fold_rank") - pl.col("baseline_rank"),
                    )
                )
                fold_rank_frames.append(fold_df)

                fold_set = set(fold_selected)
                union_len = len(base_set | fold_set)
                jaccard = len(base_set & fold_set) / union_len if union_len > 0 else 1.0
                dropped = [t for t in self.selected_targets_ if t not in fold_set]
                added = [t for t in fold_selected if t not in base_set]

                cov_df = self._coverage_by_stratum(X_test, fold_est, fold_selected)
                total_n = int(cov_df["n"].sum())
                if total_n > 0:
                    obs_cov = float((cov_df["observed_coverage"] * cov_df["n"]).sum() / total_n)
                    pred_cov = float((cov_df["predicted_coverage"] * cov_df["n"]).sum() / total_n)
                else:
                    obs_cov = float(np.mean(cov_df["observed_coverage"].to_numpy())) if len(cov_df) > 0 else 0.0
                    pred_cov = float(np.mean(cov_df["predicted_coverage"].to_numpy())) if len(cov_df) > 0 else 0.0

                fold_summaries.append(
                    {
                        "holdout_stratum": holdout_label,
                        "n_isolates": total_n,
                        "jaccard_similarity": jaccard,
                        "dropped_targets": ", ".join(dropped),
                        "added_targets": ", ".join(added),
                        "heldout_observed_coverage": obs_cov,
                        "heldout_predicted_coverage": pred_cov,
                    }
                )
                fold_prog.report(1.0)

        self.cv_rankings_ = pl.concat(fold_rank_frames, how="diagonal_relaxed")
        self.fold_summary_ = pl.DataFrame(fold_summaries)
        self.stability_metrics_ = (
            self.cv_rankings_.group_by([target_col, "baseline_rank"], maintain_order=True)
            .agg(
                pl.col("fold_rank").mean().alias("mean_cv_rank"),
                pl.col("fold_rank").min().alias("min_cv_rank"),
                pl.col("fold_rank").max().alias("max_cv_rank"),
                pl.col("fold_rank").std().fill_null(0.0).alias("rank_std"),
                pl.col("rank_shift").abs().max().alias("max_abs_rank_shift"),
                pl.col("in_fold_formulation").cast(pl.Float64).mean().alias("selection_frequency"),
            )
            .sort("baseline_rank")
        )

    def score_by_stratum(self, X: Aggregator | None = None) -> pl.DataFrame:
        """Computes observed and predicted formulation coverage per stratum.

        Args:
            X: Optional fitted ``Aggregator`` to evaluate coverage on. Defaults to ``self.aggregator_``.

        Returns:
            A Polars ``DataFrame`` containing stratum columns, ``n``, ``observed_coverage``, and ``predicted_coverage``.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
            TypeError: If ``X`` is not a fitted ``Aggregator``.
        """
        self.check_is_fitted()
        eff_X = self.aggregator_ if X is None else X
        self._validate_aggregator(eff_X)
        return self._coverage_by_stratum(eff_X, self.estimator_, self.selected_targets_)

    def stratum_trajectories(
        self,
        *,
        top_n: int | None = None,
        max_strata: int | None = 8,
        other_label: str = "Other",
    ) -> tuple[pl.DataFrame, list[str]]:
        """Computes step-by-step cumulative coverage trajectories across formulation-ranked targets per stratum.

        Args:
            top_n: Optional maximum number of top-ranked targets to include along the trajectory.
            max_strata: Optional maximum number of strata before lumping remaining strata into ``other_label``.
            other_label: Display label for lumped tail strata.

        Returns:
            A 2-tuple ``(cum_df, ordered_targets)`` where ``cum_df`` contains ``cum_estimate``,
            ``cum_lower``, ``cum_upper``, ``rank``, and ``in_formulation`` columns per stratum and step.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
        """
        self.check_is_fitted()
        target_col = self.estimator_.target_col
        ordered_targets = self.rankings_[target_col].cast(pl.String).to_list()
        if top_n is not None and top_n > 0:
            ordered_targets = ordered_targets[:top_n]
        cum_df, eff_targets = self.estimator_.cumulative_coverage(
            targets=ordered_targets,
            max_strata=max_strata,
            other_label=other_label,
        )
        rank_meta = self.rankings_.select(
            pl.col(target_col).cast(pl.String).alias(target_col),
            pl.col("rank"),
            pl.col("in_formulation"),
        )
        cum_df = cum_df.with_columns(pl.col(target_col).cast(pl.String)).join(
            rank_meta,
            on=target_col,
            how="left",
            maintain_order="left",
        )
        return cum_df, eff_targets

    def predict(self, X: Aggregator) -> np.ndarray:
        """Predicts formulation coverage fraction for each stratum in ``X``.

        Args:
            X: Fitted ``Aggregator`` instance to predict stratum coverage on.

        Returns:
            A 1D float64 NumPy array of predicted coverage proportions in ``[0, 1]`` for each stratum.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
        """
        cov_df = self.score_by_stratum(X)
        return cov_df["predicted_coverage"].to_numpy().astype(float)

    def score(self, X: Aggregator, y: Any = None, sample_weight: Any = None) -> float:
        """Computes negative sample-size-weighted Brier score of formulation coverage across strata in ``X``.

        Args:
            X: Fitted ``Aggregator`` instance to evaluate.
            y: Optional ground-truth coverage array overriding ``observed_coverage``.
            sample_weight: Optional stratum weights overriding stratum sample sizes ``n``.

        Returns:
            Negative weighted mean squared error between observed and predicted stratum coverage.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
        """
        self.check_is_fitted()
        self._validate_aggregator(X)
        cov_df = self._coverage_by_stratum(X, self.estimator_, self.selected_targets_)

        y_pred = cov_df["predicted_coverage"].to_numpy().astype(float)
        y_true = np.asarray(y, dtype=float) if y is not None else cov_df["observed_coverage"].to_numpy().astype(float)
        w = (
            np.asarray(sample_weight, dtype=float)
            if sample_weight is not None
            else np.maximum(1.0, cov_df["n"].to_numpy().astype(float))
        )
        total_w = float(np.sum(w))
        if total_w <= 0.0:
            return 0.0
        return -float(np.sum(w * (y_true - y_pred) ** 2) / total_w)

    def to_target(self, name: str | None = None, title: str | None = None) -> Target:
        """Exports the formulated top-valency targets into a binary SeroEpi ``Target``.

        Args:
            name: Optional machine identifier for the exported ``Target``.
            title: Optional human-readable display title for the exported ``Target``.

        Returns:
            A binary (non-compositional) ``Target`` matching any isolate carrying one of ``selected_targets_``.

        Raises:
            RuntimeError: If the formulation has not been fitted yet.
        """
        self.check_is_fitted()
        coerced_source = Target.coerce(
            self.aggregator_.target,
            compositional=self.aggregator_.compositional,
            negative_indicator=self.aggregator_.negative_indicator,
        )
        escaped = "|".join(re.escape(t) for t in self.selected_targets_)
        pattern = f"^(?:{escaped})$" if escaped else "^$"

        resolved_name = name or f"{coerced_source.name}_{self.valency}v"
        resolved_title = title or f"{self.valency}-Valent {self.aggregator_.target_title}"
        return Target(
            name=resolved_name,
            criteria=[
                Criterion(
                    variable=crit.variable,
                    pattern=pattern,
                    equivalence=dict(crit.equivalence),
                    negative_indicator=crit.negative_indicator,
                )
                for crit in coerced_source.criteria
            ],
            conjunction=coerced_source.conjunction,
            compositional=False,
            sep=coerced_source.sep,
            title=resolved_title,
        )

    def compare(
        self,
        other: "Formulation | BaseEstimator",
        name: str | None = None,
        *,
        alpha: float | None = None,
        null_margin: float = 0.0,
    ) -> "FormulationComparison":
        """Fits and returns a ``FormulationComparison`` contrasting this formulation with another.

        Args:
            other: Another ``Formulation`` or ``BaseEstimator`` instance to compare against.
            name: Optional display label for ``other`` in the comparison tables.
            alpha: Optional significance level override for differential prevalence FDR control.
            null_margin: Minimum absolute prevalence difference margin ``>= 0.0`` for Bayesian LFSR testing.

        Returns:
            A fitted ``FormulationComparison`` instance.

        Raises:
            RuntimeError: If this formulation has not been fitted yet.
        """
        self.check_is_fitted()
        other_name = name or (other.display_name if isinstance(other, (Formulation, BaseEstimator)) else "Comparison")
        if other_name == self.display_name:
            other_name = f"{other_name} (2)"
        comp = FormulationComparison(
            formulations={self.display_name: self, other_name: other},
            valency=self.valency,
            weighting=self.weighting,
            cv=None,
            alpha=alpha if alpha is not None else self.alpha,
            null_margin=null_margin,
        )
        other_is_fitted = getattr(other, "is_fitted_", False) and getattr(other, "aggregator_", None) is not None
        return comp.fit(None if other_is_fitted else self.aggregator_)


class FormulationComparison(SklearnBaseEstimator, MetaEstimatorMixin):
    """Meta-estimator comparing target rankings, rank shifts, concordance, and differential prevalence across models or cohorts.

    Args:
        formulations: Mapping from model/cohort labels to ``Formulation`` or ``BaseEstimator`` instances (at least 2).
        valency: Default target valency applied when wrapping raw ``BaseEstimator`` entries.
        weighting: Default stratum weighting ('pooled' or 'equal') applied when wrapping raw ``BaseEstimator`` entries.
        cv: Optional cross-validation strategy applied when fitting raw ``BaseEstimator`` entries.
        alpha: Significance level for differential prevalence intervals and FDR control.
        null_margin: Minimum absolute prevalence difference margin ``>= 0.0`` for Bayesian LFSR testing.

    Attributes:
        is_fitted_: Boolean flag indicating whether ``fit()`` has been called.
        aggregator_: Primary ``Aggregator`` bound during ``fit()``.
        formulations_: Dictionary mapping each label to its fitted ``Formulation`` instance.
        rankings_: Stacked Polars ``DataFrame`` of rankings across all compared formulations.
        rank_shifts_: Wide Polars ``DataFrame`` comparing per-target ranks and rank ranges across formulations.
        concordance_: Polars ``DataFrame`` of pairwise Spearman rho, Kendall tau, and Jaccard top-V overlap.
        differential_prevalence_: Polars ``DataFrame`` of pairwise target prevalence differences, fold changes, and FDR q-values.

    Examples:
        >>> import polars as pl
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.estimators.unpooled import UnpooledEstimator
        >>> from seroepi.estimators.formulation import FormulationComparison
        >>> from seroepi.variables import Identifier, Spatial, Trait
        >>> ds = Dataset(
        ...     pl.DataFrame({"id": ["s1", "s2", "s3", "s4"], "site": ["A", "A", "B", "B"], "k": ["K1", "K2", "K1", "K3"]}),
        ...     variables={"id": Identifier("id"), "site": Spatial("site"), "k": Trait("k", compositional=True)},
        ... )
        >>> agg = Aggregator(target=ds.var("k"), strata=[ds.var("site")]).fit(ds)
        >>> comp = FormulationComparison.from_strata(UnpooledEstimator(method="wilson"), agg, valency=2)
        >>> comp.is_fitted_
        True
    """

    is_fitted_: bool = False
    aggregator_: Aggregator
    formulations_: dict[str, Formulation]
    rankings_: pl.DataFrame
    rank_shifts_: pl.DataFrame
    concordance_: pl.DataFrame
    differential_prevalence_: pl.DataFrame

    def __init__(
        self,
        formulations: Mapping[str, Formulation | BaseEstimator],
        valency: int = 6,
        weighting: Literal["pooled", "equal"] = "pooled",
        cv: Literal["loso"] | bool | BaseCrossValidator | None = None,
        alpha: float = 0.05,
        null_margin: float = 0.0,
    ) -> None:
        """Initializes the FormulationComparison meta-estimator.

        Args:
            formulations: Mapping from model/cohort labels to ``Formulation`` or ``BaseEstimator`` instances (at least 2).
            valency: Default target valency applied when wrapping raw ``BaseEstimator`` entries.
            weighting: Default stratum weighting ('pooled' or 'equal') applied when wrapping raw ``BaseEstimator`` entries.
            cv: Optional cross-validation strategy applied when fitting raw ``BaseEstimator`` entries.
            alpha: Significance level for differential prevalence intervals and FDR control.
            null_margin: Minimum absolute prevalence difference margin ``>= 0.0`` for Bayesian LFSR testing.

        Raises:
            ValueError: If fewer than 2 formulations are provided, or if ``alpha`` or ``null_margin`` are invalid.
        """
        if len(formulations) < 2:
            raise ValueError(f"FormulationComparison requires at least 2 formulations, got {len(formulations)}.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}.")
        if null_margin < 0.0:
            raise ValueError(f"null_margin must be >= 0.0, got {null_margin}.")
        self.formulations = formulations
        self.valency = valency
        self.weighting = weighting
        self.cv = cv
        self.alpha = alpha
        self.null_margin = null_margin

    def check_is_fitted(self) -> None:
        """Validates that the formulation comparison has been fitted.

        Raises:
            RuntimeError: If ``fit()`` has not been called on this instance.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with an Aggregator before using it."
            )

    @classmethod
    def from_strata(
        cls,
        estimator: BaseEstimator,
        X: Aggregator,
        strata_values: Sequence[str] | None = None,
        *,
        stratum_col: str | None = None,
        valency: int = 6,
        weighting: Literal["pooled", "equal"] = "pooled",
        alpha: float = 0.05,
        null_margin: float = 0.0,
    ) -> Self:
        """Slices a stratified ``Aggregator`` across strata into cohort ``Formulation`` objects and fits a comparison.

        Args:
            estimator: Unfitted or fitted ``BaseEstimator`` template cloned for each stratum cohort.
            X: Stratified fitted ``Aggregator`` instance.
            strata_values: Optional subset of stratum values to compare (requires at least 2).
            stratum_col: Optional stratum column name when ``X`` has multiple strata.
            valency: Target formulation valency for each cohort.
            weighting: Stratum weighting scheme (``"pooled"`` or ``"equal"``).
            alpha: Significance level for differential prevalence intervals and FDR control.
            null_margin: Minimum absolute prevalence difference margin ``>= 0.0`` for Bayesian LFSR testing.

        Returns:
            A fitted ``FormulationComparison`` instance contrasting the selected strata cohorts.

        Raises:
            ValueError: If ``X`` is unstratified, ``stratum_col`` is missing, or fewer than 2 strata values are selected.
        """
        Formulation._validate_aggregator(X)
        if X.n_strata == 0:
            raise ValueError("FormulationComparison.from_strata requires a stratified Aggregator (got 0 strata).")
        col = stratum_col or (X.single_stratum_col if X.is_single_stratum else X.strata_cols[0])
        if col not in X.data_.columns:
            raise ValueError(f"Stratum column {col!r} not found in Aggregator columns: {X.data_.columns}.")

        available_vals = [str(v) for v in X.data_[col].cast(pl.String).unique(maintain_order=True).to_list()]
        selected_vals = [str(v) for v in strata_values] if strata_values is not None else available_vals
        if len(selected_vals) < 2:
            raise ValueError(
                f"FormulationComparison.from_strata requires at least 2 strata values, got {selected_vals!r}."
            )

        prog = current_progress()
        n_cohorts = max(1, len(selected_vals))
        cohort_forms: dict[str, Formulation] = {}
        for idx, s_val in enumerate(selected_vals):
            cohort_agg = X.filter(**{col: s_val})
            c_start = 0.85 * (idx / n_cohorts)
            c_end = 0.85 * ((idx + 1) / n_cohorts)
            with prog.subrange(
                c_start,
                c_end,
                message="Comparing Cohort Formulations",
                detail_prefix=f"Cohort {idx + 1}/{n_cohorts} ({s_val})",
            ):
                cohort_forms[s_val] = Formulation(
                    estimator=clone(estimator),
                    valency=valency,
                    weighting=weighting,
                    cv=None,
                    alpha=alpha,
                    name=s_val,
                ).fit(cohort_agg)

        comp = cls(
            formulations=cohort_forms,
            valency=valency,
            weighting=weighting,
            cv=None,
            alpha=alpha,
            null_margin=null_margin,
        )
        with prog.subrange(0.85, 1.0, message="Computing Differential Prevalence & FDR"):
            return comp.fit(X=None)

    def fit(self, X: Aggregator | None = None, y: Any = None) -> Self:
        """Fits constituent formulations (or preserves pre-fitted cohort formulations) and computes comparisons.

        Args:
            X: Optional fitted ``Aggregator`` instance used to fit any unfitted constituent formulations.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted ``FormulationComparison`` instance (``self``).

        Raises:
            ValueError: If any constituent formulation/estimator is unfitted and ``X`` is ``None``.
            TypeError: If any constituent entry is not a ``Formulation`` or ``BaseEstimator``.
        """
        prog = current_progress()
        if X is not None:
            Formulation._validate_aggregator(X)

        fitted_forms: dict[str, Formulation] = {}
        rank_frames: list[pl.DataFrame] = []
        items = list(self.formulations.items())
        n_items = max(1, len(items))

        for idx, (label, spec) in enumerate(items):
            s_start = 0.80 * (idx / n_items)
            s_end = 0.80 * ((idx + 1) / n_items)
            with prog.subrange(
                s_start,
                s_end,
                message="Fitting Formulation Comparison",
                detail_prefix=f"Model {idx + 1}/{n_items} ({label})",
            ) as sub_prog:
                if isinstance(spec, Formulation):
                    if getattr(spec, "is_fitted_", False) and getattr(spec, "aggregator_", None) is not None:
                        form = spec
                    elif X is not None:
                        form = clone(spec).fit(X, y=y)
                    else:
                        raise ValueError(
                            f"Formulation {label!r} is not fitted and X=None was passed to FormulationComparison.fit()."
                        )
                elif isinstance(spec, BaseEstimator):
                    if getattr(spec, "is_fitted_", False) and getattr(spec, "aggregator_", None) is not None:
                        form = Formulation._from_fitted_estimator(
                            spec,
                            valency=self.valency,
                            weighting=self.weighting,
                            cv=self.cv,
                            alpha=self.alpha,
                            name=label,
                        )
                    elif X is not None:
                        form = Formulation(
                            estimator=spec,
                            valency=self.valency,
                            weighting=self.weighting,
                            cv=self.cv,
                            alpha=self.alpha,
                            name=label,
                        ).fit(X, y=y)
                    else:
                        raise ValueError(
                            f"Estimator {label!r} is not fitted and X=None was passed to FormulationComparison.fit()."
                        )
                else:
                    raise TypeError(
                        f"Formulation entry {label!r} must be a Formulation or BaseEstimator, got {type(spec).__name__}."
                    )
                sub_prog.report(1.0)

            fitted_forms[label] = form
            rank_frames.append(form.rankings_.with_columns(formulation=pl.lit(label)))

        self.aggregator_ = X if X is not None else next(iter(fitted_forms.values())).aggregator_
        self.formulations_ = fitted_forms
        self.rankings_ = pl.concat(rank_frames, how="diagonal_relaxed")
        prog.report(
            0.88, message="Computing Formulation Comparison", detail="Evaluating rank shifts and concordance..."
        )
        self._compute_rank_shifts_and_concordance()
        prog.report(0.94, message="Computing Formulation Comparison", detail="Testing differential prevalence & FDR...")
        self._compute_all_differential_prevalence()
        self.is_fitted_ = True
        prog.report(1.0, message="Formulation Comparison Complete", detail="Ready")
        return self

    @staticmethod
    def _compute_bayesian_differential(
        draws_a: np.ndarray,
        draws_b: np.ndarray,
        alpha: float,
        null_margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes draw-wise difference intervals, Local False Sign Rate (lfsr), and Bayesian FDR q_value."""
        n_draws = min(draws_a.shape[0], draws_b.shape[0])
        delta_draws = draws_b[:n_draws] - draws_a[:n_draws]
        q_low = 100.0 * (alpha / 2.0)
        q_high = 100.0 * (1.0 - alpha / 2.0)
        delta_lower, delta_upper = np.percentile(delta_draws, [q_low, q_high], axis=0)

        prob_le = np.mean(delta_draws <= null_margin, axis=0)
        prob_ge = np.mean(delta_draws >= -null_margin, axis=0)
        lfsr = np.minimum(prob_le, prob_ge).astype(float)
        q_vals = Formulation._cumulative_bayesian_fdr(lfsr)
        return delta_lower, delta_upper, lfsr, q_vals

    @staticmethod
    def _compute_frequentist_differential(
        k_a: np.ndarray,
        n_a: np.ndarray,
        est_a: np.ndarray,
        low_a: np.ndarray,
        high_a: np.ndarray,
        k_b: np.ndarray,
        n_b: np.ndarray,
        est_b: np.ndarray,
        low_b: np.ndarray,
        high_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Computes Newcombe score difference intervals, two-sided Fisher's Exact p-values, and BH FDR q_values."""
        delta_est = est_b - est_a
        delta_lower = np.clip(
            delta_est - np.sqrt(np.square(est_b - low_b) + np.square(high_a - est_a)),
            -1.0,
            1.0,
        )
        delta_upper = np.clip(
            delta_est + np.sqrt(np.square(high_b - est_b) + np.square(est_a - low_a)),
            -1.0,
            1.0,
        )

        p_vals = np.ones(len(k_a), dtype=float)
        for idx in range(len(k_a)):
            na_i = int(round(max(0.0, float(n_a[idx]))))
            nb_i = int(round(max(0.0, float(n_b[idx]))))
            if na_i <= 0 or nb_i <= 0:
                p_vals[idx] = 1.0
                continue
            ka_i = int(round(np.clip(float(k_a[idx]), 0.0, float(na_i))))
            kb_i = int(round(np.clip(float(k_b[idx]), 0.0, float(nb_i))))
            table = np.array([[kb_i, nb_i - kb_i], [ka_i, na_i - ka_i]], dtype=int)
            p_vals[idx] = float(stats.fisher_exact(table, alternative="two-sided").pvalue)

        q_vals = (
            np.asarray(stats.false_discovery_control(np.clip(p_vals, 0.0, 1.0), method="bh"), dtype=float)
            if len(p_vals) > 0
            else np.array([], dtype=float)
        )
        return delta_lower, delta_upper, p_vals, q_vals

    def contrast(self, label_a: str, label_b: str) -> pl.DataFrame:
        """Computes per-target differential prevalence, fold change, rank shifts, and FDR between two formulations.

        Args:
            label_a: Key of the baseline formulation in ``self.formulations_``.
            label_b: Key of the comparison formulation in ``self.formulations_``.

        Returns:
            A Polars ``DataFrame`` containing ``delta_prevalence``, ``delta_lower``, ``delta_upper``,
            ``log2_fold_change``, ``delta_rank``, ``p_or_lfsr``, ``q_value``, and ``significant``.

        Raises:
            RuntimeError: If the comparison has not been fitted yet.
            KeyError: If either ``label_a`` or ``label_b`` is not present in ``self.formulations_``.
        """
        self.check_is_fitted()
        if label_a not in self.formulations_ or label_b not in self.formulations_:
            raise KeyError(
                f"Both {label_a!r} and {label_b!r} must be in fitted formulations: {list(self.formulations_.keys())}."
            )
        return self._contrast_pair(label_a, self.formulations_[label_a], label_b, self.formulations_[label_b])

    def _contrast_pair(
        self,
        label_a: str,
        form_a: Formulation,
        label_b: str,
        form_b: Formulation,
    ) -> pl.DataFrame:
        target_col = form_a.estimator_.target_col
        est_col_a = form_a.estimator_.estimate.column
        est_col_b = form_b.estimator_.estimate.column

        df_a = form_a.rankings_.select(
            [
                pl.col(target_col).cast(pl.String).alias(target_col),
                pl.col("event").alias("event_a"),
                pl.col("n").alias("n_a"),
                pl.col(est_col_a).alias("estimate_a"),
                pl.col("lower").alias("lower_a"),
                pl.col("upper").alias("upper_a"),
                pl.col("rank").alias("rank_a"),
            ]
        )
        df_b = form_b.rankings_.select(
            [
                pl.col(target_col).cast(pl.String).alias(target_col),
                pl.col("event").alias("event_b"),
                pl.col("n").alias("n_b"),
                pl.col(est_col_b).alias("estimate_b"),
                pl.col("lower").alias("lower_b"),
                pl.col("upper").alias("upper_b"),
                pl.col("rank").alias("rank_b"),
            ]
        )

        max_n_a = int(np.max(df_a["n_a"].to_numpy())) if len(df_a) > 0 else 0
        max_n_b = int(np.max(df_b["n_b"].to_numpy())) if len(df_b) > 0 else 0
        fallback_rank_a = len(df_a) + 1
        fallback_rank_b = len(df_b) + 1

        merged = (
            df_a.join(df_b, on=target_col, how="full", coalesce=True)
            .with_columns(
                pl.col("event_a").fill_null(0),
                pl.col("n_a").fill_null(max_n_a),
                pl.col("estimate_a").fill_null(0.0),
                pl.col("lower_a").fill_null(0.0),
                pl.col("upper_a").fill_null(0.0),
                pl.col("rank_a").fill_null(fallback_rank_a),
                pl.col("event_b").fill_null(0),
                pl.col("n_b").fill_null(max_n_b),
                pl.col("estimate_b").fill_null(0.0),
                pl.col("lower_b").fill_null(0.0),
                pl.col("upper_b").fill_null(0.0),
                pl.col("rank_b").fill_null(fallback_rank_b),
            )
            .sort("rank_a")
        )

        target_list = merged[target_col].to_list()
        est_a = merged["estimate_a"].to_numpy().astype(float)
        est_b = merged["estimate_b"].to_numpy().astype(float)
        delta_prev = est_b - est_a
        eps = 1e-6
        log2_fc = np.log2((est_b + eps) / (est_a + eps))

        draws_res_a = form_a.estimator_.target_posterior_draws(weighting=form_a.weighting)
        draws_res_b = form_b.estimator_.target_posterior_draws(weighting=form_b.weighting)

        if draws_res_a is not None and draws_res_b is not None and len(target_list) > 0:
            t_a, mat_a = draws_res_a
            t_b, mat_b = draws_res_b
            map_a = {t: i for i, t in enumerate(t_a)}
            map_b = {t: i for i, t in enumerate(t_b)}
            aligned_a = np.zeros((mat_a.shape[0], len(target_list)), dtype=float)
            aligned_b = np.zeros((mat_b.shape[0], len(target_list)), dtype=float)
            for col_i, t_name in enumerate(target_list):
                if t_name in map_a:
                    aligned_a[:, col_i] = mat_a[:, map_a[t_name]]
                if t_name in map_b:
                    aligned_b[:, col_i] = mat_b[:, map_b[t_name]]

            delta_lower, delta_upper, p_or_lfsr, q_vals = self._compute_bayesian_differential(
                aligned_a,
                aligned_b,
                alpha=self.alpha,
                null_margin=self.null_margin,
            )
            method_label = "bayesian_lfsr"
        else:
            delta_lower, delta_upper, p_or_lfsr, q_vals = self._compute_frequentist_differential(
                merged["event_a"].to_numpy().astype(float),
                merged["n_a"].to_numpy().astype(float),
                est_a,
                merged["lower_a"].to_numpy().astype(float),
                merged["upper_a"].to_numpy().astype(float),
                merged["event_b"].to_numpy().astype(float),
                merged["n_b"].to_numpy().astype(float),
                est_b,
                merged["lower_b"].to_numpy().astype(float),
                merged["upper_b"].to_numpy().astype(float),
            )
            method_label = "fisher_exact_bh"

        return merged.select(
            [
                pl.lit(label_a).alias("model_a"),
                pl.lit(label_b).alias("model_b"),
                pl.col(target_col),
                pl.col("event_a"),
                pl.col("n_a"),
                pl.col("estimate_a"),
                pl.col("event_b"),
                pl.col("n_b"),
                pl.col("estimate_b"),
                pl.Series("delta_prevalence", delta_prev, dtype=pl.Float64),
                pl.Series("delta_lower", delta_lower, dtype=pl.Float64),
                pl.Series("delta_upper", delta_upper, dtype=pl.Float64),
                pl.Series("log2_fold_change", log2_fc, dtype=pl.Float64),
                (pl.col("rank_b") - pl.col("rank_a")).alias("delta_rank"),
                pl.lit(method_label).alias("test_method"),
                pl.Series("p_or_lfsr", p_or_lfsr, dtype=pl.Float64),
                pl.Series("q_value", q_vals, dtype=pl.Float64),
                pl.Series("significant", q_vals < self.alpha, dtype=pl.Boolean),
            ]
        )

    def _compute_all_differential_prevalence(self) -> None:
        names = list(self.formulations_.keys())
        diff_frames: list[pl.DataFrame] = []
        for name_a, name_b in itertools.combinations(names, 2):
            diff_frames.append(
                self._contrast_pair(name_a, self.formulations_[name_a], name_b, self.formulations_[name_b])
            )
        self.differential_prevalence_ = (
            pl.concat(diff_frames, how="diagonal_relaxed") if diff_frames else pl.DataFrame()
        )

    def _compute_rank_shifts_and_concordance(self) -> None:
        names = list(self.formulations_.keys())
        first_form = self.formulations_[names[0]]
        target_col = first_form.estimator_.target_col
        est_col = first_form.estimator_.estimate.column

        # 1. Build wide rank_shifts_ table
        base_df = first_form.rankings_.select(
            [
                pl.col(target_col).cast(pl.String).alias(target_col),
                pl.col("rank").alias(f"rank_{names[0]}"),
                pl.col(est_col).alias(f"estimate_{names[0]}"),
            ]
        )
        for name in names[1:]:
            f_df = self.formulations_[name].rankings_.select(
                [
                    pl.col(target_col).cast(pl.String).alias(target_col),
                    pl.col("rank").alias(f"rank_{name}"),
                    pl.col(est_col).alias(f"estimate_{name}"),
                ]
            )
            base_df = base_df.join(f_df, on=target_col, how="full", coalesce=True)

        for name in names:
            fallback_rank = len(self.formulations_[name].rankings_) + 1
            base_df = base_df.with_columns(
                pl.col(f"rank_{name}").fill_null(fallback_rank),
                pl.col(f"estimate_{name}").fill_null(0.0),
            )

        rank_cols = [pl.col(f"rank_{name}") for name in names]
        base_df = base_df.with_columns(
            min_rank=pl.min_horizontal(rank_cols),
            max_rank=pl.max_horizontal(rank_cols),
        ).with_columns(
            rank_range=pl.col("max_rank") - pl.col("min_rank"),
        )
        if len(names) == 2:
            base_df = base_df.with_columns(
                delta_rank=pl.col(f"rank_{names[1]}") - pl.col(f"rank_{names[0]}"),
            )
        self.rank_shifts_ = base_df.sort(f"rank_{names[0]}")

        # 2. Build pairwise concordance_ table
        concordance_rows: list[dict[str, Any]] = []
        for name_a, name_b in itertools.combinations(names, 2):
            form_a = self.formulations_[name_a]
            form_b = self.formulations_[name_b]

            r_a = self.rank_shifts_[f"rank_{name_a}"].to_numpy().astype(float)
            r_b = self.rank_shifts_[f"rank_{name_b}"].to_numpy().astype(float)

            if len(r_a) >= 2 and np.std(r_a) > 0.0 and np.std(r_b) > 0.0:
                sp_res: Any = stats.spearmanr(r_a, r_b)
                kt_res: Any = stats.kendalltau(r_a, r_b)
                spearman_rho = float(sp_res.statistic)
                kendall_tau = float(kt_res.statistic)
            else:
                spearman_rho = 1.0
                kendall_tau = 1.0

            set_a = set(form_a.selected_targets_)
            set_b = set(form_b.selected_targets_)
            shared = [t for t in form_a.selected_targets_ if t in set_b]
            unique_a = [t for t in form_a.selected_targets_ if t not in set_b]
            unique_b = [t for t in form_b.selected_targets_ if t not in set_a]
            union_len = len(set_a | set_b)
            jaccard = len(shared) / union_len if union_len > 0 else 1.0

            concordance_rows.append(
                {
                    "model_a": name_a,
                    "model_b": name_b,
                    "spearman_rho": spearman_rho,
                    "kendall_tau": kendall_tau,
                    "top_v_overlap": len(shared),
                    "jaccard_similarity": jaccard,
                    "shared_targets": ", ".join(shared),
                    "unique_to_a": ", ".join(unique_a),
                    "unique_to_b": ", ".join(unique_b),
                }
            )

        self.concordance_ = pl.DataFrame(concordance_rows)
