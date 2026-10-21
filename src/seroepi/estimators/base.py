"""Universal base classes, adapters, and contracts for SeroEpi estimators."""

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import RegressorMixin

from seroepi.aggregate import Aggregator
from seroepi.models import StratifiedResultMixin
from seroepi.variables import Estimate, Variable


class BaseEstimator(StratifiedResultMixin, RegressorMixin, SklearnBaseEstimator, ABC):
    """Abstract base class for all SeroEpi prevalence estimators.

    Implements scikit-learn's `BaseEstimator` and `RegressorMixin` interfaces alongside
    `StratifiedResultMixin`, providing uniform fitted tabular storage (`estimates_`),
    posterior/bootstrap draw pooling (`target_posterior_draws()`), stratum pooling
    (`summarize_by_target()`, `pool_strata()`), stratum tail-lumping (`summarize_by_stratum()`,
    `lump_strata()`), and within-stratum cumulative coverage trajectories (`cumulative_coverage()`).

    Attributes:
        is_fitted_: Whether `fit()` has been executed on this estimator.
        estimates_: Fitted Polars `DataFrame` containing `[*strata_cols, "target", "event", "n", "estimate", "lower", "upper"]`.
        variables_: Mapping of output column names to semantic `Variable` instances.
        target_: Fitted target `Variable`.
        strata_: Tuple of stratification `Variable` instances.
        adjust_for_: Optional cluster-adjustment `Variable`.
        is_compositional_: Whether target categories are mutually exclusive compositional categories.
        is_padded_: Whether unobserved `(stratum, target)` cells were zero-padded.
        aggregator_: Reference to the upstream fitted `Aggregator` passed to `fit()`.
        train_draws_: Optional `(n_draws, n_obs)` matrix of posterior or bootstrap prevalence draws.
        warnings_: Tuple of deduplicated user-facing warning strings recorded during `fit()`.
        model_family: High-level estimator family label (e.g. `'Unpooled'`, `'Hierarchical Bayesian'`).
        engine: Optional inference backend label (e.g. `'SVI'`, `'MCMC'`).
        name: Optional user-assigned instance display name.
        is_bayesian: Whether this estimator produces Bayesian credible intervals (`CrI`).
        ci_prefix: Short uncertainty interval prefix (`'CI'` or `'CrI'`).
        alpha: Significance level for `(1 - alpha)` uncertainty intervals (defaults to `0.05` for 95% intervals).
    """

    is_fitted_: bool = False
    estimates_: pl.DataFrame | None = None
    variables_: dict[str, Variable] = {}
    target_: Variable
    strata_: tuple[Variable, ...] = ()
    adjust_for_: Variable | None = None
    is_compositional_: bool = False
    is_padded_: bool = False
    aggregator_: Any = None
    train_draws_: np.ndarray | None = None
    warnings_: tuple[str, ...] = ()

    model_family: str = "Estimator"
    engine: str | None = None
    name: str | None = None
    is_bayesian: bool = False
    ci_prefix: str = "CI"
    alpha: float = 0.05

    def _get_frame(self) -> pl.DataFrame:
        assert self.estimates_ is not None
        return self.estimates_

    def _set_frame(self, df: pl.DataFrame) -> None:
        self.estimates_ = df

    def _record_warning(
        self,
        message: str,
        category: type[Warning] = UserWarning,
        *,
        stacklevel: int = 3,
    ) -> None:
        """Records `message` on `self.warnings_` (deduplicated) and emits a Python warning."""
        clean = str(message).strip()
        if clean and clean not in self.warnings_:
            self.warnings_ = (*self.warnings_, clean)
        warnings.warn(clean, category, stacklevel=stacklevel)

    def _bind_aggregator(self, X: Aggregator | Any) -> pl.DataFrame:
        """Validates a fitted `Aggregator` or `TransmissionEstimator` input, binds its metadata to `self`, and returns `X.data_`."""
        if not isinstance(X, StratifiedResultMixin) or not hasattr(X, "data_"):
            raise TypeError(f"Expected Aggregator instance, got {type(X).__name__}")
        X.check_is_fitted()

        df = X.data_
        if "event" not in df.columns or "n" not in df.columns:
            raise ValueError("Aggregator data must contain 'event' and 'n' columns.")

        self.aggregator_ = X
        self.strata_ = X.strata_
        self.target_ = X.target_
        self.adjust_for_ = X.adjust_for_
        self.is_compositional_ = X.is_compositional_
        self.is_padded_ = X.is_padded_
        self.warnings_ = ()
        return df

    def _finalize_fit(
        self,
        df: pl.DataFrame,
        p: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        method: str | None,
    ) -> Self:
        """Attaches `estimate`, `lower`, and `upper` columns, registers the `Estimate` variable, and marks fitted."""
        assert self.aggregator_ is not None
        self.estimates_ = df.with_columns(
            estimate=pl.Series(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.0)),
            lower=pl.Series(np.nan_to_num(np.asarray(lower, dtype=np.float64), nan=0.0)),
            upper=pl.Series(np.nan_to_num(np.asarray(upper, dtype=np.float64), nan=0.0)),
        )
        out_vars = dict(self.aggregator_.variables_)
        out_vars["estimate"] = Estimate(
            column="estimate",
            lower="lower",
            upper="upper",
            level=1.0 - self.alpha,
            method=method,
            bounded=True,
            ci_prefix=self.ci_prefix,
            format=".1%",
            interval_description=self.describe_interval(),
        )
        self.variables_ = out_vars
        self.is_fitted_ = True
        return self

    def _predict_training_fastpath(
        self,
        X: Aggregator | Any,
        return_interval: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Validates `predict(X)` arguments and returns training estimates directly when `X is self.aggregator_`."""
        self.check_is_fitted()
        if not isinstance(X, StratifiedResultMixin) or not hasattr(X, "data_"):
            raise TypeError(f"Expected Aggregator instance, got {type(X).__name__}")
        assert self.estimates_ is not None
        if X is self.aggregator_:
            est = self.estimates_["estimate"].to_numpy()
            if return_interval:
                return est, self.estimates_["lower"].to_numpy(), self.estimates_["upper"].to_numpy()
            return est
        return None

    def _aggregate_stratum_counts(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregates total events (capped at stratum denominator `n`) per single stratum."""
        stratum_col = self.single_stratum_col
        return (
            df.group_by(stratum_col, maintain_order=True)
            .agg(
                pl.col("event").sum().alias("_raw_event"),
                pl.col("n").max().alias("n"),
            )
            .with_columns(
                event=pl.min_horizontal(pl.col("_raw_event"), pl.col("n")),
            )
            .drop("_raw_event")
        )

    @staticmethod
    def _summarize_draws(
        draws: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes mean and equal-tailed `(1 - alpha)` bounds across axis 0 of a `(n_draws, n_obs)` matrix."""
        mean_p = np.mean(draws, axis=0)
        lower_p, upper_p = np.percentile(
            draws,
            [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)],
            axis=0,
        )
        return mean_p, lower_p, upper_p

    def _cumulative_draws_by_stratum(self, df_sorted: pl.DataFrame, draws: np.ndarray) -> np.ndarray:
        """Accumulates and clips `(n_draws, len(df_sorted))` prevalence draws within each single stratum."""
        stratum_vals = df_sorted[self.single_stratum_col].cast(pl.String).to_numpy()
        cum_draws = np.zeros_like(draws)
        for s_val in dict.fromkeys(stratum_vals):
            mask = np.flatnonzero(stratum_vals == s_val)
            cum_draws[:, mask] = np.clip(np.cumsum(draws[:, mask], axis=1), 0.0, 1.0)
        return cum_draws

    def _row_draws_for_frame(self, df: pl.DataFrame) -> np.ndarray | None:
        """Returns `(n_draws, len(df))` row-level prevalence draws if the estimator supports draw simulation."""
        return None

    @property
    def ci_plural(self) -> str:
        """Plural representation of the uncertainty interval (e.g. 'CIs' or 'CrIs')."""
        return f"{self.ci_prefix}s"

    @property
    def ci_percentage(self) -> int:
        """Percentage integer of the uncertainty interval (e.g. 95)."""
        return int(round((1.0 - self.alpha) * 100))

    @property
    def display_name(self) -> str:
        """Formatted human-readable display name for the model/method."""
        if self.name:
            return self.name
        if self.engine:
            return f"{self.model_family} ({self.engine})"
        return self.model_family

    def describe_interval(self) -> str:
        """Descriptive string for uncertainty intervals (e.g. '95% CIs')."""
        return f"{self.ci_percentage}% {self.ci_plural}"

    @property
    def estimate(self) -> Estimate:
        """Returns the fitted Estimate variable definition."""
        self.check_is_fitted()
        est_var = self.variables_.get("estimate")
        if est_var is not None and isinstance(est_var, Estimate):
            return est_var
        return Estimate(
            column="estimate",
            lower="lower" if self.estimates_ is not None and "lower" in self.estimates_.columns else None,
            upper="upper" if self.estimates_ is not None and "upper" in self.estimates_.columns else None,
            ci_prefix=self.ci_prefix,
            interval_description=self.describe_interval(),
        )

    def ranked_targets(self, top_n: int | None = None) -> list[str]:
        """Returns target categories ranked descending by mean estimated prevalence across strata.

        Args:
            top_n: Optional maximum number of top-ranked targets to return.

        Returns:
            List of target category strings ordered from highest to lowest mean prevalence.
        """
        self.check_is_fitted()
        assert self.estimates_ is not None

        target_col = self.target_col
        est_col = self.estimate.column

        summary = (
            self.estimates_.lazy()
            .group_by(target_col)
            .agg(pl.col(est_col).mean().alias("_prev"))
            .sort("_prev", descending=True)
            .collect()
        )

        ranked: list[str] = [str(x) for x in summary[target_col].to_list()]
        return ranked[:top_n] if (top_n is not None and top_n > 0) else ranked

    def _require_single_stratum_frame(self, targets: Sequence[str] | None = None) -> pl.DataFrame:
        """Validates fitted single-stratum state and returns optionally target-filtered estimates."""
        self.check_is_fitted()
        if self.estimates_ is None:
            raise ValueError("Estimator estimates_ DataFrame is None.")
        if not self.is_single_stratum:
            if self.n_strata == 0:
                raise ValueError(
                    "Single-stratum aggregation requires an estimator fitted with exactly 1 stratum (got 0 strata)."
                )
            raise ValueError(
                f"Single-stratum aggregation requires an estimator fitted with exactly 1 stratum, "
                f"got {self.n_strata}: {list(self.strata_cols)}."
            )
        if targets is not None:
            return self.estimates_.filter(pl.col(self.target_col).is_in([str(t) for t in targets]))
        return self.estimates_

    @abstractmethod
    def _pool_strata_by_target(
        self,
        df: pl.DataFrame,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> pl.DataFrame:
        """Pools rows across strata for each target and computes model-specific uncertainty intervals."""
        ...

    @abstractmethod
    def _summarize_strata_and_lump(
        self,
        df: pl.DataFrame,
        x_limit: int | None,
        other_label: str,
    ) -> tuple[pl.DataFrame, list[str]]:
        """Computes stratum-level total prevalence across targets and optionally bins tail strata into 'Other'."""
        ...

    @abstractmethod
    def _compute_cumulative_coverage_columns(self, df_sorted: pl.DataFrame) -> pl.DataFrame:
        """Attaches cum_estimate, cum_lower, and cum_upper columns to a stratum-ordered DataFrame."""
        ...

    @staticmethod
    def _pool_draws_by_target(
        df: pl.DataFrame,
        p_draws: np.ndarray,
        target_col: str,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> tuple[pl.DataFrame, list[str], np.ndarray]:
        """Pools a `(n_draws, len(df))` draw matrix across strata per target using pooled or equal weights."""
        agg_df = df.group_by(target_col, maintain_order=True).agg(
            pl.col("event").sum().alias("event"),
            pl.col("n").sum().alias("n"),
        )
        target_vals = df[target_col].cast(pl.String).to_numpy()
        n_vals = df["n"].to_numpy().astype(float)
        targets_order = agg_df[target_col].cast(pl.String).to_list()

        pooled_draws = np.zeros((p_draws.shape[0], len(targets_order)), dtype=float)
        for idx, t_val in enumerate(targets_order):
            mask = target_vals == t_val
            if weighting == "equal":
                pooled_draws[:, idx] = np.mean(p_draws[:, mask], axis=1)
            else:
                w = n_vals[mask]
                w_sum = float(np.sum(w))
                if w_sum > 0.0:
                    pooled_draws[:, idx] = p_draws[:, mask] @ (w / w_sum)
                else:
                    pooled_draws[:, idx] = np.mean(p_draws[:, mask], axis=1)
        return agg_df, targets_order, pooled_draws

    def target_posterior_draws(
        self,
        targets: Sequence[str] | None = None,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> tuple[list[str], np.ndarray] | None:
        """Returns pooled posterior or bootstrap prevalence draws per target if supported by this estimator.

        Args:
            targets: Optional subset of target names to include.
            weighting: Stratum weighting scheme (`'pooled'` for sample-size weighted, `'equal'` for macro-averaged).

        Returns:
            Tuple `(target_names, draws_matrix)` of shape `(n_draws, len(target_names))`, or `None`
            for analytical frequentist estimators without draw simulation.
        """
        self.check_is_fitted()
        assert self.estimates_ is not None
        df = (
            self.estimates_.filter(pl.col(self.target_col).is_in([str(t) for t in targets]))
            if targets is not None
            else self.estimates_
        )
        p_draws = self._row_draws_for_frame(df)
        if p_draws is None:
            return None
        if df.is_empty():
            return [], np.zeros((p_draws.shape[0], 0), dtype=float)
        _, targets_order, pooled_draws = self._pool_draws_by_target(
            df,
            p_draws,
            self.target_col,
            weighting=weighting,
        )
        return targets_order, pooled_draws

    def summarize_by_target(
        self,
        targets: Sequence[str] | None = None,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> pl.DataFrame:
        """Aggregates prevalence estimates and uncertainty intervals across all strata for each target category.

        Args:
            targets: Optional sequence of target categories to include.
            weighting: Stratum aggregation scheme (`'pooled'` for sample-size weighted or `'equal'` for unweighted macro-average).

        Returns:
            Polars `DataFrame` with columns `[target_col, "event", "n", "estimate", "lower", "upper"]`
            sorted descending by estimated prevalence.

        Raises:
            ValueError: If `weighting` is invalid or `estimates_` is `None`.
        """
        if weighting not in ("pooled", "equal"):
            raise ValueError(f"weighting must be 'pooled' or 'equal', got {weighting!r}.")
        self.check_is_fitted()
        if self.estimates_ is None:
            raise ValueError("Estimator estimates_ DataFrame is None.")
        df = (
            self.estimates_.filter(pl.col(self.target_col).is_in([str(t) for t in targets]))
            if targets is not None
            else self.estimates_
        )
        est_col = self.estimate.column
        if df.is_empty():
            return pl.DataFrame(
                schema={
                    self.target_col: pl.String,
                    "event": pl.Int64,
                    "n": pl.Int64,
                    est_col: pl.Float64,
                    "lower": pl.Float64,
                    "upper": pl.Float64,
                }
            )
        if self.n_strata == 0:
            return df.select([self.target_col, "event", "n", est_col, "lower", "upper"]).sort(
                [est_col, self.target_col], descending=[True, False]
            )
        return self._pool_strata_by_target(df, weighting=weighting).sort(
            [est_col, self.target_col], descending=[True, False]
        )

    def pool_strata(
        self,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> Self:
        """Returns an unstratified clone with estimates pooled across all strata per target.

        Args:
            weighting: Stratum weighting scheme (`'pooled'` or `'equal'`).

        Returns:
            Shallow clone of `self` with `strata_ = ()` and `estimates_` set to `summarize_by_target(weighting=weighting)`.
        """
        self.check_is_fitted()
        if self.n_strata == 0:
            return self
        cloned = copy.copy(self)
        cloned.estimates_ = self.summarize_by_target(weighting=weighting)
        cloned.strata_ = ()
        return cloned

    def summarize_by_stratum(
        self,
        targets: Sequence[str] | None = None,
        *,
        x_limit: int | None = None,
        other_label: str = "Other",
    ) -> tuple[pl.DataFrame, list[str]]:
        """Computes total prevalence across targets within each single stratum using true stratum denominators.

        Ranks strata descending by total prevalence and optionally pools the tail strata beyond
        `x_limit - 1` into `other_label`.

        Args:
            targets: Optional subset of target categories to sum over within each stratum.
            x_limit: Optional maximum number of strata to display (must be `>= 2` when specified).
            other_label: Label used for the pooled tail stratum when `x_limit` is exceeded.

        Returns:
            Tuple `(stratum_df, ordered_strata)` where `stratum_df` contains stratum-level counts,
            point estimates, and uncertainty intervals, and `ordered_strata` lists the stratum keys in rank order.

        Raises:
            ValueError: If `x_limit < 2` or the estimator was not fitted with exactly 1 stratum.
        """
        if x_limit is not None and x_limit < 2:
            raise ValueError(f"x_limit must be >= 2, got {x_limit}.")
        df = self._require_single_stratum_frame(targets)
        if df.is_empty():
            empty = pl.DataFrame(
                schema={
                    self.single_stratum_col: pl.String,
                    "event": pl.Int64,
                    "n": pl.Int64,
                    self.estimate.column: pl.Float64,
                    "lower": pl.Float64,
                    "upper": pl.Float64,
                }
            )
            return empty, []
        return self._summarize_strata_and_lump(df, x_limit, other_label)

    def lump_strata(
        self,
        targets: Sequence[str] | None = None,
        *,
        x_limit: int | None = None,
        ranked_strata: Sequence[str] | None = None,
        other_label: str = "Other",
    ) -> tuple[pl.DataFrame, list[str]]:
        """Retains the top `x_limit - 1` strata and pools remaining strata into `other_label` per target.

        Args:
            targets: Optional subset of target categories to retain.
            x_limit: Optional maximum number of strata columns (`>= 2`).
            ranked_strata: Optional explicit stratum ordering overriding prevalence-based ranking.
            other_label: Display label for the pooled tail stratum.

        Returns:
            Tuple `(lumped_df, ordered_strata)` with model-faithful uncertainty intervals on the pooled `'Other'` rows.

        Raises:
            ValueError: If `x_limit < 2` or the estimator is not single-stratum.
        """
        if x_limit is not None and x_limit < 2:
            raise ValueError(f"x_limit must be >= 2, got {x_limit}.")
        df = self._require_single_stratum_frame(targets)
        stratum_col = self.single_stratum_col

        if ranked_strata is None:
            _, sorted_x = self._summarize_strata_and_lump(df, None, other_label)
        else:
            sorted_x = [str(s) for s in ranked_strata]

        if x_limit is None or len(sorted_x) <= x_limit:
            return df, sorted_x

        top_x = sorted_x[: x_limit - 1]
        other_x = sorted_x[x_limit - 1 :]

        kept_df = df.filter(pl.col(stratum_col).cast(pl.String).is_in(top_x))
        binned_df = df.filter(pl.col(stratum_col).cast(pl.String).is_in(other_x))

        other_df = self._pool_strata_by_target(binned_df).with_columns(
            pl.lit(other_label).alias(stratum_col),
        )
        combined_df = pl.concat([kept_df, other_df], how="diagonal_relaxed")
        return combined_df, [*top_x, other_label]

    def cumulative_coverage(
        self,
        targets: Sequence[str] | None = None,
        *,
        top_n: int | None = None,
        max_strata: int | None = None,
        other_label: str = "Other",
    ) -> tuple[pl.DataFrame, list[str]]:
        """Computes cumulative target coverage and model-faithful uncertainty intervals within each stratum.

        Completes missing `(stratum, target)` steps via within-stratum forward-filling so unpadded
        aggregators produce continuous step-by-step trajectories, and optionally caps the number of
        displayed strata to `max_strata` by pooling tail strata into `other_label`.

        Args:
            targets: Optional explicit ordered sequence of target categories to accumulate across.
            top_n: Optional cap on the number of top-ranked targets when `targets` is `None`.
            max_strata: Optional cap on the number of individual strata before pooling the tail into `other_label`.
            other_label: Display label for the pooled tail stratum.

        Returns:
            Tuple `(cum_df, ordered_targets)` where `cum_df` includes `cum_event`, `cum_estimate`,
            `cum_lower`, and `cum_upper` columns.
        """
        import itertools

        eff_targets = [str(t) for t in targets] if targets is not None else self.ranked_targets(top_n)
        df_filtered = self._require_single_stratum_frame(eff_targets)
        target_col = self.target_col
        stratum_col = self.single_stratum_col

        target_rank = pl.DataFrame(
            {
                target_col: eff_targets,
                "target_rank": list(range(len(eff_targets))),
            }
        )
        df_sorted = df_filtered.join(target_rank, on=target_col, how="left").sort([stratum_col, "target_rank"])

        df_sorted = (
            df_sorted.with_columns(
                pl.col("event").cum_sum().over(stratum_col).alias("_raw_cum_event"),
                pl.col("n").max().over(stratum_col).alias("max_n"),
            )
            .with_columns(
                cum_event=pl.min_horizontal(pl.col("_raw_cum_event"), pl.col("max_n")),
            )
            .drop("_raw_cum_event")
        )

        df_cum = self._compute_cumulative_coverage_columns(df_sorted)

        # Complete full (stratum x target) grid so unpadded aggregators have continuous trajectories
        strata_stats = (
            df_cum.group_by(stratum_col, maintain_order=True)
            .agg(
                pl.col("max_n").max().alias("max_n"),
                pl.col("cum_estimate").max().alias("_final_cov"),
            )
            .sort(["max_n", "_final_cov", stratum_col], descending=[True, True, False])
        )
        strata_order = [str(s) for s in strata_stats[stratum_col].to_list()]

        if len(df_cum) < len(strata_order) * len(eff_targets) and strata_order and eff_targets:
            grid_rows = list(itertools.product(strata_order, range(len(eff_targets))))
            grid_df = pl.DataFrame(
                {
                    stratum_col: [r[0] for r in grid_rows],
                    "target_rank": [r[1] for r in grid_rows],
                    target_col: [eff_targets[r[1]] for r in grid_rows],
                }
            ).join(strata_stats.select([stratum_col, "max_n"]), on=stratum_col, how="left")

            df_cum = (
                grid_df.join(
                    df_cum.drop(["max_n", target_col]),
                    on=[stratum_col, "target_rank"],
                    how="left",
                )
                .sort([stratum_col, "target_rank"])
                .with_columns(
                    pl.col("event").fill_null(0),
                    pl.col("n").fill_null(pl.col("max_n")),
                    pl.col("cum_event").forward_fill().over(stratum_col).fill_null(0),
                    pl.col("cum_estimate").forward_fill().over(stratum_col).fill_null(0.0),
                    pl.col("cum_lower").forward_fill().over(stratum_col).fill_null(0.0),
                    pl.col("cum_upper").forward_fill().over(stratum_col).fill_null(0.0),
                )
            )

        if max_strata is not None and max_strata >= 2 and len(strata_order) > max_strata:
            kept_strata = strata_order[: max_strata - 1]
            kept_df = df_cum.filter(pl.col(stratum_col).cast(pl.String).is_in(kept_strata))
            tail_df = df_cum.filter(~pl.col(stratum_col).cast(pl.String).is_in(kept_strata))

            other_df = (
                tail_df.group_by(["target_rank", target_col], maintain_order=True)
                .agg(
                    pl.lit(other_label).alias(stratum_col),
                    pl.col("event").sum().alias("event"),
                    pl.col("max_n").sum().alias("n"),
                    pl.col("max_n").sum().alias("max_n"),
                    pl.col("cum_event").sum().alias("cum_event"),
                    (
                        (pl.col("cum_estimate") * pl.col("max_n")).sum()
                        / pl.when(pl.col("max_n").sum() > 0).then(pl.col("max_n").sum()).otherwise(1.0)
                    ).alias("cum_estimate"),
                    (
                        (pl.col("cum_lower") * pl.col("max_n")).sum()
                        / pl.when(pl.col("max_n").sum() > 0).then(pl.col("max_n").sum()).otherwise(1.0)
                    ).alias("cum_lower"),
                    (
                        (pl.col("cum_upper") * pl.col("max_n")).sum()
                        / pl.when(pl.col("max_n").sum() > 0).then(pl.col("max_n").sum()).otherwise(1.0)
                    ).alias("cum_upper"),
                )
                .sort("target_rank")
            )
            if hasattr(self, "_compute_ci") and not getattr(self, "is_bootstrapped", False):
                k_o = other_df["cum_event"].to_numpy().astype(float)
                n_o = other_df["max_n"].to_numpy().astype(float)
                p_o, lo_o, hi_o = self._compute_ci(k_o, n_o)  # ty: ignore[call-non-callable]
                other_df = other_df.with_columns(
                    cum_estimate=pl.Series(np.nan_to_num(p_o, nan=0.0)),
                    cum_lower=pl.Series(np.nan_to_num(lo_o, nan=0.0)),
                    cum_upper=pl.Series(np.nan_to_num(hi_o, nan=0.0)),
                )

            df_cum = pl.concat([kept_df, other_df], how="diagonal_relaxed")

        return df_cum, eff_targets

    def contrast_label_with(
        self,
        other: "BaseEstimator",
        self_label: str | None = None,
        other_label: str | None = None,
    ) -> tuple[str, str]:
        """Infers comparative display labels contrasting `self` with `other`.

        Args:
            other: Second fitted `BaseEstimator` being contrasted against `self`.
            self_label: Optional explicit label override for `self`.
            other_label: Optional explicit label override for `other`.

        Returns:
            Tuple `(resolved_self_label, resolved_other_label)` distinguishing the two estimators
            by stratum value, cluster-adjustment state, or model display name.
        """
        if self_label is not None and other_label is not None:
            return self_label, other_label

        same_model = self.display_name == other.display_name
        self_adj = self.adjust_for_title
        other_adj = other.adjust_for_title
        same_adj = self_adj == other_adj

        self_stratum = self.infer_single_stratum_value()
        other_stratum = other.infer_single_stratum_value()
        diff_stratum = self_stratum is not None and other_stratum is not None and self_stratum != other_stratum

        if diff_stratum and same_model and same_adj:
            inferred_self = self_stratum
            inferred_other = other_stratum
        elif same_model and not same_adj:
            inferred_self = f"Adjusted ({self_adj})" if self_adj else "Unadjusted"
            inferred_other = f"Adjusted ({other_adj})" if other_adj else "Unadjusted"
        elif not same_model and same_adj:
            if self_adj:
                inferred_self = f"Adjusted ({self_adj}) {self.display_name}"
                inferred_other = f"Adjusted ({other_adj}) {other.display_name}"
            else:
                inferred_self = self.display_name
                inferred_other = other.display_name
        else:
            inferred_self = f"Adjusted ({self_adj}) {self.display_name}" if self_adj else self.display_name
            inferred_other = f"Adjusted ({other_adj}) {other.display_name}" if other_adj else other.display_name

        return self_label or inferred_self, other_label or inferred_other

    @abstractmethod
    def fit(self, X: Aggregator, y: Any = None, sample_weight: Any = None) -> Self:
        """Fits the estimator to data.

        Args:
            X: Input fitted Aggregator instance.
            y: Target values (optional if already embedded in X).
            sample_weight: Optional observation weights.

        Returns:
            Fitted estimator instance.
        """
        ...

    @abstractmethod
    def predict(self, X: Aggregator, return_interval: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        """Standard scikit-learn predict returning point estimate arrays or uncertainty intervals.

        Args:
            X: Fitted Aggregator instance.
            return_interval: If True, returns a 3-tuple (estimate, lower, upper) of 1D arrays.
                Otherwise, returns the 1D point estimate array.

        Returns:
            1D NumPy array or 3-tuple of 1D NumPy arrays.
        """
        ...

    def predict_interval(self, X: Aggregator, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Explicit method to obtain (estimate, lower, upper) uncertainty intervals.

        Args:
            X: Fitted Aggregator instance.
            alpha: Significance level (default 0.05 for 95% intervals).

        Returns:
            3-tuple of 1D NumPy arrays (estimate, lower, upper).
        """
        res = self.predict(X, return_interval=True)
        if isinstance(res, tuple) and len(res) == 3:
            return res
        point_est = np.asarray(res, dtype=float)
        return point_est, point_est, point_est

    def fit_predict(self, X: Aggregator, y: Any = None, sample_weight: Any = None) -> pl.DataFrame:
        """Fits the model and returns the fitted estimates DataFrame.

        Args:
            X: Fitted Aggregator instance.
            y: Target values.
            sample_weight: Optional observation weights.

        Returns:
            Fitted estimates Polars DataFrame containing estimate, lower, and upper bounds.
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        assert self.estimates_ is not None
        return self.estimates_

    def score(self, X: Aggregator, y: Any = None, sample_weight: Any = None) -> float:
        """Calculates negative weighted Brier score between observed events and model predictions.

        Higher scores indicate better calibration, with 0.0 being optimal.

        Args:
            X: Fitted Aggregator instance.
            y: Optional ground-truth target values (prevalence proportions).
            sample_weight: Optional sample weights (defaults to cohort size n).

        Returns:
            Negative weighted Brier score in [-1.0, 0.0].
        """
        self.check_is_fitted()
        p_pred = np.clip(np.asarray(self.predict(X), dtype=float), 0.0, 1.0)
        df = X.data_

        if y is not None:
            y_true = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
            w = np.asarray(sample_weight, dtype=float) if sample_weight is not None else np.ones(len(y_true))
        elif "event" in df.columns and "n" in df.columns:
            k = df["event"].to_numpy().astype(float)
            n = np.maximum(1e-6, df["n"].to_numpy().astype(float))
            y_true = np.clip(k / n, 0.0, 1.0)
            w = n
        elif "estimate" in df.columns:
            y_true = np.clip(df["estimate"].to_numpy().astype(float), 0.0, 1.0)
            w = np.ones(len(y_true))
        else:
            return 0.0

        total_w = float(np.sum(w))
        if total_w <= 0:
            return 0.0

        brier = float(np.sum(w * (y_true - p_pred) ** 2) / total_w)
        return -brier

    def filter(self, *predicates: pl.Expr | bool, **kwargs: Any) -> Self:
        """Filters the fitted estimator's `estimates_` and underlying `aggregator_` by predicates or kwargs.

        Args:
            *predicates: Polars boolean filter expressions (`pl.Expr`).
            **kwargs: Column equality constraints (`col_name=value`).

        Returns:
            Shallow clone of `self` with filtered `estimates_` and synchronized `aggregator_`.
        """
        cloned = super().filter(*predicates, **kwargs)
        if self.aggregator_ is not None and self.aggregator_.data_ is not None:
            agg_data_cols = set(self.aggregator_.data_.columns)
            agg_kwargs = {k: v for k, v in kwargs.items() if k in agg_data_cols}
            try:
                if agg_kwargs:
                    cloned.aggregator_ = self.aggregator_.filter(**agg_kwargs)
                elif not kwargs and predicates:
                    cloned.aggregator_ = self.aggregator_.filter(*predicates)
            except Exception:
                pass
        return cloned


class BayesianEstimator(BaseEstimator, ABC):
    """Abstract base class for hierarchical Bayesian prevalence estimators.

    Configures `is_bayesian = True` and `ci_prefix = 'CrI'` so downstream tables and plotters
    automatically label intervals as Bayesian credible intervals.

    Attributes:
        model_family: Defaults to `'Hierarchical Bayesian'`.
        engine: Inference engine identifier (`'SVI'` or `'MCMC'`).
        is_bayesian: Always `True`.
        ci_prefix: Defaults to `'CrI'`.
    """

    model_family: str = "Hierarchical Bayesian"
    engine: str | None = None
    is_bayesian: bool = True
    ci_prefix: str = "CrI"

    def describe_interval(self) -> str:
        """Descriptive string for Bayesian credible intervals."""
        engine_str = f" {self.engine}" if self.engine else ""
        return f"{self.ci_percentage}%{engine_str} credible intervals"
