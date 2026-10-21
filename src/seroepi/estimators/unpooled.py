"""Unpooled analytical and non-parametric bootstrap prevalence estimator (`UnpooledEstimator`)."""

from typing import Any, Literal, Self

import numpy as np
import polars as pl

from seroepi.aggregate import Aggregator
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.intervals import (
    agresti_coull_interval,
    bayesian_bootstrap_compositional_draws,
    bayesian_bootstrap_interval,
    bayesian_bootstrap_row_draws,
    bootstrap_interval,
    clopper_pearson_interval,
    jeffreys_interval,
    wald_interval,
    wilson_interval,
)


class UnpooledEstimator(BaseEstimator):
    """Calculates unpooled prevalence proportions and analytical or bootstrap confidence intervals.

    Supports exact Clopper-Pearson, Wilson Score, Agresti-Coull, Wald, Jeffreys,
    frequentist binomial simulation bootstrap, and Dirichlet Bayesian bootstrap intervals.

    Args:
        method: Confidence/credible interval estimation method.
        alpha: Significance level (e.g. 0.05 for 95% intervals).
        n_resamples: Number of Monte Carlo resamples for bootstrap and Bayesian bootstrap methods.
        prior_concentration: Symmetric Dirichlet prior concentration added to counts for
            `bayesian_bootstrap` (default 0.5 Jeffreys reference prior; 0.0 for strict Rubin 1981).
        random_state: Optional integer seed or NumPy Generator for reproducible resampling.
        name: Optional custom display name for the estimator.

    Attributes:
        estimates_: Fitted Polars `DataFrame` containing `[*strata_cols, "target", "event", "n", "estimate", "lower", "upper"]`.
        train_draws_: Optional `(n_resamples, n_obs)` Dirichlet draw matrix when `method="bayesian_bootstrap"`.

    Examples:
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.estimators import UnpooledEstimator
        >>> from seroepi.examples import KlebnetNeonatalSepsis
        >>> ds = KlebnetNeonatalSepsis.load()
        >>> agg = Aggregator(target=ds.var("K_locus"), strata=ds.var("Country")).fit(ds)
        >>> est = UnpooledEstimator(method="wilson", alpha=0.05).fit(agg)
        >>> est.ranked_targets(top_n=3)
        ['KL102', 'KL2', 'KL25']
    """

    model_family: str = "Unpooled"
    is_bayesian: bool = False
    ci_prefix: str = "CI"

    def __init__(
        self,
        method: Literal[
            "wilson",
            "clopper_pearson",
            "agresti_coull",
            "wald",
            "jeffreys",
            "bootstrap",
            "bayesian_bootstrap",
        ] = "wilson",
        alpha: float = 0.05,
        n_resamples: int = 1000,
        prior_concentration: float = 0.5,
        random_state: int | np.random.Generator | None = None,
        name: str | None = None,
    ) -> None:
        self.method = method
        self.alpha = alpha
        self.n_resamples = n_resamples
        self.prior_concentration = prior_concentration
        self.random_state = random_state
        self.name = name

    @property
    def _method_key(self) -> str:
        return self.method.lower() if isinstance(self.method, str) else str(self.method).lower()

    @property
    def is_bootstrapped(self) -> bool:
        """Whether this estimator uses resampling bootstrap inference."""
        return self._method_key in {"bootstrap", "bayesian_bootstrap"}

    @property
    def inference_type(self) -> Literal["analytical", "resampling"]:
        """Inference paradigm: analytical asymptotic/exact vs non-parametric resampling."""
        return "resampling" if self.is_bootstrapped else "analytical"

    @property
    def display_name(self) -> str:
        """Formatted display name for unpooled model."""
        if self.name:
            return self.name
        if self._method_key == "bayesian_bootstrap":
            return "Bayesian Bootstrap"
        if self._method_key == "bootstrap":
            return "Bootstrap"
        if self._method_key in {"wilson", "jeffreys", "wald", "clopper_pearson", "agresti_coull"}:
            return self._method_key.replace("_", "-").title()
        return self.model_family

    def describe_interval(self) -> str:
        """Descriptive string for uncertainty intervals."""
        if self._method_key == "bayesian_bootstrap":
            return f"{self.ci_percentage}% Bayesian bootstrap {self.ci_plural} ({self.n_resamples:,} resamples)"
        if self._method_key == "bootstrap":
            return f"{self.ci_percentage}% bootstrap {self.ci_plural} ({self.n_resamples:,} resamples)"
        method_title = self._method_key.replace("_", "-").title()
        return f"{self.ci_percentage}% {method_title} {self.ci_plural}"

    def _row_draws_for_frame(self, df: pl.DataFrame) -> np.ndarray | None:
        """Returns `(n_resamples, len(df))` Dirichlet bootstrap draws when `method == 'bayesian_bootstrap'`."""
        if self._method_key != "bayesian_bootstrap":
            return None
        if df.is_empty():
            return np.zeros((self.n_resamples, 0), dtype=float)
        if df is self.estimates_ and self.train_draws_ is not None:
            return self.train_draws_
        if self.is_compositional_:
            return bayesian_bootstrap_compositional_draws(
                df,
                self.strata_cols,
                n_resamples=self.n_resamples,
                prior_concentration=self.prior_concentration,
                random_state=self.random_state,
            )
        return bayesian_bootstrap_row_draws(
            df["event"].to_numpy().astype(float),
            df["n"].to_numpy().astype(float),
            n_resamples=self.n_resamples,
            prior_concentration=self.prior_concentration,
            random_state=self.random_state,
        )

    def fit(self, X: Aggregator | Any, y: Any = None, sample_weight: Any = None) -> Self:
        """Fits the unpooled prevalence estimator on a fitted `Aggregator`.

        Args:
            X: Input fitted `Aggregator` (or `TransmissionEstimator`) containing `event` and `n` columns.
            y: Ignored; present for scikit-learn pipeline compatibility.
            sample_weight: Ignored; present for scikit-learn pipeline compatibility.

        Returns:
            Fitted `self` with populated `estimates_`.
        """
        df = self._bind_aggregator(X)
        k = df["event"].to_numpy().astype(float)
        n = df["n"].to_numpy().astype(float)

        if self._method_key == "bayesian_bootstrap" and self.is_compositional_:
            self.train_draws_ = bayesian_bootstrap_compositional_draws(
                df,
                X.strata_cols,
                n_resamples=self.n_resamples,
                prior_concentration=self.prior_concentration,
                random_state=self.random_state,
            )
            safe_n = np.where(n > 0, n, 1.0)
            p = np.where(n > 0, k / safe_n, 0.0)
            _, lower, upper = self._summarize_draws(self.train_draws_, float(self.alpha))
            lower = np.where(n > 0, np.clip(lower, 0.0, 1.0), 0.0)
            upper = np.where(n > 0, np.clip(upper, 0.0, 1.0), 0.0)
        else:
            self.train_draws_ = None
            p, lower, upper = self._compute_ci(k, n)

        return self._finalize_fit(df, p, lower, upper, method=self.method)

    def predict(
        self, X: Aggregator | Any, return_interval: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts point prevalence proportions or `(estimate, lower, upper)` intervals for `X`.

        Args:
            X: Fitted `Aggregator` instance to evaluate.
            return_interval: If `True`, returns `(estimate, lower, upper)` 1D NumPy arrays;
                otherwise returns only the 1D `estimate` array.

        Returns:
            1D NumPy array of point estimates, or a 3-tuple of 1D NumPy arrays `(estimate, lower, upper)`.
        """
        fast = self._predict_training_fastpath(X, return_interval)
        if fast is not None:
            return fast

        assert self.estimates_ is not None
        assert self.aggregator_ is not None

        df = X.data_
        total_n = self.aggregator_.data_["n"].sum()
        total_events = self.aggregator_.data_["event"].sum()
        n_val = float(total_n) if total_n is not None else 0.0
        event_val = float(total_events) if total_events is not None else 0.0
        global_p = (event_val / n_val) if n_val > 0.0 else 0.0

        if "lower" in self.estimates_.columns:
            lower_val = self.estimates_["lower"].mean()
            global_lower: float = float(lower_val) if isinstance(lower_val, (int, float)) else global_p
        else:
            global_lower = global_p
        if "upper" in self.estimates_.columns:
            upper_val = self.estimates_["upper"].mean()
            global_upper: float = float(upper_val) if isinstance(upper_val, (int, float)) else global_p
        else:
            global_upper = global_p

        join_keys = [c for c in self.strata_cols if c in df.columns]
        if "target" in df.columns and "target" in self.estimates_.columns:
            join_keys.append("target")

        if join_keys:
            cols_to_join = list(dict.fromkeys(join_keys + ["estimate", "lower", "upper"]))
            merged = df.join(self.estimates_.select(cols_to_join), on=join_keys, how="left")
            p_arr = merged["estimate"].fill_null(global_p).to_numpy()
            lower = merged["lower"].fill_null(global_lower).to_numpy()
            upper = merged["upper"].fill_null(global_upper).to_numpy()
        else:
            p_arr = np.full(len(df), global_p)
            lower = np.full(len(df), global_lower)
            upper = np.full(len(df), global_upper)

        if return_interval:
            return p_arr, lower, upper
        return p_arr

    def _compute_ci(self, k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = float(self.alpha)
        key = self._method_key
        if key == "bayesian_bootstrap":
            return bayesian_bootstrap_interval(
                k,
                n,
                alpha,
                n_resamples=self.n_resamples,
                prior_concentration=self.prior_concentration,
                random_state=self.random_state,
            )
        if key == "bootstrap":
            return bootstrap_interval(
                k,
                n,
                alpha,
                n_resamples=self.n_resamples,
                random_state=self.random_state,
            )
        if key == "clopper_pearson":
            return clopper_pearson_interval(k, n, alpha)
        if key == "agresti_coull":
            return agresti_coull_interval(k, n, alpha)
        if key == "wald":
            return wald_interval(k, n, alpha)
        if key == "jeffreys":
            return jeffreys_interval(k, n, alpha)
        return wilson_interval(k, n, alpha)

    def _attach_ci_from_counts(self, df: pl.DataFrame, est_col: str) -> pl.DataFrame:
        k = df["event"].to_numpy().astype(float)
        n = df["n"].to_numpy().astype(float)
        p, lower, upper = self._compute_ci(k, n)
        return df.with_columns(
            pl.Series(est_col, np.nan_to_num(p, nan=0.0)),
            pl.Series("lower", np.nan_to_num(lower, nan=0.0)),
            pl.Series("upper", np.nan_to_num(upper, nan=0.0)),
        )

    def _pool_strata_by_target(
        self,
        df: pl.DataFrame,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> pl.DataFrame:
        target_col = self.target_col
        est_col = self.estimate.column
        if weighting == "equal":
            agg_df = (
                df.with_columns(
                    _cell_p=pl.when(pl.col("n") > 0).then(pl.col("event") / pl.col("n")).otherwise(0.0),
                    _inv_n=pl.when(pl.col("n") > 0).then(1.0 / pl.col("n")).otherwise(0.0),
                )
                .group_by(target_col, maintain_order=True)
                .agg(
                    pl.col("event").sum().alias("event"),
                    pl.col("n").sum().alias("n"),
                    pl.col("_cell_p").mean().alias("_mean_p"),
                    pl.col("_inv_n").sum().alias("_sum_inv_n"),
                    pl.len().alias("_n_strata"),
                )
            )
            mean_p = agg_df["_mean_p"].to_numpy().astype(float)
            sum_inv_n = agg_df["_inv_n" if "_inv_n" in agg_df.columns else "_sum_inv_n"].to_numpy().astype(float)
            n_strata = agg_df["_n_strata"].to_numpy().astype(float)
            n_eff = np.where(sum_inv_n > 0.0, (n_strata * n_strata) / sum_inv_n, 0.0)
            k_eff = mean_p * n_eff
            _, lower, upper = self._compute_ci(k_eff, n_eff)
            return agg_df.select([target_col, "event", "n"]).with_columns(
                pl.Series(est_col, np.nan_to_num(mean_p, nan=0.0)),
                pl.Series("lower", np.nan_to_num(lower, nan=0.0)),
                pl.Series("upper", np.nan_to_num(upper, nan=0.0)),
            )
        agg_df = df.group_by(target_col, maintain_order=True).agg(
            pl.col("event").sum().alias("event"),
            pl.col("n").sum().alias("n"),
        )
        return self._attach_ci_from_counts(agg_df, est_col)

    def _summarize_strata_and_lump(
        self,
        df: pl.DataFrame,
        x_limit: int | None,
        other_label: str,
    ) -> tuple[pl.DataFrame, list[str]]:
        stratum_col = self.single_stratum_col
        est_col = self.estimate.column
        stratum_counts = self._aggregate_stratum_counts(df)
        stratum_summary = self._attach_ci_from_counts(stratum_counts, est_col).sort(
            [est_col, stratum_col], descending=[True, False]
        )
        sorted_x = [str(x) for x in stratum_summary[stratum_col].to_list()]

        if x_limit is None or len(sorted_x) <= x_limit:
            return stratum_summary, sorted_x

        top_x = sorted_x[: x_limit - 1]
        kept_df = stratum_summary.head(x_limit - 1)
        binned_df = stratum_summary.slice(x_limit - 1)

        other_counts = binned_df.select(
            pl.lit(other_label).alias(stratum_col),
            pl.col("event").sum().alias("event"),
            pl.col("n").sum().alias("n"),
        )
        other_df = self._attach_ci_from_counts(other_counts, est_col)
        top_df = pl.concat([kept_df, other_df], how="diagonal_relaxed")
        return top_df, [*top_x, other_label]

    def _compute_cumulative_coverage_columns(self, df_sorted: pl.DataFrame) -> pl.DataFrame:
        k = df_sorted["cum_event"].to_numpy().astype(float)
        n = df_sorted["max_n"].to_numpy().astype(float)
        if self._method_key == "bayesian_bootstrap" and self.is_compositional_:
            draws = bayesian_bootstrap_compositional_draws(
                df_sorted,
                (self.single_stratum_col,),
                n_resamples=self.n_resamples,
                prior_concentration=self.prior_concentration,
                random_state=self.random_state,
            )
            cum_draws = self._cumulative_draws_by_stratum(df_sorted, draws)
            safe_n = np.where(n > 0, n, 1.0)
            p = np.where(n > 0, k / safe_n, 0.0)
            _, lower, upper = self._summarize_draws(cum_draws, float(self.alpha))
            lower = np.where(n > 0, np.clip(lower, 0.0, 1.0), 0.0)
            upper = np.where(n > 0, np.clip(upper, 0.0, 1.0), 0.0)
        else:
            p, lower, upper = self._compute_ci(k, n)
        return df_sorted.with_columns(
            cum_estimate=pl.Series(np.nan_to_num(p, nan=0.0)),
            cum_lower=pl.Series(np.nan_to_num(lower, nan=0.0)),
            cum_upper=pl.Series(np.nan_to_num(upper, nan=0.0)),
        )
