"""Hierarchical Bayesian prevalence estimators via NumPyro NUTS MCMC and SVI."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from os import cpu_count
from typing import Any, ClassVar, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from jax import random
from numpyro import diagnostics, optim, set_host_device_count
from numpyro import distributions as dist
from numpyro import sample as samp
from numpyro.infer import (
    MCMC,
    NUTS,
    SVI,
    Trace_ELBO,
    autoguide,
    init_to_median,
    init_to_sample,
    init_to_uniform,
)
from scipy.special import expit
from shiny_kit.progress import current_progress
from sklearn.preprocessing import LabelEncoder

from seroepi.aggregate import Aggregator
from seroepi.estimators.base import BayesianEstimator

set_host_device_count(cpu_count() or 8)


class BaseHierarchicalEstimator(BayesianEstimator):
    """Abstract base class for hierarchical Bayesian prevalence estimators.

    Unifies data pre-processing, compound strata grouping, label encoding,
    1D observed-cell partial pooling, posterior derivation, and scikit-learn
    prediction for hierarchical models.

    Attributes:
        engine: Inference engine identifier (``"MCMC"`` or ``"SVI"``).
        estimate_method: Method label recorded in ``estimates_["method"]``.
        likelihood: Observation likelihood kernel (``"binomial"`` or ``"beta_binomial"``).
        samples_: Dictionary of posterior parameter draws extracted after fitting.
        extra_fields_: Dictionary of sampler diagnostic fields (MCMC only).
        losses_: 1D NumPy array of variational ELBO losses per optimization step (SVI only).
        encoders_: Mapping from target/stratum column names to fitted ``LabelEncoder`` instances.
        cell_index_: Mapping from ``(target_str, group_str)`` tuples to 1D cell indices.
    """

    engine: str | None = None
    estimate_method: str = "hierarchical"
    likelihood: Literal["binomial", "beta_binomial"]
    samples_: dict[str, np.ndarray] | None = None
    extra_fields_: dict[str, np.ndarray] | None = None
    losses_: np.ndarray | None = None
    encoders_: dict[str, LabelEncoder]
    cell_index_: dict[tuple[str, str], int]

    @staticmethod
    def _binomial_model(
        target_idx: Any,
        n: Any,
        event: Any = None,
        cell_idx: Any = None,
        *,
        n_targets: int,
        n_cells: int,
    ) -> None:
        """NumPyro model for hierarchical binomial prevalence with target-by-group partial pooling."""
        alpha: Any = samp("alpha", dist.Normal(0.0, 1.5))
        b_target: Any = samp("b_target", dist.Normal(0.0, 1.0).expand([n_targets]))
        sd_group: Any = samp("sd_group", dist.HalfNormal(1.0))
        z_group: Any = samp("z_group", dist.Normal(0.0, 1.0).expand([n_cells]))
        r_cell: Any = (z_group[cell_idx] if cell_idx is not None else z_group) * sd_group
        logit_p: Any = alpha + b_target[target_idx] + r_cell
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    @staticmethod
    def _beta_binomial_model(
        target_idx: Any,
        n: Any,
        event: Any = None,
        cell_idx: Any = None,
        *,
        n_targets: int,
        n_cells: int,
    ) -> None:
        """NumPyro model for overdispersed beta-binomial prevalence with target-by-group partial pooling."""
        alpha: Any = samp("alpha", dist.Normal(0.0, 1.5))
        b_target: Any = samp("b_target", dist.Normal(0.0, 1.0).expand([n_targets]))
        sd_group: Any = samp("sd_group", dist.HalfNormal(1.0))
        z_group: Any = samp("z_group", dist.Normal(0.0, 1.0).expand([n_cells]))
        r_cell: Any = (z_group[cell_idx] if cell_idx is not None else z_group) * sd_group
        logit_p: Any = alpha + b_target[target_idx] + r_cell
        p: Any = jax.nn.sigmoid(logit_p)
        phi: Any = samp("phi", dist.Exponential(0.1))
        a: Any = (p * phi) + 1e-4
        b: Any = ((1.0 - p) * phi) + 1e-4
        samp("obs", dist.BetaBinomial(a, b, total_count=n), obs=event)

    @staticmethod
    def _compute_prevalence_draws(
        alpha: np.ndarray,
        b_target: np.ndarray,
        z_group: np.ndarray,
        sd_group: np.ndarray,
        target_idx: np.ndarray,
        cell_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computes (n_samples, n_obs) posterior prevalence draws on the probability scale."""
        alpha_col = np.asarray(alpha, dtype=np.float32)
        if alpha_col.ndim == 1:
            alpha_col = alpha_col[:, None]
        sd_col = np.asarray(sd_group, dtype=np.float32)
        if sd_col.ndim == 1:
            sd_col = sd_col[:, None]

        b_mat = np.asarray(b_target, dtype=np.float32)[:, target_idx]
        z_mat = np.asarray(z_group, dtype=np.float32)

        if cell_idx is None:
            eff = z_mat * sd_col
        else:
            safe_idx = np.where(cell_idx >= 0, cell_idx, 0)
            known_mask = (cell_idx >= 0)[None, :]
            eff = np.where(known_mask, z_mat[:, safe_idx] * sd_col, np.float32(0.0))

        logit_p = alpha_col + b_mat + eff
        return np.asarray(expit(logit_p), dtype=np.float64)

    @staticmethod
    def _summarize_posterior_draws(
        p_draws: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes posterior mean and equal-tailed (1 - alpha) credible intervals in a single quantile pass."""
        return BaseHierarchicalEstimator._summarize_draws(p_draws, alpha)

    @staticmethod
    def _transform_safe(encoder: LabelEncoder, vals: np.ndarray) -> np.ndarray:
        """Vectorized transformation mapping out-of-vocabulary categories to classes_[0]."""
        mask = np.isin(vals, encoder.classes_)
        safe_vals = np.where(mask, vals, encoder.classes_[0])
        return np.asarray(encoder.transform(safe_vals), dtype=np.int32)

    def _get_model_fn(self, *, n_targets: int, n_cells: int) -> Callable[..., None]:
        base_fn = self._beta_binomial_model if self.likelihood == "beta_binomial" else self._binomial_model
        return partial(base_fn, n_targets=n_targets, n_cells=n_cells)

    @abstractmethod
    def _run_inference(self, jax_data: dict[str, Any], *, n_targets: int, n_cells: int) -> None:
        """Abstract inference method implemented by subclasses (MCMC or SVI)."""
        raise NotImplementedError

    def _extract_target_and_group_strings(self, df: pl.DataFrame) -> tuple[str, np.ndarray, np.ndarray]:
        """Extracts string arrays for target and compound/single stratum group columns."""
        df_mod = df
        strata_cols = [s.column for s in self.strata_]
        if len(strata_cols) > 1:
            matching_strata = [c for c in strata_cols if c in df.columns]
            if len(matching_strata) == len(strata_cols):
                df_mod = df_mod.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.String) for c in strata_cols], separator="__").alias(
                        "_compound_group"
                    )
                )
                group_col = "_compound_group"
            else:
                df_mod = df_mod.with_columns(_dummy_group=pl.lit("Global"))
                group_col = "_dummy_group"
        elif strata_cols and strata_cols[0] in df.columns:
            group_col = strata_cols[0]
        else:
            df_mod = df_mod.with_columns(_dummy_group=pl.lit("Global"))
            group_col = "_dummy_group"

        target_col = "target" if "target" in df_mod.columns else "_dummy_target"
        if "target" not in df_mod.columns:
            df_mod = df_mod.with_columns(_dummy_target=pl.lit(self.target_.title or "Default"))

        target_str = df_mod[target_col].cast(pl.String).to_numpy()
        group_str = df_mod[group_col].cast(pl.String).to_numpy()
        return target_col, target_str, group_str

    def _posterior_draws_for_frame(self, df: pl.DataFrame) -> np.ndarray:
        """Extracts (n_samples, len(df)) posterior prevalence draws for rows in df."""
        assert self.samples_ is not None
        if self.train_draws_ is not None and (
            df is self.estimates_ or (self.aggregator_ is not None and df is self.aggregator_.data_)
        ):
            return self.train_draws_

        target_col, target_str, group_str = self._extract_target_and_group_strings(df)
        cell_idx = np.fromiter(
            (self.cell_index_.get((str(t), str(g)), -1) for t, g in zip(target_str, group_str, strict=False)),
            dtype=np.int32,
            count=len(df),
        )

        if self.train_draws_ is not None and np.all(cell_idx >= 0):
            return self.train_draws_[:, cell_idx]

        target_idx = self._transform_safe(self.encoders_[target_col], target_str)
        return self._compute_prevalence_draws(
            self.samples_["alpha"],
            self.samples_["b_target"],
            self.samples_["z_group"],
            self.samples_["sd_group"],
            target_idx,
            cell_idx=cell_idx,
        )

    def _row_draws_for_frame(self, df: pl.DataFrame) -> np.ndarray | None:
        return self._posterior_draws_for_frame(df)

    def fit(self, X: Aggregator | Any, y: Any = None, sample_weight: Any = None) -> Self:
        """Fits the hierarchical Bayesian estimator on a fitted ``Aggregator``.

        Encodes target and stratum categories, runs NumPyro MCMC or SVI inference,
        and derives posterior mean prevalence and equal-tailed credible intervals.

        Args:
            X: Fitted ``Aggregator`` instance containing ``event`` and ``n`` columns.
            y: Ignored; present for scikit-learn pipeline compatibility.
            sample_weight: Ignored; present for scikit-learn pipeline compatibility.

        Returns:
            The fitted hierarchical estimator instance (``self``).

        Raises:
            TypeError: If ``X`` is not an ``Aggregator`` instance.
            RuntimeError: If ``X`` has not been fitted yet.
        """
        df = self._bind_aggregator(X)

        if X.is_compositional_ and not X.is_padded_:
            self._record_warning(
                "Fitting hierarchical Bayesian estimator on unpadded compositional Aggregator. "
                "Unobserved target categories in strata will be omitted from estimation, which may bias "
                "prevalence estimates. Consider aggregating with `pad_zeros=True`.",
                UserWarning,
                stacklevel=3,
            )

        target_col, target_str, group_str = self._extract_target_and_group_strings(df)
        group_col = (
            self.strata_[0].column
            if len(self.strata_) == 1
            else ("_compound_group" if len(self.strata_) > 1 else "_dummy_group")
        )

        target_le = LabelEncoder()
        target_idx_np = np.asarray(target_le.fit_transform(target_str), dtype=np.int32)
        group_le = LabelEncoder()
        group_le.fit(group_str)
        self.encoders_ = {target_col: target_le, group_col: group_le}

        self.cell_index_ = {}
        cell_idx_np = np.empty(len(df), dtype=np.int32)
        for idx, (t_val, g_val) in enumerate(zip(target_str, group_str, strict=False)):
            key = (str(t_val), str(g_val))
            existing = self.cell_index_.get(key)
            if existing is None:
                existing = len(self.cell_index_)
                self.cell_index_[key] = existing
            cell_idx_np[idx] = existing

        n_targets = len(target_le.classes_)
        n_cells = len(self.cell_index_)
        is_bijective = n_cells == len(df)

        prog = current_progress()
        prog.report(0.05, message=f"Fitting {self.display_name}", detail="Encoding strata and preparing JAX tensors...")

        jax_data: dict[str, Any] = {
            "target_idx": jnp.asarray(target_idx_np, dtype=jnp.int32),
            "n": jnp.asarray(df["n"].to_numpy(), dtype=jnp.int32),
            "event": jnp.asarray(df["event"].to_numpy(), dtype=jnp.int32),
        }
        if not is_bijective:
            jax_data["cell_idx"] = jnp.asarray(cell_idx_np, dtype=jnp.int32)

        self._run_inference(jax_data, n_targets=n_targets, n_cells=n_cells)
        assert self.samples_ is not None

        prog.report(0.92, message=f"Fitting {self.display_name}", detail="Computing posterior prevalence intervals...")
        self.train_draws_ = self._compute_prevalence_draws(
            self.samples_["alpha"],
            self.samples_["b_target"],
            self.samples_["z_group"],
            self.samples_["sd_group"],
            target_idx_np,
            cell_idx=None if is_bijective else cell_idx_np,
        )
        mean_p, lower_p, upper_p = self._summarize_draws(self.train_draws_, self.alpha)
        self._finalize_fit(df, mean_p, lower_p, upper_p, method=self.estimate_method)
        prog.report(1.0, message=f"Fitted {self.display_name}", detail="Posterior estimation complete")
        return self

    def predict(
        self, X: Aggregator | Any, return_interval: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts posterior mean prevalence (and optional credible intervals) for ``X``.

        Args:
            X: Fitted ``Aggregator`` instance to predict on.
            return_interval: Whether to also return ``(lower, upper)`` credible interval arrays.

        Returns:
            A 1D float64 NumPy array of posterior mean prevalence estimates, or a 3-tuple
            ``(estimate, lower, upper)`` when ``return_interval=True``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
            TypeError: If ``X`` is not a fitted ``Aggregator``.
        """
        fast = self._predict_training_fastpath(X, return_interval)
        if fast is not None:
            return fast

        p_draws = self._posterior_draws_for_frame(X.data_)
        mean_p, lower_p, upper_p = self._summarize_draws(p_draws, self.alpha)

        est = np.asarray(mean_p, dtype=np.float64)
        if return_interval:
            return est, np.asarray(lower_p, dtype=np.float64), np.asarray(upper_p, dtype=np.float64)
        return est

    def _pool_strata_by_target(
        self,
        df: pl.DataFrame,
        *,
        weighting: Literal["pooled", "equal"] = "pooled",
    ) -> pl.DataFrame:
        target_col = self.target_col
        est_col = self.estimate.column
        p_draws = self._posterior_draws_for_frame(df)
        agg_df, _, pooled_draws = self._pool_draws_by_target(df, p_draws, target_col, weighting=weighting)
        mean_p, lower_p, upper_p = self._summarize_draws(pooled_draws, self.alpha)
        return agg_df.with_columns(
            pl.Series(est_col, mean_p),
            pl.Series("lower", lower_p),
            pl.Series("upper", upper_p),
        )

    def _summarize_strata_and_lump(
        self,
        df: pl.DataFrame,
        x_limit: int | None,
        other_label: str,
    ) -> tuple[pl.DataFrame, list[str]]:
        stratum_col = self.single_stratum_col
        est_col = self.estimate.column
        stratum_counts = self._aggregate_stratum_counts(df)

        p_draws = self._posterior_draws_for_frame(df)
        stratum_vals = df[stratum_col].cast(pl.String).to_numpy()
        strata_order = stratum_counts[stratum_col].cast(pl.String).to_list()

        stratum_draws = np.zeros((p_draws.shape[0], len(strata_order)), dtype=float)
        for idx, s_val in enumerate(strata_order):
            mask = stratum_vals == s_val
            stratum_draws[:, idx] = np.clip(np.sum(p_draws[:, mask], axis=1), 0.0, 1.0)

        mean_p, lower_p, upper_p = self._summarize_draws(stratum_draws, self.alpha)
        stratum_summary = stratum_counts.with_columns(
            pl.Series(est_col, mean_p),
            pl.Series("lower", lower_p),
            pl.Series("upper", upper_p),
            pl.Series("_orig_idx", list(range(len(strata_order)))),
        ).sort([est_col, stratum_col], descending=[True, False])
        sorted_indices = stratum_summary["_orig_idx"].to_numpy()
        stratum_summary = stratum_summary.drop("_orig_idx")
        sorted_draws = stratum_draws[:, sorted_indices]
        sorted_x = [str(x) for x in stratum_summary[stratum_col].to_list()]

        if x_limit is None or len(sorted_x) <= x_limit:
            return stratum_summary, sorted_x

        top_x = sorted_x[: x_limit - 1]
        kept_df = stratum_summary.head(x_limit - 1)
        binned_df = stratum_summary.slice(x_limit - 1)
        binned_draws = sorted_draws[:, x_limit - 1 :]

        w = binned_df["n"].to_numpy().astype(float)
        w_sum = float(np.sum(w))
        other_draw = (binned_draws @ (w / w_sum)) if w_sum > 0.0 else np.mean(binned_draws, axis=1)
        mean_o, low_o, up_o = self._summarize_draws(other_draw[:, None], self.alpha)

        other_df = binned_df.select(
            pl.lit(other_label).alias(stratum_col),
            pl.col("event").sum().alias("event"),
            pl.col("n").sum().alias("n"),
        ).with_columns(
            pl.Series(est_col, mean_o),
            pl.Series("lower", low_o),
            pl.Series("upper", up_o),
        )
        top_df = pl.concat([kept_df, other_df], how="diagonal_relaxed")
        return top_df, [*top_x, other_label]

    def _compute_cumulative_coverage_columns(self, df_sorted: pl.DataFrame) -> pl.DataFrame:
        p_draws = self._posterior_draws_for_frame(df_sorted)
        cum_draws = self._cumulative_draws_by_stratum(df_sorted, p_draws)
        mean_p, lower_p, upper_p = self._summarize_draws(cum_draws, self.alpha)
        return df_sorted.with_columns(
            cum_estimate=pl.Series(mean_p),
            cum_lower=pl.Series(lower_p),
            cum_upper=pl.Series(upper_p),
        )


class HierarchicalMCMCEstimator(BaseHierarchicalEstimator):
    """Hierarchical Bayesian model for prevalence estimation fitted via NumPyro NUTS MCMC.

    Models target effects and random group interactions, estimating full posterior distributions
    and offering formal MCMC convergence diagnostics (R-hat, effective sample size).

    Args:
        num_samples: Number of posterior samples to draw per chain.
        num_chains: Number of parallel MCMC chains.
        num_warmup: Number of warmup/burn-in steps.
        likelihood: Binomial or overdispersed beta_binomial likelihood kernel.
        alpha: Significance level for credible intervals (default 0.05 for 95% CrIs).
        chain_method: MCMC chain execution strategy ('auto', 'parallel', 'vectorized', or 'sequential').
        target_accept_prob: NUTS target acceptance probability (default 0.8).
        max_tree_depth: NUTS maximum tree depth (default 10).
        init_strategy: Parameter initialization strategy ('median', 'uniform', 'sample', or custom callable).
        random_state: Random seed for reproducibility.
        name: Optional custom display name for the estimator.

    Attributes:
        samples_: Dictionary mapping model parameter names to NumPy arrays of posterior draws.
        extra_fields_: Dictionary of NUTS sampler diagnostic fields (e.g., ``diverging``).
        estimates_: Polars ``DataFrame`` containing posterior mean prevalence and credible intervals.

    Examples:
        >>> import polars as pl
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.estimators.hierarchical import HierarchicalMCMCEstimator
        >>> from seroepi.variables import Identifier, Spatial, Trait
        >>> ds = Dataset(
        ...     pl.DataFrame({"id": ["s1", "s2", "s3"], "region": ["A", "A", "B"], "k": ["K1", "K2", "K1"]}),
        ...     variables={"id": Identifier("id"), "region": Spatial("region"), "k": Trait("k", compositional=True)},
        ... )
        >>> agg = Aggregator(target=ds.var("k"), strata=[ds.var("region")], pad_zeros=True).fit(ds)
        >>> est = HierarchicalMCMCEstimator(num_warmup=50, num_samples=50, num_chains=1).fit(agg)
        >>> est.is_fitted_
        True
    """

    engine: str | None = "MCMC"
    estimate_method: str = "mcmc_nuts"
    _MCMC_CACHE: ClassVar[dict[tuple[Any, ...], MCMC]] = {}

    def __init__(
        self,
        num_samples: int = 1500,
        num_chains: int = 4,
        num_warmup: int = 1000,
        likelihood: Literal["binomial", "beta_binomial"] = "binomial",
        alpha: float = 0.05,
        chain_method: Literal["auto", "parallel", "vectorized", "sequential"] = "auto",
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        init_strategy: Literal["median", "uniform", "sample"] | Callable[..., Any] = "median",
        random_state: int | None = 42,
        name: str | None = None,
    ) -> None:
        """Initializes the MCMC hierarchical Bayesian prevalence estimator.

        Args:
            num_samples: Number of posterior samples to draw per chain.
            num_chains: Number of parallel MCMC chains.
            num_warmup: Number of warmup/burn-in steps.
            likelihood: Binomial or overdispersed beta_binomial likelihood kernel.
            alpha: Significance level for credible intervals (default 0.05 for 95% CrIs).
            chain_method: MCMC chain execution strategy ('auto', 'parallel', 'vectorized', or 'sequential').
            target_accept_prob: NUTS target acceptance probability (default 0.8).
            max_tree_depth: NUTS maximum tree depth (default 10).
            init_strategy: Parameter initialization strategy ('median', 'uniform', 'sample', or custom callable).
            random_state: Random seed for reproducibility.
            name: Optional custom display name for the estimator.
        """
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.likelihood = likelihood
        self.alpha = alpha
        self.chain_method = chain_method
        self.target_accept_prob = target_accept_prob
        self.max_tree_depth = max_tree_depth
        self.init_strategy = init_strategy
        self.random_state = random_state
        self.name = name

    def _resolve_chain_method(self) -> str:
        """Determines the optimal chain execution method based on active JAX devices."""
        if self.chain_method != "auto":
            return self.chain_method
        if self.num_chains <= 1:
            return "sequential"
        if jax.device_count() >= self.num_chains:
            return "parallel"
        return "vectorized"

    def _resolve_init_strategy(self) -> tuple[Callable[..., Any], str | None]:
        """Resolves the NumPyro initialization function and its cacheable string key."""
        if callable(self.init_strategy):
            return self.init_strategy, None
        if self.init_strategy == "sample":
            return init_to_sample, "sample"
        if self.init_strategy == "uniform":
            return init_to_uniform, "uniform"
        return init_to_median, "median"

    def _get_or_create_mcmc(self, *, n_targets: int, n_cells: int, has_cell_idx: bool) -> MCMC:
        """Retrieves a cached compiled MCMC runner or creates a new one with jit_model_args=True."""
        resolved_chain_method = self._resolve_chain_method()
        init_strat, init_key = self._resolve_init_strategy()
        cache_key = (
            self.likelihood,
            n_targets,
            n_cells,
            has_cell_idx,
            self.num_warmup,
            self.num_samples,
            self.num_chains,
            resolved_chain_method,
            float(self.target_accept_prob),
            int(self.max_tree_depth),
            init_key,
        )
        if init_key is not None and cache_key in self._MCMC_CACHE:
            return self._MCMC_CACHE[cache_key]

        model_fn = self._get_model_fn(n_targets=n_targets, n_cells=n_cells)
        kernel = NUTS(
            model_fn,
            init_strategy=init_strat,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False,
            chain_method=resolved_chain_method,
            jit_model_args=True,
        )
        if init_key is not None:
            self._MCMC_CACHE[cache_key] = mcmc
        return mcmc

    def _run_inference(self, jax_data: dict[str, Any], *, n_targets: int, n_cells: int) -> None:
        prog = current_progress()
        seed = self.random_state if self.random_state is not None else 42
        rng_key = random.PRNGKey(seed)
        mcmc = self._get_or_create_mcmc(
            n_targets=n_targets,
            n_cells=n_cells,
            has_cell_idx="cell_idx" in jax_data,
        )
        prog.report(
            0.15,
            message=f"Fitting {self.display_name}",
            detail=f"Running NUTS ({self.num_warmup:,} warmup + {self.num_samples:,} samples × {self.num_chains} chains)...",
            force=True,
        )
        mcmc.run(rng_key, **jax_data)
        prog.report(
            0.86, message=f"Fitting {self.display_name}", detail="Extracting MCMC posterior samples...", force=True
        )
        self.samples_ = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        self.extra_fields_ = {k: np.asarray(v) for k, v in mcmc.get_extra_fields().items()}

    def diagnostics(self) -> pl.DataFrame:
        """Returns MCMC convergence diagnostics (R-hat, ESS, mean, std) as a Polars ``DataFrame``.

        Returns:
            A Polars ``DataFrame`` with one row per model parameter and sampler diagnostic statistic.

        Raises:
            RuntimeError: If the estimator has not been fitted or no posterior samples are available.
        """
        self.check_is_fitted()
        if not self.samples_:
            raise RuntimeError("No posterior samples found.")

        summary_dict = diagnostics.summary(self.samples_, prob=0.95, group_by_chain=False)

        rows: list[dict[str, Any]] = []
        for param, stats_dict in summary_dict.items():
            np_stats = {k: np.asarray(v, dtype=float) for k, v in stats_dict.items()}
            param_shape = np_stats["mean"].shape

            if len(param_shape) == 0:
                row: dict[str, Any] = {"Parameter": param}
                row.update({k: float(v) for k, v in np_stats.items()})
                rows.append(row)
            else:
                it = np.nditer(np.empty(param_shape), flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    idx_str = ",".join(map(str, idx))
                    row = {"Parameter": f"{param}[{idx_str}]"}
                    row.update({k: float(v[idx]) for k, v in np_stats.items()})
                    rows.append(row)

        if self.extra_fields_ is not None:
            for field, values in self.extra_fields_.items():
                val_array = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "Parameter": f"sampler_{field}",
                        "mean": float(np.mean(val_array)),
                        "std": float(np.std(val_array)),
                        "sum": float(np.sum(val_array)),
                    }
                )

        return pl.DataFrame(rows)


class HierarchicalSVIEstimator(BaseHierarchicalEstimator):
    """Hierarchical Bayesian model for prevalence estimation fitted via fast Stochastic Variational Inference (SVI).

    Uses an automatic variational guide (`AutoNormal`) and Adam optimization to achieve fast approximate
    posterior draws on large datasets without full MCMC simulation.

    Args:
        num_steps: Number of optimization steps for variational ELBO.
        learning_rate: Step size for Adam optimizer.
        num_samples: Number of posterior predictive draws from the fitted guide.
        likelihood: Binomial or overdispersed beta_binomial likelihood kernel.
        alpha: Significance level for credible intervals (default 0.05 for 95% CrIs).
        random_state: Random seed for reproducibility.
        name: Optional custom display name for the estimator.

    Attributes:
        losses_: 1D NumPy array of variational ELBO loss values recorded at each optimization step.
        samples_: Dictionary mapping model parameter names to NumPy arrays of variational posterior draws.
        estimates_: Polars ``DataFrame`` containing posterior mean prevalence and credible intervals.

    Examples:
        >>> import polars as pl
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.estimators.hierarchical import HierarchicalSVIEstimator
        >>> from seroepi.variables import Identifier, Spatial, Trait
        >>> ds = Dataset(
        ...     pl.DataFrame({"id": ["s1", "s2", "s3"], "region": ["A", "A", "B"], "k": ["K1", "K2", "K1"]}),
        ...     variables={"id": Identifier("id"), "region": Spatial("region"), "k": Trait("k", compositional=True)},
        ... )
        >>> agg = Aggregator(target=ds.var("k"), strata=[ds.var("region")], pad_zeros=True).fit(ds)
        >>> est = HierarchicalSVIEstimator(num_steps=200, num_samples=100).fit(agg)
        >>> est.diagnostics().columns
        ['step', 'elbo_loss']
    """

    engine: str | None = "SVI"
    estimate_method: str = "svi"

    def __init__(
        self,
        num_steps: int = 3000,
        learning_rate: float = 0.01,
        num_samples: int = 1500,
        likelihood: Literal["binomial", "beta_binomial"] = "binomial",
        alpha: float = 0.05,
        random_state: int | None = 42,
        name: str | None = None,
    ) -> None:
        """Initializes the SVI hierarchical Bayesian prevalence estimator.

        Args:
            num_steps: Number of optimization steps for variational ELBO.
            learning_rate: Step size for Adam optimizer.
            num_samples: Number of posterior predictive draws from the fitted guide.
            likelihood: Binomial or overdispersed beta_binomial likelihood kernel.
            alpha: Significance level for credible intervals (default 0.05 for 95% CrIs).
            random_state: Random seed for reproducibility.
            name: Optional custom display name for the estimator.
        """
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.likelihood = likelihood
        self.alpha = alpha
        self.random_state = random_state
        self.name = name

    def _run_inference(self, jax_data: dict[str, Any], *, n_targets: int, n_cells: int) -> None:
        prog = current_progress()
        seed = self.random_state if self.random_state is not None else 42
        rng_key = random.PRNGKey(seed)
        opt_key, pred_key = random.split(rng_key)

        model_fn = self._get_model_fn(n_targets=n_targets, n_cells=n_cells)
        guide = autoguide.AutoNormal(model_fn)
        optimizer = optim.Adam(step_size=self.learning_rate)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

        if not prog.is_active or self.num_steps <= 100:
            svi_result = svi.run(opt_key, num_steps=self.num_steps, progress_bar=False, **jax_data)
            self.losses_ = np.asarray(svi_result.losses, dtype=float)
            svi_params = svi_result.params
        else:
            prog.report(
                0.10, message=f"Fitting {self.display_name}", detail="Initializing variational guide...", force=True
            )
            svi_state = svi.init(opt_key, **jax_data)
            n_chunks = min(15, max(2, self.num_steps // 100))
            chunk_size = self.num_steps // n_chunks
            rem_steps = self.num_steps % n_chunks

            def _step_fn(state: Any, _: Any) -> tuple[Any, Any]:
                return svi.update(state, **jax_data)

            run_chunk = jax.jit(lambda st: jax.lax.scan(_step_fn, st, None, length=chunk_size))
            loss_parts: list[np.ndarray] = []
            done = 0
            for _ in range(n_chunks):
                svi_state, chunk_losses = run_chunk(svi_state)
                c_np = np.asarray(chunk_losses, dtype=float)
                loss_parts.append(c_np)
                done += chunk_size
                last_loss = float(c_np[-1]) if len(c_np) > 0 else 0.0
                prog.report(
                    0.10 + 0.76 * (done / self.num_steps),
                    message=f"Fitting {self.display_name}",
                    detail=f"Step {done:,}/{self.num_steps:,} (ELBO: {last_loss:,.1f})",
                )
            if rem_steps > 0:
                svi_state, rem_losses = jax.lax.scan(_step_fn, svi_state, None, length=rem_steps)
                loss_parts.append(np.asarray(rem_losses, dtype=float))
            self.losses_ = np.concatenate(loss_parts) if loss_parts else np.array([], dtype=float)
            svi_params = svi.get_params(svi_state)

        prog.report(
            0.88, message=f"Fitting {self.display_name}", detail=f"Sampling {self.num_samples:,} posterior draws..."
        )
        posterior_draws = guide.sample_posterior(
            pred_key,
            svi_params,
            sample_shape=(self.num_samples,),
        )
        self.samples_ = {k: np.asarray(v) for k, v in posterior_draws.items()}

    def diagnostics(self) -> pl.DataFrame:
        """Returns SVI optimization ELBO loss history as a formatted Polars ``DataFrame``.

        Returns:
            A Polars ``DataFrame`` with columns ``["step", "elbo_loss"]``.

        Raises:
            RuntimeError: If the estimator has not been fitted or no loss history is available.
        """
        self.check_is_fitted()
        if not hasattr(self, "losses_") or self.losses_ is None:
            raise RuntimeError("No optimization loss history found.")
        return pl.DataFrame(
            {
                "step": np.arange(len(self.losses_)),
                "elbo_loss": np.asarray(self.losses_, dtype=float),
            }
        )
