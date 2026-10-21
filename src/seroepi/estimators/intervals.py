"""Vectorized analytical, bootstrap, and Dirichlet Bayesian bootstrap confidence/credible interval functions."""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from shiny_kit.progress import current_progress


def wilson_interval(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized Wilson score binomial confidence intervals.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CIs).

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    safe_n = np.where(n > 0, n, 1)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    denom = safe_n + z2
    p_tilde = (k + 0.5 * z2) / denom
    margin = (z / denom) * np.sqrt(safe_n * p_hat * (1.0 - p_hat) + 0.25 * z2)
    lower = np.where(n > 0, np.clip(p_tilde - margin, 0.0, 1.0), 0.0)
    upper = np.where(n > 0, np.clip(p_tilde + margin, 0.0, 1.0), 0.0)
    return p_hat, lower, upper


def clopper_pearson_interval(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized exact Clopper-Pearson binomial confidence intervals via the Beta distribution.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CIs).

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    safe_n = np.where(n > 0, n, 1)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    lower = np.zeros_like(p_hat, dtype=float)
    upper = np.ones_like(p_hat, dtype=float)

    mask_k_gt_0 = (k > 0) & (n > 0)
    if np.any(mask_k_gt_0):
        lower[mask_k_gt_0] = stats.beta.ppf(alpha / 2.0, k[mask_k_gt_0], safe_n[mask_k_gt_0] - k[mask_k_gt_0] + 1)

    mask_k_lt_n = (k < safe_n) & (n > 0)
    if np.any(mask_k_lt_n):
        upper[mask_k_lt_n] = stats.beta.ppf(1.0 - alpha / 2.0, k[mask_k_lt_n] + 1, safe_n[mask_k_lt_n] - k[mask_k_lt_n])

    lower = np.where(n > 0, lower, 0.0)
    upper = np.where(n > 0, upper, 0.0)
    return p_hat, lower, upper


def agresti_coull_interval(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized Agresti-Coull binomial confidence intervals.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CIs).

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    safe_n = np.where(n > 0, n, 1)
    n_tilde = safe_n + z2
    k_tilde = k + 0.5 * z2
    p_tilde = k_tilde / n_tilde
    margin = z * np.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    lower = np.where(n > 0, np.clip(p_tilde - margin, 0.0, 1.0), 0.0)
    upper = np.where(n > 0, np.clip(p_tilde + margin, 0.0, 1.0), 0.0)
    return p_hat, lower, upper


def wald_interval(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized asymptotic Wald binomial confidence intervals.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CIs).

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    safe_n = np.where(n > 0, n, 1)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    margin = z * np.sqrt(np.where(n > 0, p_hat * (1.0 - p_hat) / safe_n, 0.0))
    lower = np.where(n > 0, np.clip(p_hat - margin, 0.0, 1.0), 0.0)
    upper = np.where(n > 0, np.clip(p_hat + margin, 0.0, 1.0), 0.0)
    return p_hat, lower, upper


def jeffreys_interval(k: np.ndarray, n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized Jeffreys equal-tailed credible intervals with a ``Beta(0.5, 0.5)`` prior.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CrIs).

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    safe_n = np.where(n > 0, n, 1)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    lower = np.zeros_like(p_hat, dtype=float)
    upper = np.ones_like(p_hat, dtype=float)

    mask_k_gt_0 = (k > 0) & (n > 0)
    if np.any(mask_k_gt_0):
        lower[mask_k_gt_0] = stats.beta.ppf(
            alpha / 2.0, k[mask_k_gt_0] + 0.5, safe_n[mask_k_gt_0] - k[mask_k_gt_0] + 0.5
        )

    mask_k_lt_n = (k < safe_n) & (n > 0)
    if np.any(mask_k_lt_n):
        upper[mask_k_lt_n] = stats.beta.ppf(
            1.0 - alpha / 2.0, k[mask_k_lt_n] + 0.5, safe_n[mask_k_lt_n] - k[mask_k_lt_n] + 0.5
        )

    lower = np.where(n > 0, lower, 0.0)
    upper = np.where(n > 0, upper, 0.0)
    return p_hat, lower, upper


def bootstrap_interval(
    k: np.ndarray,
    n: np.ndarray,
    alpha: float,
    n_resamples: int = 1000,
    random_state: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes vectorized parametric binomial simulation bootstrap confidence intervals.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CIs).
        n_resamples: Number of bootstrap Monte Carlo resamples to draw per row.
        random_state: Optional integer seed or NumPy ``Generator`` for reproducible simulation.

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.
    """
    safe_n = np.where(n > 0, n, 1)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    simulations = np.random.default_rng(random_state).binomial(safe_n.astype(int), p_hat, size=(n_resamples, len(k)))
    sim_p = simulations / safe_n
    lower, upper = np.percentile(sim_p, [100 * (alpha / 2.0), 100 * (1.0 - alpha / 2.0)], axis=0)
    return p_hat, np.where(n > 0, lower, 0.0), np.where(n > 0, upper, 0.0)


def bayesian_bootstrap_row_draws(
    k: np.ndarray,
    n: np.ndarray,
    n_resamples: int = 1000,
    prior_concentration: float = 0.5,
    random_state: Any = None,
) -> np.ndarray:
    """Draws ``(n_resamples, len(k))`` 2-component Beta/Dirichlet proportions via ``scipy.stats.dirichlet.rvs``.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        n_resamples: Number of posterior Dirichlet draws per row.
        prior_concentration: Symmetric Dirichlet prior concentration added to positive and negative counts.
        random_state: Optional integer seed or NumPy ``Generator`` for reproducibility.

    Returns:
        A 2D float64 NumPy array of shape ``(n_resamples, len(k))`` containing posterior prevalence draws.

    Raises:
        ValueError: If ``prior_concentration < 0.0``.
    """
    if prior_concentration < 0.0:
        raise ValueError(f"prior_concentration must be >= 0.0, got {prior_concentration}.")
    rng = np.random.default_rng(random_state)
    draws = np.zeros((n_resamples, len(k)), dtype=float)
    for idx in range(len(k)):
        n_i = float(n[idx])
        if n_i <= 0.0:
            continue
        k_i = float(np.clip(k[idx], 0.0, n_i))
        a1 = k_i + prior_concentration
        a2 = (n_i - k_i) + prior_concentration
        if a1 <= 0.0:
            draws[:, idx] = 0.0
        elif a2 <= 0.0:
            draws[:, idx] = 1.0
        else:
            draws[:, idx] = stats.dirichlet.rvs([a1, a2], size=n_resamples, random_state=rng)[:, 0]
    return draws


def bayesian_bootstrap_interval(
    k: np.ndarray,
    n: np.ndarray,
    alpha: float,
    n_resamples: int = 1000,
    prior_concentration: float = 0.5,
    random_state: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Bayesian bootstrap intervals sampled directly via ``scipy.stats.dirichlet.rvs``.

    Args:
        k: 1D NumPy array of observed event counts.
        n: 1D NumPy array of total sample denominators.
        alpha: Two-sided significance level in ``(0, 1)`` (e.g., ``0.05`` for 95% CrIs).
        n_resamples: Number of posterior Dirichlet draws per row.
        prior_concentration: Symmetric Dirichlet prior concentration parameter ``>= 0.0``.
        random_state: Optional integer seed or NumPy ``Generator`` for reproducibility.

    Returns:
        A 3-tuple ``(p_hat, lower, upper)`` of 1D float64 NumPy arrays clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``prior_concentration < 0.0``.
    """
    safe_n = np.where(n > 0, n, 1.0)
    p_hat = np.where(n > 0, k / safe_n, 0.0)
    if len(k) == 0:
        return p_hat, np.zeros_like(p_hat), np.zeros_like(p_hat)
    draws = bayesian_bootstrap_row_draws(
        k,
        n,
        n_resamples=n_resamples,
        prior_concentration=prior_concentration,
        random_state=random_state,
    )
    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)
    lower, upper = np.percentile(draws, [q_low, q_high], axis=0)
    lower = np.where(n > 0, np.clip(lower, 0.0, 1.0), 0.0)
    upper = np.where(n > 0, np.clip(upper, 0.0, 1.0), 0.0)
    return p_hat, lower, upper


def bayesian_bootstrap_compositional_draws(
    df: pl.DataFrame,
    strata_cols: tuple[str, ...],
    n_resamples: int = 1000,
    prior_concentration: float = 0.5,
    random_state: Any = None,
) -> np.ndarray:
    """Draws joint stratum-level simplex proportions ``(n_resamples, len(df))`` via ``scipy.stats.dirichlet.rvs``.

    Within each stratum, category event counts ``k_s`` and any unobserved remainder
    ``max(0, n_s - sum(k_s))`` are jointly sampled on the probability simplex so that
    cumulative target coverage and stratum pooling preserve exact compositional negative
    covariance.

    Args:
        df: Aggregated Polars ``DataFrame`` containing ``"event"`` and ``"n"`` columns.
        strata_cols: Tuple of stratum column names used to partition rows into independent simplices.
        n_resamples: Number of joint Dirichlet simplex draws to simulate per stratum.
        prior_concentration: Symmetric Dirichlet prior concentration added to each category.
        random_state: Optional integer seed or NumPy ``Generator`` for reproducibility.

    Returns:
        A 2D float64 NumPy array of shape ``(n_resamples, len(df))`` aligned to the rows of ``df``.

    Raises:
        ValueError: If ``prior_concentration < 0.0``.
    """
    if prior_concentration < 0.0:
        raise ValueError(f"prior_concentration must be >= 0.0, got {prior_concentration}.")
    n_rows = len(df)
    draws = np.zeros((n_resamples, n_rows), dtype=float)
    if n_rows == 0:
        return draws

    k_arr = df["event"].to_numpy().astype(float)
    n_arr = df["n"].to_numpy().astype(float)
    rng = np.random.default_rng(random_state)

    active_strata = [c for c in strata_cols if c in df.columns]
    if active_strata:
        groups: dict[tuple[Any, ...], list[int]] = {}
        strata_tuples = df.select(active_strata).iter_rows()
        for row_idx, key in enumerate(strata_tuples):
            groups.setdefault(key, []).append(row_idx)
        group_indices = list(groups.values())
    else:
        group_indices = [list(range(n_rows))]

    prog = current_progress()
    n_groups = max(1, len(group_indices))
    for g_idx, idx_list in enumerate(group_indices):
        row_idx = np.asarray(idx_list, dtype=int)
        n_s = float(np.max(n_arr[row_idx])) if len(row_idx) > 0 else 0.0
        if n_s <= 0.0:
            continue
        k_s = np.clip(k_arr[row_idx], 0.0, n_s)
        k_len = len(k_s)
        rem = max(0.0, n_s - float(np.sum(k_s)))

        if rem > 0.0 or k_len == 1:
            alpha_vec = np.append(k_s + prior_concentration, rem + prior_concentration)
        else:
            alpha_vec = k_s + prior_concentration

        pos_mask = alpha_vec > 0.0
        n_pos = int(np.sum(pos_mask))
        if n_pos == 0:
            continue
        full_simplex = np.zeros((n_resamples, len(alpha_vec)), dtype=float)
        if n_pos == 1:
            full_simplex[:, pos_mask] = 1.0
        else:
            full_simplex[:, pos_mask] = stats.dirichlet.rvs(
                alpha_vec[pos_mask],
                size=n_resamples,
                random_state=rng,
            )
        draws[:, row_idx] = full_simplex[:, :k_len]
        if prog.is_active:
            prog.report(
                (g_idx + 1) / n_groups,
                message="Fitting Bayesian Bootstrap",
                detail=f"Simulating Dirichlet simplex draws ({g_idx + 1}/{n_groups} strata)",
            )

    return draws
