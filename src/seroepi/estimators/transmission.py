"""Scikit-learn compatible estimator for estimating pathogen transmission between isolates."""

import copy
import itertools
import re
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from sklearn.neighbors import NearestNeighbors

from seroepi.dataset import Dataset
from seroepi.models import StratifiedResultMixin
from seroepi.variables import (
    Count,
    Criterion,
    Metric,
    Spatial,
    Target,
    Taxonomy,
    Temporal,
    Trait,
    Variable,
)

_EARTH_RADIUS_KM: float = 6371.0088

_DURATION_UNIT_DAYS: dict[str, float] = {
    "h": 1.0 / 24.0,
    "hr": 1.0 / 24.0,
    "hour": 1.0 / 24.0,
    "hours": 1.0 / 24.0,
    "d": 1.0,
    "day": 1.0,
    "days": 1.0,
    "w": 7.0,
    "wk": 7.0,
    "week": 7.0,
    "weeks": 7.0,
    "mo": 30.4375,
    "month": 30.4375,
    "months": 30.4375,
    "q": 91.3125,
    "quarter": 91.3125,
    "quarters": 91.3125,
    "y": 365.25,
    "yr": 365.25,
    "year": 365.25,
    "years": 365.25,
}


def parse_temporal_cutoff_days(cutoff: str | float | timedelta) -> float:
    """Converts a temporal cutoff specification into a non-negative float number of days.

    Supports numeric days (``int`` / ``float``), ``datetime.timedelta``, and Polars-style
    duration strings such as ``'14d'``, ``'4w'``, ``'3mo'``, ``'1q'``, ``'1y'``, or ``'48h'``.

    Args:
        cutoff: Temporal cutoff as numeric days, ``timedelta``, or a duration string.

    Returns:
        Equivalent non-negative duration in days as a float.

    Raises:
        ValueError: If ``cutoff`` is negative, empty, or contains an unrecognized duration unit.
        TypeError: If ``cutoff`` is not a ``str``, ``float``, ``int``, or ``timedelta``.
    """
    if isinstance(cutoff, timedelta):
        days = cutoff.total_seconds() / 86400.0
        if days < 0:
            raise ValueError(f"temporal_cutoff must be non-negative, got {cutoff!r}.")
        return float(days)

    if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool):
        if float(cutoff) < 0:
            raise ValueError(f"temporal_cutoff must be non-negative, got {cutoff!r}.")
        return float(cutoff)

    if isinstance(cutoff, str):
        raw = cutoff.strip().lower()
        if not raw:
            raise ValueError("temporal_cutoff string cannot be empty.")
        try:
            val = float(raw)
            if val < 0:
                raise ValueError(f"temporal_cutoff must be non-negative, got {cutoff!r}.")
            return val
        except ValueError:
            pass

        tokens = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*([a-z]+)", raw)
        if not tokens or "".join(f"{v}{u}" for v, u in tokens) != re.sub(r"\s+", "", raw):
            raise ValueError(
                f"Invalid temporal_cutoff string {cutoff!r}. Expected format like '14d', '4w', '3mo', '1y', or '48h'."
            )
        total_days = 0.0
        for val_str, unit in tokens:
            val = float(val_str)
            if val < 0:
                raise ValueError(f"temporal_cutoff must be non-negative, got {cutoff!r}.")
            if unit not in _DURATION_UNIT_DAYS:
                raise ValueError(
                    f"Unsupported duration unit {unit!r} in temporal_cutoff {cutoff!r}. "
                    f"Supported units: {sorted(_DURATION_UNIT_DAYS.keys())}."
                )
            total_days += val * _DURATION_UNIT_DAYS[unit]
        return total_days

    raise TypeError(f"temporal_cutoff must be a str, float, int, or timedelta, got {type(cutoff).__name__}.")


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Computes vectorized great-circle Haversine distance in kilometers between coordinate arrays.

    Args:
        lat1: Array of source latitudes in decimal degrees.
        lon1: Array of source longitudes in decimal degrees.
        lat2: Array of target latitudes in decimal degrees.
        lon2: Array of target longitudes in decimal degrees.

    Returns:
        NumPy array of great-circle distances in kilometers (km).
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class TransmissionEstimator(StratifiedResultMixin, NearestNeighbors):
    """Scikit-learn compatible neighbor & cluster estimator for pathogen transmission on a `Dataset`.

    Combines pairwise genomic distances (`Dataset.distances`) with clonal/lineage stratification
    (`strata`) and epidemiological `Spatial` and `Temporal` variables to infer transmission links,
    connected transmission clusters (`labels_`), isolate-level summaries (`isolates_`), pairwise
    transmission edge lists (`pairs_`), and stratum-level transmission tables (`data_`).

    Args:
        n_neighbors: Number of nearest transmission neighbors to retrieve in `kneighbors()`.
        radius: Maximum genomic distance (e.g. SNPs or cgMLST alleles) for a transmission link.
        strata: Clonal, spatial, temporal, or metadata `Variable`(s) used to block candidate
            links and stratify the output `data_` summary table.
        spatial: Explicit `Spatial` variable (categorical locality at a `Spatial.Resolution`) or
            a pair of `(lat, lon)` `Spatial(resolution=Spatial.Resolution.POINT)` variables.
            If omitted, any `Spatial` variables inside `strata` are used automatically.
        temporal: Explicit `Temporal` variable for date/time transmission constraints.
            If omitted, any `Temporal` variable inside `strata` is used automatically.
        spatial_resolution: Optional `Spatial.Resolution` override (e.g. `SITE`, `CITY`, `COUNTRY`,
            `REGION`, `CONTINENT`, `GLOBAL`) applied to categorical `Spatial` variables.
        temporal_resolution: Optional `Temporal.Resolution` override (e.g. `DAY`, `EPI_WEEK`,
            `WEEK`, `MONTH`, `QUARTER`, `YEAR`) applied when `temporal_cutoff` is not specified.
        spatial_cutoff: Optional maximum geographic Haversine distance in kilometers (km) between
            isolates when `POINT` coordinates (or country centroids) are used.
        temporal_cutoff: Optional maximum time elapsed between linked isolates (e.g. `'14d'`,
            `'4w'`, `'3mo'`, `'1y'`, `timedelta(days=30)`, or numeric days).
        directed: If True, requires `source_time <= target_time` for temporal links and counts
            putative secondary recipients (`in_degree >= 1`) as `event` in `data_`.
        target: Optional `Target`, `Criterion`, or `Variable`. Defaults to a binary
            `'transmission'` target so `event / n` estimates the stratum transmission proportion.
        filter_qc: If True (default), filters the input `Dataset` by its registered `QC` variables.
        pad_zeros: If True, zero-pads unobserved strata combinations in `data_`.
        include_singletons: If False (default), unlinked singleton isolates receive `-1` in
            `labels_` (`None` in `isolates_['cluster']`). If True, singletons receive unique
            cluster IDs after multi-isolate clusters.
        cluster_prefix: String prefix for formatted cluster identifiers (default `'TC_'`).
        algorithm: NearestNeighbors algorithm when fitted on raw feature matrices.
        leaf_size: Leaf size passed to BallTree or KDTree for raw matrix inputs.
        metric: Distance metric for raw matrix inputs (defaults to `'precomputed'`).
        p: Power parameter for Minkowski metric on raw matrix inputs.
        metric_params: Additional keyword arguments for metric function on raw matrix inputs.
        n_jobs: Number of parallel jobs for neighbor search on raw matrix inputs.

    Attributes:
        is_fitted_: Boolean flag indicating whether ``fit()`` has been called.
        data_: Stratified summary Polars ``DataFrame`` compatible with ``Aggregator`` and ``BaseEstimator``.
        pairs_: Polars ``DataFrame`` of inferred isolate-to-isolate transmission links within ``radius``.
        isolates_: Polars ``DataFrame`` of per-isolate cluster assignments, degrees, and nearest neighbors.
        labels_: 1D int64 NumPy array of cluster IDs for each isolate (``-1`` for singletons unless ``include_singletons=True``).
        sample_ids_: 1D string NumPy array of fitted isolate identifiers.

    Examples:
        >>> import polars as pl
        >>> from scipy.sparse import csr_matrix
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.dist import GenomicDistances
        >>> from seroepi.estimators.transmission import TransmissionEstimator
        >>> from seroepi.variables import Identifier, Spatial
        >>> dist = GenomicDistances(csr_matrix([[0.0, 2.0, 50.0], [2.0, 0.0, 50.0], [50.0, 50.0, 0.0]]), ("s1", "s2", "s3"))
        >>> ds = Dataset(
        ...     pl.DataFrame({"id": ["s1", "s2", "s3"], "site": ["W1", "W1", "W1"]}),
        ...     variables={"id": Identifier("id"), "site": Spatial("site")},
        ...     distances=dist,
        ... )
        >>> te = TransmissionEstimator(radius=10.0, strata=[ds.var("site")]).fit(ds)
        >>> te.labels_.tolist()
        [0, 0, -1]
    """

    is_fitted_: bool = False
    data_: pl.DataFrame
    pairs_: pl.DataFrame
    isolates_: pl.DataFrame
    labels_: np.ndarray
    sample_ids_: np.ndarray
    variables_: dict[str, Variable]
    target_: Variable
    strata_: tuple[Variable, ...]
    adjust_for_: Variable | None = None
    is_compositional_: bool = False
    is_padded_: bool = False

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        radius: float = 20.0,
        strata: Sequence[Variable] | Variable | None = None,
        spatial: Sequence[Spatial] | Spatial | None = None,
        temporal: Temporal | None = None,
        spatial_resolution: Spatial.Resolution | str | None = None,
        temporal_resolution: Temporal.Resolution | str | None = None,
        spatial_cutoff: float | None = None,
        temporal_cutoff: str | float | timedelta | None = None,
        directed: bool = False,
        target: Target | Criterion | Variable | None = None,
        filter_qc: bool = True,
        pad_zeros: bool = False,
        include_singletons: bool = False,
        cluster_prefix: str = "TC_",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        metric: str | Callable[..., Any] = "precomputed",
        p: float = 2,
        metric_params: dict[str, Any] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        """Initializes the TransmissionEstimator.

        Args:
            n_neighbors: Number of nearest transmission neighbors to retrieve in ``kneighbors()``.
            radius: Maximum genomic distance for a transmission link.
            strata: Optional blocking/stratification ``Variable`` or sequence of ``Variable`` instances.
            spatial: Optional explicit ``Spatial`` variable or ``(lat, lon)`` pair.
            temporal: Optional explicit ``Temporal`` variable.
            spatial_resolution: Optional ``Spatial.Resolution`` override.
            temporal_resolution: Optional ``Temporal.Resolution`` override.
            spatial_cutoff: Optional maximum Haversine distance in kilometers (km).
            temporal_cutoff: Optional maximum elapsed duration between linked isolates.
            directed: Whether to enforce ``source_time <= target_time`` and count recipients as ``event``.
            target: Optional ``Target``, ``Criterion``, or ``Variable``.
            filter_qc: Whether to apply dataset QC filters before fitting.
            pad_zeros: Whether to zero-pad unobserved strata combinations in ``data_``.
            include_singletons: Whether unlinked singleton isolates receive unique cluster IDs instead of ``-1``.
            cluster_prefix: String prefix for formatted cluster names (e.g., ``"TC_"``).
            algorithm: ``NearestNeighbors`` algorithm for raw matrix inputs.
            leaf_size: Leaf size for ``BallTree`` or ``KDTree`` on raw matrix inputs.
            metric: Distance metric for raw matrix inputs.
            p: Power parameter for Minkowski metric on raw matrix inputs.
            metric_params: Additional metric keyword arguments for raw matrix inputs.
            n_jobs: Number of parallel jobs for raw matrix neighbor searches.

        Raises:
            TypeError: If ``strata``, ``spatial``, ``temporal``, or ``target`` have invalid types.
            ValueError: If ``spatial_cutoff`` or ``temporal_cutoff`` are negative or malformed.
        """
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.strata = self._normalize_strata(strata)
        self.spatial = self._normalize_spatial(spatial)
        if temporal is not None and not isinstance(temporal, Temporal):
            raise TypeError(
                f"temporal must be a Temporal Variable instance, got {type(temporal).__name__}. "
                "Use dataset.var(...) to retrieve it."
            )
        self.temporal = temporal
        self.spatial_resolution = spatial_resolution
        self.temporal_resolution = temporal_resolution
        if spatial_cutoff is not None and float(spatial_cutoff) < 0:
            raise ValueError(f"spatial_cutoff must be non-negative, got {spatial_cutoff!r}.")
        self.spatial_cutoff = float(spatial_cutoff) if spatial_cutoff is not None else None
        if temporal_cutoff is not None:
            _ = parse_temporal_cutoff_days(temporal_cutoff)
        self.temporal_cutoff = temporal_cutoff
        self.directed = directed
        if target is not None and not isinstance(target, (Target, Criterion, Variable)):
            raise TypeError(f"target must be a Variable, Criterion, or Target instance, got {type(target).__name__}.")
        self.target = target
        self.filter_qc = filter_qc
        self.pad_zeros = pad_zeros
        self.include_singletons = include_singletons
        self.cluster_prefix = cluster_prefix

    def _get_frame(self) -> pl.DataFrame:
        return self.data_

    def _set_frame(self, df: pl.DataFrame) -> None:
        self.data_ = df

    @staticmethod
    def _normalize_strata(strata: Sequence[Variable] | Variable | None) -> tuple[Variable, ...]:
        """Normalizes `strata` into a tuple of `Variable` instances."""
        if strata is None:
            return ()
        if isinstance(strata, Variable):
            return (strata,)
        if isinstance(strata, (list, tuple, Sequence)):
            for idx, s in enumerate(strata):
                if not isinstance(s, Variable):
                    raise TypeError(
                        f"Strata item at index {idx} must be a Variable instance, got {type(s).__name__}. "
                        "Use dataset.var(...) to retrieve it."
                    )
            return tuple(strata)
        raise TypeError(f"strata must be a Variable or Sequence of Variables, got {type(strata).__name__}.")

    @staticmethod
    def _normalize_spatial(spatial: Sequence[Spatial] | Spatial | None) -> tuple[Spatial, ...]:
        """Normalizes `spatial` into a tuple of `Spatial` instances."""
        if spatial is None:
            return ()
        if isinstance(spatial, Spatial):
            return (spatial,)
        if isinstance(spatial, Variable):
            raise TypeError(f"spatial must be a Spatial Variable instance, got {type(spatial).__name__}.")
        if isinstance(spatial, (list, tuple, Sequence)):
            for idx, s in enumerate(spatial):
                if not isinstance(s, Spatial):
                    raise TypeError(
                        f"spatial item at index {idx} must be a Spatial Variable instance, got {type(s).__name__}."
                    )
            return tuple(spatial)
        raise TypeError(
            f"spatial must be a Spatial Variable or Sequence of Spatial Variables, got {type(spatial).__name__}."
        )

    @staticmethod
    def _coerce_spatial_resolution(res: Spatial.Resolution | str | None) -> Spatial.Resolution | None:
        if res is None:
            return None
        if isinstance(res, Spatial.Resolution):
            return res
        return Spatial.Resolution(str(res).lower())

    @staticmethod
    def _coerce_temporal_resolution(res: Temporal.Resolution | str | None) -> Temporal.Resolution | None:
        if res is None:
            return None
        if isinstance(res, Temporal.Resolution):
            return res
        return Temporal.Resolution(str(res).lower())

    def fit(self, X: Dataset | Any, y: Any = None) -> Self:
        """Fits the transmission model on a ``Dataset`` (or precomputed/feature matrix ``X``).

        Args:
            X: A ``Dataset`` with attached ``GenomicDistances``, or a 2D precomputed/feature matrix.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted ``TransmissionEstimator`` instance (``self``).

        Raises:
            ValueError: If ``X`` is a ``Dataset`` without attached ``GenomicDistances``.
        """
        if not isinstance(X, Dataset):
            return self._fit_raw_matrix(X, y)

        if X.distances is None:
            raise ValueError(
                "Cannot fit TransmissionEstimator on a Dataset without attached GenomicDistances. "
                "Call dataset.with_distances(...) first."
            )

        self._dataset_mode = True
        ds = X.qc() if self.filter_qc else X
        df = ds.data
        id_col = ds.identifier.column
        self.id_col_ = id_col

        # 1. Resolve Spatial & Temporal variables and Resolution overrides
        sp_res_override = self._coerce_spatial_resolution(self.spatial_resolution)
        tm_res_override = self._coerce_temporal_resolution(self.temporal_resolution)

        strata_vars: list[Variable] = []
        for s in self.strata:
            if isinstance(s, Spatial) and sp_res_override is not None and s.resolution != Spatial.Resolution.POINT:
                strata_vars.append(s.copy(resolution=sp_res_override))
            elif isinstance(s, Temporal) and tm_res_override is not None:
                strata_vars.append(s.copy(resolution=tm_res_override))
            else:
                strata_vars.append(s)

        explicit_spatial = list(self.spatial)
        if not explicit_spatial:
            explicit_spatial = [s for s in strata_vars if isinstance(s, Spatial)]
        else:
            explicit_spatial = [
                s.copy(resolution=sp_res_override)
                if sp_res_override is not None and s.resolution != Spatial.Resolution.POINT
                else s
                for s in explicit_spatial
            ]

        temporal_var = self.temporal
        if temporal_var is None:
            temporal_var = next((s for s in strata_vars if isinstance(s, Temporal)), None)
        elif tm_res_override is not None:
            temporal_var = temporal_var.copy(resolution=tm_res_override)

        # 2. Resolve optional Target
        resolved_target: Target | None = None
        is_compositional = False
        if self.target is not None:
            resolved_target = Target.coerce(self.target)
            is_compositional = resolved_target.is_compositional
            if not is_compositional:
                # Binary target: restrict transmission evaluation to target-positive isolates
                df = df.filter(resolved_target.to_expr())
            target_var = Trait(column="target", title=resolved_target.title, compositional=is_compositional)
        else:
            target_var = Trait(column="target", title="Transmission", compositional=False)

        n_samples = df.height
        sample_ids = df[id_col].cast(pl.String).to_numpy()
        self.sample_ids_ = sample_ids
        self.n_samples_fit_ = n_samples
        self.n_features_in_ = n_samples

        # 3. Project strata, spatial, and temporal columns on df
        proj_exprs: list[pl.Expr] = []
        strata_cols = [s.column for s in strata_vars]
        for s in strata_vars:
            if isinstance(s, Spatial) and s.resolution == Spatial.Resolution.GLOBAL:
                proj_exprs.append(pl.lit("Global").alias(s.column))
            else:
                proj_exprs.append(s.to_expr())

        for sp_var in explicit_spatial:
            if sp_var.column not in strata_cols and sp_var.resolution != Spatial.Resolution.POINT:
                if sp_var.resolution == Spatial.Resolution.GLOBAL:
                    proj_exprs.append(pl.lit("Global").alias(sp_var.column))
                else:
                    proj_exprs.append(sp_var.to_expr())

        if temporal_var is not None and temporal_var.column not in strata_cols:
            proj_exprs.append(temporal_var.to_expr())

        if temporal_var is not None:
            # Parse raw timestamps BEFORE truncating temporal column in df
            parsed_ts_series = df.select(temporal_var.parse_expr().alias("_parsed_ts"))["_parsed_ts"]
            ts_epoch_days = (
                (parsed_ts_series.dt.epoch(time_unit="us").cast(pl.Float64) / (86400.0 * 1_000_000.0))
                .fill_null(float("nan"))
                .to_numpy()
            )
        else:
            ts_epoch_days = None

        if resolved_target is not None and is_compositional:
            proj_exprs.append(resolved_target.to_category_expr().alias("_target_cat"))

        if proj_exprs:
            df_proj = df.with_columns(*proj_exprs)
        else:
            df_proj = df

        if resolved_target is not None and is_compositional:
            df_proj = df_proj.filter(pl.col("_target_cat").is_not_null())
            sample_ids = df_proj[id_col].cast(pl.String).to_numpy()
            if ts_epoch_days is not None:
                valid_mask = df.select(resolved_target.to_category_expr().is_not_null().alias("_m"))["_m"].to_numpy()
                ts_epoch_days = ts_epoch_days[valid_mask]
            n_samples = df_proj.height
            self.sample_ids_ = sample_ids
            self.n_samples_fit_ = n_samples
            self.n_features_in_ = n_samples

        # 4. Extract candidate off-diagonal genomic pairs aligned to sample_ids
        r_idx, c_idx, g_dist = self._extract_aligned_genomic_edges(ds, sample_ids)

        # 5. Apply Clonal/Strata, Spatial, and Temporal constraints to candidate edges
        r_idx, c_idx, g_dist, t_dist, s_dist = self._filter_candidate_edges(
            df_raw=df,
            df_proj=df_proj,
            r_idx=r_idx,
            c_idx=c_idx,
            g_dist=g_dist,
            strata_vars=strata_vars,
            explicit_spatial=explicit_spatial,
            temporal_var=temporal_var,
            ts_epoch_days=ts_epoch_days,
            has_comp_target=bool(resolved_target is not None and is_compositional),
        )

        # 6. Store sorted per-isolate valid neighbor structures (across all valid genomic distances)
        self._build_neighbor_structures(
            n_samples=n_samples,
            r_idx=r_idx,
            c_idx=c_idx,
            g_dist=g_dist,
            t_dist=t_dist,
            s_dist=s_dist,
        )

        # 7. Threshold by self.radius to build clusters (labels_), isolates_, pairs_, and stratified data_
        self._build_transmission_outputs(
            df_proj=df_proj,
            id_col=id_col,
            strata_vars=strata_vars,
            target_var=target_var,
            resolved_target=resolved_target,
            is_compositional=is_compositional,
        )

        self.is_fitted_ = True
        return self

    def _fit_raw_matrix(self, X: Any, y: Any = None) -> Self:
        """Fallback when fitted on a raw distance or feature matrix instead of a `Dataset`."""
        self._dataset_mode = False
        super().fit(X, y)
        n_samples = int(getattr(self, "n_samples_fit_", X.shape[0]))
        self.sample_ids_ = np.asarray([str(i) for i in range(n_samples)], dtype=str)
        self.id_col_ = "sample_id"

        r_graph = super().radius_neighbors_graph(radius=self.radius, mode="distance")
        coo = r_graph.tocoo()
        off_diag = coo.row != coo.col
        r_idx = coo.row[off_diag]
        c_idx = coo.col[off_diag]
        g_dist = np.asarray(coo.data[off_diag], dtype=np.float64)
        t_dist = np.full(len(r_idx), np.nan, dtype=np.float64)
        s_dist = np.full(len(r_idx), np.nan, dtype=np.float64)

        self._build_neighbor_structures(
            n_samples=n_samples,
            r_idx=r_idx,
            c_idx=c_idx,
            g_dist=g_dist,
            t_dist=t_dist,
            s_dist=s_dist,
        )
        df_proj = pl.DataFrame({self.id_col_: self.sample_ids_})
        target_var = Trait(column="target", title="Transmission", compositional=False)
        self._build_transmission_outputs(
            df_proj=df_proj,
            id_col=self.id_col_,
            strata_vars=[],
            target_var=target_var,
            resolved_target=None,
            is_compositional=False,
        )
        self.is_fitted_ = True
        return self

    @staticmethod
    def _extract_aligned_genomic_edges(
        ds: Dataset,
        sample_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts directed off-diagonal edges `(r_idx, c_idx, g_dist)` aligned to `sample_ids`."""
        assert ds.distances is not None
        n = len(sample_ids)
        if n == 0 or ds.distances.n_samples == 0:
            return (
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.float64),
            )

        id_to_pos: dict[str, int] = {str(s): i for i, s in enumerate(sample_ids)}
        dist_samples = ds.distances.samples
        global_map = np.array([id_to_pos.get(str(s), -1) for s in dist_samples], dtype=np.intp)

        coo = ds.distances.array.tocoo()
        r_mapped = global_map[coo.row]
        c_mapped = global_map[coo.col]
        valid = (r_mapped >= 0) & (c_mapped >= 0) & (r_mapped != c_mapped)
        if not np.any(valid):
            return (
                np.array([], dtype=np.intp),
                np.array([], dtype=np.intp),
                np.array([], dtype=np.float64),
            )

        r_sub = r_mapped[valid]
        c_sub = c_mapped[valid]
        v_sub = np.asarray(coo.data[valid], dtype=np.float64)

        # Symmetrize so both (i, j) and (j, i) are present prior to directional filtering
        all_r = np.concatenate([r_sub, c_sub])
        all_c = np.concatenate([c_sub, r_sub])
        all_v = np.concatenate([v_sub, v_sub])

        linear = all_r.astype(np.int64) * n + all_c.astype(np.int64)
        _, uniq_idx = np.unique(linear, return_index=True)
        uniq_idx.sort()
        return all_r[uniq_idx], all_c[uniq_idx], all_v[uniq_idx]

    def _resolve_coordinates_arrays(
        self,
        df_raw: pl.DataFrame,
        explicit_spatial: list[Spatial],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extracts `(lat, lon)` float64 arrays when point coordinates or country centroids are used."""
        point_vars = [s for s in explicit_spatial if s.resolution == Spatial.Resolution.POINT]
        if len(point_vars) >= 2:
            lat_col, lon_col = point_vars[0].column, point_vars[1].column
            if any("lon" in point_vars[0].column.lower() and "lat" in point_vars[1].column.lower() for _ in [0]):
                lat_col, lon_col = point_vars[1].column, point_vars[0].column
            lat_arr = df_raw[lat_col].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
            lon_arr = df_raw[lon_col].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
            return lat_arr, lon_arr

        if self.spatial_cutoff is not None:
            # Check for centroid_lat / centroid_lon or lat / lon columns in df_raw
            for lat_c, lon_c in (("latitude", "longitude"), ("lat", "lon"), ("centroid_lat", "centroid_lon")):
                if lat_c in df_raw.columns and lon_c in df_raw.columns:
                    lat_arr = df_raw[lat_c].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
                    lon_arr = df_raw[lon_c].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
                    return lat_arr, lon_arr

            # If a COUNTRY spatial variable is present and CountryData is built, project centroids
            country_vars = [s for s in explicit_spatial if s.resolution == Spatial.Resolution.COUNTRY]
            if country_vars:
                from seroepi.country import CountryData

                if CountryData.is_built():
                    cd = CountryData.get_instance()
                    c_var = country_vars[0]
                    iso_df = df_raw.select(c_var.iso3_expr(strict=True, country_data=cd).alias("_iso3")).join(
                        cd.data.select(["iso3", "centroid_lat", "centroid_lon"]),
                        left_on="_iso3",
                        right_on="iso3",
                        how="left",
                    )
                    lat_arr = iso_df["centroid_lat"].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
                    lon_arr = iso_df["centroid_lon"].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy()
                    return lat_arr, lon_arr

        return None

    def _filter_candidate_edges(
        self,
        *,
        df_raw: pl.DataFrame,
        df_proj: pl.DataFrame,
        r_idx: np.ndarray,
        c_idx: np.ndarray,
        g_dist: np.ndarray,
        strata_vars: list[Variable],
        explicit_spatial: list[Spatial],
        temporal_var: Temporal | None,
        ts_epoch_days: np.ndarray | None,
        has_comp_target: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filters candidate genomic edges by clonal/strata, spatial, and temporal rules."""
        m = len(r_idx)
        if m == 0:
            empty_f = np.array([], dtype=np.float64)
            return r_idx, c_idx, g_dist, empty_f, empty_f

        keep = np.ones(m, dtype=bool)
        t_dist = np.full(m, np.nan, dtype=np.float64)
        s_dist = np.full(m, np.nan, dtype=np.float64)

        # 1. Compositional target equality (e.g. target=ds.var("ST"))
        if has_comp_target and "_target_cat" in df_proj.columns:
            t_vals = df_proj["_target_cat"].cast(pl.String).to_numpy()
            t_valid = df_proj["_target_cat"].is_not_null().to_numpy()
            keep &= t_valid[r_idx] & t_valid[c_idx] & (t_vals[r_idx] == t_vals[c_idx])

        # 2. Non-spatial, non-temporal blocking strata equality (e.g. Taxonomy clone/ST)
        for s_var in strata_vars:
            if isinstance(s_var, (Spatial, Temporal)):
                continue
            col_vals = df_proj[s_var.column].cast(pl.String).to_numpy()
            col_valid = df_proj[s_var.column].is_not_null().to_numpy()
            keep &= col_valid[r_idx] & col_valid[c_idx] & (col_vals[r_idx] == col_vals[c_idx])

        # 3. Spatial constraints (Point Haversine km cutoff and/or Categorical Resolution equality)
        coords = self._resolve_coordinates_arrays(
            df_raw if df_raw.height == df_proj.height else df_proj, explicit_spatial
        )
        if coords is not None:
            lat_arr, lon_arr = coords
            s_dist = haversine_distance_km(lat_arr[r_idx], lon_arr[r_idx], lat_arr[c_idx], lon_arr[c_idx])
            if self.spatial_cutoff is not None:
                keep &= ~np.isnan(s_dist) & (s_dist <= self.spatial_cutoff)

        cat_spatial_vars = [s for s in explicit_spatial if s.resolution != Spatial.Resolution.POINT]
        # Apply categorical spatial equality unless overridden by a numeric spatial_cutoff on coordinates
        if cat_spatial_vars and (self.spatial_cutoff is None or coords is None):
            for sp_var in cat_spatial_vars:
                if sp_var.resolution == Spatial.Resolution.GLOBAL:
                    continue
                sp_vals = df_proj[sp_var.column].cast(pl.String).to_numpy()
                sp_valid = df_proj[sp_var.column].is_not_null().to_numpy()
                keep &= sp_valid[r_idx] & sp_valid[c_idx] & (sp_vals[r_idx] == sp_vals[c_idx])

        # 4. Temporal constraints (Explicit temporal_cutoff duration or Temporal.Resolution bucket match)
        if temporal_var is not None and ts_epoch_days is not None:
            raw_dt = ts_epoch_days[c_idx] - ts_epoch_days[r_idx]
            t_dist = raw_dt if self.directed else np.abs(raw_dt)

            if self.temporal_cutoff is not None:
                cutoff_days = parse_temporal_cutoff_days(self.temporal_cutoff)
                if self.directed:
                    keep &= ~np.isnan(raw_dt) & (raw_dt >= 0.0) & (raw_dt <= cutoff_days)
                else:
                    keep &= ~np.isnan(raw_dt) & (np.abs(raw_dt) <= cutoff_days)
            else:
                if temporal_var.resolution != Temporal.Resolution.UNKNOWN:
                    tm_vals = df_proj[temporal_var.column].cast(pl.String).to_numpy()
                    tm_valid = df_proj[temporal_var.column].is_not_null().to_numpy()
                    keep &= tm_valid[r_idx] & tm_valid[c_idx] & (tm_vals[r_idx] == tm_vals[c_idx])
                if self.directed:
                    keep &= ~np.isnan(raw_dt) & (raw_dt >= 0.0)

        return r_idx[keep], c_idx[keep], g_dist[keep], t_dist[keep], s_dist[keep]

    def _build_neighbor_structures(
        self,
        *,
        n_samples: int,
        r_idx: np.ndarray,
        c_idx: np.ndarray,
        g_dist: np.ndarray,
        t_dist: np.ndarray,
        s_dist: np.ndarray,
    ) -> None:
        """Builds per-isolate sorted neighbor arrays for fast `kneighbors` and `radius_neighbors` queries."""
        self._neighbor_indices: list[np.ndarray] = [np.array([], dtype=np.intp) for _ in range(n_samples)]
        self._neighbor_distances: list[np.ndarray] = [np.array([], dtype=np.float64) for _ in range(n_samples)]
        self._edge_r = r_idx
        self._edge_c = c_idx
        self._edge_g = g_dist
        self._edge_t = t_dist
        self._edge_s = s_dist

        if len(r_idx) == 0 or n_samples == 0:
            return

        # Lexicographical sort by (r_idx, g_dist, abs(t_dist), s_dist, c_idx)
        t_sort = np.nan_to_num(np.abs(t_dist), nan=np.inf)
        s_sort = np.nan_to_num(s_dist, nan=np.inf)
        order = np.lexsort((c_idx, s_sort, t_sort, g_dist, r_idx))

        r_sorted = r_idx[order]
        c_sorted = c_idx[order]
        g_sorted = g_dist[order]

        unique_r, split_pts = np.unique(r_sorted, return_index=True)
        c_groups = np.split(c_sorted, split_pts[1:])
        g_groups = np.split(g_sorted, split_pts[1:])

        for row_i, c_arr, g_arr in zip(unique_r, c_groups, g_groups, strict=True):
            self._neighbor_indices[int(row_i)] = c_arr
            self._neighbor_distances[int(row_i)] = g_arr

    def _build_transmission_outputs(
        self,
        *,
        df_proj: pl.DataFrame,
        id_col: str,
        strata_vars: list[Variable],
        target_var: Variable,
        resolved_target: Target | None,
        is_compositional: bool,
    ) -> None:
        """Computes connected transmission clusters (`labels_`), `pairs_`, `isolates_`, and `data_`."""
        n_samples = len(self.sample_ids_)
        strata_cols = [s.column for s in strata_vars]

        # 1. Filter valid edges to those within self.radius
        r_limit = float(self.radius) if self.radius is not None else float("inf")
        if len(self._edge_g) > 0:
            rad_mask = self._edge_g <= r_limit
            r_rad = self._edge_r[rad_mask]
            c_rad = self._edge_c[rad_mask]
            g_rad = self._edge_g[rad_mask]
            t_rad = self._edge_t[rad_mask]
            s_rad = self._edge_s[rad_mask]
        else:
            r_rad = np.array([], dtype=np.intp)
            c_rad = np.array([], dtype=np.intp)
            g_rad = np.array([], dtype=np.float64)
            t_rad = np.array([], dtype=np.float64)
            s_rad = np.array([], dtype=np.float64)

        # 2. Connected components for transmission clusters (weakly connected in directed mode)
        if n_samples == 0:
            self.labels_ = np.array([], dtype=np.int64)
            comp_sizes = np.array([], dtype=np.int64)
            cluster_names: list[str | None] = []
        else:
            if len(r_rad) > 0:
                sym_r = np.concatenate([r_rad, c_rad])
                sym_c = np.concatenate([c_rad, r_rad])
                adj = coo_array(
                    (np.ones(len(sym_r), dtype=np.int8), (sym_r, sym_c)),
                    shape=(n_samples, n_samples),
                ).tocsr()
                _, raw_labels = sp_connected_components(csgraph=adj, directed=False, return_labels=True)
            else:
                raw_labels = np.arange(n_samples, dtype=np.int32)

            counts = np.bincount(raw_labels)
            comp_sizes = counts[raw_labels].astype(np.int64)

            # Rank multi-isolate components (size >= 2) deterministically by (-size, first_index)
            multi_comps: list[tuple[int, int, int]] = []
            single_comps: list[tuple[int, int]] = []
            seen_comp: dict[int, int] = {}
            for idx, lbl in enumerate(raw_labels):
                lbl_int = int(lbl)
                if lbl_int not in seen_comp:
                    seen_comp[lbl_int] = idx
                    sz = int(counts[lbl_int])
                    if sz >= 2:
                        multi_comps.append((-sz, idx, lbl_int))
                    else:
                        single_comps.append((idx, lbl_int))

            multi_comps.sort()
            remap: dict[int, int] = {}
            for new_id, (_, _, old_lbl) in enumerate(multi_comps):
                remap[old_lbl] = new_id

            if self.include_singletons:
                offset = len(multi_comps)
                for s_idx, (_, old_lbl) in enumerate(single_comps):
                    remap[old_lbl] = offset + s_idx
            else:
                for _, old_lbl in single_comps:
                    remap[old_lbl] = -1

            self.labels_ = np.array([remap[int(lbl)] for lbl in raw_labels], dtype=np.int64)
            cluster_names = [f"{self.cluster_prefix}{cid + 1}" if cid >= 0 else None for cid in self.labels_]

        # 3. Compute isolate degrees, nearest neighbors, and build self.isolates_
        out_deg = (
            np.bincount(r_rad, minlength=n_samples).astype(np.int64) if n_samples > 0 else np.array([], dtype=np.int64)
        )
        in_deg = (
            np.bincount(c_rad, minlength=n_samples).astype(np.int64) if n_samples > 0 else np.array([], dtype=np.int64)
        )
        degree = in_deg + out_deg if self.directed else out_deg
        is_transmitted = (in_deg >= 1) if self.directed else (degree >= 1)

        nearest_id: list[str | None] = [None] * n_samples
        nearest_dist: list[float | None] = [None] * n_samples
        for i in range(n_samples):
            nbr_i = self._neighbor_indices[i]
            dist_i = self._neighbor_distances[i]
            if len(dist_i) > 0 and dist_i[0] <= r_limit:
                nearest_id[i] = str(self.sample_ids_[int(nbr_i[0])])
                nearest_dist[i] = float(dist_i[0])

        isolate_cols: dict[str, Any] = {
            id_col: pl.Series(id_col, self.sample_ids_, dtype=pl.String),
            "cluster": pl.Series("cluster", cluster_names, dtype=pl.String),
            "cluster_id": pl.Series("cluster_id", self.labels_, dtype=pl.Int64),
            "cluster_size": pl.Series("cluster_size", comp_sizes, dtype=pl.Int64),
            "degree": pl.Series("degree", degree, dtype=pl.Int64),
        }
        if self.directed:
            isolate_cols["in_degree"] = pl.Series("in_degree", in_deg, dtype=pl.Int64)
            isolate_cols["out_degree"] = pl.Series("out_degree", out_deg, dtype=pl.Int64)
        isolate_cols["nearest_neighbor"] = pl.Series("nearest_neighbor", nearest_id, dtype=pl.String)
        isolate_cols["nearest_distance"] = pl.Series("nearest_distance", nearest_dist, dtype=pl.Float64)
        isolate_cols["is_transmitted"] = pl.Series("is_transmitted", is_transmitted, dtype=pl.Boolean)

        for col in strata_cols:
            if col in df_proj.columns:
                isolate_cols[col] = df_proj[col]
        if is_compositional and "_target_cat" in df_proj.columns:
            isolate_cols["target"] = df_proj["_target_cat"].cast(pl.String)

        self.isolates_ = pl.DataFrame(isolate_cols)

        # 4. Build self.pairs_ DataFrame (canonical undirected pairs if directed=False)
        if len(r_rad) > 0:
            if not self.directed:
                canon = r_rad < c_rad
                p_r = r_rad[canon]
                p_c = c_rad[canon]
                p_g = g_rad[canon]
                p_t = t_rad[canon]
                p_s = s_rad[canon]
            else:
                p_r, p_c, p_g, p_t, p_s = r_rad, c_rad, g_rad, t_rad, s_rad

            pair_cols: dict[str, Any] = {
                "source": pl.Series("source", self.sample_ids_[p_r], dtype=pl.String),
                "target": pl.Series("target", self.sample_ids_[p_c], dtype=pl.String),
                "genomic_distance": pl.Series("genomic_distance", p_g, dtype=pl.Float64),
                "temporal_distance_days": pl.Series(
                    "temporal_distance_days",
                    [None if np.isnan(x) else float(x) for x in p_t],
                    dtype=pl.Float64,
                ),
                "spatial_distance_km": pl.Series(
                    "spatial_distance_km",
                    [None if np.isnan(x) else float(x) for x in p_s],
                    dtype=pl.Float64,
                ),
                "cluster": pl.Series("cluster", [cluster_names[int(i)] for i in p_r], dtype=pl.String),
            }
            for col in strata_cols:
                if col in df_proj.columns:
                    pair_cols[col] = df_proj[col].gather(p_r)
            self.pairs_ = pl.DataFrame(pair_cols)
        else:
            empty_pair_schema: dict[str, Any] = {
                "source": pl.String,
                "target": pl.String,
                "genomic_distance": pl.Float64,
                "temporal_distance_days": pl.Float64,
                "spatial_distance_km": pl.Float64,
                "cluster": pl.String,
            }
            for col in strata_cols:
                if col in df_proj.columns:
                    empty_pair_schema[col] = df_proj.schema[col]
            self.pairs_ = pl.DataFrame(schema=empty_pair_schema)

        # 5. Build stratified summary DataFrame self.data_
        self._build_stratified_summary(
            strata_vars=strata_vars,
            target_var=target_var,
            resolved_target=resolved_target,
            is_compositional=is_compositional,
        )

    def _build_stratified_summary(
        self,
        *,
        strata_vars: list[Variable],
        target_var: Variable,
        resolved_target: Target | None,
        is_compositional: bool,
    ) -> None:
        """Aggregates isolate and pair transmission metrics into `self.data_`."""
        strata_cols = [s.column for s in strata_vars]
        target_default_name = resolved_target.name if resolved_target is not None else "transmission"

        iso_df = self.isolates_
        if not is_compositional:
            iso_df = iso_df.with_columns(pl.lit(target_default_name).alias("target"))

        group_cols = [*strata_cols, "target"]

        if iso_df.height == 0:
            empty_schema: dict[str, Any] = {s.column: s.dtype for s in strata_vars}
            empty_schema.update(
                {
                    "target": pl.String,
                    "event": pl.Int64,
                    "n": pl.Int64,
                    "n_clusters": pl.Int64,
                    "mean_cluster_size": pl.Float64,
                    "max_cluster_size": pl.Int64,
                    "n_pairs": pl.Int64,
                    "mean_distance": pl.Float64,
                }
            )
            self.data_ = pl.DataFrame(schema=empty_schema)
        else:
            # Compute per-group cluster stats from unique multi-isolate clusters (cluster_id >= 0)
            multi_iso = iso_df.filter(pl.col("cluster_id") >= 0)
            if multi_iso.height > 0:
                cluster_sizes_by_grp = (
                    multi_iso.group_by([*group_cols, "cluster_id"])
                    .agg(pl.len().cast(pl.Int64).alias("_c_size"))
                    .group_by(group_cols)
                    .agg(
                        pl.len().cast(pl.Int64).alias("n_clusters"),
                        pl.col("_c_size").mean().cast(pl.Float64).alias("mean_cluster_size"),
                        pl.col("_c_size").max().cast(pl.Int64).alias("max_cluster_size"),
                    )
                )
            else:
                cluster_sizes_by_grp = None

            # Compute per-group pair counts and mean genomic distance
            if self.pairs_.height > 0:
                pairs_with_tgt = self.pairs_
                if is_compositional:
                    # Join target category from source isolate
                    src_tgt = iso_df.select([pl.col(self.id_col_).alias("source"), pl.col("target")])
                    pairs_with_tgt = pairs_with_tgt.join(src_tgt, on="source", how="left")
                else:
                    pairs_with_tgt = pairs_with_tgt.with_columns(pl.lit(target_default_name).alias("target"))

                pair_stats_by_grp = pairs_with_tgt.group_by(group_cols).agg(
                    pl.len().cast(pl.Int64).alias("n_pairs"),
                    pl.col("genomic_distance").mean().cast(pl.Float64).alias("mean_distance"),
                )
            else:
                pair_stats_by_grp = None

            if is_compositional:
                # For compositional target, denominator n is the stratum total (matching Aggregator contract)
                strata_n = (
                    iso_df.group_by(strata_cols).agg(pl.len().cast(pl.Int64).alias("n"))
                    if strata_cols
                    else iso_df.select(pl.len().cast(pl.Int64).alias("n"))
                )
                events_df = iso_df.group_by(group_cols, maintain_order=True).agg(
                    pl.col("is_transmitted").cast(pl.Int64).sum().alias("event"),
                )
                base_summary = (
                    events_df.join(strata_n, on=strata_cols, how="left")
                    if strata_cols
                    else events_df.join(strata_n, how="cross")
                )
            else:
                base_summary = iso_df.group_by(group_cols, maintain_order=True).agg(
                    pl.col("is_transmitted").cast(pl.Int64).sum().alias("event"),
                    pl.len().cast(pl.Int64).alias("n"),
                )

            if cluster_sizes_by_grp is not None:
                base_summary = base_summary.join(cluster_sizes_by_grp, on=group_cols, how="left")
            else:
                base_summary = base_summary.with_columns(
                    pl.lit(0, dtype=pl.Int64).alias("n_clusters"),
                    pl.lit(0.0, dtype=pl.Float64).alias("mean_cluster_size"),
                    pl.lit(0, dtype=pl.Int64).alias("max_cluster_size"),
                )

            if pair_stats_by_grp is not None:
                base_summary = base_summary.join(pair_stats_by_grp, on=group_cols, how="left")
            else:
                base_summary = base_summary.with_columns(
                    pl.lit(0, dtype=pl.Int64).alias("n_pairs"),
                    pl.lit(None, dtype=pl.Float64).alias("mean_distance"),
                )

            if self.pad_zeros and strata_cols:
                unique_strata = [iso_df[s].drop_nulls().unique(maintain_order=True).to_list() for s in strata_cols]
                unique_targets = iso_df["target"].drop_nulls().unique(maintain_order=True).to_list()
                grid_rows = list(itertools.product(*unique_strata, unique_targets))
                if grid_rows:
                    grid_df = pl.DataFrame(grid_rows, schema=group_cols, orient="row")
                    if is_compositional:
                        strata_n = iso_df.group_by(strata_cols).agg(pl.len().cast(pl.Int64).alias("n"))
                        base_summary = grid_df.join(base_summary.drop("n"), on=group_cols, how="left").join(
                            strata_n, on=strata_cols, how="left"
                        )
                    else:
                        base_summary = grid_df.join(base_summary, on=group_cols, how="left")

            self.data_ = base_summary.with_columns(
                pl.col("event").fill_null(0).cast(pl.Int64),
                pl.col("n").fill_null(0).cast(pl.Int64),
                pl.col("n_clusters").fill_null(0).cast(pl.Int64),
                pl.col("mean_cluster_size").fill_null(0.0).cast(pl.Float64),
                pl.col("max_cluster_size").fill_null(0).cast(pl.Int64),
                pl.col("n_pairs").fill_null(0).cast(pl.Int64),
            ).select(
                [
                    *strata_cols,
                    "target",
                    "event",
                    "n",
                    "n_clusters",
                    "mean_cluster_size",
                    "max_cluster_size",
                    "n_pairs",
                    "mean_distance",
                ]
            )

        out_vars: dict[str, Variable] = {s.column: s for s in strata_vars}
        out_vars["target"] = target_var
        out_vars["event"] = Count("event", title="Transmitted Isolates", is_denominator=False)
        out_vars["n"] = Count("n", title="Total Isolates", is_denominator=True)
        out_vars["n_clusters"] = Count("n_clusters", title="Transmission Clusters", is_denominator=False)
        out_vars["mean_cluster_size"] = Metric("mean_cluster_size", title="Mean Cluster Size")
        out_vars["max_cluster_size"] = Count("max_cluster_size", title="Max Cluster Size", is_denominator=False)
        out_vars["n_pairs"] = Count("n_pairs", title="Transmission Pairs", is_denominator=False)
        out_vars["mean_distance"] = Metric("mean_distance", title="Mean Genomic Distance")

        self.variables_ = out_vars
        self.target_ = target_var
        self.strata_ = tuple(strata_vars)
        self.adjust_for_ = None
        self.is_compositional_ = is_compositional
        self.is_padded_ = bool(self.pad_zeros)

    # -------------------------------------------------------------------------
    # Scikit-Learn NearestNeighbors Graph & Query API
    # -------------------------------------------------------------------------

    def radius_neighbors(
        self,
        X: Dataset | Any = None,
        radius: float | None = None,
        return_distance: bool = True,
        sort_results: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Finds all transmission neighbors within a given genomic ``radius`` for each isolate.

        Respects all configured ``strata``, ``Spatial``, and ``Temporal`` cutoffs/resolutions.

        Args:
            X: Optional query ``Dataset`` or feature matrix. Defaults to the training data when ``None``.
            radius: Optional genomic distance threshold override. Defaults to ``self.radius``.
            return_distance: Whether to return both ``(distances, indices)`` object arrays.
            sort_results: Whether to sort each isolate's neighbors in ascending distance order.

        Returns:
            A 2-tuple ``(neigh_dist, neigh_ind)`` of 1D object arrays when ``return_distance=True``,
            or ``neigh_ind`` alone when ``return_distance=False``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
        """
        self.check_is_fitted()
        if X is not None and not isinstance(X, Dataset) and not getattr(self, "_dataset_mode", True):
            return super().radius_neighbors(
                X=X,
                radius=radius,
                return_distance=return_distance,
                sort_results=sort_results,
            )

        target_est = self if X is None else copy.copy(self).fit(X)
        raw_r = self.radius if radius is None else radius
        r_val = float(raw_r) if raw_r is not None else float("inf")
        n = len(target_est._neighbor_indices)

        neigh_dist_list: list[np.ndarray] = []
        neigh_ind_list: list[np.ndarray] = []
        for i in range(n):
            dists = target_est._neighbor_distances[i]
            inds = target_est._neighbor_indices[i]
            if len(dists) == 0:
                neigh_dist_list.append(np.array([], dtype=np.float64))
                neigh_ind_list.append(np.array([], dtype=np.intp))
                continue
            # _neighbor_distances[i] is already sorted in ascending order
            cutoff_idx = int(np.searchsorted(dists, r_val, side="right"))
            d_sub = dists[:cutoff_idx].copy()
            i_sub = inds[:cutoff_idx].copy()
            if not sort_results and len(i_sub) > 1:
                idx_order = np.argsort(i_sub)
                i_sub = i_sub[idx_order]
                d_sub = d_sub[idx_order]
            neigh_dist_list.append(d_sub)
            neigh_ind_list.append(i_sub)

        neigh_ind = np.empty(n, dtype=object)
        neigh_ind[:] = neigh_ind_list
        if not return_distance:
            return neigh_ind

        neigh_dist = np.empty(n, dtype=object)
        neigh_dist[:] = neigh_dist_list
        return neigh_dist, neigh_ind

    def radius_neighbors_graph(
        self,
        X: Dataset | Any = None,
        radius: float | None = None,
        mode: Literal["connectivity", "distance"] = "connectivity",
        sort_results: bool = False,
    ) -> csr_matrix:
        """Computes the (weighted) sparse graph of isolate transmission neighbors within ``radius``.

        Args:
            X: Optional query ``Dataset`` or feature matrix. Defaults to the training data when ``None``.
            radius: Optional genomic distance threshold override. Defaults to ``self.radius``.
            mode: Edge weight mode (``"connectivity"`` for binary ``1.0`` weights or ``"distance"`` for genomic distances).
            sort_results: Whether to sort column indices by distance within each CSR row.

        Returns:
            A square SciPy ``csr_matrix`` of shape ``(n_samples, n_samples)``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
            ValueError: If ``mode`` is not ``"connectivity"`` or ``"distance"``.
        """
        self.check_is_fitted()
        if mode not in ("connectivity", "distance"):
            raise ValueError(f"Unsupported mode={mode!r}. Expected 'connectivity' or 'distance'.")

        dists, inds = self.radius_neighbors(
            X=X,
            radius=radius,
            return_distance=True,
            sort_results=sort_results,
        )
        n = len(inds)
        counts = np.array([len(ix) for ix in inds], dtype=np.intp)
        indptr = np.zeros(n + 1, dtype=np.intp)
        np.cumsum(counts, out=indptr[1:])
        if indptr[-1] == 0:
            return csr_matrix((n, n), dtype=np.float64)

        indices = np.concatenate(list(inds)).astype(np.intp)
        if mode == "connectivity":
            data = np.ones(len(indices), dtype=np.float64)
        else:
            data = np.concatenate(list(dists)).astype(np.float64)

        return csr_matrix((data, indices, indptr), shape=(n, n))

    def kneighbors(
        self,
        X: Dataset | Any = None,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Finds the K-nearest transmission neighbors for each isolate.

        Isolates with fewer than ``n_neighbors`` valid transmission candidates within ``self.radius``
        have remaining slots padded with ``-1`` in ``indices`` and ``np.nan`` in ``distances``.

        Args:
            X: Optional query ``Dataset`` or feature matrix. Defaults to the training data when ``None``.
            n_neighbors: Optional number of neighbors ``k`` to retrieve per isolate. Defaults to ``self.n_neighbors``.
            return_distance: Whether to return ``(distances, indices)`` or ``indices`` only.

        Returns:
            A 2-tuple ``(dist_mat, ind_mat)`` of shape ``(n_samples, n_neighbors)`` when ``return_distance=True``,
            or ``ind_mat`` alone when ``return_distance=False``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
            ValueError: If ``n_neighbors <= 0``.
        """
        self.check_is_fitted()
        if X is not None and not isinstance(X, Dataset) and not getattr(self, "_dataset_mode", True):
            return super().kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance)

        target_est = self if X is None else copy.copy(self).fit(X)
        raw_k = self.n_neighbors if n_neighbors is None else n_neighbors
        k = int(raw_k) if raw_k is not None else 5
        if k <= 0:
            raise ValueError(f"Expected n_neighbors > 0, got {k}.")

        n = len(target_est._neighbor_indices)
        r_val = float(self.radius) if self.radius is not None else np.inf

        dist_mat = np.full((n, k), np.nan, dtype=np.float64)
        ind_mat = np.full((n, k), -1, dtype=np.intp)

        for i in range(n):
            dists = target_est._neighbor_distances[i]
            inds = target_est._neighbor_indices[i]
            if len(dists) == 0:
                continue
            cutoff_idx = int(np.searchsorted(dists, r_val, side="right"))
            take = min(k, cutoff_idx)
            if take > 0:
                dist_mat[i, :take] = dists[:take]
                ind_mat[i, :take] = inds[:take]

        if return_distance:
            return dist_mat, ind_mat
        return ind_mat

    def kneighbors_graph(
        self,
        X: Dataset | Any = None,
        n_neighbors: int | None = None,
        mode: Literal["connectivity", "distance"] = "connectivity",
    ) -> csr_matrix:
        """Computes the (weighted) sparse graph of K-nearest transmission neighbors.

        Args:
            X: Optional query ``Dataset`` or feature matrix. Defaults to the training data when ``None``.
            n_neighbors: Optional number of neighbors ``k`` per isolate. Defaults to ``self.n_neighbors``.
            mode: Edge weight mode (``"connectivity"`` for binary ``1.0`` weights or ``"distance"`` for genomic distances).

        Returns:
            A square SciPy ``csr_matrix`` of shape ``(n_samples, n_samples)``.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
            ValueError: If ``mode`` is not ``"connectivity"`` or ``"distance"``.
        """
        self.check_is_fitted()
        if mode not in ("connectivity", "distance"):
            raise ValueError(f"Unsupported mode={mode!r}. Expected 'connectivity' or 'distance'.")

        dist_mat, ind_mat = self.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=True)
        n = ind_mat.shape[0]
        row_list: list[np.ndarray] = []
        col_list: list[np.ndarray] = []
        val_list: list[np.ndarray] = []

        for i in range(n):
            valid = ind_mat[i] >= 0
            if np.any(valid):
                cols = ind_mat[i, valid]
                row_list.append(np.full(len(cols), i, dtype=np.intp))
                col_list.append(cols)
                if mode == "connectivity":
                    val_list.append(np.ones(len(cols), dtype=np.float64))
                else:
                    val_list.append(dist_mat[i, valid])

        if not row_list:
            return csr_matrix((n, n), dtype=np.float64)

        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
        return coo_array((vals, (rows, cols)), shape=(n, n)).tocsr()

    def fit_predict(self, X: Dataset | Any, y: Any = None) -> np.ndarray:
        """Fits the transmission model on ``X`` and returns integer cluster ``labels_``.

        Args:
            X: A ``Dataset`` with attached ``GenomicDistances``, or a 2D precomputed/feature matrix.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            1D int64 NumPy array of cluster labels for each isolate.
        """
        return self.fit(X, y).labels_

    def to_dataset(
        self,
        dataset: Dataset,
        name: str = "transmission_cluster",
    ) -> Dataset:
        """Returns a new ``Dataset`` with the fitted transmission cluster labels joined as a ``TRANSMISSION_CLONE`` column.

        Args:
            dataset: Target ``Dataset`` to enrich with fitted transmission cluster labels.
            name: Column name for the new ``Taxonomy(resolution=TRANSMISSION_CLONE)`` variable.

        Returns:
            A new ``Dataset`` containing the joined ``name`` column and registered ``Taxonomy`` variable.

        Raises:
            RuntimeError: If the estimator has not been fitted yet.
        """
        self.check_is_fitted()
        id_col = dataset.identifier.column
        cluster_df = self.isolates_.select(
            [
                pl.col(self.id_col_).alias(id_col),
                pl.col("cluster").cast(pl.Categorical).alias(name),
            ]
        )
        lf = dataset.data.lazy()
        if name in dataset.data.columns:
            lf = lf.drop(name)
        new_data = lf.join(cluster_df.lazy(), on=id_col, how="left").collect()

        new_vars = dict(dataset.variables)
        new_vars[name] = Taxonomy(
            column=name,
            title=name,
            resolution=Taxonomy.Resolution.TRANSMISSION_CLONE,
            dtype=pl.Categorical(),
        )
        return Dataset(
            data=new_data,
            name=dataset.name,
            variables=new_vars,
            distances=dataset.distances,
        )
