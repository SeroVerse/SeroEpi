"""Dedicated aggregation engine, parameter validation, and structured Aggregates result container."""

import copy
import itertools
import re
from collections.abc import Sequence
from typing import Any, Literal, Self

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from seroepi.dataset import Dataset
from seroepi.models import StratifiedResultMixin
from seroepi.variables import (
    QC,
    Count,
    Criterion,
    Target,
    Temporal,
    Trait,
    Variable,
)


class Aggregator(StratifiedResultMixin, BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for dataset aggregation, temporal windowing, and stratification.

    Transforms an isolate-level `Dataset` into a structured contingency table (`data_`) with columns
    `[*strata_cols, "target", "event", "n"]`, supporting both mutually exclusive compositional traits
    (e.g. capsule K-locus serotypes) and binary/compound `Target` definitions, with optional cluster
    deduplication (`adjust_for`), Cartesian zero-padding (`pad_zeros`), and rolling temporal windows (`every`).

    Args:
        target: Target `Variable`, `Criterion`, or composite `Target` to aggregate.
        strata: Optional stratification `Variable` or sequence of `Variable` instances (e.g. `Country`, `Year`).
        adjust_for: Optional cluster/outbreak `Variable`; when provided, `event` and `n` count unique clusters
            rather than raw isolate rows.
        negative_indicator: Optional string, regex pattern, or sequence of strings treated as missing/negative
            calls for compositional targets.
        pad_zeros: Whether to zero-pad unobserved `(stratum, target)` cells via Cartesian product grid.
            Recommended (`pad_zeros=True`) prior to fitting hierarchical Bayesian models.
        filter_qc: Whether to automatically apply all `QC` variables registered on the input `Dataset`.
        every: Optional Polars duration string (e.g. `"1y"`, `"6mo"`) for dynamic temporal windowing.
        period: Optional Polars window length string when using sliding temporal windows (defaults to `every`).
        offset: Optional Polars window offset string for dynamic temporal grouping.
        closed: Boundary closure rule (`"left"`, `"right"`, `"none"`, or `"both"`) for temporal windows.
        upsample: Whether to upsample missing regular temporal intervals with zero counts.
        compositional: Optional explicit override for whether `target` is treated as compositional.

    Attributes:
        data_: Fitted Polars `DataFrame` containing `[*strata_cols, "target", "event", "n"]`.
        variables_: Mapping of output column names to semantic `Variable` instances.
        target_: Resolved output `Trait` variable representing the `"target"` column.
        strata_: Tuple of stratification `Variable` instances.
        adjust_for_: Optional cluster-adjustment `Variable`.
        is_compositional_: Whether `data_` represents mutually exclusive compositional categories.
        is_padded_: Whether unobserved cells were zero-padded or upsampled.

    Examples:
        >>> from seroepi.aggregate import Aggregator
        >>> from seroepi.dataset import Dataset
        >>> ds = Dataset.example()
        >>> agg = Aggregator(
        ...     target=ds.var("K_locus"),
        ...     strata=ds.var("Country"),
        ...     pad_zeros=True,
        ... ).fit(ds)
        >>> agg.data_.columns
        ['Country', 'target', 'event', 'n']
    """

    is_fitted_: bool = False
    data_: pl.DataFrame
    variables_: dict[str, Variable]
    target_: Variable
    strata_: tuple[Variable, ...]
    adjust_for_: Variable | None
    is_compositional_: bool
    is_padded_: bool

    def __init__(
        self,
        target: Target | Criterion | Variable,
        strata: Sequence[Variable] | Variable | None = None,
        adjust_for: Variable | None = None,
        negative_indicator: str | re.Pattern[str] | Sequence[str] | None = None,
        pad_zeros: bool = False,
        filter_qc: bool = True,
        every: str | None = None,
        period: str | None = None,
        offset: str | None = None,
        closed: Literal["left", "right", "none", "both"] = "left",
        upsample: bool = False,
        compositional: bool | None = None,
    ) -> None:
        if not isinstance(target, (Target, Criterion, Variable)):
            raise TypeError(
                f"Aggregator target must be a Variable, Criterion, or Target instance, got {type(target).__name__}. "
                "Use dataset.var(...) to retrieve it."
            )
        if adjust_for is not None and not isinstance(adjust_for, Variable):
            raise TypeError(
                f"Aggregator adjust_for must be a Variable instance, got {type(adjust_for).__name__}. "
                "Use dataset.var(...) to retrieve it."
            )
        self.target = target
        self.strata = self._normalize_strata(strata)
        self.adjust_for = adjust_for
        self.negative_indicator = negative_indicator
        self.pad_zeros = pad_zeros
        self.filter_qc = filter_qc
        self.every = every
        self.period = period
        self.offset = offset
        self.closed = closed
        self.upsample = upsample
        self.compositional = compositional

    def _get_frame(self) -> pl.DataFrame:
        return self.data_

    def _set_frame(self, df: pl.DataFrame) -> None:
        self.data_ = df

    @staticmethod
    def _normalize_strata(strata: Sequence[Variable] | Variable | None) -> tuple[Variable, ...]:
        """Normalizes strata parameter into a tuple of strictly Variable items."""
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

    def fit_transform(self, X: Dataset, y: Any = None, **fit_params: Any) -> Self:
        """Fits the aggregator on `X` and returns the fitted `Aggregator` instance.

        Args:
            X: Input isolate-level `Dataset`.
            y: Ignored; present for scikit-learn pipeline compatibility.
            **fit_params: Additional fit keyword parameters (unused).

        Returns:
            Fitted `self`.
        """
        return self.fit(X, y)

    def transform(self, X: Dataset) -> Self:
        """Clones and fits the aggregator on `X` without mutating `self` in-place.

        Args:
            X: Input isolate-level `Dataset` to aggregate using this instance's configuration.

        Returns:
            A newly fitted `Aggregator` clone containing the aggregated `data_` for `X`.

        Raises:
            RuntimeError: If `self` has not been fitted yet.
        """
        self.check_is_fitted()
        return copy.copy(self).fit(X)

    def fit(self, X: Dataset, y: Any = None) -> Self:
        """Executes QC filtering, target resolution, strata projection, windowing, and zero-padding on `X`.

        Args:
            X: Input isolate-level `Dataset`.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            Fitted `self` with populated `data_`, `variables_`, `target_`, and `strata_`.

        Raises:
            KeyError: If `adjust_for` or any required variable column is missing from `X`.
        """
        schema = X.data.schema

        # 1. Coerce target into canonical Target and build output target Variable
        resolved_target = Target.coerce(
            self.target,
            compositional=self.compositional,
            negative_indicator=self.negative_indicator,
        )
        is_compositional = resolved_target.is_compositional
        target_var = Trait(column="target", title=resolved_target.title, compositional=is_compositional)

        # 2. Base Frame Preparation: QC Filter, Strata & Temporal Projections, Cluster Resolution
        lf, strata_vars, strata_cols, temporal_var, cluster_col = self._prepare_base_frame(X, schema)

        # 3. Denominator & Event Aggregation
        denom_expr = pl.col(cluster_col).n_unique().alias("n") if cluster_col else pl.len().alias("n")

        valid_lf: pl.LazyFrame | None = None
        strata_n: pl.LazyFrame | None = None
        target_col: str | None = None

        if is_compositional:
            target_col = "target"
            cat_expr = resolved_target.to_category_expr().alias("target")
            lf = lf.with_columns(cat_expr)
            valid_lf = lf.filter(pl.col("target").is_not_null())
            comp_target_expr = pl.col("target")

            event_expr = pl.col(cluster_col).n_unique().alias("event") if cluster_col else pl.len().alias("event")

            # Stratum denominators (computed over all QC-passed samples in the stratum/window)
            if self.every and temporal_var:
                by_cols = [c for c in strata_cols if c != temporal_var.column]
                strata_n = (
                    lf.sort(temporal_var.column)
                    .group_by_dynamic(
                        index_column=temporal_var.column,
                        every=self.every,
                        period=self.period or self.every,
                        offset=self.offset,
                        closed=self.closed,
                        group_by=by_cols if by_cols else None,
                    )
                    .agg(denom_expr)
                )
                comp_by_cols = [c for c in strata_cols if c != temporal_var.column] + ["target"]
                events_lf = (
                    valid_lf.with_columns(comp_target_expr)
                    .sort(temporal_var.column)
                    .group_by_dynamic(
                        index_column=temporal_var.column,
                        every=self.every,
                        period=self.period or self.every,
                        offset=self.offset,
                        closed=self.closed,
                        group_by=comp_by_cols,
                    )
                    .agg(event_expr)
                )
            else:
                strata_n = lf.group_by(strata_cols).agg(denom_expr) if strata_cols else lf.select(denom_expr)
                group_keys = strata_cols + [comp_target_expr]
                events_lf = valid_lf.group_by(group_keys).agg(event_expr)

            # Join component events with stratum denominators
            if strata_cols:
                agg_lf = events_lf.join(strata_n, on=strata_cols, how="left")
            else:
                agg_lf = events_lf.join(strata_n, how="cross")

            tname = None
        else:
            tname = resolved_target.name
            lf = lf.with_columns(resolved_target.to_expr())
            event_expr = (
                pl.col(cluster_col).filter(pl.col(tname) > 0).n_unique().alias("event")
                if cluster_col
                else pl.col(tname).cast(pl.Float64).sum().alias("event")
            )

            if self.every and temporal_var:
                by_cols = [c for c in strata_cols if c != temporal_var.column]
                agg_lf = (
                    lf.sort(temporal_var.column)
                    .group_by_dynamic(
                        index_column=temporal_var.column,
                        every=self.every,
                        period=self.period or self.every,
                        offset=self.offset,
                        closed=self.closed,
                        group_by=by_cols if by_cols else None,
                    )
                    .agg([event_expr, denom_expr])
                    .with_columns(target=pl.lit(tname))
                )
            else:
                base_agg = (
                    lf.group_by(strata_cols).agg([event_expr, denom_expr])
                    if strata_cols
                    else lf.select([event_expr, denom_expr])
                )
                agg_lf = base_agg.with_columns(target=pl.lit(tname))

        # 4. Apply Zero-Padding or Temporal Upsampling
        agg_lf = self._apply_padding(
            agg_lf=agg_lf,
            lf=lf,
            strata=strata_vars,
            temporal_var=temporal_var,
            target_name=tname,
            is_compositional=is_compositional,
            target_col=target_col,
            valid_lf=valid_lf,
            strata_n=strata_n,
        )

        agg_lf = agg_lf.select(strata_cols + ["target", "event", "n"])

        # 5. Build Output Variables Mapping & Store Fitted Attributes
        out_vars = self._build_variables_map(strata_vars, target_var)
        self.data_ = agg_lf.collect()
        self.variables_ = out_vars
        self.target_ = target_var
        self.strata_ = tuple(strata_vars)
        self.adjust_for_ = self.adjust_for
        self.is_compositional_ = is_compositional
        self.is_padded_ = bool(self.pad_zeros or self.upsample)
        self.is_fitted_ = True
        return self

    def _prepare_base_frame(
        self, dataset: Dataset, schema: Any
    ) -> tuple[pl.LazyFrame, list[Variable], list[str], Temporal | None, str | None]:
        """Applies QC filters, resolves and projects strata/temporal columns, and validates clustering."""
        lf = dataset.data.lazy()

        # 1. Apply Quality Control filters
        if self.filter_qc:
            qc_expr = [v.filter_expr() for v in dataset.variables.values() if isinstance(v, QC)]
            if qc_expr:
                lf = lf.filter(*qc_expr)

        # 2. Project strata polymorphically
        strata_vars = list(self._normalize_strata(self.strata))
        strata_cols = [s.column for s in strata_vars]
        temporal_var = next((s for s in strata_vars if isinstance(s, Temporal)), None)

        if strata_vars:
            projections = [
                s.parse_expr() if isinstance(s, Temporal) and self.every else s.to_expr() for s in strata_vars
            ]
            lf = lf.with_columns(*projections)

        # 3. Validate cluster column for adjustment
        cluster_col = None
        if self.adjust_for:
            cluster_col = self.adjust_for.column
            if cluster_col not in dataset.data.columns and cluster_col not in schema.names():
                raise KeyError(
                    f"Adjustment variable '{cluster_col}' was not found in dataset columns: {dataset.data.columns}."
                )

        return lf, strata_vars, strata_cols, temporal_var, cluster_col

    def _apply_padding(
        self,
        agg_lf: pl.LazyFrame,
        lf: pl.LazyFrame,
        strata: Sequence[Variable],
        temporal_var: Temporal | None,
        target_name: str | None,
        is_compositional: bool,
        target_col: str | None = None,
        valid_lf: pl.LazyFrame | None = None,
        strata_n: pl.LazyFrame | None = None,
    ) -> pl.LazyFrame:
        """Unified zero-padding and temporal upsampling engine."""
        strata_cols = [s.column for s in strata]
        temporal_interval = self.every or (temporal_var.resolution.interval if temporal_var else None)
        upsampled = False

        # Temporal upsample: completes missing regular time-steps within observed groups
        if (self.upsample or (self.pad_zeros and temporal_var)) and temporal_var and temporal_interval:
            df_tmp = agg_lf.collect()
            if len(df_tmp) > 0 and temporal_var.column in df_tmp.columns:
                time_dtype = df_tmp[temporal_var.column].dtype
                if time_dtype in (pl.Date, pl.Datetime):
                    by_cols = [c for c in strata_cols if c != temporal_var.column]
                    if is_compositional and "target" in df_tmp.columns:
                        by_cols = by_cols + ["target"]
                    df_tmp = (
                        df_tmp.sort(temporal_var.column)
                        .upsample(
                            time_column=temporal_var.column,
                            every=temporal_interval,
                            group_by=by_cols if by_cols else None,
                        )
                        .with_columns(
                            pl.col("event").fill_null(0),
                            pl.col("n").fill_null(0),
                            pl.col("target").fill_null(target_name) if not is_compositional else pl.col("target"),
                            *(pl.col(c).forward_fill() for c in by_cols),
                        )
                    )
                    agg_lf = df_tmp.lazy()
                    upsampled = True

        # Cartesian product grid padding: fills unobserved strata (and target components)
        if not upsampled and self.pad_zeros and strata_cols:
            unique_strata = [lf.select(pl.col(s).drop_nulls().unique()).collect()[s].to_list() for s in strata_cols]
            if is_compositional and target_col and valid_lf is not None:
                unique_targets = (
                    valid_lf.select(pl.col(target_col).cast(pl.Utf8).alias("target").drop_nulls().unique())
                    .collect()["target"]
                    .to_list()
                )
                all_dims = unique_strata + [unique_targets]
                all_cols = strata_cols + ["target"]
                grid_rows = list(itertools.product(*all_dims))
                grid_df = pl.DataFrame(grid_rows, schema=all_cols, orient="row").lazy()

                assert strata_n is not None
                agg_lf = (
                    grid_df.join(agg_lf.drop("n"), on=strata_cols + ["target"], how="left")
                    .join(strata_n, on=strata_cols, how="left")
                    .with_columns(
                        pl.col("event").fill_null(0),
                        pl.col("n").fill_null(0),
                    )
                )
            else:
                grid_rows = list(itertools.product(*unique_strata))
                grid_df = pl.DataFrame(grid_rows, schema=strata_cols, orient="row").lazy()
                agg_lf = grid_df.join(agg_lf, on=strata_cols, how="left").with_columns(
                    pl.col("event").fill_null(0),
                    pl.col("n").fill_null(0),
                    pl.col("target").fill_null(target_name),
                )

        return agg_lf

    @staticmethod
    def _build_variables_map(
        strata_vars: list[Variable],
        target_var: Variable,
    ) -> dict[str, Variable]:
        """Constructs the output variable mapping including strata, target, and metrics."""
        out_vars: dict[str, Variable] = {}
        for s in strata_vars:
            out_vars[s.column] = s
        out_vars["target"] = target_var
        out_vars["event"] = Count("event", is_denominator=False)
        out_vars["n"] = Count("n", is_denominator=True)
        return out_vars
