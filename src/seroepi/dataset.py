from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl
from sklearn.base import BaseEstimator, ClusterMixin

from seroepi.dist import GenomicDistances
from seroepi.models import DataModel
from seroepi.variables import (
    QC,
    Count,
    Estimate,
    Identifier,
    Metric,
    Resistance,
    Spatial,
    Taxonomy,
    Temporal,
    Trait,
    Variable,
    Virulence,
)


@dataclass(frozen=True, slots=True)
class Dataset(DataModel):
    """Immutable, in-memory isolate dataset container for bacterial genomic epidemiology.

    Wraps an isolate-level Polars `DataFrame`, a strictly validated `variables` dictionary
    containing exactly one `Identifier` variable, and an optional sparse `GenomicDistances`
    matrix synchronized with the dataset's sample identifiers.

    Attributes:
        data: Isolate-level Polars `DataFrame`.
        variables: Mapping of physical column names to semantic `Variable` instances.
        name: Human-readable dataset name.
        distances: Optional sparse pairwise `GenomicDistances` graph attached to the isolates.

    Examples:
        >>> from seroepi.examples import KlebnetNeonatalSepsis
        >>> ds = KlebnetNeonatalSepsis.load()
        >>> qc_ds = ds.qc().filter(Region="Sub-Saharan Africa")
        >>> k_var = qc_ds.var("K_locus")
        >>> k_var.title
        'Capsule'
    """

    distances: GenomicDistances | None = None

    def __post_init__(self) -> None:
        """Synchronizes variable dtypes and validates that exactly one Identifier variable exists."""
        DataModel.__post_init__(self)
        id_vars = [v for v in self.variables.values() if isinstance(v, Identifier)]
        if len(id_vars) != 1:
            raise ValueError(
                f"Dataset must have exactly one Identifier column, found {len(id_vars)}: {[v.column for v in id_vars]}"
            )

    @property
    def identifier(self) -> Identifier:
        """Unique primary sample `Identifier` variable for this dataset."""
        return next(v for v in self.variables.values() if isinstance(v, Identifier))

    @property
    def sample_ids(self) -> pl.Series:
        """Polars `Series` of primary sample identifier values."""
        return self.data[self.identifier.column]

    @property
    def qc_variables(self) -> tuple[QC, ...]:
        """Tuple of all registered `QC` variables in this dataset."""
        return tuple(v for v in self.variables.values() if isinstance(v, QC))

    @property
    def pure_traits(self) -> tuple[Trait, ...]:
        """Tuple of primary `Trait` variables (excluding `Resistance` and `Virulence`)."""
        return tuple(
            v
            for v in self.variables.values()
            if isinstance(v, Trait) and not isinstance(v, (Resistance, Virulence)) and v.column in self.data.columns
        )

    def role_counts(self) -> dict[str, int]:
        """Counts registered semantic `Variable` types in this dataset.

        Returns:
            Alphabetically sorted dictionary mapping `Variable` subclass names to their column counts.
        """
        counts: dict[str, int] = {}
        for v in self.variables.values():
            cls_name = v.__class__.__name__
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return dict(sorted(counts.items()))

    def variable_groups(
        self,
        *,
        include_spatial_temporal: bool = True,
        include_metrics: bool = True,
        annotate_mode: bool = False,
    ) -> dict[str, dict[str, str]]:
        """Groups non-Identifier/non-QC variables by semantic category for UI selectors.

        Args:
            include_spatial_temporal: Whether to include `Spatial` and `Temporal` variables.
            include_metrics: Whether to include `Count` and `Metric` variables.
            annotate_mode: Whether to append `[Compositional]` or `[Binary]` tags to labels.

        Returns:
            Nested dictionary mapping category headings to `{column_name: display_label}` dicts.
        """
        groups: dict[str, dict[str, str]] = {
            "Spatial & Temporal": {},
            "Traits (Serotypes / Targets)": {},
            "Resistance & Virulence": {},
            "Taxonomy & Lineages": {},
            "Counts & Metrics": {},
            "Other Variables": {},
        }
        for col, var in self.variables.items():
            if isinstance(var, (Identifier, QC)) or col not in self.data.columns:
                continue
            base_lbl = f"{var.column} ({var.title})" if var.title != var.column else var.column
            lbl = f"{base_lbl} [{'Compositional' if var.is_compositional else 'Binary'}]" if annotate_mode else base_lbl
            if isinstance(var, (Spatial, Temporal)):
                if include_spatial_temporal:
                    groups["Spatial & Temporal"][col] = lbl
            elif isinstance(var, (Resistance, Virulence)):
                groups["Resistance & Virulence"][col] = lbl
            elif isinstance(var, Trait):
                groups["Traits (Serotypes / Targets)"][col] = lbl
            elif isinstance(var, Taxonomy):
                groups["Taxonomy & Lineages"][col] = lbl
            elif isinstance(var, (Count, Metric)):
                if include_metrics:
                    groups["Counts & Metrics"][col] = lbl
            else:
                groups["Other Variables"][col] = lbl

        return {k: v for k, v in groups.items() if v}

    def select_column_group(self, group: str) -> pl.DataFrame:
        """Returns a column projection of `self.data` for the requested semantic variable group.

        Args:
            group: Semantic group key (`"all"`, `"core"`, `"amr_vir"`, `"traits"`, or `"qc_metrics"`).

        Returns:
            Polars `DataFrame` containing the primary `Identifier` column and all matching group columns.
        """
        if group == "all":
            return self.data

        id_col = self.identifier.column
        selected_cols: list[str] = [id_col]
        for col, var in self.variables.items():
            if col == id_col or col not in self.data.columns:
                continue
            if group == "core" and isinstance(var, (Taxonomy, Spatial, Temporal)):
                selected_cols.append(col)
            elif group == "amr_vir" and isinstance(var, (Resistance, Virulence)):
                selected_cols.append(col)
            elif group == "traits" and isinstance(var, Trait) and not isinstance(var, (Resistance, Virulence)):
                selected_cols.append(col)
            elif group == "qc_metrics" and isinstance(var, (QC, Count, Metric)):
                selected_cols.append(col)

        return self.data.select(selected_cols)

    def filter(self, *predicates: pl.Expr | bool, **kwargs: Any) -> "Dataset":
        """Filters dataset rows by Polars predicates or column equality kwargs and subsets `distances`.

        Args:
            *predicates: Polars boolean filter expressions (`pl.Expr`).
            **kwargs: Column equality constraints (`col_name=value`).

        Returns:
            A new `Dataset` containing the matching rows and correspondingly subsetted `GenomicDistances`.

        Raises:
            KeyError: If any keyword column name is not present in `self.data.columns`.
        """
        exprs: list[pl.Expr | bool] = list(predicates)
        for col_name, val in kwargs.items():
            if col_name not in self.data.columns:
                raise KeyError(f"Column '{col_name}' not found in Dataset columns: {self.data.columns}")
            exprs.append(pl.col(col_name) == val)

        if not exprs:
            return self

        filtered_df = self.data.filter(*exprs)
        sub_dist = (
            self.distances.subset(filtered_df[self.identifier.column].to_list()) if self.distances is not None else None
        )
        return Dataset(
            data=filtered_df,
            name=self.name,
            variables=self.variables,
            distances=sub_dist,
        )

    def qc(self) -> "Dataset":
        """Filters the dataset by all registered `QC` variable thresholds.

        Returns:
            A new `Dataset` retaining only isolates that pass all registered quality-control criteria.
        """
        qc_exprs = [v.filter_expr() for v in self.qc_variables]
        if not qc_exprs:
            return self
        return self.filter(*qc_exprs)

    def with_distances(self, distances: GenomicDistances | None) -> "Dataset":
        """Attaches or replaces the pairwise `GenomicDistances` matrix on this dataset.

        Args:
            distances: `GenomicDistances` instance (or `None` to strip attached distances).

        Returns:
            A new `Dataset` sharing `data` and `variables` with `distances` attached.
        """
        return Dataset(
            data=self.data,
            name=self.name,
            variables=self.variables,
            distances=distances,
        )

    def rename(
        self,
        mapping: Mapping[str, str] | None = None,
        *,
        infer_types: bool = True,
        coalesce: bool = False,
        strict: bool = True,
        **kwargs: str,
    ) -> "Dataset":
        """Renames columns in the dataset and updates their associated semantic Variables.

        Preserves existing `Variable` subclasses and custom titles (updating `.title` to
        the new column name if it previously matched the old column name). When
        `infer_types=True` and a renamed column was a generic untyped `Variable`,
        automatically infers its semantic subclass from the new column name via
        `Variable.guess()` while upholding the single-`Identifier` invariant.

        Args:
            mapping: Dictionary mapping `{old_col: new_col}`.
            infer_types: If True (default), re-runs `Variable.guess(new_col)` when the
                source variable is a base untyped `Variable`.
            coalesce: If False (default), raises `ValueError` when a target column name
                collides with an existing column (except simultaneous swaps like
                `{'a': 'b', 'b': 'a'}`). If True, merges colliding columns via
                `pl.coalesce(target, source)` and drops the redundant source column.
            strict: If True (default), raises `KeyError` when any `old_col` is not found
                in `self.data.columns`. If False, silently ignores missing source columns.
            **kwargs: Additional `old_col="new_col"` keyword mappings.

        Returns:
            A new `Dataset` with renamed DataFrame columns and updated `variables`.
        """
        combined: dict[str, str] = {}
        if mapping is not None:
            combined.update(mapping)
        if kwargs:
            combined.update(kwargs)

        if not combined:
            return self

        active_map: dict[str, str] = {}
        for old_col, new_col in combined.items():
            if not isinstance(new_col, str) or not new_col.strip():
                raise ValueError(f"Target column name for '{old_col}' cannot be empty or whitespace.")
            clean_new = new_col.strip()
            if old_col not in self.data.columns:
                if strict:
                    raise KeyError(f"Column '{old_col}' not found in Dataset columns: {self.data.columns}")
                continue
            if old_col != clean_new:
                active_map[old_col] = clean_new

        if not active_map:
            return self

        unchanged_cols = set(self.data.columns) - set(active_map.keys())
        target_to_sources: dict[str, list[str]] = {}
        for src, dst in active_map.items():
            target_to_sources.setdefault(dst, []).append(src)

        colliding_targets = [dst for dst, srcs in target_to_sources.items() if dst in unchanged_cols or len(srcs) > 1]
        if colliding_targets and not coalesce:
            raise ValueError(
                f"Target column(s) {colliding_targets} already exist or are mapped from multiple columns. "
                "Pass `coalesce=True` to merge values into the target column."
            )

        def _transform_var(old_var: Variable, old_col: str, new_col: str) -> Variable:
            new_title = new_col if old_var.title == old_col else old_var.title
            if infer_types and type(old_var) is Variable:
                guessed = Variable.guess(new_col)
                if not isinstance(guessed, Identifier):
                    resolved_title = new_title if new_title != new_col else guessed.title
                    return guessed.copy(
                        title=resolved_title,
                        dtype=old_var.dtype,
                        negative_indicator=old_var.negative_indicator or guessed.negative_indicator,
                    )
            return old_var.copy(column=new_col, title=new_title)

        df = self.data
        if not colliding_targets:
            # Pure 1-to-1 rename (supports simultaneous swaps like {'a': 'b', 'b': 'a'})
            df = df.rename(active_map)
            new_vars: dict[str, Variable] = {}
            inv_map = {dst: src for src, dst in active_map.items()}
            for col in df.columns:
                if col in inv_map:
                    src_col = inv_map[col]
                    old_var = self.variables.get(src_col, Variable(src_col))
                    new_vars[col] = _transform_var(old_var, src_col, col)
                else:
                    new_vars[col] = self.variables.get(col, Variable(col))
        else:
            # Coalesce colliding targets and rename non-colliding targets
            pure_rename_map: dict[str, str] = {}
            coalesce_exprs: list[pl.Expr] = []
            cols_to_drop: list[str] = []
            coalesced_from: dict[str, list[str]] = {}

            for dst, srcs in target_to_sources.items():
                if dst in unchanged_cols or len(srcs) > 1:
                    if dst in unchanged_cols:
                        chain = [dst, *srcs]
                        cols_to_drop.extend(srcs)
                    else:
                        # Multiple srcs mapping to a brand-new dst: rename first src -> dst, coalesce rest
                        primary_src = srcs[0]
                        pure_rename_map[primary_src] = dst
                        chain = [primary_src, *srcs[1:]]
                        cols_to_drop.extend(srcs[1:])
                    chain_dtypes = [df.schema[c] for c in chain]
                    if len(set(chain_dtypes)) > 1:
                        target_dtype = chain_dtypes[0]
                        if any(dt in (pl.String, pl.Utf8, pl.Categorical) for dt in chain_dtypes):
                            coalesced_col = pl.coalesce([pl.col(c).cast(pl.String) for c in chain])
                            if target_dtype == pl.Categorical:
                                coalesced_col = coalesced_col.cast(pl.Categorical)
                        else:
                            coalesced_col = pl.coalesce([pl.col(c).cast(target_dtype, strict=False) for c in chain])
                    else:
                        coalesced_col = pl.coalesce([pl.col(c) for c in chain])
                    coalesce_exprs.append(coalesced_col.alias(chain[0]))
                    coalesced_from[dst] = chain
                else:
                    pure_rename_map[srcs[0]] = dst

            if coalesce_exprs:
                df = df.with_columns(coalesce_exprs)
            if cols_to_drop:
                df = df.drop(cols_to_drop)
            if pure_rename_map:
                df = df.rename(pure_rename_map)

            inv_pure = {dst: src for src, dst in pure_rename_map.items()}
            new_vars = {}
            for col in df.columns:
                if col in coalesced_from:
                    chain = coalesced_from[col]
                    anchor_src = chain[0]
                    base_var = self.variables.get(anchor_src, Variable(anchor_src))
                    # If any source in the coalesce chain was the Identifier, preserve Identifier on col
                    if any(isinstance(self.variables.get(s), Identifier) for s in chain):
                        new_vars[col] = Identifier(
                            column=col,
                            title=col if base_var.title == anchor_src else base_var.title,
                            dtype=base_var.dtype,
                        )
                    else:
                        resolved_var = _transform_var(base_var, anchor_src, col)
                        if type(resolved_var) is Variable:
                            for s in chain[1:]:
                                cand = self.variables.get(s)
                                if cand is not None and type(cand) is not Variable and not isinstance(cand, Identifier):
                                    resolved_var = _transform_var(cand, s, col)
                                    break
                        new_vars[col] = resolved_var
                elif col in inv_pure:
                    src_col = inv_pure[col]
                    old_var = self.variables.get(src_col, Variable(src_col))
                    new_vars[col] = _transform_var(old_var, src_col, col)
                else:
                    new_vars[col] = self.variables.get(col, Variable(col))

        return Dataset(
            data=df,
            name=self.name,
            variables=new_vars,
            distances=self.distances,
        )

    def enrich(
        self,
        other: "Dataset",
        name: str | None = None,
        *,
        on: str | None = None,
    ) -> "Dataset":
        """Enriches the dataset with variables and columns from another Dataset.

        Performs a left join anchored to `self.identifier` and `other.identifier`
        (or an explicit `on` column present in both datasets), deduplicating the
        right-hand dataset on the join key so row counts are strictly preserved.
        For columns that exist in both datasets, null values in `self` are filled with
        values from `other` (coalesce).

        Args:
            other: The secondary Dataset providing additional attributes or metadata.
            name: Optional name for the resulting Dataset. Defaults to `self.name`.
            on: Optional shared column name to join on instead of `identifier`.

        Returns:
            A new Dataset with enriched attributes, preserved single Identifier,
            and preserved GenomicDistances.
        """
        if on is not None and on in self.data.columns and on in other.data.columns:
            l_key = on
            r_key = on
        else:
            l_key = self.identifier.column
            r_key = other.identifier.column

        shared_cols = [c for c in other.data.columns if c in self.data.columns and c != r_key and c != l_key]
        new_cols = [c for c in other.data.columns if c not in self.data.columns and c != r_key]

        other_sub = other.data.select([r_key] + shared_cols + new_cols).unique(
            subset=[r_key], keep="first", maintain_order=True
        )
        joined = self.data.join(other_sub, left_on=l_key, right_on=r_key, how="left", suffix="_other")

        if shared_cols:
            coalesce_exprs = [pl.coalesce(pl.col(c), pl.col(f"{c}_other")).alias(c) for c in shared_cols]
            drop_suffixes = [f"{c}_other" for c in shared_cols]
            joined = joined.with_columns(coalesce_exprs).drop(drop_suffixes)

        new_vars = dict(self.variables)
        for c in new_cols:
            var = other.variables.get(c)
            if var is not None:
                if isinstance(var, Identifier):
                    new_vars[c] = Variable(column=var.column, title=var.title, dtype=var.dtype)
                else:
                    new_vars[c] = var
            elif c not in new_vars:
                new_vars[c] = Variable(column=c)

        id_col = self.identifier.column
        resolved_distances = self.distances
        if resolved_distances is not None and other.distances is not None:
            combined_dist = resolved_distances.concat(other.distances).subset(joined[id_col].to_list())
            resolved_distances = combined_dist if len(combined_dist) > 0 else None
        elif resolved_distances is None and other.distances is not None:
            sub_dist = other.distances.subset(joined[id_col].to_list())
            resolved_distances = sub_dist if len(sub_dist) > 0 else None

        return Dataset(
            data=joined,
            name=name or self.name,
            variables=new_vars,
            distances=resolved_distances,
        )

    def concat(
        self,
        *others: "Dataset",
        name: str | None = None,
        deduplicate: bool = True,
        source_col: str | None = None,
    ) -> "Dataset":
        """Vertically stacks samples from one or more other Datasets into a combined Dataset.

        Automatically aligns differing Identifier column names onto `self.identifier.column`,
        optionally deduplicates overlapping sample IDs by coalescing non-null column values
        in dataset order, merges any attached `GenomicDistances` into a unified sparse
        distance graph, and optionally records dataset provenance in `source_col`.

        Args:
            *others: One or more Datasets to concatenate with this dataset.
            name: Optional name for the resulting Dataset. Defaults to `self.name`.
            deduplicate: If True, collapses duplicate sample IDs by coalescing the first
                non-null value in each column across datasets.
            source_col: Optional column name (e.g. 'source_dataset') to record the source
                Dataset name for each isolate row.

        Returns:
            A new combined Dataset containing all sample rows, with missing columns
            filled with nulls across distinct schemas, and merged GenomicDistances.

        Raises:
            TypeError: If any item in `others` is not a Dataset.
        """
        if not others and not source_col:
            return self

        for idx, other in enumerate(others):
            if not isinstance(other, Dataset):
                raise TypeError(
                    f"All arguments to concat must be Dataset instances, got {type(other).__name__} at index {idx}"
                )

        id_col = self.identifier.column
        id_dtype = self.identifier.dtype
        all_datasets = (self, *others)

        aligned_frames: list[pl.DataFrame] = []
        for ds_item in all_datasets:
            df_item = ds_item.data
            other_id_col = ds_item.identifier.column
            if other_id_col != id_col:
                if id_col in df_item.columns:
                    df_item = df_item.drop(id_col)
                df_item = df_item.rename({other_id_col: id_col})
            df_item = df_item.with_columns(pl.col(id_col).cast(id_dtype))
            if source_col:
                df_item = df_item.with_columns(pl.lit(ds_item.name).cast(pl.Categorical).alias(source_col))
            aligned_frames.append(df_item)

        combined_df = pl.concat(aligned_frames, how="diagonal_relaxed")

        if deduplicate and combined_df[id_col].n_unique() < len(combined_df):
            other_cols = [c for c in combined_df.columns if c != id_col]
            combined_df = (
                combined_df.group_by(id_col, maintain_order=True)
                .agg([pl.col(c).drop_nulls().first().alias(c) for c in other_cols])
                .select([id_col, *other_cols])
            )

        new_vars = dict(self.variables)
        for other in others:
            other_id_col = other.identifier.column
            for col, var in other.variables.items():
                if col == other_id_col or col in new_vars:
                    continue
                if isinstance(var, Identifier):
                    new_vars[col] = Variable(column=var.column, title=var.title, dtype=var.dtype)
                else:
                    new_vars[col] = var

        if source_col:
            new_vars[source_col] = Variable(
                column=source_col,
                title="Source Dataset",
                dtype=pl.Categorical(),
            )

        avail_dists = [d.distances for d in all_datasets if d.distances is not None and len(d.distances) > 0]
        resolved_distances: GenomicDistances | None = None
        if avail_dists:
            merged_dist = avail_dists[0].concat(*avail_dists[1:]).subset(combined_df[id_col].to_list())
            if len(merged_dist) > 0:
                resolved_distances = merged_dist

        return Dataset(
            data=combined_df,
            name=name or self.name,
            variables={k: v for k, v in new_vars.items() if k in combined_df.columns},
            distances=resolved_distances,
        )

    def with_clusters(
        self,
        clusterer: ClusterMixin | BaseEstimator,
        name: str = "cluster",
        allow_dense: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        """Runs clustering on attached distances and joins cluster labels to the dataset.

        When `clusterer` is a `TransmissionEstimator`, fits it directly on `self` (so
        clonal `strata`, `Spatial`, and `Temporal` cutoffs/resolutions are applied alongside
        genomic distances) and registers the column with `Taxonomy.Resolution.TRANSMISSION_CLONE`.

        Args:
            clusterer: A scikit-learn compatible clustering estimator or `TransmissionEstimator`.
            name: Column name for the resulting cluster labels in the dataset.
            allow_dense: Whether to allow automatic dense conversion for matrices > 5,000 samples.
            **kwargs: Additional keyword arguments passed to clusterer fit/fit_predict.

        Returns:
            A new Dataset with the cluster column joined and registered in variables.
        """
        if self.distances is None:
            raise ValueError("Cannot cluster dataset: no distances attached. Call with_distances() first.")

        to_dataset_fn = getattr(clusterer, "to_dataset", None)
        fit_fn = getattr(clusterer, "fit", None)
        if callable(to_dataset_fn) and callable(fit_fn):
            fitted_trans: Any = fit_fn(self, **kwargs)
            return fitted_trans.to_dataset(self, name=name)

        cluster_series = self.distances.cluster(
            clusterer,
            allow_dense=allow_dense,
            **kwargs,
        )

        id_col = self.identifier.column
        clusters_df = pl.DataFrame(
            {
                id_col: self.distances.samples,
                name: pl.Series(name, cluster_series, dtype=pl.Categorical),
            }
        )
        lf = self.data.lazy()
        if name in self.data.columns:
            lf = lf.drop(name)
        new_data = lf.join(clusters_df.lazy(), on=id_col, how="left").collect()

        new_vars = dict(self.variables)
        new_vars[name] = Taxonomy(
            column=name,
            title=name,
            resolution=Taxonomy.Resolution.GENOMIC_CLONE,
            dtype=pl.Categorical(),
        )

        return Dataset(
            data=new_data,
            name=self.name,
            variables=new_vars,
            distances=self.distances,
        )

    def resolve_coordinates(
        self,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        output_col: str = "country",
        country_data: Any = None,
        fallback_nearest_km: float = 25.0,
    ) -> "Dataset":
        """Reverse geocodes GPS coordinates to ISO3 country codes via CountryData.

        Args:
            lat_col: Name of latitude column.
            lon_col: Name of longitude column.
            output_col: Name of the resulting country column to create.
            country_data: Optional CountryData instance. If None, uses CountryData.get_instance().
            fallback_nearest_km: Distance tolerance in km to match coastal points to nearest country.

        Returns:
            New Dataset with resolved country column and registered Spatial variable.
        """
        from seroepi.country import CountryData

        cd = country_data or (CountryData.get_instance() if CountryData.is_built() else CountryData.build())
        latname = self.var(lat_col).column if lat_col in self.variables else lat_col
        lonname = self.var(lon_col).column if lon_col in self.variables else lon_col

        resolved_df = cd.reverse_geocode(
            self.data,
            lat_col=latname,
            lon_col=lonname,
            output_col=output_col,
            fallback_nearest_km=fallback_nearest_km,
        )
        new_vars = dict(self.variables)
        new_vars[output_col] = Spatial(column=output_col, resolution=Spatial.Resolution.COUNTRY)

        return Dataset(
            data=resolved_df,
            variables=new_vars,
            name=self.name,
            distances=self.distances,
        )

    def with_spatial_metadata(
        self,
        country_col: str = "country",
        country_data: Any = None,
    ) -> "Dataset":
        """Enriches the dataset with country demographics, UN subregions, continents, and centroids.

        Args:
            country_col: Column name containing country names or ISO3 codes.
            country_data: Optional CountryData instance. If None, uses CountryData.get_instance().

        Returns:
            New Dataset enriched with spatial metadata columns.
        """
        from seroepi.country import CountryData

        cd = country_data or (CountryData.get_instance() if CountryData.is_built() else CountryData.build())
        actual_col = self.var(country_col).column if country_col in self.variables else country_col

        metadata_cols = [
            "iso3",
            "region",
            "subregion",
            "continent",
            "capital_city",
            "centroid_lat",
            "centroid_lon",
            "population",
            "life_expectancy",
            "life_expectancy_year",
            "income_level",
            "lending_type",
        ]
        available_meta = [c for c in metadata_cols if c in cd.data.columns]
        dim_df = cd.data.select(available_meta)

        spatial_iso = Spatial(column=actual_col)
        df_base = self.data.drop([c for c in available_meta if c in self.data.columns and c != actual_col])
        df_with_key = df_base.with_columns(spatial_iso.iso3_expr(strict=True, country_data=cd).alias("_iso3_join_key"))

        joined = df_with_key.join(dim_df, left_on="_iso3_join_key", right_on="iso3", how="left")
        if "iso3" not in joined.columns:
            joined = joined.rename({"_iso3_join_key": "iso3"})
        else:
            joined = joined.drop("_iso3_join_key")

        new_vars = dict(self.variables)
        if "iso3" in joined.columns and "iso3" != self.identifier.column:
            new_vars["iso3"] = Spatial(column="iso3", title="ISO3 Country Code", resolution=Spatial.Resolution.COUNTRY)
        new_vars["region"] = Spatial(column="region", resolution=Spatial.Resolution.CONTINENT)
        new_vars["subregion"] = Spatial(column="subregion", resolution=Spatial.Resolution.REGION)
        new_vars["continent"] = Spatial(column="continent", resolution=Spatial.Resolution.CONTINENT)
        new_vars["capital_city"] = Spatial(column="capital_city", resolution=Spatial.Resolution.CITY)
        new_vars["centroid_lat"] = Spatial(column="centroid_lat", resolution=Spatial.Resolution.POINT)
        new_vars["centroid_lon"] = Spatial(column="centroid_lon", resolution=Spatial.Resolution.POINT)
        new_vars["population"] = Count(column="population")
        new_vars["life_expectancy"] = Estimate(column="life_expectancy", title="life_expectancy", bounded=False)
        new_vars["life_expectancy_year"] = Temporal(column="life_expectancy_year", resolution=Temporal.Resolution.YEAR)
        new_vars["income_level"] = Variable(column="income_level")
        new_vars["lending_type"] = Variable(column="lending_type")

        return Dataset(
            data=joined,
            variables={k: v for k, v in new_vars.items() if k in joined.columns},
            name=self.name,
            distances=self.distances,
        )
