"""Genomic distance clustering and CountryData spatial enrichment subcomponent (`EnrichClusterPanel`)."""

from typing import Any

from htmltools import TagChild
from shiny import render
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_output, bind_task
from shiny_kit.progress import current_progress
from shiny_kit.ui import Icon
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from seroepi.app.models import SessionState
from seroepi.dataset import Dataset
from seroepi.dist import ConnectedComponentsClusterer
from seroepi.variables import Spatial


class EnrichClusterPanel(Component[SessionState]):
    """Genomic distance clustering and CountryData spatial enrichment.

    Provides two interactive enrichment workflows on the active `Dataset`:
    1. **Clusters**: Partitions the attached `GenomicDistances` matrix using Connected
       Components, DBSCAN, or Single-Linkage Agglomerative Clustering.
    2. **Country Data**: Joins UN/World Bank `CountryData` metadata or reverse-geocodes
       latitude/longitude columns into ISO3 country designations.
    """

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.navset_pill(
            shiny_ui.nav_panel(
                "Clusters",
                shiny_ui.output_ui("cluster_tab_ui"),
                value="Clusters",
                icon=Icon.lucide("dna", size=14),
            ),
            shiny_ui.nav_panel(
                "Country Data",
                shiny_ui.output_ui("country_tab_ui"),
                value="Country Data",
                icon=Icon.lucide("globe", size=14),
            ),
            id="enrich_cluster_nav",
        )

    @bind_output(render.ui, "cluster_tab_ui")
    def render_cluster_tab_ui(self) -> TagChild:
        """Renders the genomic distance clustering controls when a distance matrix is attached.

        Returns:
            UI fragment for cluster column naming, algorithm selection, and SNP threshold input.
        """
        ds = self.state.raw_dataset
        if ds is None:
            return shiny_ui.p(
                "Load a dataset first to generate genomic clusters.",
                class_="text-muted small mt-2",
            )

        if ds.distances is None:
            return shiny_ui.div(
                shiny_ui.span(
                    Icon.lucide("circle-alert", size=14),
                    " No GenomicDistances Attached",
                    class_="badge bg-warning-subtle text-warning-emphasis p-2 w-100 mb-2",
                ),
                shiny_ui.p(
                    "Attach a pairwise distance matrix during upload or load an Example / Pathogenwatch "
                    "dataset with distances enabled to generate genomic cluster columns.",
                    class_="text-muted small mb-0",
                ),
                class_="mt-2",
            )

        n_dist = ds.distances.n_samples
        n_total = len(ds)
        return shiny_ui.div(
            shiny_ui.span(
                Icon.lucide("dna", size=14),
                f" Distances: {n_dist:,} / {n_total:,} isolates ({n_dist:,} × {n_dist:,})",
                class_="badge bg-info-subtle text-info-emphasis p-2 w-100 mb-2",
            ),
            shiny_ui.input_text("cluster_col_name", "Cluster Column Name", value="cluster"),
            shiny_ui.input_select(
                "cluster_algorithm",
                "Clustering Algorithm",
                choices={
                    "connected_components": "Connected Components (SNP / Distance Graph)",
                    "dbscan": "DBSCAN (Precomputed Distance Matrix)",
                    "agglomerative": "Agglomerative (Single-Linkage)",
                },
            ),
            shiny_ui.input_numeric("cluster_threshold", "Distance / SNP Threshold", value=20.0, min=0.0, step=1.0),
            shiny_ui.panel_conditional(
                "input.cluster_algorithm === 'dbscan'",
                shiny_ui.input_numeric(
                    "cluster_min_samples", "Minimum Samples per Core Cluster", value=2, min=1, step=1
                ),
            ),
            shiny_ui.input_action_button(
                "btn_run_clustering",
                "Generate Clusters",
                icon=Icon.lucide("network", size=15),
                class_="btn-primary w-100 mt-2",
            ),
            class_="mt-2",
        )

    @bind_output(render.ui, "country_tab_ui")
    def render_country_tab_ui(self) -> TagChild:
        """Renders the `CountryData` join and GPS reverse-geocoding controls.

        Returns:
            UI fragment for spatial enrichment and coordinate resolution.
        """
        ds = self.state.raw_dataset
        if ds is None:
            return shiny_ui.p(
                "Load a dataset first to enrich with CountryData.",
                class_="text-muted small mt-2",
            )

        all_cols = {c: c for c in ds.data.columns}
        country_candidates = [
            v.column
            for v in ds.variables.values()
            if isinstance(v, Spatial) and v.resolution == Spatial.Resolution.COUNTRY
        ]
        default_country_col = country_candidates[0] if country_candidates else next(iter(all_cols.keys()), "")

        point_cols = [
            v.column
            for v in ds.variables.values()
            if isinstance(v, Spatial) and v.resolution == Spatial.Resolution.POINT
        ]
        lat_candidates = [
            c
            for c in [*point_cols, *ds.data.columns]
            if c.lower() in ("lat", "latitude") or c.lower().endswith(("_lat", "_latitude"))
        ]
        lon_candidates = [
            c
            for c in [*point_cols, *ds.data.columns]
            if c.lower() in ("lon", "lng", "longitude") or c.lower().endswith(("_lon", "_lng", "_longitude"))
        ]

        return shiny_ui.div(
            shiny_ui.p("1. Enrich from Country / ISO3 Column", class_="fw-semibold small mb-1"),
            shiny_ui.p(
                "Joins UN subregions, continents, capital coordinates, population, and World Bank life expectancy.",
                class_="text-muted small mb-2",
            ),
            shiny_ui.input_select(
                "spatial_country_col",
                "Country / Location Column",
                choices=all_cols,
                selected=default_country_col,
            ),
            shiny_ui.input_action_button(
                "btn_enrich_spatial",
                "Add CountryData Metadata",
                icon=Icon.lucide("globe", size=15),
                class_="btn-primary btn-sm w-100 mb-3",
            ),
            shiny_ui.hr(class_="my-2"),
            shiny_ui.p("2. Reverse Geocode GPS Coordinates", class_="fw-semibold small mb-1"),
            shiny_ui.input_select(
                "geocode_lat_col",
                "Latitude Column",
                choices=all_cols,
                selected=lat_candidates[0] if lat_candidates else next(iter(all_cols.keys()), ""),
            ),
            shiny_ui.input_select(
                "geocode_lon_col",
                "Longitude Column",
                choices=all_cols,
                selected=lon_candidates[0] if lon_candidates else next(iter(all_cols.keys()), ""),
            ),
            shiny_ui.input_text("geocode_out_col", "Output Country Column", value="country"),
            shiny_ui.input_numeric(
                "geocode_fallback_km", "Coastal Fallback Tolerance (km)", value=25.0, min=0.0, step=5.0
            ),
            shiny_ui.input_action_button(
                "btn_resolve_coords",
                "Resolve GPS to Country (ISO3)",
                icon=Icon.lucide("globe", size=15),
                class_="btn-outline-primary btn-sm w-100 mt-1",
            ),
            class_="mt-2",
        )

    @bind_task("btn_run_clustering", on_success="_on_clustered")
    def task_run_clustering(self) -> tuple[Dataset, str, int]:
        """Executes the selected clustering algorithm on `raw_dataset.distances`.

        Returns:
            Tuple of `(clustered_dataset, column_name, n_clusters)`.

        Raises:
            ValueError: If no active dataset or `GenomicDistances` matrix is available.
        """
        ds = self.state.raw_dataset
        if ds is None or ds.distances is None:
            raise ValueError("Active dataset does not have an attached GenomicDistances matrix.")

        prog = current_progress()
        col_name = self.val("cluster_col_name", "cluster")
        algo = self.val("cluster_algorithm", "connected_components")
        threshold = float(self.val("cluster_threshold", 20.0))

        prog.report(
            0.15,
            message="Clustering genomic distances",
            detail=f"Algorithm: {algo} (threshold={threshold:g})",
        )
        if algo == "dbscan":
            min_samples = int(self.val("cluster_min_samples", 2))
            clusterer: Any = DBSCAN(eps=threshold, min_samples=min_samples, metric="precomputed")
        elif algo == "agglomerative":
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=threshold,
                metric="precomputed",
                linkage="single",
            )
        else:
            clusterer = ConnectedComponentsClusterer(threshold=threshold)

        prog.report(0.45, detail=f"Computing clusters across {len(ds):,} isolates...")
        clustered_ds = ds.with_clusters(clusterer, name=col_name, allow_dense=True)
        n_clusters = int(clustered_ds.data[col_name].drop_nulls().n_unique())
        prog.report(1.0, detail=f"Found {n_clusters:,} clusters")
        return clustered_ds, col_name, n_clusters

    def _on_clustered(self, res: tuple[Dataset, str, int]) -> str:
        clustered_ds, col_name, n_clusters = res
        self.state.register_dataset(clustered_ds, reset_filters=False)
        return f"Added cluster column '{col_name}' ({n_clusters:,} distinct clusters)."

    @bind_task("btn_resolve_coords", on_success="_on_geocoded")
    def task_resolve_coordinates(self) -> tuple[Dataset, str]:
        """Reverse-geocodes latitude and longitude columns into an ISO3 country `Spatial` column.

        Returns:
            Tuple of `(resolved_dataset, output_column_name)`.

        Raises:
            ValueError: If no active dataset or coordinate columns are selected.
        """
        ds = self.state.raw_dataset
        if ds is None:
            raise ValueError("No active dataset available.")
        lat_col = self.val("geocode_lat_col")
        lon_col = self.val("geocode_lon_col")
        out_col = self.val("geocode_out_col", "country")
        fallback_km = float(self.val("geocode_fallback_km", 25.0))
        if not lat_col or not lon_col:
            raise ValueError("Select both Latitude and Longitude columns.")

        prog = current_progress()
        prog.report(
            0.20,
            message="Reverse geocoding coordinates",
            detail=f"Matching ({lat_col}, {lon_col}) across {len(ds):,} isolates...",
        )
        resolved = ds.resolve_coordinates(
            lat_col=lat_col,
            lon_col=lon_col,
            output_col=out_col,
            fallback_nearest_km=fallback_km,
        )
        prog.report(1.0, detail=f"Resolved into '{out_col}'")
        return resolved, out_col

    def _on_geocoded(self, res: tuple[Dataset, str]) -> str:
        resolved_ds, out_col = res
        self.state.register_dataset(resolved_ds, reset_filters=False)
        return f"Resolved GPS coordinates into Spatial column '{out_col}'."

    @bind_task(
        "btn_enrich_spatial",
        register_to="datasets",
        reset_filters=False,
        notify_success="Enriched dataset with CountryData (ISO3, UN regions, continents, population, life expectancy).",
    )
    def task_enrich_spatial(self) -> Dataset:
        """Joins UN/World Bank `CountryData` onto `raw_dataset` using the selected country column.

        Returns:
            Spatially enriched `Dataset` instance.

        Raises:
            ValueError: If no active dataset or country column is selected.
        """
        ds = self.state.raw_dataset
        if ds is None:
            raise ValueError("No active dataset available.")
        country_col = self.val("spatial_country_col")
        if not country_col:
            raise ValueError("Please select a Country / ISO3 column to join against CountryData.")
        return ds.with_spatial_metadata(country_col=country_col)
