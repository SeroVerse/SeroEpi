"""Top-level `DatasetTab` composing all dataset & target subcomponents."""

from typing import Any

from htmltools import TagChild
from shiny import render
from shiny import ui as shiny_ui
from shiny_kit.component import Tab, subcomponent
from shiny_kit.decorators import bind_download, bind_effect, bind_output
from shiny_kit.ui import Icon

from seroepi.app.dataset.composer import TargetComposer
from seroepi.app.dataset.enrich import EnrichClusterPanel
from seroepi.app.dataset.explorer import DataFrameExplorer
from seroepi.app.dataset.filters import FilterPanel
from seroepi.app.dataset.loader import DatasetLoader
from seroepi.app.dataset.merger import DatasetMerger
from seroepi.app.models import SessionState


class DatasetTab(Tab[SessionState]):
    """Dataset & Targets.

    Composes `DatasetLoader`, `FilterPanel`, `EnrichClusterPanel`, `DataFrameExplorer`,
    `TargetComposer`, and `DatasetMerger` into the primary data & target engineering page.

    Attributes:
        icon (Tag): Lucide `"database"` navbar icon.
        loader (DatasetLoader): Sidebar subcomponent for ingesting datasets from examples, files, or Pathogenwatch.
        filters (FilterPanel): Sidebar subcomponent for toggling QC rules and semantic variable filters.
        enrich (EnrichClusterPanel): Sidebar subcomponent for genomic clustering and `CountryData` spatial enrichment.
        explorer (DataFrameExplorer): Main-panel DataGrid explorer with semantic column-group filtering.
        composer (TargetComposer): Main-panel interactive `Target` composer and repertoire manager.
        merger (DatasetMerger): Main-panel multi-dataset concatenation, enrichment, and column alignment tool.
    """

    icon = Icon.lucide("database")

    loader = subcomponent(DatasetLoader)
    filters = subcomponent(FilterPanel)
    enrich = subcomponent(EnrichClusterPanel)
    explorer = subcomponent(DataFrameExplorer)
    composer = subcomponent(TargetComposer)
    merger = subcomponent(DatasetMerger)

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                shiny_ui.output_ui("active_dataset_bar"),
                shiny_ui.accordion(
                    shiny_ui.accordion_panel("Load Data", cls.loader.ui(), icon=Icon.lucide("database", size=16)),
                    shiny_ui.accordion_panel("Variable Filters", cls.filters.ui(), icon=Icon.lucide("funnel", size=16)),
                    shiny_ui.accordion_panel("Enrich & Cluster", cls.enrich.ui(), icon=Icon.lucide("network", size=16)),
                    id="dataset_sidebar_accordion",
                    open=["Load Data"],
                    multiple=True,
                ),
                width=400,
            ),
            shiny_ui.navset_card_underline(
                shiny_ui.nav_panel(
                    "Dataset Explorer",
                    cls.explorer.ui(),
                    value="sub_explorer",
                    icon=Icon.lucide("table", size=16),
                ),
                shiny_ui.nav_panel(
                    "Target Composer",
                    cls.composer.ui(),
                    value="sub_composer",
                    icon=Icon.lucide("target", size=16),
                ),
                shiny_ui.nav_panel(
                    "Dataset Merger",
                    cls.merger.ui(),
                    value="sub_merger",
                    icon=Icon.lucide("layers", size=16),
                ),
                id="dataset_sub_nav",
            ),
        )

    @bind_output(render.ui, "active_dataset_bar")
    def render_active_dataset_bar(self) -> TagChild:
        """Renders the top-of-sidebar switcher card for the active `Dataset` with `.joblib` export and delete controls.

        Returns:
            UI card fragment summarizing the active dataset's row count, column count, and distance matrix status.
        """
        if self.state.datasets.count == 0:
            return shiny_ui.div(
                shiny_ui.span(
                    Icon.lucide("circle-alert", size=14),
                    " No Dataset Loaded",
                    class_="badge bg-secondary-subtle text-secondary-emphasis p-2 w-100",
                ),
                class_="mb-2",
            )

        ds = self.state.raw_dataset
        filt_ds = self.state.active_dataset
        assert ds is not None and filt_ds is not None

        names = self.state.datasets.names
        active_key = self.state.datasets.active_key or names[0]
        dist_badge = (
            f"{filt_ds.distances.n_samples:,}×{filt_ds.distances.n_samples:,} CSR"
            if filt_ds.distances is not None
            else "No Distances"
        )
        dist_cls = "bg-info-subtle text-info-emphasis" if filt_ds.distances is not None else "bg-light text-muted"

        return shiny_ui.card(
            shiny_ui.card_body(
                shiny_ui.input_select(
                    "active_dataset_select",
                    "Active Dataset",
                    choices={n: n for n in names},
                    selected=active_key,
                ),
                shiny_ui.div(
                    shiny_ui.span(
                        f"{len(filt_ds):,} / {len(ds):,} rows",
                        class_="badge bg-primary-subtle text-primary-emphasis me-1",
                    ),
                    shiny_ui.span(
                        f"{ds.data.width} cols",
                        class_="badge bg-secondary-subtle text-secondary-emphasis me-1",
                    ),
                    shiny_ui.span(dist_badge, class_=f"badge {dist_cls}"),
                    class_="d-flex flex-wrap gap-1 mb-2",
                ),
                shiny_ui.div(
                    shiny_ui.download_button(
                        "btn_export_joblib",
                        ".joblib",
                        icon=Icon.lucide("download", size=13),
                        class_="btn-sm btn-outline-primary flex-fill",
                    ),
                    shiny_ui.input_action_button(
                        "btn_remove_dataset",
                        "Remove",
                        icon=Icon.lucide("trash-2", size=13),
                        class_="btn-sm btn-outline-danger",
                    ),
                    class_="d-flex gap-2",
                ),
                class_="p-2",
            ),
            class_="mb-2 border-primary-subtle shadow-sm",
        )

    @bind_effect("active_dataset_select")
    def _on_select_active_dataset(self) -> None:
        selected = self.val("active_dataset_select")
        if selected and selected in self.state.datasets.items and selected != self.state.datasets.active_key:
            self.state.datasets.select(selected)

    @bind_effect("btn_remove_dataset")
    def _on_remove_dataset(self) -> None:
        key = self.state.datasets.active_key
        if key:
            self.state.datasets.remove(key)
            self.notify(f"Removed dataset '{key}'.")

    @bind_download(
        "btn_export_joblib",
        filename=lambda self: f"{(self.state.active_dataset.name if self.state.active_dataset else 'dataset')}.joblib",
    )
    def export_joblib(self) -> Any:
        """Streams the active `Dataset` serialized as a `.joblib` byte stream for browser download.

        Yields:
            Serialized `.joblib` bytes of `self.state.active_dataset`.
        """
        ds = self.state.active_dataset
        if ds is not None:
            raw_bytes = ds.save(path=None)
            if isinstance(raw_bytes, bytes):
                yield raw_bytes
