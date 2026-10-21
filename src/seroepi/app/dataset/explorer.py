"""Polars DataFrame Explorer subcomponent (`DataFrameExplorer`)."""

from typing import Any, ClassVar

import polars as pl
from htmltools import TagChild
from shiny import render, req
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_datagrid, bind_output
from shiny_kit.ui import EmptyState, Icon

from seroepi.app.models import SessionState


class DataFrameExplorer(Component[SessionState]):
    """Interactive Polars DataGrid view with semantic column-group filtering and role badges.

    Attributes:
        empty_state (ClassVar[EmptyState]): Placeholder card rendered when no dataset is loaded.
    """

    empty_state: ClassVar[EmptyState] = EmptyState(
        icon="database",
        title="Welcome to SeroEpi Dataset Explorer",
        message=(
            "Use the left sidebar to load a benchmark surveillance dataset, "
            "upload local genotype / isolate tables, load a saved .joblib dataset, or fetch a collection "
            "directly from Pathogenwatch."
        ),
    )

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.TagList(
            shiny_ui.output_ui("explorer_header_ui"),
            shiny_ui.output_data_frame("dataset_grid"),
        )

    @bind_output(render.ui, "explorer_header_ui")
    def render_explorer_header_ui(self) -> TagChild:
        """Renders the dataset summary header bar, semantic role count badges, and column-group selector.

        Returns:
            Header toolbar UI tag or `empty_state.ui()` when no dataset is active.
        """
        ds = self.state.raw_dataset
        filt_ds = self.state.active_dataset
        if ds is None or filt_ds is None:
            return self.empty_state.ui()

        raw_n, filt_n = len(ds), len(filt_ds)
        pct = (100.0 * filt_n / raw_n) if raw_n > 0 else 0.0
        n_active_filters = len(self.state.active_filter_exprs)

        role_badges = [
            shiny_ui.span(f"{cls_name}: {cnt}", class_="badge bg-secondary-subtle text-secondary-emphasis me-1")
            for cls_name, cnt in filt_ds.role_counts().items()
        ]

        selected_group = self.val("explorer_col_group", "all", isolate=True)
        return shiny_ui.div(
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.span(Icon.lucide("database", size=16), f" {filt_ds.name}", class_="fw-bold me-2"),
                    shiny_ui.span(f"{filt_n:,} / {raw_n:,} Isolates ({pct:.1f}%)", class_="badge bg-primary me-1 p-2"),
                    shiny_ui.span(
                        f"{n_active_filters} Active Filter(s)",
                        class_=(
                            "badge bg-warning-subtle text-warning-emphasis me-1 p-2"
                            if n_active_filters > 0
                            else "badge bg-secondary-subtle text-secondary-emphasis me-1 p-2"
                        ),
                    ),
                    *role_badges,
                    class_="d-flex flex-wrap align-items-center gap-1",
                ),
                shiny_ui.div(
                    shiny_ui.input_select(
                        "explorer_col_group",
                        label=None,
                        choices={
                            "all": f"All Columns ({filt_ds.data.width})",
                            "core": "Core Metadata (ID, Taxonomy, Spatial, Temporal)",
                            "traits": "Traits (Serotypes / Targets)",
                            "amr_vir": "Resistance & Virulence",
                            "qc_metrics": "QC & Numeric Metrics",
                        },
                        selected=selected_group,
                        width="280px",
                    ),
                ),
                class_="d-flex flex-wrap justify-content-between align-items-center mb-2 gap-2",
            ),
        )

    @bind_datagrid("dataset_grid", filters=True, height="600px")
    def render_dataset_grid(self) -> pl.DataFrame:
        """Slices the active `Dataset` to the selected semantic column group for the DataGrid.

        Returns:
            Polars `DataFrame` containing the columns belonging to the selected group.
        """
        filt_ds = self.state.active_dataset
        req(filt_ds is not None)
        assert filt_ds is not None
        return filt_ds.select_column_group(self.val("explorer_col_group", "all"))
