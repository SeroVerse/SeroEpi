"""SeroEpi explorer components (`TableExplorer`, `PlotterExplorer`) subclassing `shiny-kit` bases."""

from typing import Any, ClassVar, Protocol

import altair as alt
import polars as pl
import shinywidgets
from htmltools import TagChild
from shiny import req
from shiny_kit.explorers import BasePlotterExplorer
from shiny_kit.explorers import TableExplorer as BaseTableExplorer
from shiny_kit.widgets import WidgetSelector

from seroepi.app.decorators import bind_altair
from seroepi.app.widgets import PlotterWidgetSelector


class PlotterProtocol(Protocol):
    """Structural protocol for Altair chart builders used by `PlotterExplorer`."""

    def plot(self, *args: Any, **kwargs: Any) -> "alt.TopLevelMixin": ...


class TableExplorer[M](BaseTableExplorer[M]):
    """Multi-view Polars DataFrame and CSV export explorer specialized for SeroEpi models."""

    def resolve_table(self, subject: M, mode: str) -> pl.DataFrame | None:
        """Returns the Polars `DataFrame` for `subject` and `mode`."""
        raise NotImplementedError


class PlotterExplorer[M](BasePlotterExplorer[M]):
    """Interactive Altair plotter explorer with `PlotterWidgetSelector` controls and container auto-sizing.

    Subclasses [`shiny_kit.explorers.BasePlotterExplorer`](file:///Users/tsta0015/Programming/shiny-kit/src/shiny_kit/explorers.py)
    to bind [`PlotterWidgetSelector`](file:///Users/tsta0015/Programming/SeroEpi/src/seroepi/app/widgets.py)
    and render Altair charts via [`bind_altair`](file:///Users/tsta0015/Programming/SeroEpi/src/seroepi/app/decorators.py).
    """

    widget_selector_cls: ClassVar[type[WidgetSelector]] = PlotterWidgetSelector

    def render_chart_widget(self) -> TagChild:
        """Renders the `shinywidgets.output_widget` container for the Altair chart."""
        return shinywidgets.output_widget(
            self.chart_output_id,
            width="100%",
            height=self.chart_height,
            fill=True,
            fillable=True,
        )

    @bind_altair("main_altair_chart")
    def main_altair_chart(self) -> alt.TopLevelMixin:
        """Reactively renders the active Altair chart fitted to its container."""
        chart = self.current_chart()
        req(chart is not None)
        assert chart is not None
        return chart


__all__ = ["PlotterExplorer", "PlotterProtocol", "TableExplorer"]
