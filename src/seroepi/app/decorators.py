"""SeroEpi server binding decorators (`bind_altair`, `bind_datagrid`)."""

from typing import TYPE_CHECKING, Any

import shinywidgets
from shiny_kit.decorators import bind_datagrid, bind_output

from seroepi.plotting.base import BasePlotter

if TYPE_CHECKING:
    from shiny_kit.component import Component


class bind_altair(bind_output):
    """Descriptor & decorator binding a method returning an Altair chart to `shinywidgets.render_altair`.

    Subclasses [`shiny_kit.decorators.bind_output`](file:///Users/tsta0015/Programming/shiny-kit/src/shiny_kit/decorators.py)
    and overrides `transform()` to adapt the chart's dimensions to its parent container via
    [`BasePlotter.fit_to_container`](file:///Users/tsta0015/Programming/SeroEpi/src/seroepi/plotting/base.py#L464).

    Attributes:
        autosize (bool): Whether to automatically adapt the chart to its container dimensions.
    """

    def __init__(
        self,
        id_: str | None = None,
        *,
        autosize: bool = True,
    ) -> None:
        """Initializes `bind_altair` with `shinywidgets.render_altair` and optional container auto-sizing.

        Args:
            id_: Optional output widget ID. Defaults to the decorated method's `__name__`.
            autosize: Whether to automatically adapt the chart to its container dimensions.
        """
        try:
            from altair.jupyter import JupyterChart

            JupyterChart.enable_offline(True)
        except Exception:
            pass
        super().__init__(shinywidgets.render_altair, id_=id_)
        self.autosize = autosize

    def transform(self, component: "Component[Any]", value: Any) -> Any:
        """Resizes the returned Altair chart to fit the active client container dimensions."""
        if not self.autosize or value is None:
            return value
        w: float | None = None
        h: float | None = None
        sess = getattr(component, "_session", None)
        target_id = self.target_id
        if sess is not None and hasattr(sess, "clientdata"):
            try:
                w = sess.clientdata.output_width(target_id)
            except Exception:
                w = None
            try:
                h = sess.clientdata.output_height(target_id)
            except Exception:
                h = None
        return BasePlotter.fit_to_container(value, width=w, height=h)


__all__ = ["bind_altair", "bind_datagrid"]
