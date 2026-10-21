from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from htmltools import TagChild
from shiny import ui as shiny_ui
from shiny_kit.widgets import ParameterSpec, WidgetSelector


class EstimatorWidgetSelector(WidgetSelector):
    """Specialized `WidgetSelector` for estimator classes with constrained numeric inputs."""

    default_exclude: ClassVar[set[str]] = {"name"}
    non_ui_types: ClassVar[tuple[type[Any], ...]] = (np.random.Generator,)

    _STEP_100_PARAMS: ClassVar[set[str]] = {"num_samples", "num_warmup", "num_steps", "n_resamples"}
    _MIN_1_INT_PARAMS: ClassVar[set[str]] = {
        "num_samples",
        "num_warmup",
        "num_steps",
        "n_resamples",
        "num_chains",
        "max_tree_depth",
    }

    def __init__(self) -> None:
        super().__init__()
        self._register_estimator_rules()

    def _register_estimator_rules(self) -> None:
        @self.register(lambda s: s.name == "alpha")
        def _build_alpha(input_id: str, spec: ParameterSpec) -> TagChild:
            val = float(spec.default) if isinstance(spec.default, (int, float)) else 0.05
            return shiny_ui.input_numeric(
                input_id,
                label=self._create_label(spec),
                value=val,
                min=0.001,
                max=0.5,
                step=0.01,
            )

        @self.register(lambda s: s.name == "target_accept_prob")
        def _build_accept_prob(input_id: str, spec: ParameterSpec) -> TagChild:
            val = float(spec.default) if isinstance(spec.default, (int, float)) else 0.8
            return shiny_ui.input_numeric(
                input_id,
                label=self._create_label(spec),
                value=val,
                min=0.1,
                max=0.99,
                step=0.05,
            )

        @self.register(lambda s: s.name == "learning_rate")
        def _build_lr(input_id: str, spec: ParameterSpec) -> TagChild:
            val = float(spec.default) if isinstance(spec.default, (int, float)) else 0.01
            return shiny_ui.input_numeric(
                input_id,
                label=self._create_label(spec),
                value=val,
                min=0.0001,
                max=1.0,
                step=0.005,
            )

        @self.register(lambda s: s.name in self._MIN_1_INT_PARAMS)
        def _build_count(input_id: str, spec: ParameterSpec) -> TagChild:
            val = int(spec.default) if isinstance(spec.default, int) else 1
            step = 100 if spec.name in self._STEP_100_PARAMS else 1
            return shiny_ui.input_numeric(
                input_id,
                label=self._create_label(spec),
                value=val,
                min=1,
                step=step,
            )


class PlotterWidgetSelector(WidgetSelector):
    """Specialized `WidgetSelector` for plotter classes with Vega palettes, hex color pickers, and grouped styling."""

    default_exclude: ClassVar[set[str]] = {
        "y_order",
        "x_order",
        "targets",
        "format_",
        "width",
        "height",
        "center_width",
        "top_height",
        "right_width",
        "side_width",
        "theme",
    }

    VEGA_COLOR_SCHEMES: ClassVar[dict[str, str]] = {
        "tableau10": "Tableau 10 (Categorical)",
        "category10": "Category 10 (Categorical)",
        "dark2": "Dark 2 (Categorical)",
        "set2": "Set 2 (Categorical)",
        "viridis": "Viridis (Sequential)",
        "magma": "Magma (Sequential)",
        "inferno": "Inferno (Sequential)",
        "plasma": "Plasma (Sequential)",
        "cividis": "Cividis (Sequential)",
        "turbo": "Turbo (Sequential)",
        "blues": "Blues (Sequential)",
        "tealblues": "Teal-Blues (Sequential)",
        "greens": "Greens (Sequential)",
        "oranges": "Oranges (Sequential)",
        "reds": "Reds (Sequential)",
        "purples": "Purples (Sequential)",
        "warmgreys": "Warm Greys (Sequential)",
        "spectral": "Spectral (Diverging)",
        "rdylbu": "Red-Yellow-Blue (Diverging)",
    }

    _STYLING_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "_width",
        "_height",
        "_size",
        "_color",
        "_opacity",
        "_angle",
        "_padding",
        "_spacing",
    )
    _STYLING_NAMES: ClassVar[set[str]] = {"width", "height", "spacing", "padding", "opacity", "font"}

    def __init__(self) -> None:
        super().__init__()
        self._register_plotter_rules()

    @staticmethod
    def _is_hex_color_param(spec: ParameterSpec) -> bool:
        return isinstance(spec.default, str) and spec.default.startswith("#") and len(spec.default) in (4, 7)

    @classmethod
    def _is_styling_param(cls, spec: ParameterSpec) -> bool:
        return (
            spec.name in cls._STYLING_NAMES
            or spec.name.endswith(cls._STYLING_SUFFIXES)
            or cls._is_hex_color_param(spec)
        )

    def _register_plotter_rules(self) -> None:
        @self.register(lambda s: s.name == "color_scheme")
        def _build_color_scheme(input_id: str, spec: ParameterSpec) -> TagChild:
            choices = dict(self.VEGA_COLOR_SCHEMES)
            default_scheme = str(spec.default) if spec.default else "tableau10"
            if default_scheme not in choices:
                choices[default_scheme] = default_scheme.title()
            return shiny_ui.input_select(
                input_id,
                label=self._create_label(spec),
                choices=choices,
                selected=default_scheme,
            )

        @self.register(self._is_hex_color_param)
        def _build_hex_color(input_id: str, spec: ParameterSpec) -> TagChild:
            default_hex = (
                str(spec.default) if isinstance(spec.default, str) and spec.default.startswith("#") else "#4c78a8"
            )
            sync_js = (
                "const t=this.parentElement.querySelector('input[type=text]');"
                "if(t){t.value=this.value;t.dispatchEvent(new Event('input',{bubbles:true}));"
                "t.dispatchEvent(new Event('change',{bubbles:true}));}"
            )
            return shiny_ui.div(
                shiny_ui.tags.label(self._create_label(spec), class_="form-label mb-1"),
                shiny_ui.div(
                    shiny_ui.tags.input(
                        type="color",
                        value=default_hex,
                        oninput=sync_js,
                        class_="form-control form-control-color me-2",
                        style="width: 2.5rem; min-width: 2.5rem; padding: 0.2rem;",
                        title="Pick color",
                    ),
                    shiny_ui.div(
                        shiny_ui.input_text(input_id, label=None, value=default_hex),
                        class_="flex-grow-1",
                    ),
                    class_="d-flex align-items-center",
                ),
                class_="mb-3",
            )

    def build_form(
        self,
        target_cls: type[Any],
        *,
        exclude: set[str] | None = None,
        choices: Mapping[str, Sequence[str] | Mapping[str, str]] | None = None,
        prefix: str = "",
        group_styling: bool = True,
    ) -> list[TagChild]:
        """Builds Plotter input widgets, grouping low-level styling & dimension controls into a collapsible accordion."""
        specs = self.introspect(target_cls, exclude=exclude)
        if not group_styling:
            return [self._build_single_widget(spec, choices=choices, prefix=prefix) for spec in specs]

        primary_widgets: list[TagChild] = []
        styling_widgets: list[TagChild] = []

        for spec in specs:
            w = self._build_single_widget(spec, choices=choices, prefix=prefix)
            if self._is_styling_param(spec):
                styling_widgets.append(w)
            else:
                primary_widgets.append(w)

        if styling_widgets:
            accordion_id = f"{prefix}styling_accordion" if prefix else "plotter_styling_accordion"
            primary_widgets.append(
                shiny_ui.accordion(
                    shiny_ui.accordion_panel(
                        "Styling & Dimensions",
                        *styling_widgets,
                        value="styling_panel",
                    ),
                    id=accordion_id,
                    open=False,
                    class_="mt-2",
                )
            )

        return primary_widgets
