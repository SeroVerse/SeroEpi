"""Prevalence Aggregation, Estimation, and Interactive Plotting tab module."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import altair as alt
import polars as pl
from htmltools import TagChild
from shiny import render
from shiny import ui as shiny_ui
from shiny_kit.component import Component, Tab, subcomponent
from shiny_kit.decorators import bind_effect, bind_output, bind_task
from shiny_kit.explorers import ArtifactSwitcher
from shiny_kit.state import Requirement
from shiny_kit.ui import Icon

from seroepi.aggregate import Aggregator
from seroepi.app.components import PlotterExplorer, TableExplorer
from seroepi.app.models import SessionState
from seroepi.app.widgets import EstimatorWidgetSelector
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.hierarchical import (
    HierarchicalMCMCEstimator,
    HierarchicalSVIEstimator,
)
from seroepi.estimators.unpooled import UnpooledEstimator
from seroepi.plotting.bar import BarPlotter
from seroepi.plotting.base import BasePlotter, UnstratifiedPlotter
from seroepi.plotting.choropleth import ChoroplethPlotter
from seroepi.plotting.heatmap import HeatmapPlotter
from seroepi.plotting.pyramid import PyramidPlotter
from seroepi.plotting.upset import UpSetPlotter
from seroepi.variables import QC, Identifier, Variable

# ---------------------------------------------------------------------------
# 1. Model & Plotter Registries
# ---------------------------------------------------------------------------

ESTIMATOR_REGISTRY: dict[str, tuple[str, type[BaseEstimator]]] = {
    "unpooled": ("Unpooled (Analytical / Bootstrap)", UnpooledEstimator),
    "svi": ("Hierarchical Bayesian (SVI)", HierarchicalSVIEstimator),
    "mcmc": ("Hierarchical Bayesian (MCMC)", HierarchicalMCMCEstimator),
}

PLOTTER_REGISTRY: dict[str, tuple[str, type[BasePlotter]]] = {
    "bar": ("Prevalence Bar Chart", BarPlotter),
    "choropleth": ("Spatial Choropleth Map", ChoroplethPlotter),
    "heatmap": ("Prevalence Heatmap", HeatmapPlotter),
    "upset": ("UpSet Co-occurrence", UpSetPlotter),
    "pyramid": ("Pyramid Comparison", PyramidPlotter),
}


# ---------------------------------------------------------------------------
# 2. Subcomponent: Aggregation Sidebar Panel
# ---------------------------------------------------------------------------


class AggregatorSwitcher(ArtifactSwitcher[Aggregator]):
    """Reusable `ArtifactSwitcher` configured for fitted `Aggregator` instances."""

    registry_kind = "aggregators"
    label = "Fitted Aggregators"
    enable_delete = True
    enable_download = True
    enable_upload = False
    file_extension = ".joblib"
    download_button_label = "Save (.joblib)"

    def summary_ui(self, item: Aggregator) -> TagChild:
        """Renders a compact summary badge card for the currently active `Aggregator`.

        Args:
            item: Active fitted `Aggregator` instance.

        Returns:
            UI card fragment summarizing the aggregator's target, strata, row count, and mode.
        """
        return shiny_ui.div(
            shiny_ui.div(
                Icon.lucide("circle-check", size=14, class_="text-success me-1"),
                shiny_ui.strong("Fitted Aggregator", class_="small"),
                class_="d-flex align-items-center mb-1",
            ),
            shiny_ui.div(item.summary_label, class_="small text-body fw-semibold"),
            shiny_ui.div(
                f"{len(item.data_):,} rows • {'Compositional' if item.is_compositional else 'Binary'}",
                class_="text-muted",
                style="font-size: 0.78rem;",
            ),
            class_="p-2 bg-body-tertiary rounded border border-success-subtle mb-2",
        )


class AggregationPanel(Component[SessionState]):
    """Sidebar subcomponent configuring and fitting `Aggregator` instances from cached Targets.

    Attributes:
        switcher (AggregatorSwitcher): Subcomponent for switching or deleting fitted `Aggregator` instances.
    """

    switcher = subcomponent(AggregatorSwitcher)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(
            cls.switcher.ui(),
            shiny_ui.output_ui("agg_form_ui"),
            shiny_ui.input_action_button(
                "btn_aggregate",
                "Aggregate Data",
                icon=Icon.lucide("calculator", size=16),
                class_="btn-primary w-100 mt-2",
            ),
        )

    @bind_output(render.ui)
    def agg_form_ui(self) -> TagChild:
        """Renders the target, stratification, adjustment, and zero-padding input controls.

        Returns:
            UI form fragment for `Aggregator` configuration.
        """
        ds = self.state.active_dataset
        targets = self.state.targets.items
        if ds is None or not targets:
            return shiny_ui.p(
                "Load a dataset and build at least one Target in the Dataset tab first.",
                class_="text-muted small mb-2",
            )

        target_choices: dict[str, str] = {
            name: f"{name} ({'Compositional' if tgt.is_compositional else 'Binary'})" for name, tgt in targets.items()
        }
        selected_target = self.state.targets.active_key or next(iter(target_choices.keys()), None)

        var_choices: dict[str, str] = {}
        for col, var in ds.variables.items():
            if isinstance(var, (Identifier, QC)):
                continue
            var_choices[col] = f"{var.title} ({type(var).__name__})"

        adjust_choices: dict[str, str] = {"": "(None — Unadjusted)", **var_choices}

        return shiny_ui.div(
            shiny_ui.input_select(
                "agg_target",
                "Target (from Target Composer)",
                choices=target_choices,
                selected=selected_target,
            ),
            shiny_ui.input_selectize(
                "agg_strata",
                "Stratify By (Optional)",
                choices=var_choices,
                multiple=True,
                options={"placeholder": "Unstratified (0 strata)..."},
            ),
            shiny_ui.input_select(
                "agg_adjust",
                "Adjust For (Optional)",
                choices=adjust_choices,
                selected="",
            ),
            shiny_ui.input_switch(
                "agg_pad_zeros",
                "Pad Zeroes (Zero-fill missing strata)",
                value=True,
            ),
        )

    @bind_task("btn_aggregate", register_to="aggregators")
    def run_aggregation(self) -> Aggregator:
        """Executes or loads from cache the configured `Aggregator` on the active dataset.

        Returns:
            Fitted `Aggregator` instance.

        Raises:
            ValueError: If no active dataset or target is selected.
        """
        ds = self.state.active_dataset
        if ds is None:
            raise ValueError("No active dataset available.")

        tgt_key = self.val("agg_target")
        if not tgt_key:
            raise ValueError("Please select a Target from the Target Composer.")
        target = self.state.targets.get(str(tgt_key))
        if target is None:
            raise ValueError(f"Target '{tgt_key}' not found in Target Composer.")

        raw_strata = self.val("agg_strata", default=())
        strata_cols = [str(c) for c in (raw_strata if isinstance(raw_strata, (list, tuple)) else [raw_strata]) if c]
        strata_vars: list[Variable] = [ds.var(c) for c in strata_cols if c in ds.variables]

        adjust_col = self.val("agg_adjust", default="")
        adjust_var: Variable | None = (
            ds.var(str(adjust_col)) if adjust_col and str(adjust_col) in ds.variables else None
        )

        pad_zeros = bool(self.val("agg_pad_zeros", default=True))

        agg = Aggregator(
            target=target,
            strata=strata_vars if strata_vars else None,
            adjust_for=adjust_var,
            pad_zeros=pad_zeros,
        )
        return self.state.cache.fit_or_load(agg, ds)


# ---------------------------------------------------------------------------
# 3. Subcomponent: Estimation Sidebar Panel (Train, Switch, Save, Load)
# ---------------------------------------------------------------------------


class EstimatorSwitcher(ArtifactSwitcher[BaseEstimator]):
    """Reusable `ArtifactSwitcher` configured for fitted `BaseEstimator` models."""

    registry_kind = "estimators"
    label = "Trained / Loaded Estimators"
    enable_delete = True
    enable_download = True
    enable_upload = False
    file_extension = ".joblib"
    download_button_label = "Save (.joblib)"


class EstimationPanel(Component[SessionState]):
    """Sidebar subcomponent managing estimator training, hyperparameter selection, and `.joblib` I/O.

    Attributes:
        switcher (EstimatorSwitcher): Subcomponent for switching, exporting, or deleting trained estimators.
    """

    switcher = subcomponent(EstimatorSwitcher)

    def __init__(self) -> None:
        """Initializes `EstimationPanel` with a default `EstimatorWidgetSelector`."""
        super().__init__()
        self._selector = EstimatorWidgetSelector.default()

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(
            cls.switcher.ui(),
            shiny_ui.output_ui("train_estimator_section_ui"),
            shiny_ui.hr(class_="my-3"),
            shiny_ui.div(
                shiny_ui.p(
                    Icon.lucide("folder-open", size=14, class_="me-1"),
                    shiny_ui.strong("Load Saved Estimator"),
                    class_="small mb-1 d-flex align-items-center",
                ),
                shiny_ui.input_file(
                    "est_file_upload",
                    label=None,
                    accept=[".joblib", ".pkl"],
                    multiple=False,
                    placeholder="Select .joblib model...",
                ),
            ),
        )

    @bind_output(render.ui)
    def train_estimator_section_ui(self) -> TagChild:
        """Renders the estimator family selector, optional custom name input, and train button.

        Returns:
            UI fragment for estimator training or an informational prompt when no aggregator is active.
        """
        agg = self.state.active_aggregator
        if agg is None:
            return shiny_ui.div(
                shiny_ui.div(
                    Icon.lucide("info", size=15, class_="me-1 text-secondary"),
                    shiny_ui.span(
                        "Fit an Aggregator above to unlock estimator training, or load a saved .joblib estimator below.",
                        class_="small text-muted",
                    ),
                    class_="p-2 bg-body-tertiary rounded border",
                ),
            )

        est_choices = {key: label for key, (label, _) in ESTIMATOR_REGISTRY.items()}
        return shiny_ui.div(
            shiny_ui.div(
                shiny_ui.span("Active Input: ", class_="text-muted small"),
                shiny_ui.strong(agg.summary_label, class_="small"),
                class_="mb-2",
            ),
            shiny_ui.input_select(
                "est_type",
                "Estimator Model",
                choices=est_choices,
                selected="unpooled",
            ),
            shiny_ui.input_text(
                "est_custom_name",
                "Estimator Name (Optional)",
                placeholder="Auto-named if left blank...",
            ),
            shiny_ui.output_ui("est_hyperparams_ui"),
            shiny_ui.input_action_button(
                "btn_train_estimator",
                "Train Estimator",
                icon=Icon.lucide("play", size=16),
                class_="btn-success w-100 mt-2",
            ),
        )

    @bind_output(render.ui)
    def est_hyperparams_ui(self) -> TagChild:
        """Renders the auto-introspected hyperparameter form for the selected estimator class.

        Returns:
            Collapsible `ui.accordion` containing estimator hyperparameter inputs.
        """
        est_key = str(self.val("est_type", default="unpooled"))
        if est_key not in ESTIMATOR_REGISTRY:
            return shiny_ui.div()
        est_cls = ESTIMATOR_REGISTRY[est_key][1]
        widgets = self._selector.build_form(est_cls, prefix=f"est_{est_key}_")
        return shiny_ui.accordion(
            shiny_ui.accordion_panel(
                "Model Hyperparameters",
                *widgets,
                value="hyperparams_panel",
            ),
            id="est_hyperparams_acc",
            open=False,
            class_="mt-2 mb-2",
        )

    @bind_task("btn_train_estimator", register_to="estimators")
    def run_estimation(self) -> BaseEstimator:
        """Trains or loads from cache the selected `BaseEstimator` on the active `Aggregator`.

        Returns:
            Fitted `BaseEstimator` instance.

        Raises:
            ValueError: If no active `Aggregator` is available or the estimator key is invalid.
        """
        agg = self.state.active_aggregator
        if agg is None:
            raise ValueError("Please fit an Aggregator before training an estimator.")

        est_key = str(self.val("est_type", default="unpooled"))
        if est_key not in ESTIMATOR_REGISTRY:
            raise ValueError(f"Unknown estimator type: '{est_key}'.")

        est_cls = ESTIMATOR_REGISTRY[est_key][1]
        kwargs = self._selector.extract_kwargs(est_cls, self.input, prefix=f"est_{est_key}_")
        estimator = est_cls(**kwargs)
        fitted = self.state.cache.fit_or_load(estimator, agg)

        custom_name = str(self.val("est_custom_name", default="") or "").strip()
        result = copy.copy(fitted)
        result.name = custom_name if custom_name else None
        return result

    @bind_effect("est_file_upload")
    def _on_upload_estimator_file(self) -> None:
        file_info = self.val("est_file_upload", strip=False)
        if not file_info or not isinstance(file_info, list):
            return
        entry = file_info[0]
        datapath = str(entry.get("datapath", ""))
        filename = str(entry.get("name", "uploaded_estimator.joblib"))
        if datapath:
            self.load_estimator_task.invoke(datapath, filename)

    @bind_task(register_to="estimators")
    def load_estimator_task(self, datapath: str, filename: str) -> BaseEstimator:
        """Deserializes and validates an uploaded `.joblib` estimator artifact.

        Args:
            datapath: Temporary filesystem path of the uploaded file.
            filename: Original user-facing filename used to derive a fallback display name.

        Returns:
            Loaded and validated `BaseEstimator` instance.
        """
        est = BaseEstimator.load(Path(datapath))
        est.check_is_fitted()
        if not est.name:
            stem = Path(filename).stem
            est.name = f"{est.display_name} [{stem}]"
        return est


# ---------------------------------------------------------------------------
# 4. Subcomponent: Estimates & Diagnostics DataFrame Explorer (Main Tab 1)
# ---------------------------------------------------------------------------


class EstimatesExplorer(TableExplorer[tuple[BaseEstimator | None, Aggregator | None]]):
    """Main-panel `TableExplorer` displaying Estimates, Aggregated Counts, and Bayesian Diagnostics tables."""

    empty_icon = "table"
    empty_title = "No Aggregated Data or Estimates Yet"
    empty_message = "Aggregate data and train (or load) an Estimator in the sidebar to inspect prevalence estimates."
    csv_prefix = "seroepi"

    def __init__(self) -> None:
        super().__init__()
        self._last_est_version: int = 0

    def active_subject(self) -> tuple[BaseEstimator | None, Aggregator | None] | None:
        """Returns the `(active_estimator, active_aggregator)` pair, or `None` if neither exists.

        Returns:
            Tuple of `(BaseEstimator | None, Aggregator | None)` or `None`.
        """
        est = self.state.active_estimator
        agg = self.state.active_aggregator
        if est is None and agg is None:
            return None
        return est, agg

    def table_views(self, subject: tuple[BaseEstimator | None, Aggregator | None]) -> dict[str, str]:
        """Returns available DataGrid view modes for `subject`.

        Args:
            subject: Tuple of `(active_estimator, active_aggregator)`.

        Returns:
            Mapping from view mode keys (`"estimates"`, `"aggregated"`, `"diagnostics"`) to labels.
        """
        est, agg = subject
        choices: dict[str, str] = {}
        if est is not None:
            choices["estimates"] = "Prevalence Estimates"
        if (est is not None and est.aggregator_ is not None) or agg is not None:
            choices["aggregated"] = "Aggregated Counts"
        if est is not None and est.is_bayesian and hasattr(est, "diagnostics"):
            choices["diagnostics"] = "Bayesian Diagnostics"
        return choices

    def resolve_selected_mode(
        self,
        subject: tuple[BaseEstimator | None, Aggregator | None],
        choices: dict[str, str],
        current_mode: str | None,
    ) -> str:
        """Selects `"estimates"` automatically whenever a newly trained estimator is registered.

        Args:
            subject: Tuple of `(active_estimator, active_aggregator)`.
            choices: Available table view choices.
            current_mode: Currently selected table view key.

        Returns:
            Resolved table view key.
        """
        est, _ = subject
        est_ver = self.state.estimators.version
        if est is not None and est_ver != self._last_est_version:
            self._last_est_version = est_ver
            return "estimates"
        return super().resolve_selected_mode(subject, choices, current_mode)

    def toolbar_badges_ui(self, subject: tuple[BaseEstimator | None, Aggregator | None]) -> Sequence[TagChild]:
        """Renders summary badges for the active estimator or aggregator in the table toolbar.

        Args:
            subject: Tuple of `(active_estimator, active_aggregator)`.

        Returns:
            Sequence of badge UI tags.
        """
        est, agg = subject
        if est is not None:
            return (
                shiny_ui.span(est.display_name, class_="badge bg-primary me-1"),
                shiny_ui.span(est.summary_label, class_="badge bg-secondary me-1"),
                shiny_ui.span(est.describe_interval(), class_="badge bg-info text-dark me-1"),
            )
        if agg is not None:
            return (shiny_ui.span(f"Aggregator: {agg.summary_label}", class_="badge bg-secondary me-1"),)
        return ()

    def resolve_table(
        self,
        subject: tuple[BaseEstimator | None, Aggregator | None],
        mode: str,
    ) -> pl.DataFrame | None:
        """Extracts the Polars DataFrame corresponding to `mode` from `subject`.

        Args:
            subject: Tuple of `(active_estimator, active_aggregator)`.
            mode: Selected view key (`"estimates"`, `"aggregated"`, or `"diagnostics"`).

        Returns:
            Polars `DataFrame` to display in the grid, or `None`.
        """
        est, agg = subject
        if mode == "aggregated":
            source_agg = est.aggregator_ if (est is not None and est.aggregator_ is not None) else agg
            return source_agg.data_ if source_agg is not None else None
        if mode == "diagnostics" and est is not None and hasattr(est, "diagnostics"):
            return est.diagnostics()  # ty: ignore[call-non-callable]
        if est is not None and est.estimates_ is not None:
            return est.estimates_
        if agg is not None:
            return agg.data_
        return None


# ---------------------------------------------------------------------------
# 5. Subcomponent: Interactive Plots Explorer (Main Tab 2)
# ---------------------------------------------------------------------------


class PlotsExplorer(PlotterExplorer[BaseEstimator]):
    """Main-panel `PlotterExplorer` dynamically populating plotters and `PlotterWidgetSelector` controls."""

    plotter_registry = PLOTTER_REGISTRY
    param_prefix = "plot"
    download_prefix = "seroepi"
    sidebar_width = 310
    chart_height = "75vh"
    empty_icon = "chart-bar"
    empty_title = "No Active Estimator to Plot"
    empty_message = "Train or load an Estimator in the sidebar to interactively explore epidemiological plots."

    @classmethod
    def available_plot_choices(cls, estimator: BaseEstimator | None, n_estimators: int) -> dict[str, str]:
        """Returns mapping of plot keys to display labels supported by `estimator` and registry state.

        Args:
            estimator: Active fitted `BaseEstimator`, or `None`.
            n_estimators: Total number of estimators currently registered in session state.

        Returns:
            Dictionary mapping compatible plotter keys to human-readable display labels.
        """
        if estimator is None:
            return {}
        return {
            key: label
            for key, (label, plotter_cls) in PLOTTER_REGISTRY.items()
            if plotter_cls.can_plot(estimator, n_estimators=n_estimators)
        }

    def active_subject(self) -> BaseEstimator | None:
        """Returns the currently active `BaseEstimator` from `SessionState`.

        Returns:
            Active `BaseEstimator` or `None`.
        """
        return self.state.active_estimator

    def available_choices(self, subject: BaseEstimator) -> dict[str, str]:
        """Returns compatible plotter choices for `subject` given the session's estimator count.

        Args:
            subject: Active `BaseEstimator`.

        Returns:
            Dictionary of `{plot_key: label}`.
        """
        return self.available_plot_choices(subject, self.state.estimators.count)

    def selector_header_ui(self, subject: BaseEstimator) -> TagChild:
        """Renders an unlock status indicator for multi-estimator plotters (such as `PyramidPlotter`).

        Args:
            subject: Active `BaseEstimator`.

        Returns:
            UI indicator badge showing whether multi-estimator comparison plots are unlocked.
        """
        n_est = self.state.estimators.count
        multi_est_labels = [
            label
            for _, (label, cls) in PLOTTER_REGISTRY.items()
            if not cls.can_plot(subject, n_estimators=1) and cls.can_plot(subject, n_estimators=2)
        ]
        multi_names = ", ".join(multi_est_labels)
        if not multi_names:
            return shiny_ui.div()
        if n_est >= 2:
            return shiny_ui.div(
                Icon.lucide("git-compare", size=13, class_="me-1 text-success"),
                shiny_ui.span(f"{multi_names} unlocked ({n_est} estimators)", class_="small text-success"),
                class_="d-flex align-items-center mb-2",
            )
        return shiny_ui.div(
            Icon.lucide("lock", size=13, class_="me-1 text-muted"),
            shiny_ui.span(f"Train/load a 2nd estimator to unlock {multi_names}", class_="small text-muted"),
            class_="d-flex align-items-center mb-2",
        )

    @staticmethod
    def _stratum_slice_choices(est: BaseEstimator) -> dict[str, str]:
        """Builds dropdown choices for pooling or isolating a single stratum on a stratified estimator."""
        choices: dict[str, str] = {"__pooled__": "Pooled (All Strata)"}
        if est.n_strata == 1 and est.estimates_ is not None:
            col = est.single_stratum_col
            if col in est.estimates_.columns:
                vals = est.estimates_[col].drop_nulls().cast(pl.String).unique(maintain_order=True).to_list()
                for v in vals:
                    choices[str(v)] = f"{est.single_stratum_title}: {v}"
        return choices

    @staticmethod
    def _resolve_unstratified_side(est: BaseEstimator, slice_val: str | None) -> BaseEstimator:
        """Converts a potentially stratified estimator into a 1D unstratified estimator for PyramidPlotter."""
        if est.n_strata == 0:
            return est
        if slice_val and slice_val != "__pooled__" and est.n_strata == 1:
            col = est.single_stratum_col
            if est.estimates_ is not None and col in est.estimates_.columns:
                return est.filter(pl.col(col).cast(pl.String) == slice_val)
        return est.pool_strata()

    def render_extra_controls(self, subject: BaseEstimator, plot_key: str) -> TagChild:
        """Renders left/right estimator and stratum slice selectors when `"pyramid"` is active.

        Args:
            subject: Active `BaseEstimator`.
            plot_key: Currently selected plotter key.

        Returns:
            Extra sidebar controls for `PyramidPlotter`, or an empty `div` for other plotters.
        """
        if plot_key != "pyramid":
            return shiny_ui.div()

        names = self.state.estimators.names
        if not names:
            return shiny_ui.div()

        default_left = names[0]
        default_right = names[1] if len(names) > 1 else names[0]
        cur_left = self.val("pyramid_left", isolate=True)
        cur_right = self.val("pyramid_right", isolate=True)
        sel_left = str(cur_left) if cur_left in names else default_left
        sel_right = str(cur_right) if cur_right in names else default_right

        left_est = self.state.estimators.get(sel_left)
        right_est = self.state.estimators.get(sel_right)

        controls: list[TagChild] = [
            shiny_ui.input_select(
                "pyramid_left",
                "Left Estimator",
                choices={n: n for n in names},
                selected=sel_left,
            ),
        ]
        if left_est is not None and left_est.n_strata >= 1:
            controls.append(
                shiny_ui.input_select(
                    "pyramid_left_slice",
                    "Left Stratum Slice",
                    choices=self._stratum_slice_choices(left_est),
                    selected="__pooled__",
                )
            )

        controls.append(
            shiny_ui.input_select(
                "pyramid_right",
                "Right Estimator",
                choices={n: n for n in names},
                selected=sel_right,
            )
        )
        if right_est is not None and right_est.n_strata >= 1:
            controls.append(
                shiny_ui.input_select(
                    "pyramid_right_slice",
                    "Right Stratum Slice",
                    choices=self._stratum_slice_choices(right_est),
                    selected="__pooled__",
                )
            )

        return shiny_ui.div(*controls, shiny_ui.hr(class_="my-2"))

    def build_chart(self, plotter: BasePlotter, subject: BaseEstimator, plot_key: str) -> alt.TopLevelMixin | None:
        """Constructs the Altair chart for `subject` using `plotter`, handling pooling and dual-estimator plots.

        Args:
            plotter: Configured `BasePlotter` instance.
            subject: Active `BaseEstimator`.
            plot_key: Registry key of the active plotter.

        Returns:
            Rendered Altair chart, or `None` if `plotter` cannot plot `subject`.
        """
        if not plotter.can_plot(subject, n_estimators=self.state.estimators.count):
            return None

        if isinstance(plotter, PyramidPlotter):
            names = self.state.estimators.names
            if not names:
                return None
            left_name = str(self.val("pyramid_left", default=names[0]))
            right_name = str(self.val("pyramid_right", default=names[1] if len(names) > 1 else names[0]))
            left_raw = self.state.estimators.get(left_name) or subject
            right_raw = self.state.estimators.get(right_name) or subject

            left_slice = str(self.val("pyramid_left_slice", default="__pooled__"))
            right_slice = str(self.val("pyramid_right_slice", default="__pooled__"))
            left_ready = self._resolve_unstratified_side(left_raw, left_slice)
            right_ready = self._resolve_unstratified_side(right_raw, right_slice)
            return plotter.plot(left_ready, right_ready)

        if isinstance(plotter, UnstratifiedPlotter) and subject.n_strata > 0:
            return plotter.plot(subject.pool_strata())

        if isinstance(plotter, ChoroplethPlotter):
            return plotter.plot(subject, param_suffix=f"_{uuid4().hex[:6]}")

        return plotter.plot(subject)


# ---------------------------------------------------------------------------
# 6. Top-Level EstimationTab (Composed of 4 Focused Subcomponents)
# ---------------------------------------------------------------------------


class EstimationTab(Tab[SessionState]):
    """Prevalence Estimation.

    Top-level tab module orchestrating `AggregationPanel`, `EstimationPanel`,
    `EstimatesExplorer`, and `PlotsExplorer` for stratified prevalence modeling
    and interactive visualization.

    Attributes:
        icon (Tag): Lucide `"trending-up"` navbar icon.
        requires (ClassVar[tuple[Requirement, ...]]): Prerequisites requiring at least one
            loaded `Dataset` and one `Target` in the Target Composer.
        agg_panel (AggregationPanel): Sidebar panel for configuring and fitting `Aggregator` instances.
        est_panel (EstimationPanel): Sidebar panel for training or loading `BaseEstimator` models.
        estimates_explorer (EstimatesExplorer): Main-panel DataGrid explorer for estimates and diagnostics.
        plots_explorer (PlotsExplorer): Main-panel interactive Altair plotter workspace.
    """

    icon = Icon.lucide("trending-up")
    requires: ClassVar[tuple[Requirement, ...]] = (
        Requirement("datasets", min_count=1, label="Loaded Dataset", target_tab="dataset_tab"),
        Requirement("targets", min_count=1, label="Target in Target Composer", target_tab="dataset_tab"),
    )

    agg_panel = subcomponent(AggregationPanel)
    est_panel = subcomponent(EstimationPanel)
    estimates_explorer = subcomponent(EstimatesExplorer)
    plots_explorer = subcomponent(PlotsExplorer)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                shiny_ui.accordion(
                    shiny_ui.accordion_panel(
                        "1. Prevalence Aggregation",
                        cls.agg_panel.ui(),
                        value="agg_accordion_panel",
                        icon=Icon.lucide("calculator", size=16),
                    ),
                    shiny_ui.accordion_panel(
                        "2. Prevalence Estimation",
                        cls.est_panel.ui(),
                        value="est_accordion_panel",
                        icon=Icon.lucide("trending-up", size=16),
                    ),
                    id="estimation_sidebar_accordion",
                    open=["agg_accordion_panel", "est_accordion_panel"],
                    multiple=True,
                ),
                width=360,
            ),
            shiny_ui.navset_card_tab(
                shiny_ui.nav_panel(
                    "Estimates Explorer",
                    cls.estimates_explorer.ui(),
                    value="estimates_dataframe_panel",
                    icon=Icon.lucide("table", size=16),
                ),
                shiny_ui.nav_panel(
                    "Plots",
                    cls.plots_explorer.ui(),
                    value="estimates_plots_panel",
                    icon=Icon.lucide("chart-bar", size=16),
                ),
                id="estimation_main_tabs",
            ),
        )
