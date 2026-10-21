"""Vaccine Formulation Design, LOSO Stability, and Cohort/Model Differential Prevalence tab module."""

import copy
from collections.abc import Callable, Sequence
from typing import ClassVar

import altair as alt
import polars as pl
from htmltools import TagChild
from shiny import reactive, render
from shiny import ui as shiny_ui
from shiny_kit.component import Component, Tab, subcomponent
from shiny_kit.decorators import bind_effect, bind_output, bind_task, reactive_calc
from shiny_kit.explorers import ArtifactSwitcher
from shiny_kit.state import Requirement
from shiny_kit.ui import Icon, KPICard, KPIRow

from seroepi.app.components import PlotterExplorer, TableExplorer
from seroepi.app.models import SessionState
from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.formulation import Formulation, FormulationComparison
from seroepi.plotting.base import BasePlotter
from seroepi.plotting.formulation import (
    DifferentialPrevalencePlotter,
    FormulationCoveragePlotter,
    FormulationStabilityPlotter,
    RankShiftPlotter,
)
from seroepi.plotting.upset import UpSetPlotter

# ---------------------------------------------------------------------------
# 1. Plotter Registries for Formulation & Comparison Workflows
# ---------------------------------------------------------------------------

FORMULATION_PLOTTER_REGISTRY: dict[str, tuple[str, type[BasePlotter]]] = {
    "coverage": ("Coverage & Stratum Trajectories", FormulationCoveragePlotter),
    "stability": ("LOSO Stability & Stratum Coverage", FormulationStabilityPlotter),
}

COMPARISON_PLOTTER_REGISTRY: dict[str, tuple[str, type[BasePlotter]]] = {
    "differential": ("Differential Prevalence (Volcano / Forest)", DifferentialPrevalencePlotter),
    "rank_shift": ("Target Rank Shifts (Bump Chart)", RankShiftPlotter),
    "upset": ("Target Repertoire Overlap (UpSet)", UpSetPlotter),
}


# ---------------------------------------------------------------------------
# 2. Subcomponent 1: Formulation Designer & LOSO Stability Panel
# ---------------------------------------------------------------------------


class FormulationSwitcher(ArtifactSwitcher[Formulation]):
    """Sidebar switcher for registered `Formulation` artifacts with `.joblib` & Target Composer export."""

    registry_kind = "formulations"
    label = "Registered Formulations"
    enable_delete = True
    enable_download = True
    file_extension = ".joblib"
    download_button_label = "Save (.joblib)"

    def extra_buttons_ui(self, item: Formulation) -> Sequence[TagChild]:
        """Renders an extra button exporting the active `Formulation` into the Target Composer.

        Args:
            item: Active fitted `Formulation` instance.

        Returns:
            Single-element tuple containing the `"Export to Target Composer"` action button.
        """
        return (
            shiny_ui.input_action_button(
                "btn_export_target",
                "Export to Target Composer",
                icon=Icon.lucide("target", size=14),
                class_="btn-sm btn-outline-success flex-grow-1",
            ),
        )

    @bind_effect("btn_export_target")
    def _on_export_target(self) -> None:
        form = self.registry.active
        if form is None:
            self.notify("Fit or select a Formulation first.", type_="warning")
            return
        tgt = form.to_target()
        self.state.targets.register(tgt.name, tgt, set_active=True)
        self.notify(
            f"Registered binary coverage target '{tgt.name}' ({tgt.title}) in Target Composer!",
            type_="message",
        )


class DesignerPlotsExplorer(PlotterExplorer[Formulation]):
    """Interactive Altair plot explorer for the active `Formulation`."""

    plotter_registry = FORMULATION_PLOTTER_REGISTRY
    default_plot_key = "coverage"
    param_prefix = "fplot"
    download_prefix = "seroepi_formulation"
    empty_icon = "shield-check"
    empty_title = "No Active Formulation"
    empty_message = "Click 'Design & Fit Formulation' in the sidebar to rank targets and evaluate coverage."

    def active_subject(self) -> Formulation | None:
        """Returns the currently active `Formulation` from `SessionState`.

        Returns:
            Active `Formulation` instance or `None`.
        """
        return self.state.active_formulation


class DesignerTableExplorer(TableExplorer[Formulation]):
    """Interactive DataGrid explorer for `Formulation` rankings, stratum coverage, and LOSO CV."""

    csv_prefix = "seroepi_formulation"
    empty_icon = "table"
    empty_title = "No Active Formulation"
    empty_message = "Fit a Formulation to inspect rankings and stability tables."

    def active_subject(self) -> Formulation | None:
        """Returns the currently active `Formulation` from `SessionState`.

        Returns:
            Active `Formulation` instance or `None`.
        """
        return self.state.active_formulation

    def table_views(self, subject: Formulation) -> dict[str, str]:
        """Returns available DataGrid view modes for `subject`.

        Args:
            subject: Active `Formulation` instance.

        Returns:
            Mapping from table mode keys to human-readable labels.
        """
        choices: dict[str, str] = {
            "rankings": "Global Rankings & FDR",
            "stratum_coverage": "Stratum Coverage",
        }
        if subject.aggregator_.n_strata >= 1:
            choices["stratum_trajectories"] = "Stratum Coverage Trajectories"
        if len(subject.stability_metrics_) > 0:
            choices["stability"] = "LOSO Target Stability"
        if len(subject.fold_summary_) > 0:
            choices["folds"] = "LOSO Held-Out Folds"
        return choices

    def resolve_table(self, subject: Formulation, mode: str) -> pl.DataFrame | None:
        """Resolves the Polars DataFrame for `mode` from `subject`.

        Args:
            subject: Active `Formulation` instance.
            mode: Selected table view key.

        Returns:
            Polars `DataFrame` to display in the grid.
        """
        if mode == "stratum_coverage":
            return subject.score_by_stratum()
        if mode == "stratum_trajectories" and subject.aggregator_.n_strata >= 1:
            cum_df, _ = subject.stratum_trajectories()
            return cum_df
        if mode == "stability" and len(subject.stability_metrics_) > 0:
            return subject.stability_metrics_
        if mode == "folds" and len(subject.fold_summary_) > 0:
            return subject.fold_summary_
        return subject.rankings_


class FormulationDesignerPanel(Component[SessionState]):
    """Interactive subcomponent for designing K-valent formulations, LOSO CV, and Target Composer export.

    Attributes:
        switcher (FormulationSwitcher): Sidebar artifact switcher for saved formulations.
        plots_explorer (DesignerPlotsExplorer): Main-panel Altair plotter explorer for coverage and stability.
        tables_explorer (DesignerTableExplorer): Main-panel DataGrid explorer for rankings and LOSO tables.
    """

    switcher = subcomponent(FormulationSwitcher)
    plots_explorer = subcomponent(DesignerPlotsExplorer)
    tables_explorer = subcomponent(DesignerTableExplorer)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                cls.switcher.ui(),
                shiny_ui.output_ui("designer_config_ui"),
                shiny_ui.input_action_button(
                    "btn_fit_formulation",
                    "Design & Fit Formulation",
                    icon=Icon.lucide("shield-check", size=16),
                    class_="btn-primary w-100 mt-2",
                ),
                width=360,
            ),
            shiny_ui.div(
                shiny_ui.output_ui("designer_kpi_ui"),
                shiny_ui.navset_card_tab(
                    shiny_ui.nav_panel(
                        "Interactive Plots",
                        cls.plots_explorer.ui(),
                        value="designer_plots_tab",
                        icon=Icon.lucide("chart-bar", size=16),
                    ),
                    shiny_ui.nav_panel(
                        "DataFrames Explorer",
                        cls.tables_explorer.ui(),
                        value="designer_tables_tab",
                        icon=Icon.lucide("table", size=16),
                    ),
                    id="designer_workspace_tabs",
                ),
            ),
        )

    @bind_output(render.ui)
    def designer_config_ui(self) -> TagChild:
        """Renders sidebar configuration inputs for valency, weighting, FDR alpha, LOSO CV, and target pinning.

        Returns:
            UI form fragment for `Formulation` design.
        """
        est_names = self.state.estimators.names
        if not est_names:
            return shiny_ui.p(
                "Train or load at least one Estimator in the Prevalence Estimation tab first.",
                class_="text-muted small mb-2",
            )

        cur_est_key = self.val("form_est_select", isolate=True)
        selected_est_key = (
            str(cur_est_key) if cur_est_key in est_names else (self.state.estimators.active_key or est_names[0])
        )
        est = self.state.estimators.get(selected_est_key)
        target_choices: dict[str, str] = {}
        if est is not None:
            for t in est.ranked_targets(None):
                target_choices[str(t)] = str(t)

        return shiny_ui.div(
            shiny_ui.input_select(
                "form_est_select",
                "Source Prevalence Estimator",
                choices={name: name for name in est_names},
                selected=selected_est_key,
            ),
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.input_numeric(
                        "form_valency",
                        "Valency (K)",
                        value=6,
                        min=1,
                        max=max(1, len(target_choices) or 50),
                        step=1,
                    ),
                    class_="col-6",
                ),
                shiny_ui.div(
                    shiny_ui.input_select(
                        "form_weighting",
                        "Stratum Weighting",
                        choices={
                            "pooled": "Pooled (Isolate-Weighted)",
                            "equal": "Equal (Stratum-Balanced)",
                        },
                        selected="pooled",
                    ),
                    class_="col-6",
                ),
                class_="row g-2",
            ),
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.input_numeric(
                        "form_min_prev",
                        "Min. Burden (τ)",
                        value=0.0,
                        min=0.0,
                        max=0.95,
                        step=0.01,
                    ),
                    class_="col-6",
                ),
                shiny_ui.div(
                    shiny_ui.input_numeric(
                        "form_alpha",
                        "FDR Alpha (α)",
                        value=0.05,
                        min=0.001,
                        max=0.50,
                        step=0.01,
                    ),
                    class_="col-6",
                ),
                class_="row g-2",
            ),
            shiny_ui.input_select(
                "form_cv",
                "Cross-Validation (Geographic / Stratum Stability)",
                choices={
                    "loso": "Leave-One-Stratum-Out (LOSO)",
                    "none": "None (Skip Cross-Validation)",
                },
                selected="loso",
            ),
            shiny_ui.input_selectize(
                "form_manual_targets",
                "Manual Target Overrides / Pinning (Optional)",
                choices=target_choices,
                multiple=True,
                options={"placeholder": "Auto-select greedy top-K targets..."},
            ),
            shiny_ui.input_text(
                "form_custom_name",
                "Formulation Name (Optional)",
                placeholder="Auto-named if left blank...",
            ),
        )

    @bind_task("btn_fit_formulation", register_to="formulations")
    def run_formulation(self) -> Formulation:
        """Fits or loads from cache the configured `Formulation` on the selected estimator's `Aggregator`.

        Returns:
            Fitted `Formulation` instance.

        Raises:
            ValueError: If no fitted `BaseEstimator` with an attached `Aggregator` is selected.
        """
        est_key = self.val("form_est_select")
        est = self.state.estimators.get(str(est_key)) if est_key else self.state.active_estimator
        if est is None or est.aggregator_ is None:
            raise ValueError("Please select a fitted Estimator before designing a Formulation.")

        valency = max(1, int(self.val("form_valency", default=6) or 6))
        weighting_raw = str(self.val("form_weighting", default="pooled"))
        weighting = "equal" if weighting_raw == "equal" else "pooled"
        min_prev = float(self.val("form_min_prev", default=0.0) or 0.0)
        alpha = float(self.val("form_alpha", default=0.05) or 0.05)
        cv_raw = str(self.val("form_cv", default="loso"))
        cv_arg = "loso" if cv_raw == "loso" else None

        raw_manual = self.val("form_manual_targets", default=())
        manual_list = [str(t) for t in (raw_manual if isinstance(raw_manual, (list, tuple)) else [raw_manual]) if t]
        targets_arg = manual_list if manual_list else None
        if targets_arg is not None:
            valency = len(targets_arg)

        custom_name = str(self.val("form_custom_name", default="") or "").strip() or None

        form = Formulation(
            estimator=est,
            valency=valency,
            weighting=weighting,
            targets=targets_arg,
            cv=cv_arg,
            min_prevalence=min_prev,
            alpha=alpha,
            name=custom_name,
        )
        fitted = self.state.cache.fit_or_load(form, est.aggregator_)
        result = copy.copy(fitted)
        result.name = custom_name
        return result

    @bind_output(render.ui)
    def designer_kpi_ui(self) -> TagChild:
        """Renders a 4-card KPI summary row for the active `Formulation`.

        Returns:
            `KPIRow` UI tag summarizing valency, cumulative coverage, FDR significance, and LOSO coverage.
        """
        form = self.state.active_formulation
        if form is None:
            return shiny_ui.div()

        est_col = form.estimator_.estimate.column
        in_df = form.rankings_.filter(pl.col("in_formulation"))
        cum_cov = min(1.0, max(0.0, float(in_df[est_col].sum()))) if len(in_df) > 0 else 0.0
        n_sig = int(form.rankings_["significant"].sum()) if "significant" in form.rankings_.columns else 0
        selected_str = ", ".join(form.selected_targets_[:8])
        if len(form.selected_targets_) > 8:
            selected_str += f" (+{len(form.selected_targets_) - 8} more)"

        if len(form.fold_summary_) > 0:
            mean_heldout = float(form.fold_summary_["heldout_predicted_coverage"].to_numpy().mean())
            mean_jaccard = float(form.fold_summary_["jaccard_similarity"].to_numpy().mean())
            loso_val = f"{mean_heldout:.1%} (J={mean_jaccard:.2f})"
            loso_sub = f"Across {len(form.fold_summary_)} held-out strata"
        else:
            loso_val = "N/A"
            loso_sub = "Unstratified or CV disabled"

        return KPIRow(
            KPICard(
                label="Selected Targets",
                value=f"{form.valency}-Valent",
                subtitle=selected_str,
                subtitle_class="text-primary",
                truncate_subtitle=True,
                col_width=4,
            ),
            KPICard(
                label="Cumulative Coverage",
                value=f"{cum_cov:.1%}",
                subtitle=f"Weighting: {form.weighting}",
                value_class="fs-5 text-success",
                col_width=3,
            ),
            KPICard(
                label=f"FDR Sig. (q < {form.alpha})",
                value=f"{n_sig} / {len(form.rankings_)}",
                subtitle=f"Threshold τ = {form.min_prevalence:.1%}",
                col_width=2,
            ),
            KPICard(
                label="LOSO Held-Out Coverage",
                value=loso_val,
                subtitle=loso_sub,
                value_class="fs-6",
                col_width=3,
            ),
        ).ui()


# ---------------------------------------------------------------------------
# 3. Subcomponent 2: Cohort & Model Contrasts Panel (FormulationComparison)
# ---------------------------------------------------------------------------


class ComparisonPlotsExplorer(PlotterExplorer[FormulationComparison]):
    """Interactive Altair plot explorer for `FormulationComparison` contrasts and rank shifts."""

    plotter_registry = COMPARISON_PLOTTER_REGISTRY
    default_plot_key = "differential"
    param_prefix = "cplot"
    download_prefix = "seroepi_comparison"
    empty_icon = "git-compare"
    empty_title = "No Active Cohort or Formulation Comparison"
    empty_message = "Select 2+ cohorts or saved formulations in the sidebar and click 'Run Comparison & Contrast'."

    def __init__(self) -> None:
        super().__init__()
        self._get_comparison: Callable[[], FormulationComparison | None] = lambda: None
        self._get_pair: Callable[[], tuple[str, str] | None] = lambda: None

    def bind_comparison(
        self,
        get_comp: Callable[[], FormulationComparison | None],
        get_pair: Callable[[], tuple[str, str] | None],
    ) -> None:
        """Wires reactive getters for the active comparison and selected contrast pair."""
        self._get_comparison = get_comp
        self._get_pair = get_pair

    def active_subject(self) -> FormulationComparison | None:
        """Returns the currently active `FormulationComparison` instance.

        Returns:
            Active `FormulationComparison` or `None`.
        """
        return self._get_comparison()

    def build_chart(
        self,
        plotter: BasePlotter,
        subject: FormulationComparison,
        plot_key: str,
    ) -> alt.TopLevelMixin | None:
        """Constructs the comparison Altair chart, forwarding the selected `(label_a, label_b)` pair when applicable.

        Args:
            plotter: Configured `BasePlotter` instance.
            subject: Active `FormulationComparison`.
            plot_key: Registry key of the active plotter.

        Returns:
            Rendered Altair chart.
        """
        if isinstance(plotter, DifferentialPrevalencePlotter):
            pair = self._get_pair()
            if pair is not None:
                return plotter.plot(subject, label_a=pair[0], label_b=pair[1])
        return plotter.plot(subject)


class ComparisonTableExplorer(TableExplorer[FormulationComparison]):
    """Interactive DataGrid explorer for `FormulationComparison` differential prevalence & concordance."""

    csv_prefix = "seroepi_comparison"
    empty_icon = "table"
    empty_title = "No Active Comparison"
    empty_message = "Run a comparison in the sidebar to inspect differential prevalence and concordance tables."

    def __init__(self) -> None:
        super().__init__()
        self._get_comparison: Callable[[], FormulationComparison | None] = lambda: None
        self._get_pair: Callable[[], tuple[str, str] | None] = lambda: None

    def bind_comparison(
        self,
        get_comp: Callable[[], FormulationComparison | None],
        get_pair: Callable[[], tuple[str, str] | None],
    ) -> None:
        """Wires reactive getters for the active comparison and selected contrast pair.

        Args:
            get_comp: Zero-argument callable returning the active `FormulationComparison | None`.
            get_pair: Zero-argument callable returning the selected `(label_a, label_b)` pair.
        """
        self._get_comparison = get_comp
        self._get_pair = get_pair

    def active_subject(self) -> FormulationComparison | None:
        """Returns the currently active `FormulationComparison` instance.

        Returns:
            Active `FormulationComparison` or `None`.
        """
        return self._get_comparison()

    def table_views(self, subject: FormulationComparison) -> dict[str, str]:
        """Returns available DataGrid view modes for `subject`.

        Args:
            subject: Active `FormulationComparison`.

        Returns:
            Mapping from table mode keys to human-readable labels.
        """
        return {
            "differential": "Differential Prevalence (A vs. B)",
            "concordance": "Pairwise Concordance",
            "rank_shifts": "Wide Rank Shifts",
            "rankings": "All Cohort / Model Rankings",
        }

    def resolve_table(self, subject: FormulationComparison, mode: str) -> pl.DataFrame | None:
        """Resolves the Polars DataFrame corresponding to `mode` from `subject`.

        Args:
            subject: Active `FormulationComparison`.
            mode: Selected table view key.

        Returns:
            Polars `DataFrame` to display in the grid.
        """
        if mode == "concordance":
            return subject.concordance_
        if mode == "rank_shifts":
            return subject.rank_shifts_
        if mode == "rankings":
            return subject.rankings_
        pair = self._get_pair()
        if pair is not None:
            return subject.contrast(pair[0], pair[1])
        return subject.differential_prevalence_


class FormulationComparisonPanel(Component[SessionState]):
    """Interactive subcomponent for Cohort Contrasts (from_strata) and Multi-Model FormulationComparison.

    Attributes:
        plots_explorer (ComparisonPlotsExplorer): Main-panel Altair plotter explorer for contrasts and rank shifts.
        tables_explorer (ComparisonTableExplorer): Main-panel DataGrid explorer for differential prevalence and concordance.
    """

    plots_explorer = subcomponent(ComparisonPlotsExplorer)
    tables_explorer = subcomponent(ComparisonTableExplorer)

    def __init__(self) -> None:
        super().__init__()
        self._active_comparison: reactive.Value[FormulationComparison | None] = reactive.Value(None)
        self.plots_explorer.bind_comparison(self.active_comparison, self.selected_contrast_pair)
        self.tables_explorer.bind_comparison(self.active_comparison, self.selected_contrast_pair)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                shiny_ui.input_radio_buttons(
                    "comp_mode",
                    "Comparison Workflow",
                    choices={
                        "cohorts": "Cohort / Stratum Contrast (from_strata)",
                        "models": "Saved Formulations / Estimators",
                    },
                    selected="cohorts",
                ),
                shiny_ui.output_ui("comp_source_ui"),
                shiny_ui.hr(class_="my-2"),
                shiny_ui.div(
                    shiny_ui.div(
                        shiny_ui.input_numeric("comp_valency", "Valency (K)", value=6, min=1, max=50, step=1),
                        class_="col-6",
                    ),
                    shiny_ui.div(
                        shiny_ui.input_select(
                            "comp_weighting",
                            "Weighting",
                            choices={"pooled": "Pooled", "equal": "Equal"},
                            selected="pooled",
                        ),
                        class_="col-6",
                    ),
                    class_="row g-2",
                ),
                shiny_ui.div(
                    shiny_ui.div(
                        shiny_ui.input_numeric(
                            "comp_null_margin",
                            "Null Margin (δ)",
                            value=0.0,
                            min=0.0,
                            max=0.50,
                            step=0.01,
                        ),
                        class_="col-6",
                    ),
                    shiny_ui.div(
                        shiny_ui.input_numeric(
                            "comp_alpha",
                            "FDR Alpha (α)",
                            value=0.05,
                            min=0.001,
                            max=0.50,
                            step=0.01,
                        ),
                        class_="col-6",
                    ),
                    class_="row g-2",
                ),
                shiny_ui.input_action_button(
                    "btn_run_comparison",
                    "Run Comparison & Contrast",
                    icon=Icon.lucide("git-compare", size=16),
                    class_="btn-primary w-100 mt-2",
                ),
                shiny_ui.output_ui("contrast_pair_selector_ui"),
                width=360,
            ),
            shiny_ui.div(
                shiny_ui.output_ui("comparison_kpi_ui"),
                shiny_ui.navset_card_tab(
                    shiny_ui.nav_panel(
                        "Interactive Plots",
                        cls.plots_explorer.ui(),
                        value="comparison_plots_tab",
                        icon=Icon.lucide("chart-bar", size=16),
                    ),
                    shiny_ui.nav_panel(
                        "DataFrames Explorer",
                        cls.tables_explorer.ui(),
                        value="comparison_tables_tab",
                        icon=Icon.lucide("table", size=16),
                    ),
                    id="comparison_workspace_tabs",
                ),
            ),
        )

    @bind_output(render.ui)
    def comp_source_ui(self) -> TagChild:
        mode = str(self.val("comp_mode", default="cohorts"))
        if mode == "cohorts":
            stratified_ests = {
                name: name
                for name, est in self.state.estimators.items.items()
                if est.aggregator_ is not None and est.aggregator_.n_strata >= 1
            }
            if not stratified_ests:
                return shiny_ui.div(
                    Icon.lucide("circle-alert", size=14, class_="text-warning me-1"),
                    shiny_ui.span(
                        "Cohort contrasts require a stratified Estimator (e.g. stratified by Country or Source). "
                        "Switch to 'Saved Formulations / Estimators' or fit a stratified Estimator first.",
                        class_="small text-muted",
                    ),
                    class_="p-2 bg-light rounded border",
                )

            cur_est = self.val("comp_cohort_est", isolate=True)
            sel_est_key = str(cur_est) if cur_est in stratified_ests else next(iter(stratified_ests.keys()))
            est = self.state.estimators.get(sel_est_key)
            assert est is not None and est.aggregator_ is not None
            agg = est.aggregator_

            strata_cols = list(agg.strata_cols)
            cur_col = self.val("comp_stratum_col", isolate=True)
            sel_col = str(cur_col) if cur_col in strata_cols else strata_cols[0]

            # Order strata values by descending isolate count n
            top_strata_df = agg.data_.group_by(sel_col).agg(n=pl.col("n").max()).sort("n", descending=True)
            strata_vals = [str(v) for v in top_strata_df[sel_col].to_list() if v is not None]
            strata_choices = {
                str(row[sel_col]): f"{row[sel_col]} (n={int(row['n']):,})"
                for row in top_strata_df.to_dicts()
                if row[sel_col] is not None
            }
            default_selected = strata_vals[:2] if len(strata_vals) >= 2 else strata_vals

            controls: list[TagChild] = [
                shiny_ui.input_select(
                    "comp_cohort_est",
                    "Stratified Estimator",
                    choices=stratified_ests,
                    selected=sel_est_key,
                ),
            ]
            if len(strata_cols) > 1:
                controls.append(
                    shiny_ui.input_select(
                        "comp_stratum_col",
                        "Stratum Variable",
                        choices={c: c for c in strata_cols},
                        selected=sel_col,
                    )
                )
            controls.append(
                shiny_ui.input_selectize(
                    "comp_strata_values",
                    "Select Cohorts / Strata (2+)",
                    choices=strata_choices,
                    selected=default_selected,
                    multiple=True,
                )
            )
            return shiny_ui.div(*controls)

        # Mode == "models": combine registered Formulations and Estimators
        model_choices: dict[str, str] = {}
        for name in self.state.formulations.names:
            model_choices[f"form::{name}"] = f"[Formulation] {name}"
        for name in self.state.estimators.names:
            model_choices[f"est::{name}"] = f"[Estimator] {name}"

        keys = list(model_choices.keys())
        default_sel = keys[:2] if len(keys) >= 2 else keys
        return shiny_ui.div(
            shiny_ui.input_selectize(
                "comp_saved_models",
                "Select Formulations / Estimators (2+)",
                choices=model_choices,
                selected=default_sel,
                multiple=True,
            )
        )

    @bind_task("btn_run_comparison", on_success="_on_comparison_done")
    def run_comparison_task(self) -> FormulationComparison:
        mode = str(self.val("comp_mode", default="cohorts"))
        valency = max(1, int(self.val("comp_valency", default=6) or 6))
        weighting_raw = str(self.val("comp_weighting", default="pooled"))
        weighting = "equal" if weighting_raw == "equal" else "pooled"
        null_margin = float(self.val("comp_null_margin", default=0.0) or 0.0)
        alpha = float(self.val("comp_alpha", default=0.05) or 0.05)

        if mode == "cohorts":
            est_key = self.val("comp_cohort_est")
            est = self.state.estimators.get(str(est_key)) if est_key else self.state.active_estimator
            if est is None or est.aggregator_ is None or est.aggregator_.n_strata == 0:
                raise ValueError("Please select a stratified Estimator for Cohort / Stratum Contrasts.")
            agg = est.aggregator_
            strata_cols = list(agg.strata_cols)
            col_raw = self.val("comp_stratum_col")
            sel_col = str(col_raw) if (col_raw and str(col_raw) in strata_cols) else strata_cols[0]

            raw_vals = self.val("comp_strata_values", default=())
            sel_vals = [str(v) for v in (raw_vals if isinstance(raw_vals, (list, tuple)) else [raw_vals]) if v]
            if len(sel_vals) < 2:
                raise ValueError("Please select at least 2 cohorts/strata to compare.")

            return self.state.cache.call_or_load(
                FormulationComparison.from_strata,
                est,
                agg,
                sel_vals,
                stratum_col=sel_col,
                valency=valency,
                weighting=weighting,
                alpha=alpha,
                null_margin=null_margin,
            )

        raw_models = self.val("comp_saved_models", default=())
        sel_keys = [str(k) for k in (raw_models if isinstance(raw_models, (list, tuple)) else [raw_models]) if k]
        if len(sel_keys) < 2:
            raise ValueError("Please select at least 2 saved Formulations or Estimators to compare.")

        specs: dict[str, Formulation | BaseEstimator] = {}
        for key in sel_keys:
            if key.startswith("form::"):
                f_name = key.removeprefix("form::")
                f_obj = self.state.formulations.get(f_name)
                if f_obj is not None:
                    specs[f_name] = f_obj
            elif key.startswith("est::"):
                e_name = key.removeprefix("est::")
                e_obj = self.state.estimators.get(e_name)
                if e_obj is not None:
                    specs[e_name] = e_obj

        if len(specs) < 2:
            raise ValueError("Could not resolve at least 2 valid models from the registry.")

        comp = FormulationComparison(
            formulations=specs,
            valency=valency,
            weighting=weighting,
            cv=None,
            alpha=alpha,
            null_margin=null_margin,
        )
        return self.state.cache.fit_or_load(comp, None)

    def _on_comparison_done(self, comp: FormulationComparison) -> None:
        self._active_comparison.set(comp)

    @reactive_calc
    def active_comparison(self) -> FormulationComparison | None:
        """Reactively returns the most recently computed `FormulationComparison`.

        Returns:
            Active `FormulationComparison` instance or `None`.
        """
        return self._active_comparison.get()

    @reactive_calc
    def selected_contrast_pair(self) -> tuple[str, str] | None:
        """Reactively resolves the active `(label_a, label_b)` pairwise contrast labels.

        Returns:
            Tuple of `(baseline_label, comparison_label)` or `None` if no comparison is active.
        """
        comp = self.active_comparison()
        if comp is None:
            return None
        names = list(comp.formulations_.keys())
        if len(names) < 2:
            return None
        raw_a = self.val("contrast_label_a")
        raw_b = self.val("contrast_label_b")
        label_a = str(raw_a) if (raw_a and str(raw_a) in names) else names[0]
        label_b = (
            str(raw_b)
            if (raw_b and str(raw_b) in names and str(raw_b) != label_a)
            else next((n for n in names if n != label_a), names[1])
        )
        return label_a, label_b

    @bind_output(render.ui)
    def contrast_pair_selector_ui(self) -> TagChild:
        """Renders dropdown selectors for choosing the active `(Baseline A, Comparison B)` pair.

        Returns:
            UI fragment with `contrast_label_a` and `contrast_label_b` selects.
        """
        comp = self.active_comparison()
        if comp is None:
            return shiny_ui.div()
        names = list(comp.formulations_.keys())
        if len(names) < 2:
            return shiny_ui.div()

        pair = self.selected_contrast_pair()
        sel_a = pair[0] if pair else names[0]
        sel_b = pair[1] if pair else names[1]
        choices = {n: n for n in names}

        return shiny_ui.div(
            shiny_ui.hr(class_="my-2"),
            shiny_ui.p(
                Icon.lucide("sliders-horizontal", size=14, class_="me-1"),
                shiny_ui.strong("Active Pairwise Contrast"),
                class_="small mb-1 d-flex align-items-center",
            ),
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.input_select("contrast_label_a", "Baseline (A)", choices=choices, selected=sel_a),
                    class_="col-6",
                ),
                shiny_ui.div(
                    shiny_ui.input_select("contrast_label_b", "Comparison (B)", choices=choices, selected=sel_b),
                    class_="col-6",
                ),
                class_="row g-2",
            ),
        )

    @bind_output(render.ui)
    def comparison_kpi_ui(self) -> TagChild:
        """Renders a 4-card KPI summary row for the active pairwise contrast.

        Returns:
            `KPIRow` UI tag summarizing top-K overlap, rank concordance, differential targets, and inference engine.
        """
        comp = self.active_comparison()
        pair = self.selected_contrast_pair()
        if comp is None or pair is None:
            return shiny_ui.div()

        label_a, label_b = pair
        diff_df = comp.contrast(label_a, label_b)
        n_sig = int(diff_df["significant"].sum()) if "significant" in diff_df.columns else 0
        method_str = str(diff_df["test_method"][0]) if len(diff_df) > 0 else "fisher_exact_bh"
        method_badge = "Bayesian LFSR" if method_str == "bayesian_lfsr" else "Fisher's Exact + BH"

        conc_row = comp.concordance_.filter(
            ((pl.col("model_a") == label_a) & (pl.col("model_b") == label_b))
            | ((pl.col("model_a") == label_b) & (pl.col("model_b") == label_a))
        )
        if len(conc_row) > 0:
            jaccard = float(conc_row["jaccard_similarity"][0])
            overlap = int(conc_row["top_v_overlap"][0])
            rho = float(conc_row["spearman_rho"][0])
            tau = float(conc_row["kendall_tau"][0])
            shared = str(conc_row["shared_targets"][0]) or "None"
        else:
            jaccard, overlap, rho, tau, shared = 0.0, 0, 0.0, 0.0, "None"

        return KPIRow(
            KPICard(
                label=f"Top-{comp.valency} Overlap ({label_a} vs {label_b})",
                value=f"{overlap}/{comp.valency} (J={jaccard:.2f})",
                subtitle=f"Shared: {shared}",
                value_class="fs-5 text-primary",
                truncate_subtitle=True,
                col_width=4,
            ),
            KPICard(
                label="Rank Concordance",
                value=f"ρ = {rho:.2f} · τ = {tau:.2f}",
                subtitle="Spearman ρ & Kendall τ",
                col_width=3,
            ),
            KPICard(
                label=f"Differential Targets (q < {comp.alpha})",
                value=f"{n_sig} / {len(diff_df)}",
                subtitle=f"Null margin δ = {comp.null_margin:.1%}",
                value_class="fs-5 text-danger",
                col_width=3,
            ),
            KPICard(
                label="Inference Engine",
                value=method_badge,
                subtitle=f"{len(comp.formulations_)} cohorts/models",
                value_class="fs-6 text-success",
                col_width=2,
            ),
        ).ui()


# ---------------------------------------------------------------------------
# 4. Top-Level FormulationTab
# ---------------------------------------------------------------------------


class FormulationTab(Tab[SessionState]):
    """Formulation & Contrasts.

    Top-level tab module composing `FormulationDesignerPanel` (for K-valent vaccine/phage
    formulation design and Leave-One-Stratum-Out stability analysis) and
    `FormulationComparisonPanel` (for cohort/stratum contrasts and multi-model concordance).

    Attributes:
        icon (Tag): Lucide `"shield-check"` navbar icon.
        requires (ClassVar[tuple[Requirement, ...]]): Prerequisites requiring at least one
            loaded `Dataset` and one trained `BaseEstimator`.
        designer_panel (FormulationDesignerPanel): Subcomponent for K-valent formulation design and LOSO CV.
        comparison_panel (FormulationComparisonPanel): Subcomponent for cohort and model differential prevalence contrasts.
    """

    icon = Icon.lucide("shield-check")
    requires: ClassVar[tuple[Requirement, ...]] = (
        Requirement("datasets", min_count=1, label="Loaded Dataset", target_tab="dataset_tab"),
        Requirement("estimators", min_count=1, label="Trained Estimator", target_tab="estimation_tab"),
    )

    designer_panel = subcomponent(FormulationDesignerPanel)
    comparison_panel = subcomponent(FormulationComparisonPanel)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.navset_card_underline(
            shiny_ui.nav_panel(
                "1. Formulation Designer & Stability",
                cls.designer_panel.ui(),
                value="formulation_designer_subtab",
                icon=Icon.lucide("shield-check", size=16),
            ),
            shiny_ui.nav_panel(
                "2. Cohort & Model Contrasts (FDR / LFSR)",
                cls.comparison_panel.ui(),
                value="formulation_comparison_subtab",
                icon=Icon.lucide("git-compare", size=16),
            ),
            id="formulation_main_navset",
        )
