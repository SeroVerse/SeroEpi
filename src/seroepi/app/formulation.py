"""Multi-Valent Formulation Design, LOSO Stability, and Cohort/Model Contrasts tab module."""

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
from shiny_kit.ui import Icon

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

ALL_FORMULATION_PLOTTERS: dict[str, tuple[str, type[BasePlotter]]] = {
    **FORMULATION_PLOTTER_REGISTRY,
    **COMPARISON_PLOTTER_REGISTRY,
}

type FormulationSubject = tuple[Formulation | None, FormulationComparison | None]


# ---------------------------------------------------------------------------
# 2. Subcomponent 1: Formulation Design Sidebar Panel
# ---------------------------------------------------------------------------


class FormulationSwitcher(ArtifactSwitcher[Formulation]):
    """Sidebar switcher for registered `Formulation` artifacts with `.joblib` & Target Composer export."""

    registry_kind = "formulations"
    label = "Registered Formulations"
    enable_delete = True
    enable_download = True
    file_extension = ".joblib"
    download_button_label = "Save (.joblib)"

    def summary_ui(self, item: Formulation) -> TagChild:
        """Renders a compact summary badge card for the currently active `Formulation`.

        Args:
            item: Active fitted `Formulation` instance.

        Returns:
            UI card fragment summarizing valency, cumulative coverage, FDR significance, and LOSO status.
        """
        est_col = item.estimator_.estimate.column
        in_df = item.rankings_.filter(pl.col("in_formulation"))
        cum_cov = min(1.0, max(0.0, float(in_df[est_col].sum()))) if len(in_df) > 0 else 0.0
        n_sig = int(item.rankings_["significant"].sum()) if "significant" in item.rankings_.columns else 0
        loso_txt = ""
        if len(item.fold_summary_) > 0:
            mean_heldout = float(item.fold_summary_["heldout_predicted_coverage"].to_numpy().mean())
            loso_txt = f" • LOSO {mean_heldout:.1%}"

        return shiny_ui.div(
            shiny_ui.div(
                Icon.lucide("shield-check", size=14, class_="text-success me-1"),
                shiny_ui.strong(f"{item.valency}-Valent Formulation ({item.weighting})", class_="small"),
                class_="d-flex align-items-center mb-1",
            ),
            shiny_ui.div(
                ", ".join(item.selected_targets_[:6])
                + (f" (+{len(item.selected_targets_) - 6} more)" if len(item.selected_targets_) > 6 else ""),
                class_="small text-body fw-semibold text-truncate",
            ),
            shiny_ui.div(
                f"{cum_cov:.1%} coverage • {n_sig}/{len(item.rankings_)} FDR sig.{loso_txt}",
                class_="text-muted",
                style="font-size: 0.78rem;",
            ),
            class_="p-2 bg-body-tertiary rounded border border-success-subtle mb-2",
        )

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


class FormulationDesignPanel(Component[SessionState]):
    """Sidebar subcomponent configuring and fitting K-valent `Formulation` models with LOSO CV.

    Attributes:
        switcher (FormulationSwitcher): Subcomponent for switching, exporting, or deleting saved formulations.
    """

    switcher = subcomponent(FormulationSwitcher)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(
            cls.switcher.ui(),
            shiny_ui.output_ui("designer_config_ui"),
            shiny_ui.input_action_button(
                "btn_fit_formulation",
                "Design & Fit Formulation",
                icon=Icon.lucide("shield-check", size=16),
                class_="btn-primary w-100 mt-2",
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


# ---------------------------------------------------------------------------
# 3. Subcomponent 2: Cohort & Model Contrasts Sidebar Panel
# ---------------------------------------------------------------------------


class ContrastsPanel(Component[SessionState]):
    """Sidebar subcomponent for running Cohort Contrasts (`from_strata`) and Multi-Model `FormulationComparison`."""

    def __init__(self) -> None:
        super().__init__()
        self._active_comparison: reactive.Value[FormulationComparison | None] = reactive.Value(None)
        self._comparison_version: reactive.Value[int] = reactive.Value(0)

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.div(
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
                class_="btn-success w-100 mt-2",
            ),
            shiny_ui.output_ui("contrast_pair_selector_ui"),
        )

    @bind_output(render.ui)
    def comp_source_ui(self) -> TagChild:
        """Renders cohort or saved-model selection controls based on `comp_mode`."""
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
                    class_="p-2 bg-body-tertiary rounded border",
                )

            cur_est = self.val("comp_cohort_est", isolate=True)
            sel_est_key = str(cur_est) if cur_est in stratified_ests else next(iter(stratified_ests.keys()))
            est = self.state.estimators.get(sel_est_key)
            assert est is not None and est.aggregator_ is not None
            agg = est.aggregator_

            strata_cols = list(agg.strata_cols)
            cur_col = self.val("comp_stratum_col", isolate=True)
            sel_col = str(cur_col) if cur_col in strata_cols else strata_cols[0]

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
        """Executes or loads from cache the configured `FormulationComparison`."""
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
        with reactive.isolate():
            self._comparison_version.set(self._comparison_version.get() + 1)

    @reactive_calc
    def active_comparison(self) -> FormulationComparison | None:
        """Reactively returns the most recently computed `FormulationComparison`."""
        return self._active_comparison.get()

    @reactive_calc
    def comparison_version(self) -> int:
        """Reactively returns the monotonically increasing version counter for completed comparisons."""
        return self._comparison_version.get()

    @reactive_calc
    def selected_contrast_pair(self) -> tuple[str, str] | None:
        """Reactively resolves the active `(label_a, label_b)` pairwise contrast labels."""
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
        """Renders dropdown selectors for choosing the active `(Baseline A, Comparison B)` pair."""
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


# ---------------------------------------------------------------------------
# 4. Subcomponent 3: Unified Formulation & Contrasts DataFrame Explorer (Main Tab 1)
# ---------------------------------------------------------------------------


class FormulationExplorer(TableExplorer[FormulationSubject]):
    """Main-panel `TableExplorer` displaying Formulation rankings/LOSO tables and Contrast tables."""

    csv_prefix = "seroepi_formulation"
    empty_icon = "shield-check"
    empty_title = "No Active Formulation or Contrast Yet"
    empty_message = (
        "Design a K-valent Formulation or run a Cohort/Model Contrast in the sidebar to inspect rankings, "
        "LOSO stability, and differential prevalence tables."
    )

    def __init__(self) -> None:
        super().__init__()
        self._get_comparison: Callable[[], FormulationComparison | None] = lambda: None
        self._get_comp_version: Callable[[], int] = lambda: 0
        self._get_pair: Callable[[], tuple[str, str] | None] = lambda: None
        self._last_form_version: int = 0
        self._last_comp_version: int = 0

    def bind_comparison(
        self,
        get_comp: Callable[[], FormulationComparison | None],
        get_comp_version: Callable[[], int],
        get_pair: Callable[[], tuple[str, str] | None],
    ) -> None:
        """Wires reactive getters for the active comparison, its version counter, and the contrast pair."""
        self._get_comparison = get_comp
        self._get_comp_version = get_comp_version
        self._get_pair = get_pair

    def active_subject(self) -> FormulationSubject | None:
        """Returns the `(active_formulation, active_comparison)` pair, or `None` if neither exists."""
        form = self.state.active_formulation
        comp = self._get_comparison()
        if form is None and comp is None:
            return None
        return form, comp

    def table_views(self, subject: FormulationSubject) -> dict[str, str]:
        """Returns available DataGrid view modes for `subject`."""
        form, comp = subject
        choices: dict[str, str] = {}
        if form is not None:
            choices["rankings"] = "Global Rankings & FDR"
            choices["stratum_coverage"] = "Stratum Coverage"
            if form.aggregator_.n_strata >= 1:
                choices["stratum_trajectories"] = "Stratum Coverage Trajectories"
            if len(form.stability_metrics_) > 0:
                choices["stability"] = "LOSO Target Stability"
            if len(form.fold_summary_) > 0:
                choices["folds"] = "LOSO Held-Out Folds"
        if comp is not None:
            choices["differential"] = "Contrast: Differential Prevalence (A vs. B)"
            choices["concordance"] = "Contrast: Pairwise Concordance"
            choices["rank_shifts"] = "Contrast: Wide Rank Shifts"
            choices["comp_rankings"] = "Contrast: All Cohort / Model Rankings"
        return choices

    def resolve_selected_mode(
        self,
        subject: FormulationSubject,
        choices: dict[str, str],
        current_mode: str | None,
    ) -> str:
        """Automatically switches between Formulation and Contrast table views when either workflow runs."""
        form, comp = subject
        form_ver = self.state.formulations.version
        comp_ver = self._get_comp_version()

        if comp is not None and comp_ver != self._last_comp_version:
            self._last_comp_version = comp_ver
            self._last_form_version = form_ver
            if "differential" in choices:
                return "differential"
        if form is not None and form_ver != self._last_form_version:
            self._last_form_version = form_ver
            if "rankings" in choices:
                return "rankings"
        return super().resolve_selected_mode(subject, choices, current_mode)

    def toolbar_badges_ui(self, subject: FormulationSubject) -> Sequence[TagChild]:
        """Renders compact inline summary badges in the table toolbar matching `EstimatesExplorer`."""
        form, comp = subject
        mode = str(self.val("table_view_mode", default=""))
        is_contrast_mode = mode in ("differential", "concordance", "rank_shifts", "comp_rankings") or form is None

        if is_contrast_mode and comp is not None:
            pair = self._get_pair()
            if pair is None:
                return ()
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
            else:
                jaccard, overlap, rho, tau = 0.0, 0, 0.0, 0.0

            return (
                shiny_ui.span(
                    f"{label_a} vs {label_b}: {overlap}/{comp.valency} Overlap (J={jaccard:.2f})",
                    class_="badge bg-primary me-1",
                ),
                shiny_ui.span(
                    f"ρ = {rho:.2f} · τ = {tau:.2f}",
                    class_="badge bg-secondary me-1",
                ),
                shiny_ui.span(
                    f"Diff. Targets (q < {comp.alpha}): {n_sig}/{len(diff_df)}",
                    class_="badge bg-danger-subtle text-danger-emphasis me-1",
                ),
                shiny_ui.span(
                    method_badge,
                    class_="badge bg-info-subtle text-info-emphasis me-1",
                ),
            )

        if form is not None:
            est_col = form.estimator_.estimate.column
            in_df = form.rankings_.filter(pl.col("in_formulation"))
            cum_cov = min(1.0, max(0.0, float(in_df[est_col].sum()))) if len(in_df) > 0 else 0.0
            n_sig = int(form.rankings_["significant"].sum()) if "significant" in form.rankings_.columns else 0
            badges: list[TagChild] = [
                shiny_ui.span(f"{form.valency}-Valent ({form.weighting})", class_="badge bg-primary me-1"),
                shiny_ui.span(f"Coverage: {cum_cov:.1%}", class_="badge bg-success-subtle text-success-emphasis me-1"),
                shiny_ui.span(
                    f"FDR Sig (q < {form.alpha}): {n_sig}/{len(form.rankings_)}",
                    class_="badge bg-secondary me-1",
                ),
            ]
            if len(form.fold_summary_) > 0:
                mean_heldout = float(form.fold_summary_["heldout_predicted_coverage"].to_numpy().mean())
                mean_jaccard = float(form.fold_summary_["jaccard_similarity"].to_numpy().mean())
                badges.append(
                    shiny_ui.span(
                        f"LOSO: {mean_heldout:.1%} (J={mean_jaccard:.2f})",
                        class_="badge bg-info-subtle text-info-emphasis me-1",
                    )
                )
            return tuple(badges)

        return ()

    def resolve_table(self, subject: FormulationSubject, mode: str) -> pl.DataFrame | None:
        """Resolves the Polars DataFrame for `mode` from `subject`."""
        form, comp = subject
        if mode in ("differential", "concordance", "rank_shifts", "comp_rankings") and comp is not None:
            if mode == "concordance":
                return comp.concordance_
            if mode == "rank_shifts":
                return comp.rank_shifts_
            if mode == "comp_rankings":
                return comp.rankings_
            pair = self._get_pair()
            if pair is not None:
                return comp.contrast(pair[0], pair[1])
            return comp.differential_prevalence_

        if form is not None:
            if mode == "stratum_coverage":
                return form.score_by_stratum()
            if mode == "stratum_trajectories" and form.aggregator_.n_strata >= 1:
                cum_df, _ = form.stratum_trajectories()
                return cum_df
            if mode == "stability" and len(form.stability_metrics_) > 0:
                return form.stability_metrics_
            if mode == "folds" and len(form.fold_summary_) > 0:
                return form.fold_summary_
            return form.rankings_

        if comp is not None:
            pair = self._get_pair()
            return comp.contrast(pair[0], pair[1]) if pair is not None else comp.differential_prevalence_
        return None


# ---------------------------------------------------------------------------
# 5. Subcomponent 4: Unified Interactive Plots Explorer (Main Tab 2)
# ---------------------------------------------------------------------------


class FormulationPlotsExplorer(PlotterExplorer[FormulationSubject]):
    """Main-panel `PlotterExplorer` dynamically populating Formulation and Contrast plotters."""

    plotter_registry = ALL_FORMULATION_PLOTTERS
    default_plot_key = "coverage"
    param_prefix = "fplot"
    download_prefix = "seroepi_formulation"
    sidebar_width = 310
    chart_height = "75vh"
    empty_icon = "chart-bar"
    empty_title = "No Active Formulation or Contrast to Plot"
    empty_message = (
        "Design a Formulation or run a Cohort/Model Contrast in the sidebar to explore coverage, "
        "stability, volcano, bump, and UpSet plots."
    )

    def __init__(self) -> None:
        super().__init__()
        self._get_comparison: Callable[[], FormulationComparison | None] = lambda: None
        self._get_comp_version: Callable[[], int] = lambda: 0
        self._get_pair: Callable[[], tuple[str, str] | None] = lambda: None
        self._last_form_version: int = 0
        self._last_comp_version: int = 0

    def bind_comparison(
        self,
        get_comp: Callable[[], FormulationComparison | None],
        get_comp_version: Callable[[], int],
        get_pair: Callable[[], tuple[str, str] | None],
    ) -> None:
        """Wires reactive getters for the active comparison, its version counter, and the contrast pair."""
        self._get_comparison = get_comp
        self._get_comp_version = get_comp_version
        self._get_pair = get_pair

    def active_subject(self) -> FormulationSubject | None:
        """Returns the `(active_formulation, active_comparison)` pair, or `None` if neither exists."""
        form = self.state.active_formulation
        comp = self._get_comparison()
        if form is None and comp is None:
            return None
        return form, comp

    def available_choices(self, subject: FormulationSubject) -> dict[str, str]:
        """Returns compatible plotter choices across active `Formulation` and `FormulationComparison` objects."""
        form, comp = subject
        choices: dict[str, str] = {}
        if form is not None and form.is_fitted_:
            for key, (label, _) in FORMULATION_PLOTTER_REGISTRY.items():
                choices[key] = label
        if comp is not None and comp.is_fitted_:
            for key, (label, _) in COMPARISON_PLOTTER_REGISTRY.items():
                choices[key] = label
        return choices

    @bind_output(render.ui)
    def plot_selector_ui(self) -> TagChild:
        """Renders the Plot Type dropdown, auto-switching to `"differential"` or `"coverage"` on new runs."""
        subject = self.active_subject()
        if subject is None:
            return shiny_ui.p(self.empty_message, class_="text-muted small mb-0")

        form, comp = subject
        choices = self.available_choices(subject)
        current_plot = self.val("plot_type", isolate=True)
        fallback = self.default_plot_key if self.default_plot_key in choices else next(iter(choices.keys()), None)
        selected_plot = str(current_plot) if current_plot in choices else fallback

        form_ver = self.state.formulations.version
        comp_ver = self._get_comp_version()
        if comp is not None and comp_ver != self._last_comp_version:
            self._last_comp_version = comp_ver
            self._last_form_version = form_ver
            if "differential" in choices:
                selected_plot = "differential"
        elif form is not None and form_ver != self._last_form_version:
            self._last_form_version = form_ver
            if "coverage" in choices:
                selected_plot = "coverage"

        header = self.selector_header_ui(subject)
        return shiny_ui.div(
            header,
            shiny_ui.input_select(
                "plot_type",
                "Plot Type",
                choices=choices,
                selected=selected_plot,
            ),
        )

    def selector_header_ui(self, subject: FormulationSubject) -> TagChild:
        """Renders an unlock status indicator for Cohort & Model Contrast plotters."""
        _, comp = subject
        if comp is not None:
            n_models = len(comp.formulations_)
            return shiny_ui.div(
                Icon.lucide("git-compare", size=13, class_="me-1 text-success"),
                shiny_ui.span(
                    f"Contrast plots unlocked ({n_models} cohorts/models)",
                    class_="small text-success",
                ),
                class_="d-flex align-items-center mb-2",
            )
        return shiny_ui.div(
            Icon.lucide("lock", size=13, class_="me-1 text-muted"),
            shiny_ui.span(
                "Run Cohort & Model Contrasts in sidebar to unlock Volcano, Bump & UpSet plots",
                class_="small text-muted",
            ),
            class_="d-flex align-items-center mb-2",
        )

    def build_chart(
        self,
        plotter: BasePlotter,
        subject: FormulationSubject,
        plot_key: str,
    ) -> alt.TopLevelMixin | None:
        """Constructs the Altair chart from either the active `Formulation` or `FormulationComparison`."""
        form, comp = subject
        if plot_key in COMPARISON_PLOTTER_REGISTRY:
            if comp is None or not comp.is_fitted_:
                return None
            if isinstance(plotter, DifferentialPrevalencePlotter):
                pair = self._get_pair()
                if pair is not None:
                    return plotter.plot(comp, label_a=pair[0], label_b=pair[1])
                return plotter.plot(comp)
            if isinstance(plotter, (RankShiftPlotter, UpSetPlotter)):
                return plotter.plot(comp)
            return None

        if form is None or not form.is_fitted_:
            return None
        if isinstance(plotter, (FormulationCoveragePlotter, FormulationStabilityPlotter)):
            return plotter.plot(form)
        return None


# ---------------------------------------------------------------------------
# 6. Top-Level FormulationTab (Composed of 4 Focused Subcomponents)
# ---------------------------------------------------------------------------


class FormulationTab(Tab[SessionState]):
    """Formulation & Contrasts.

    Top-level tab module orchestrating `FormulationDesignPanel`, `ContrastsPanel`,
    `FormulationExplorer`, and `FormulationPlotsExplorer` inside a unified sidebar +
    underline-card workspace matching `DatasetTab` and `EstimationTab`.

    Attributes:
        icon (Tag): Lucide `"shield-check"` navbar icon.
        requires (ClassVar[tuple[Requirement, ...]]): Prerequisites requiring at least one
            loaded `Dataset` and one trained `BaseEstimator`.
        design_panel (FormulationDesignPanel): Sidebar accordion panel for K-valent formulation design and LOSO CV.
        contrasts_panel (ContrastsPanel): Sidebar accordion panel for cohort and model differential contrasts.
        formulation_explorer (FormulationExplorer): Main-panel DataGrid explorer for rankings, LOSO, and contrasts.
        plots_explorer (FormulationPlotsExplorer): Main-panel interactive Altair plotter workspace.
    """

    icon = Icon.lucide("shield-check")
    requires: ClassVar[tuple[Requirement, ...]] = (
        Requirement("datasets", min_count=1, label="Loaded Dataset", target_tab="dataset_tab"),
        Requirement("estimators", min_count=1, label="Trained Estimator", target_tab="estimation_tab"),
    )

    design_panel = subcomponent(FormulationDesignPanel)
    contrasts_panel = subcomponent(ContrastsPanel)
    formulation_explorer = subcomponent(FormulationExplorer)
    plots_explorer = subcomponent(FormulationPlotsExplorer)

    def __init__(self) -> None:
        super().__init__()
        self.formulation_explorer.bind_comparison(
            self.contrasts_panel.active_comparison,
            self.contrasts_panel.comparison_version,
            self.contrasts_panel.selected_contrast_pair,
        )
        self.plots_explorer.bind_comparison(
            self.contrasts_panel.active_comparison,
            self.contrasts_panel.comparison_version,
            self.contrasts_panel.selected_contrast_pair,
        )

    @classmethod
    def _render_ui(cls) -> TagChild:
        return shiny_ui.layout_sidebar(
            shiny_ui.sidebar(
                shiny_ui.accordion(
                    shiny_ui.accordion_panel(
                        "1. Formulation Design",
                        cls.design_panel.ui(),
                        value="form_design_accordion_panel",
                        icon=Icon.lucide("shield-check", size=16),
                    ),
                    shiny_ui.accordion_panel(
                        "2. Cohort & Model Contrasts",
                        cls.contrasts_panel.ui(),
                        value="form_contrasts_accordion_panel",
                        icon=Icon.lucide("git-compare", size=16),
                    ),
                    id="formulation_sidebar_accordion",
                    open=["form_design_accordion_panel", "form_contrasts_accordion_panel"],
                    multiple=True,
                ),
                width=360,
            ),
            shiny_ui.navset_card_underline(
                shiny_ui.nav_panel(
                    "Formulation Explorer",
                    cls.formulation_explorer.ui(),
                    value="formulation_dataframe_panel",
                    icon=Icon.lucide("table", size=16),
                ),
                shiny_ui.nav_panel(
                    "Plots",
                    cls.plots_explorer.ui(),
                    value="formulation_plots_panel",
                    icon=Icon.lucide("chart-bar", size=16),
                ),
                id="formulation_main_tabs",
            ),
        )
