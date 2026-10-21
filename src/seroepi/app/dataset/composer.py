"""Interactive Target Composer subcomponent (`TargetComposer`)."""

from typing import Any, Literal

import altair as alt
import polars as pl
from htmltools import TagChild
from shiny import reactive, render, req
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_datagrid, bind_effect, bind_output, reactive_calc
from shiny_kit.ui import Icon, StagedList
from shinywidgets import output_widget

from seroepi.app.decorators import bind_altair
from seroepi.app.models import SessionState
from seroepi.variables import Criterion, Target


class TargetComposer(Component[SessionState]):
    """Interactive composer, live Altair/Polars previewer, and cached `Target` repertoire manager.

    Allows users to stage one or more [`Criterion`](file:///Users/tsta0015/Programming/SeroEpi/src/seroepi/variables.py)
    rules with regex matching, negative indicators, and equivalence aliases, preview the
    resulting compositional or binary [`Target`](file:///Users/tsta0015/Programming/SeroEpi/src/seroepi/variables.py)
    distribution in real time on the active `Dataset`, and register targets into `state.targets`.
    """

    def __init__(self) -> None:
        """Initializes `TargetComposer` with an empty reactive `StagedList[Criterion]`."""
        super().__init__()
        self._staged: StagedList[Criterion] = StagedList()

    @property
    def staged_criteria(self) -> list[Criterion]:
        """Reactively returns the ordered list of staged `Criterion` items.

        Returns:
            List of staged `Criterion` instances.
        """
        return self._staged.items

    @staged_criteria.setter
    def staged_criteria(self, items: list[Criterion]) -> None:
        self._staged.set(items)

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.layout_columns(
            # --- Left Column: Target Composer ---
            shiny_ui.card(
                shiny_ui.card_header(Icon.lucide("sparkles", size=16), " Target Composer", class_="fw-bold"),
                shiny_ui.card_body(
                    shiny_ui.output_ui("target_composer_vars_ui"),
                    shiny_ui.layout_columns(
                        shiny_ui.input_text(
                            "crit_pattern", "Regex Pattern", value=".*", placeholder="e.g. .* or ^(typeA|typeB)$"
                        ),
                        shiny_ui.input_text(
                            "crit_neg_ind", "Negative Indicator", value="", placeholder="Default (- / 0 / none)"
                        ),
                        col_widths=(7, 5),
                        fill=False,
                    ),
                    shiny_ui.input_text(
                        "crit_equiv",
                        "Equivalence Mapping (Optional, raw=normalized)",
                        value="",
                        placeholder="e.g. alias1=canonical, alias2=canonical",
                    ),
                    StagedList.action_buttons(
                        add_id="btn_add_criterion",
                        clear_id="btn_clear_criteria",
                        pop_id="btn_pop_criterion",
                        add_label="Add Criterion",
                    ),
                    shiny_ui.output_ui("staged_criteria_list_ui"),
                    shiny_ui.hr(class_="my-2"),
                    shiny_ui.p("3. Target Settings & Registration", class_="fw-semibold small mb-1"),
                    shiny_ui.layout_columns(
                        shiny_ui.input_text("target_name_input", "Target ID / Key", placeholder="Auto from criteria"),
                        shiny_ui.input_text("target_title_input", "Display Title", placeholder="Auto from criteria"),
                        col_widths=(6, 6),
                        fill=False,
                    ),
                    shiny_ui.layout_columns(
                        shiny_ui.input_radio_buttons(
                            "target_conjunction",
                            "Conjunction",
                            choices={"or": "OR (Union)", "and": "AND (Intersection)"},
                            selected="or",
                            inline=True,
                        ),
                        shiny_ui.input_text("target_sep", "Token Separator", value=" "),
                        col_widths=(8, 4),
                        fill=False,
                    ),
                    shiny_ui.input_select(
                        "target_compositional",
                        "Target Mode",
                        choices={
                            "auto": "Auto (Infer from Variable Types)",
                            "true": "Compositional (Categorical Breakdown)",
                            "false": "Binary Endpoint (Positive vs. Negative)",
                        },
                        selected="auto",
                    ),
                    shiny_ui.input_action_button(
                        "btn_register_target",
                        "Save Target to Repertoire",
                        icon=Icon.lucide("target", size=16),
                        class_="btn-primary w-100 mt-2",
                    ),
                    fillable=False,
                ),
                class_="shadow-sm",
                fill=False,
            ),
            # --- Right Column: Live Preview & Repertoire ---
            shiny_ui.div(
                shiny_ui.card(
                    shiny_ui.card_header(
                        Icon.lucide("chart-bar", size=16), " Live Target Preview (Active Dataset)", class_="fw-bold"
                    ),
                    shiny_ui.card_body(
                        shiny_ui.output_ui("target_preview_summary_ui"),
                        output_widget("target_preview_chart", width="100%", height="240px", fill=True, fillable=True),
                        shiny_ui.output_data_frame("target_preview_grid"),
                        fillable=False,
                    ),
                    class_="shadow-sm mb-3",
                    fill=False,
                ),
                shiny_ui.card(
                    shiny_ui.card_header(
                        shiny_ui.div(
                            shiny_ui.span(
                                Icon.lucide("layers", size=16), " Cached Target Repertoire", class_="fw-bold"
                            ),
                            shiny_ui.output_ui("repertoire_count_badge"),
                            class_="d-flex justify-content-between align-items-center",
                        )
                    ),
                    shiny_ui.card_body(
                        shiny_ui.output_ui("repertoire_controls_ui"),
                        shiny_ui.output_data_frame("repertoire_grid"),
                        fillable=False,
                    ),
                    class_="shadow-sm",
                    fill=False,
                ),
            ),
            col_widths=(5, 7),
            fill=False,
        )

    @bind_effect()
    def _auto_seed_default_criterion(self) -> None:
        """Seeds the first antigenic `Trait` into `staged_criteria` when a dataset is first loaded."""
        ds = self.state.raw_dataset
        if ds is None:
            return
        with reactive.isolate():
            if not self.staged_criteria and ds.pure_traits:
                self.staged_criteria = [Criterion(variable=ds.pure_traits[0])]

    @reactive_calc
    def staged_target(self) -> tuple[Target | None, str | None]:
        """Constructs the live `Target` from staged `Criterion` items and Composer settings."""
        ds = self.state.active_dataset
        if ds is None:
            return None, "Load a dataset first to compose targets."

        specs = self.staged_criteria
        if not specs:
            return None, "Stage at least one Variable Criterion on the left to preview a Target."

        try:
            criteria = [c.with_variable(ds.var(c.column)) for c in specs if c.column in ds.variables]
            if not criteria:
                return None, "None of the staged criteria match columns in the active dataset."

            raw_name = self.val("target_name_input", "")
            raw_title = self.val("target_title_input", "")
            conj: Literal["or", "and"] = "and" if self.val("target_conjunction", "or") == "and" else "or"
            comp_raw = self.val("target_compositional", "auto")
            comp_val: bool | None = True if comp_raw == "true" else (False if comp_raw == "false" else None)
            sep_val = self.val("target_sep", " ", strip=False) or " "

            if not raw_name:
                raw_name = (
                    criteria[0].column
                    if (len(criteria) == 1 and criteria[0].pattern_str == ".*")
                    else ("_AND_" if conj == "and" else "_OR_").join(c.column for c in criteria)
                )

            if not raw_title:
                raw_title = (
                    criteria[0].title
                    if (len(criteria) == 1 and criteria[0].pattern_str == ".*")
                    else (" & " if conj == "and" else " | ").join(c.title for c in criteria)
                )

            return (
                Target(
                    name=raw_name,
                    criteria=criteria,
                    conjunction=conj,
                    compositional=comp_val,
                    sep=sep_val,
                    title=raw_title,
                ),
                None,
            )
        except Exception as exc:
            return None, str(exc)

    @bind_output(render.ui, "target_composer_vars_ui")
    def render_target_composer_vars_ui(self) -> TagChild:
        """Renders the quick preset generator and criterion variable selector dropdowns.

        Returns:
            UI fragment for selecting candidate variables from `active_dataset`.
        """
        ds = self.state.active_dataset
        if ds is None:
            return shiny_ui.p(
                "Load a dataset from the left sidebar first to compose and register Targets.",
                class_="text-muted small mb-2",
            )

        candidates = ds.variable_groups(
            include_spatial_temporal=False,
            include_metrics=False,
            annotate_mode=True,
        )
        prev_preset = self.val("preset_var_col", isolate=True)
        prev_crit = self.val("crit_var_col", isolate=True)

        return shiny_ui.div(
            shiny_ui.p("1. Quick Preset Generator", class_="fw-semibold small mb-1"),
            shiny_ui.input_select(
                "preset_var_col",
                "Candidate Variable",
                choices=candidates,
                selected=prev_preset if (prev_preset and prev_preset in ds.variables) else None,
            ),
            shiny_ui.div(
                shiny_ui.input_action_button(
                    "btn_preset_stage", "Stage in Composer", class_="btn-sm btn-outline-secondary flex-fill"
                ),
                shiny_ui.input_action_button(
                    "btn_preset_quick_add",
                    "Quick-Add Target",
                    icon=Icon.lucide("plus", size=14),
                    class_="btn-sm btn-outline-primary flex-fill",
                ),
                shiny_ui.input_action_button(
                    "btn_preset_all_traits",
                    "Add All Traits",
                    icon=Icon.lucide("layers", size=14),
                    class_="btn-sm btn-outline-success",
                ),
                class_="d-flex flex-wrap gap-1 mb-2",
            ),
            shiny_ui.hr(class_="my-2"),
            shiny_ui.p("2. Compose Criteria", class_="fw-semibold small mb-1"),
            shiny_ui.input_select(
                "crit_var_col",
                "Criterion Variable",
                choices=candidates,
                selected=prev_crit if (prev_crit and prev_crit in ds.variables) else None,
            ),
        )

    @bind_output(render.ui, "staged_criteria_list_ui")
    def render_staged_criteria_list_ui(self) -> TagChild:
        """Renders the numbered badge list of staged `Criterion` items.

        Returns:
            UI fragment displaying staged criteria or an empty-state prompt.
        """
        ds = self.state.active_dataset
        return self._staged.render_badges(
            lambda idx, c: shiny_ui.div(
                shiny_ui.span(f"#{idx}", class_="badge bg-primary me-1"),
                shiny_ui.span(
                    ds.var(c.column).role.upper() if (ds is not None and c.column in ds.variables) else "VAR",
                    class_="badge bg-secondary-subtle text-secondary-emphasis me-1",
                ),
                shiny_ui.code(c.describe(), class_="small"),
                class_="p-1 mb-1 border rounded bg-body-tertiary",
            ),
            empty_text="No criteria staged yet. Click 'Stage in Composer' or 'Add Criterion'.",
        )

    @bind_output(render.ui, "target_preview_summary_ui")
    def render_target_preview_summary_ui(self) -> TagChild:
        """Renders summary badges for the live-staged `Target` (mode, coverage percentage, category count).

        Returns:
            UI badge row or warning alert if no valid target is staged.
        """
        ds = self.state.active_dataset
        target, err = self.staged_target()
        if err or target is None or ds is None:
            return shiny_ui.div(
                Icon.lucide("triangle-alert", size=15),
                f" {err or 'No target staged.'}",
                class_="alert alert-warning py-2 px-3 small mb-2",
            )

        _, stats = target.evaluate(ds, top_n=25)
        mode_lbl = "Compositional (Multi-Category)" if stats["is_compositional"] else "Binary Endpoint (+/-)"
        mode_cls = "bg-info-subtle text-info-emphasis" if stats["is_compositional"] else "bg-warning-subtle text-dark"

        return shiny_ui.div(
            shiny_ui.span(f"ID: {target.name}", class_="badge bg-dark me-1 p-2"),
            shiny_ui.span(f"Title: {target.title}", class_="badge bg-secondary me-1 p-2"),
            shiny_ui.span(mode_lbl, class_=f"badge {mode_cls} me-1 p-2"),
            shiny_ui.span(
                f"Typeable / Positive: {stats['matched_n']:,} / {stats['total_n']:,} ({stats['coverage_pct']:.1f}%)",
                class_="badge bg-success-subtle text-success-emphasis me-1 p-2",
            ),
            shiny_ui.span(
                f"Unique Categories: {stats['n_categories']:,}",
                class_="badge bg-primary-subtle text-primary-emphasis p-2",
            ),
            class_="d-flex flex-wrap gap-1 mb-2",
        )

    @bind_altair("target_preview_chart")
    def render_target_preview_chart(self) -> alt.TopLevelMixin:
        """Renders a horizontal Altair bar chart of the top-12 categories for the staged `Target`.

        Returns:
            Configured Altair bar chart.
        """
        from seroepi.plotting.bar import BarPlotter

        ds = self.state.active_dataset
        target, err = self.staged_target()
        req(ds is not None and target is not None and err is None)
        assert ds is not None and target is not None

        df, _ = target.evaluate(ds, top_n=12)
        chart_df = df.rename({"Prevalence (%)": "Prevalence_Pct"})
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusEnd=3, color="#2563eb")
            .encode(
                x=alt.X("Prevalence_Pct:Q", title="Isolate Prevalence (%)"),
                y=alt.Y("Category:N", sort="-x", title=target.title),
                tooltip=[
                    alt.Tooltip("Category:N", title="Category"),
                    alt.Tooltip("Isolates:Q", title="Isolates", format=","),
                    alt.Tooltip("Prevalence_Pct:Q", title="Prevalence (%)", format=".2f"),
                ],
            )
            .properties(width=460, height=190)
        )
        return BarPlotter(theme=self.state.color_mode).configure_chart(chart)

    @bind_datagrid("target_preview_grid", filters=False, height="220px")
    def render_target_preview_grid(self) -> pl.DataFrame:
        """Renders the top-100 category breakdown table for the staged `Target`.

        Returns:
            Polars `DataFrame` of category counts and percentages.
        """
        ds = self.state.active_dataset
        target, err = self.staged_target()
        req(ds is not None and target is not None and err is None)
        assert ds is not None and target is not None
        df, _ = target.evaluate(ds, top_n=100)
        return df

    @bind_output(render.ui, "repertoire_count_badge")
    def render_repertoire_count_badge(self) -> TagChild:
        """Renders the count badge for `state.targets`.

        Returns:
            UI `span` badge showing the number of cached targets.
        """
        cnt = self.state.targets.count
        return shiny_ui.span(f"{cnt} Cached", class_=f"badge {'bg-success' if cnt > 0 else 'bg-secondary'}")

    @bind_output(render.ui, "repertoire_controls_ui")
    def render_repertoire_controls_ui(self) -> TagChild:
        """Renders controls for selecting, editing, removing, or clearing cached `Target` items.

        Returns:
            UI toolbar fragment for managing the cached target repertoire.
        """
        if self.state.targets.count == 0:
            return shiny_ui.p(
                "No Targets saved in repertoire yet. Use Quick Presets or the Target Composer on the left to add Targets.",
                class_="text-muted small mb-2",
            )

        names = self.state.targets.names
        active = self.state.targets.active_key or names[0]
        return shiny_ui.div(
            shiny_ui.input_select(
                "repertoire_target_select",
                label=None,
                choices={n: f"{n} ({self.state.targets.items[n].title})" for n in names},
                selected=active,
                width="260px",
            ),
            shiny_ui.input_action_button(
                "btn_rep_load_composer",
                "Load into Composer",
                icon=Icon.lucide("pencil", size=13),
                class_="btn-sm btn-outline-primary",
            ),
            shiny_ui.input_action_button(
                "btn_rep_remove",
                "Remove",
                icon=Icon.lucide("trash-2", size=13),
                class_="btn-sm btn-outline-danger",
            ),
            shiny_ui.input_action_button(
                "btn_rep_clear",
                "Clear All",
                class_="btn-sm btn-outline-secondary",
            ),
            class_="d-flex flex-wrap align-items-center gap-2 mb-2",
        )

    @bind_datagrid("repertoire_grid", filters=False, height="230px")
    def render_repertoire_grid(self) -> pl.DataFrame:
        """Renders the summary DataGrid of all registered `Target` items evaluated against `active_dataset`.

        Returns:
            Polars `DataFrame` summarizing each target in `state.targets`.
        """
        return Target.summarize_repertoire(self.state.targets.items, self.state.active_dataset)

    @bind_effect("btn_preset_stage")
    def _on_preset_stage(self) -> None:
        ds = self.state.active_dataset
        col = self.val("preset_var_col")
        if ds is not None and col and col in ds.variables:
            self._staged.set([Criterion(variable=ds.var(col))])

    @bind_effect("btn_preset_quick_add")
    def _on_preset_quick_add(self) -> None:
        ds = self.state.active_dataset
        col = self.val("preset_var_col")
        if ds is not None and col and col in ds.variables:
            tgt = Target.coerce(ds.var(col))
            self.state.targets.register(tgt.name, tgt, set_active=True)
            self.notify(f"Registered Target '{tgt.name}' ({tgt.title}) in repertoire.")

    @bind_effect("btn_preset_all_traits")
    def _on_preset_all_traits(self) -> None:
        ds = self.state.active_dataset
        if ds is None:
            return
        traits = ds.pure_traits
        for var in traits:
            tgt = Target.coerce(var)
            self.state.targets.register(tgt.name, tgt, set_active=True)
        if traits:
            self.notify(f"Registered {len(traits)} Trait Target(s) in repertoire.")
        else:
            self.notify("No Trait variables found in the active dataset.", type_="warning")

    @bind_effect("btn_add_criterion")
    def _on_add_criterion(self) -> None:
        ds = self.state.active_dataset
        col = self.val("crit_var_col")
        if ds is None or not col or col not in ds.variables:
            return
        crit = Criterion.from_inputs(
            variable=ds.var(col),
            pattern=self.val("crit_pattern", ".*"),
            equivalence_text=self.val("crit_equiv", ""),
            negative_indicator=self.val("crit_neg_ind", ""),
        )
        self._staged.add(crit)

    @bind_effect("btn_pop_criterion")
    def _on_pop_criterion(self) -> None:
        self._staged.pop()

    @bind_effect("btn_clear_criteria")
    def _on_clear_criteria(self) -> None:
        self._staged.clear()

    @bind_effect("btn_register_target")
    def _on_register_target(self) -> None:
        target, err = self.staged_target()
        if err or target is None:
            self.notify(err or "Cannot register invalid Target.", type_="error")
            return
        self.state.targets.register(target.name, target, set_active=True)
        self.notify(f"Saved Target '{target.name}' ({target.title}) to Target Repertoire.")

    @bind_effect("repertoire_target_select")
    def _on_select_repertoire_target(self) -> None:
        selected = self.val("repertoire_target_select")
        if selected and selected in self.state.targets.items and selected != self.state.targets.active_key:
            self.state.targets.select(selected)

    @bind_effect("btn_rep_load_composer")
    def _on_rep_load_composer(self) -> None:
        key = self.val("repertoire_target_select")
        tgt = self.state.targets.get(key) if key else None
        if tgt is None:
            return

        self._staged.set(tgt.criteria)
        shiny_ui.update_text("target_name_input", value=tgt.name, session=self.session)
        shiny_ui.update_text("target_title_input", value=tgt.title, session=self.session)
        shiny_ui.update_radio_buttons("target_conjunction", selected=tgt.conjunction, session=self.session)
        comp_choice = "auto" if tgt.compositional is None else ("true" if tgt.compositional else "false")
        shiny_ui.update_select("target_compositional", selected=comp_choice, session=self.session)
        shiny_ui.update_text("target_sep", value=tgt.sep, session=self.session)
        self.notify(f"Loaded Target '{tgt.name}' into Composer.")

    @bind_effect("btn_rep_remove")
    def _on_rep_remove(self) -> None:
        key = self.val("repertoire_target_select")
        if key:
            self.state.targets.remove(key)
            self.notify(f"Removed Target '{key}' from repertoire.")

    @bind_effect("btn_rep_clear")
    def _on_rep_clear(self) -> None:
        self.state.targets.clear()
        self.notify("Cleared Target repertoire.")
