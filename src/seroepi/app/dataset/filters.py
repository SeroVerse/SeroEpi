"""Toggleable QC rule switches and semantic Variable filter builder (`FilterPanel`)."""

from typing import Any, Literal

import polars as pl
from htmltools import TagChild
from shiny import reactive, render
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_effect, bind_output
from shiny_kit.ui import Icon

from seroepi.app.models import SessionState
from seroepi.dataset import Dataset
from seroepi.variables import (
    Count,
    Metric,
    Resistance,
    Temporal,
    VariableFilter,
    Virulence,
)


class FilterPanel(Component[SessionState]):
    """Interactive QC rule switches and semantic Variable filter builder.

    Provides toggle switches for pre-configured `QC` variables on the active `Dataset`
    alongside a role-aware `VariableFilter` editor supporting categorical values,
    numeric ranges, temporal year ranges, and `Resistance`/`Virulence` semantic states.
    """

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.output_ui("filters_panel_ui")

    @bind_output(render.ui, "filters_panel_ui")
    def render_filters_panel_ui(self) -> TagChild:
        """Renders the QC switch list and variable filter selector for `raw_dataset`.

        Returns:
            UI fragment containing QC switches and the variable filter builder.
        """
        ds = self.state.raw_dataset
        if ds is None:
            return shiny_ui.p(
                "Load a dataset to configure QC and semantic Variable filters.",
                class_="text-muted small",
            )

        with reactive.isolate():
            qc_map = self.state.qc_enabled_map

        qc_items: list[TagChild] = []
        for idx, qv in enumerate(ds.qc_variables):
            if qv.column not in ds.data.columns:
                continue
            pass_n = int(ds.data.select(qv.filter_expr().fill_null(False).sum()).item() or 0)
            qc_items.append(
                shiny_ui.div(
                    shiny_ui.input_switch(
                        f"qc_switch_{idx}",
                        label=shiny_ui.span(
                            shiny_ui.code(qv.describe(), class_="small"),
                            shiny_ui.span(
                                f"{pass_n:,}/{len(ds):,}",
                                class_="badge bg-secondary-subtle text-secondary-emphasis ms-1",
                            ),
                        ),
                        value=qc_map.get(qv.column, False),
                    ),
                    class_="mb-1",
                )
            )

        qc_section: TagChild = (
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.span("Quality Control (QC) Rules", class_="fw-semibold small"),
                    shiny_ui.div(
                        shiny_ui.input_action_button(
                            "btn_qc_all_on",
                            "All On",
                            class_="btn-xs btn-outline-primary py-0 px-2 me-1",
                        ),
                        shiny_ui.input_action_button(
                            "btn_qc_all_off",
                            "Reset",
                            class_="btn-xs btn-outline-secondary py-0 px-2",
                        ),
                    ),
                    class_="d-flex justify-content-between align-items-center mb-2",
                ),
                *qc_items,
                shiny_ui.hr(class_="my-2"),
            )
            if qc_items
            else shiny_ui.p("No pre-configured QC variables in this dataset.", class_="text-muted small mb-2")
        )

        return shiny_ui.div(
            qc_section,
            shiny_ui.p("Add Semantic Variable Filter", class_="fw-semibold small mb-1"),
            shiny_ui.input_select(
                "add_filter_col",
                "Variable",
                choices=ds.variable_groups(),
            ),
            shiny_ui.output_ui("filter_value_editor_ui"),
            shiny_ui.input_action_button(
                "btn_add_var_filter",
                "Add / Update Filter",
                icon=Icon.lucide("plus", size=14),
                class_="btn-outline-primary btn-sm w-100 mt-2",
            ),
            shiny_ui.output_ui("active_filters_and_commit_ui"),
        )

    @bind_output(render.ui, "filter_value_editor_ui")
    def render_filter_value_editor_ui(self) -> TagChild:
        """Renders role-specific value controls (sliders, presence modes, or selectize) for the chosen variable.

        Returns:
            Role-appropriate UI input controls for the selected column.
        """
        ds = self.state.raw_dataset
        col = self.val("add_filter_col")
        if ds is None or not col or col not in ds.variables or col not in ds.data.columns:
            return None

        var = ds.var(col)
        series = ds.data[col]

        if isinstance(var, (Resistance, Virulence)):
            unique_vals = series.drop_nulls().cast(pl.Utf8).unique().sort().to_list()[:200]
            mode_choices: dict[str, str] = {}
            if isinstance(var, Resistance):
                mode_choices.update(
                    {
                        "resistant": "Resistant (R)",
                        "intermediate": "Intermediate (I)",
                        "sensitive": "Sensitive (S)",
                        "non_susceptible": "Non-susceptible (I or R)",
                    }
                )
            elif isinstance(var, Virulence):
                mode_choices.update(
                    {
                        "intact": "Intact (Functional)",
                        "defective": "Defective (Truncated or Incomplete)",
                        "truncated": "Truncated",
                        "incomplete": "Incomplete",
                    }
                )
            mode_choices.update(
                {
                    "positive": "Positive (Present / Non-negative)",
                    "negative": "Negative (Absent / '-' / Null)",
                    "values": "Match Specific Alleles / Values",
                }
            )
            return shiny_ui.div(
                shiny_ui.input_select(
                    "filter_presence_mode",
                    "Filter Mode",
                    choices=mode_choices,
                ),
                shiny_ui.input_selectize(
                    "filter_cat_values",
                    "Specific Values (if mode = Specific Values)",
                    choices={v: v for v in unique_vals},
                    multiple=True,
                ),
            )

        if isinstance(var, Temporal):
            years = (
                series.dt.year().drop_nulls()
                if series.dtype in (pl.Date, pl.Datetime)
                else series.cast(pl.Utf8).str.extract(r"(\d{4})", 1).cast(pl.Int64, strict=False).drop_nulls()
            )
            if len(years) > 0:
                y_min, y_max = int(years.min() or 2000), int(years.max() or 2025)
                if y_min == y_max:
                    y_min, y_max = y_min - 1, y_max + 1
                return shiny_ui.input_slider(
                    "filter_year_range", "Year Range", min=y_min, max=y_max, value=[y_min, y_max], sep=""
                )

        if isinstance(var, (Count, Metric)) or series.dtype.is_numeric():
            num_s = series.cast(pl.Float64, strict=False).drop_nulls()
            if len(num_s) > 0:
                v_min, v_max = float(num_s.min() or 0.0), float(num_s.max() or 1.0)
                if v_min == v_max:
                    v_max = v_min + 1.0
                return shiny_ui.input_slider(
                    "filter_num_range", "Value Range", min=v_min, max=v_max, value=[v_min, v_max]
                )

        unique_vals = series.drop_nulls().cast(pl.Utf8).unique().sort().to_list()[:500]
        return shiny_ui.input_selectize(
            "filter_cat_values",
            f"Allowed Values ({len(unique_vals)} unique)",
            choices={v: v for v in unique_vals},
            multiple=True,
        )

    @bind_output(render.ui, "active_filters_and_commit_ui")
    def render_active_filters_and_commit_ui(self) -> TagChild:
        """Renders active `VariableFilter` badges, toggle/remove actions, and dataset commit buttons.

        Returns:
            UI fragment for managing active variable filters and saving filtered dataset copies.
        """
        ds = self.state.raw_dataset
        if ds is None:
            return None

        v_filters = self.state.variable_filters
        active_cards = [
            shiny_ui.div(
                shiny_ui.div(
                    shiny_ui.span(
                        "ON" if spec.enabled else "OFF",
                        class_=(
                            "badge bg-success-subtle text-success-emphasis me-1"
                            if spec.enabled
                            else "badge bg-secondary-subtle text-muted me-1"
                        ),
                    ),
                    shiny_ui.span(
                        spec.variable.role.upper(),
                        class_="badge bg-secondary-subtle text-secondary-emphasis border me-1",
                    ),
                    shiny_ui.code(spec.describe(), class_="small"),
                    class_="text-truncate",
                ),
                class_="p-1 mb-1 border rounded bg-body-tertiary",
            )
            for spec in v_filters.values()
        ]

        filter_mgmt_ui: TagChild = (
            shiny_ui.div(
                shiny_ui.hr(class_="my-2"),
                shiny_ui.p("Active Variable Filters", class_="fw-semibold small mb-1"),
                *active_cards,
                shiny_ui.input_select(
                    "manage_filter_col",
                    "Select Filter to Toggle / Remove",
                    choices={c: f"{c} ({'Enabled' if s.enabled else 'Paused'})" for c, s in v_filters.items()},
                ),
                shiny_ui.div(
                    shiny_ui.input_action_button(
                        "btn_toggle_vfilter", "Toggle On/Off", class_="btn-sm btn-outline-primary flex-fill"
                    ),
                    shiny_ui.input_action_button("btn_remove_vfilter", "Remove", class_="btn-sm btn-outline-danger"),
                    shiny_ui.input_action_button(
                        "btn_clear_vfilters", "Clear All", class_="btn-sm btn-outline-secondary"
                    ),
                    class_="d-flex gap-1 mt-1",
                ),
            )
            if v_filters
            else None
        )

        commit_ui: TagChild = (
            shiny_ui.div(
                shiny_ui.hr(class_="my-2"),
                shiny_ui.input_text("save_filtered_name", "Save Filtered Dataset As", value=f"{ds.name}_filtered"),
                shiny_ui.div(
                    shiny_ui.input_action_button(
                        "btn_overwrite_filtered",
                        "Update Active",
                        icon=Icon.lucide("check-check", size=14),
                        class_="btn-sm btn-primary flex-fill",
                    ),
                    shiny_ui.input_action_button(
                        "btn_save_new_filtered",
                        "Save Copy",
                        icon=Icon.lucide("plus", size=14),
                        class_="btn-sm btn-outline-primary flex-fill",
                    ),
                    class_="d-flex gap-1",
                ),
            )
            if self.state.active_filter_exprs
            else None
        )

        return shiny_ui.div(filter_mgmt_ui, commit_ui)

    def _set_all_qc(self, enabled: bool) -> None:
        ds = self.state.raw_dataset
        if ds is not None:
            self.state.qc_enabled_map = {v.column: enabled for v in ds.qc_variables}
            for idx, _ in enumerate(ds.qc_variables):
                shiny_ui.update_switch(f"qc_switch_{idx}", value=enabled, session=self.session)

    @bind_effect("btn_qc_all_on")
    def _on_qc_all_on(self) -> None:
        self._set_all_qc(True)

    @bind_effect("btn_qc_all_off")
    def _on_qc_all_off(self) -> None:
        self._set_all_qc(False)

    @bind_effect()
    def _sync_individual_qc_switches(self) -> None:
        ds = self.state.raw_dataset
        if ds is None:
            return
        with reactive.isolate():
            current = dict(self.state.qc_enabled_map)
        changed = False
        for idx, qv in enumerate(ds.qc_variables):
            sw_id = f"qc_switch_{idx}"
            if sw_id in self.input:
                val = bool(self.input[sw_id]())
                if current.get(qv.column, False) != val:
                    current[qv.column] = val
                    changed = True
        if changed:
            self.state.qc_enabled_map = current

    @bind_effect("btn_add_var_filter")
    def _on_add_var_filter(self) -> None:
        ds = self.state.raw_dataset
        col = self.val("add_filter_col")
        if ds is None or not col or col not in ds.variables:
            return

        var = ds.var(col)
        series = ds.data[col]
        updated = dict(self.state.variable_filters)

        if isinstance(var, (Resistance, Virulence)):
            mode_raw = str(self.val("filter_presence_mode", "positive"))
            valid_modes: dict[
                str,
                Literal[
                    "positive",
                    "negative",
                    "resistant",
                    "intermediate",
                    "sensitive",
                    "non_susceptible",
                    "intact",
                    "defective",
                    "truncated",
                    "incomplete",
                    "values",
                ],
            ] = {
                "positive": "positive",
                "negative": "negative",
                "resistant": "resistant",
                "intermediate": "intermediate",
                "sensitive": "sensitive",
                "non_susceptible": "non_susceptible",
                "intact": "intact",
                "defective": "defective",
                "truncated": "truncated",
                "incomplete": "incomplete",
                "values": "values",
            }
            mode = valid_modes.get(mode_raw, "positive")
            vals = tuple(self.val("filter_cat_values", ()))
            updated[col] = VariableFilter(
                column=col, variable=var, kind="presence", presence_mode=mode, selected_values=vals
            )
        elif isinstance(var, Temporal) and (yr := self.val("filter_year_range")) and len(yr) == 2:
            updated[col] = VariableFilter(
                column=col, variable=var, kind="temporal", year_range=(int(yr[0]), int(yr[1]))
            )
        elif (
            (isinstance(var, (Count, Metric)) or series.dtype.is_numeric())
            and (nr := self.val("filter_num_range"))
            and len(nr) == 2
        ):
            updated[col] = VariableFilter(
                column=col, variable=var, kind="numeric", numeric_range=(float(nr[0]), float(nr[1]))
            )
        else:
            vals = tuple(self.val("filter_cat_values", ()))
            if not vals:
                self.notify("Select at least one category value for the filter.", type_="warning")
                return
            updated[col] = VariableFilter(column=col, variable=var, kind="categorical", selected_values=vals)

        self.state.variable_filters = updated

    @bind_effect("btn_toggle_vfilter")
    def _on_toggle_vfilter(self) -> None:
        col = self.val("manage_filter_col")
        current = dict(self.state.variable_filters)
        if col and col in current:
            current[col].enabled = not current[col].enabled
            self.state.variable_filters = current

    @bind_effect("btn_remove_vfilter")
    def _on_remove_vfilter(self) -> None:
        col = self.val("manage_filter_col")
        current = dict(self.state.variable_filters)
        if col and col in current:
            current.pop(col, None)
            self.state.variable_filters = current

    @bind_effect("btn_clear_vfilters")
    def _on_clear_vfilters(self) -> None:
        self.state.variable_filters = {}

    @bind_effect("btn_overwrite_filtered")
    def _on_overwrite_filtered(self) -> None:
        filt_ds = self.state.active_dataset
        if filt_ds is not None:
            self.state.register_dataset(filt_ds)
            self.notify(f"Updated active dataset '{filt_ds.name}' to {len(filt_ds):,} filtered isolates.")

    @bind_effect("btn_save_new_filtered")
    def _on_save_new_filtered(self) -> None:
        filt_ds = self.state.active_dataset
        if filt_ds is None:
            return
        new_name = self.val("save_filtered_name", f"{filt_ds.name}_filtered")
        named_ds = Dataset(
            data=filt_ds.data,
            variables=filt_ds.variables,
            name=new_name,
            distances=filt_ds.distances,
        )
        self.state.register_dataset(named_ds, name=new_name)
        self.notify(f"Saved filtered dataset '{new_name}' ({len(named_ds):,} rows).")
