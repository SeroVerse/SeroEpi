"""Multi-dataset vertical concatenation (.concat), horizontal enrichment (.enrich), and column alignment (`DatasetMerger`)."""

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
from htmltools import TagChild
from shiny import reactive, render, req
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_datagrid, bind_effect, bind_output, reactive_calc
from shiny_kit.ui import EmptyState, Icon, StagedList

from seroepi.app.models import SessionState
from seroepi.dataset import Dataset


@dataclass(frozen=True, slots=True)
class StagedRename:
    """Single staged column rename rule (`dataset_name`: `old_col` -> `new_col`).

    Attributes:
        dataset_name (str): Name of the registered `Dataset` to apply the rename to.
        old_col (str): Existing column name in `dataset_name`.
        new_col (str): Target column name to rename or coalesce `old_col` into.
    """

    dataset_name: str
    old_col: str
    new_col: str


def _normalize_col_token(col: str) -> str:
    """Normalizes a column name into a lowercase alphanumeric/underscore token for case/fuzzy matching."""
    return re.sub(r"[\s_\-\.]+", "_", col.strip().lower()).strip("_")


class DatasetMerger(Component[SessionState]):
    """Interactive multi-dataset vertical concatenation (.concat), horizontal enrichment (.enrich), and column renaming.

    Attributes:
        empty_state (ClassVar[EmptyState]): Placeholder card rendered when fewer than 2 datasets are loaded.
    """

    empty_state: ClassVar[EmptyState] = EmptyState(
        icon="layers",
        title="Load at Least Two Datasets to Merge",
        message=(
            "Use the Load Data panel in the sidebar to load multiple datasets (e.g. multiple "
            "Pathogenwatch collections, uploaded isolate tables, or .joblib files). Once two or more "
            "datasets are registered in the session, return here to align column names (.rename), "
            "stack cohorts (.concat), or join metadata columns (.enrich)."
        ),
        dashed=True,
    )

    def __init__(self) -> None:
        super().__init__()
        self._staged_renames: StagedList[StagedRename] = StagedList()

    @property
    def staged_renames(self) -> list[StagedRename]:
        """Reactively returns the ordered list of staged column renames."""
        return self._staged_renames.items

    @staged_renames.setter
    def staged_renames(self, items: list[StagedRename]) -> None:
        self._staged_renames.set(items)

    @classmethod
    def _render_ui(cls) -> Any:
        return shiny_ui.output_ui("merger_main_ui")

    def _apply_renames_to_dataset(self, ds: Dataset, renames: list[StagedRename] | None = None) -> Dataset:
        """Applies all staged renames targeting `ds.name` onto `ds` with coalescing enabled."""
        rules = renames if renames is not None else self.staged_renames
        mapping = {
            r.old_col: r.new_col
            for r in rules
            if r.dataset_name == ds.name and r.old_col in ds.data.columns and r.old_col != r.new_col
        }
        if not mapping:
            return ds
        return ds.rename(mapping, infer_types=True, coalesce=True, strict=False)

    def _active_merge_datasets(self) -> list[Dataset]:
        """Returns the ordered list of raw `Dataset` instances currently selected for the merge operation."""
        mode = str(self.val("merge_mode", "concat"))
        if mode == "enrich":
            p_name = str(self.val("enrich_primary_ds", ""))
            s_name = str(self.val("enrich_secondary_ds", ""))
            out: list[Dataset] = []
            for n in (p_name, s_name):
                if n:
                    ds = self.state.datasets.get(n)
                    if ds is not None and ds not in out:
                        out.append(ds)
            return out

        selected_names = list(self.val("concat_datasets") or [])
        return [d for n in selected_names if (d := self.state.datasets.get(n)) is not None]

    def compute_auto_alignments(self) -> list[StagedRename]:
        """Detects case/punctuation-normalized column matches across the selected datasets."""
        ds_list = self._active_merge_datasets()
        if not ds_list:
            return []

        current_staged = self.staged_renames
        staged_keys = {(r.dataset_name, r.old_col) for r in current_staged}

        primary = self._apply_renames_to_dataset(ds_list[0], current_staged)
        primary_norm: dict[str, str] = {}
        for col in primary.data.columns:
            if col == primary.identifier.column:
                continue
            norm = _normalize_col_token(col)
            if norm and norm not in primary_norm:
                primary_norm[norm] = col

        suggestions: list[StagedRename] = []

        # 1. Check within-dataset case collisions (e.g. a dataset having both 'Country' and 'country')
        for raw_ds in ds_list:
            ds_eff = self._apply_renames_to_dataset(raw_ds, current_staged)
            seen_norm: dict[str, str] = {}
            # Prefer canonical column from primary if present, otherwise first Title/uppercase column
            ordered_cols = sorted(
                (c for c in ds_eff.data.columns if c != ds_eff.identifier.column),
                key=lambda c: 0 if c in primary_norm.values() else (1 if c[:1].isupper() else 2),
            )
            for col in ordered_cols:
                norm = _normalize_col_token(col)
                if not norm:
                    continue
                if norm not in seen_norm:
                    seen_norm[norm] = col
                else:
                    target_col = seen_norm[norm]
                    if col != target_col and (raw_ds.name, col) not in staged_keys:
                        suggestions.append(StagedRename(dataset_name=raw_ds.name, old_col=col, new_col=target_col))
                        staged_keys.add((raw_ds.name, col))

        # 2. Check cross-dataset alignments onto primary dataset's schema
        for sec_raw in ds_list[1:]:
            sec_eff = self._apply_renames_to_dataset(sec_raw, current_staged)
            for col in sec_eff.data.columns:
                if col == sec_eff.identifier.column or col in primary.data.columns:
                    continue
                norm = _normalize_col_token(col)
                if norm in primary_norm:
                    target_col = primary_norm[norm]
                    if target_col != col and (sec_raw.name, col) not in staged_keys:
                        suggestions.append(StagedRename(dataset_name=sec_raw.name, old_col=col, new_col=target_col))
                        staged_keys.add((sec_raw.name, col))

        return suggestions

    @reactive_calc
    def staged_merge(self) -> tuple[Dataset | None, dict[str, Any], str | None]:
        """Computes a live preview of the configured merge operation (including staged renames) and schema diagnostics."""
        if self.state.datasets.count < 2:
            return None, {}, "Load at least 2 datasets into the session to merge."

        mode = str(self.val("merge_mode", "concat"))
        out_name = str(self.val("merge_output_name", "")).strip()
        renames = self.staged_renames

        if mode == "enrich":
            p_name = str(self.val("enrich_primary_ds", ""))
            s_name = str(self.val("enrich_secondary_ds", ""))
            if not p_name or not s_name:
                return None, {}, "Select both a primary and secondary dataset."
            if p_name == s_name:
                return None, {}, "Primary and secondary datasets must be distinct."
            raw_primary = self.state.datasets.get(p_name)
            raw_secondary = self.state.datasets.get(s_name)
            if raw_primary is None or raw_secondary is None:
                return None, {}, "Selected datasets are no longer available."

            try:
                ds_primary = self._apply_renames_to_dataset(raw_primary, renames)
                ds_secondary = self._apply_renames_to_dataset(raw_secondary, renames)
                resolved_name = out_name or f"{ds_primary.name}_enriched"
                merged = ds_primary.enrich(ds_secondary, name=resolved_name)
            except Exception as exc:
                return None, {}, str(exc)

            p_ids = set(ds_primary.sample_ids.cast(pl.String).to_list())
            s_ids = set(ds_secondary.sample_ids.cast(pl.String).to_list())
            overlap = len(p_ids & s_ids)
            p_cols = set(ds_primary.data.columns) - {ds_primary.identifier.column}
            s_cols = set(ds_secondary.data.columns) - {ds_secondary.identifier.column}
            meta = {
                "mode": "Horizontal Join (.enrich)",
                "input_rows": len(ds_primary),
                "output_rows": len(merged),
                "overlap_ids": overlap,
                "id_alignment": f"{ds_primary.identifier.column} ← {ds_secondary.identifier.column}",
                "shared_cols": len(p_cols & s_cols),
                "added_cols": len(s_cols - p_cols),
                "renamed_cols": len(renames),
                "dist_samples": merged.distances.n_samples if merged.distances is not None else 0,
                "default_name": resolved_name,
            }
            return merged, meta, None

        selected_names = list(self.val("concat_datasets") or [])
        valid_datasets = [self.state.datasets.get(n) for n in selected_names]
        raw_list = [d for d in valid_datasets if d is not None]
        if len(raw_list) < 2:
            return None, {}, "Select at least 2 datasets to concatenate."

        dedup = bool(self.val("concat_deduplicate", True))
        add_source = bool(self.val("concat_add_source", True))
        resolved_name = out_name or "_".join(d.name for d in raw_list[:2]) + (
            f"_plus{len(raw_list) - 2}" if len(raw_list) > 2 else "_merged"
        )

        try:
            ds_list = [self._apply_renames_to_dataset(d, renames) for d in raw_list]
            primary = ds_list[0]
            others = ds_list[1:]
            merged = primary.concat(
                *others,
                name=resolved_name,
                deduplicate=dedup,
                source_col="source_dataset" if add_source else None,
            )
        except Exception as exc:
            return None, {}, str(exc)

        total_input_rows = sum(len(d) for d in ds_list)
        all_id_sets = [set(d.sample_ids.cast(pl.String).to_list()) for d in ds_list]
        union_ids = set().union(*all_id_sets)
        overlap_count = total_input_rows - len(union_ids)
        id_cols_used = " / ".join(dict.fromkeys(d.identifier.column for d in ds_list))
        col_sets = [set(d.data.columns) - {d.identifier.column} for d in ds_list]
        shared_cols = set.intersection(*col_sets) if col_sets else set()
        union_cols = set.union(*col_sets) if col_sets else set()

        meta = {
            "mode": "Vertical Stack (.concat)",
            "input_rows": total_input_rows,
            "output_rows": len(merged),
            "overlap_ids": overlap_count,
            "id_alignment": f"{id_cols_used} → {primary.identifier.column}",
            "shared_cols": len(shared_cols),
            "added_cols": len(union_cols - shared_cols),
            "renamed_cols": len(renames),
            "dist_samples": merged.distances.n_samples if merged.distances is not None else 0,
            "default_name": resolved_name,
        }
        return merged, meta, None

    @bind_output(render.ui, "merger_main_ui")
    def render_merger_main_ui(self) -> TagChild:
        """Renders the two-column merge configuration and live schema preview layout.

        Returns:
            Two-column UI layout or `empty_state.ui()` when fewer than 2 datasets are registered.
        """
        names = self.state.datasets.names
        if len(names) < 2:
            return self.empty_state.ui()

        choices = {
            n: f"{n} ({len(self.state.datasets.items[n]):,} rows, {self.state.datasets.items[n].data.width} cols)"
            for n in names
        }
        default_pair = list(names[:2])

        with reactive.isolate():
            curr_mode = str(self.val("merge_mode", "concat"))
            if curr_mode not in ("concat", "enrich"):
                curr_mode = "concat"
            curr_concat = [n for n in (self.val("concat_datasets") or []) if n in choices]
            selected_concat = curr_concat if len(curr_concat) >= 2 else default_pair
            curr_dedup = bool(self.val("concat_deduplicate", True))
            curr_source = bool(self.val("concat_add_source", True))
            curr_p = str(self.val("enrich_primary_ds", ""))
            selected_p = curr_p if curr_p in choices else default_pair[0]
            curr_s = str(self.val("enrich_secondary_ds", ""))
            selected_s = curr_s if (curr_s in choices and curr_s != selected_p) else default_pair[1]
            curr_rename_ds = str(self.val("rename_ds_select", ""))
            selected_rename_ds = curr_rename_ds if curr_rename_ds in choices else default_pair[1]
            curr_out_name = str(self.val("merge_output_name", ""))

        return shiny_ui.layout_columns(
            shiny_ui.card(
                shiny_ui.card_header(
                    shiny_ui.span(Icon.lucide("layers", size=16), " Merge & Column Alignment", class_="fw-semibold")
                ),
                shiny_ui.card_body(
                    shiny_ui.input_radio_buttons(
                        "merge_mode",
                        "Merge Operation",
                        choices={
                            "concat": "Vertical Stack (.concat — combine isolate rows)",
                            "enrich": "Horizontal Join (.enrich — join columns by ID)",
                        },
                        selected=curr_mode,
                    ),
                    shiny_ui.panel_conditional(
                        "input.merge_mode === 'concat'",
                        shiny_ui.input_selectize(
                            "concat_datasets",
                            "Datasets to Stack (in priority order)",
                            choices=choices,
                            selected=selected_concat,
                            multiple=True,
                        ),
                        shiny_ui.input_switch(
                            "concat_deduplicate",
                            "Deduplicate Overlapping Sample IDs (Coalesce non-nulls)",
                            value=curr_dedup,
                        ),
                        shiny_ui.input_switch(
                            "concat_add_source",
                            "Add 'source_dataset' Provenance Column",
                            value=curr_source,
                        ),
                    ),
                    shiny_ui.panel_conditional(
                        "input.merge_mode === 'enrich'",
                        shiny_ui.input_select(
                            "enrich_primary_ds",
                            "Primary Dataset (Left Anchor)",
                            choices=choices,
                            selected=selected_p,
                        ),
                        shiny_ui.input_select(
                            "enrich_secondary_ds",
                            "Secondary Dataset (Metadata / Columns to Join)",
                            choices=choices,
                            selected=selected_s,
                        ),
                    ),
                    shiny_ui.hr(class_="my-2"),
                    shiny_ui.div(
                        shiny_ui.span(
                            Icon.lucide("sliders-horizontal", size=14),
                            " Column Alignment & Rename (.rename)",
                            class_="fw-semibold small",
                        ),
                        shiny_ui.input_action_button(
                            "btn_auto_align_cols",
                            "Auto-Align Matches",
                            icon=Icon.lucide("sparkles", size=13),
                            class_="btn-sm btn-outline-info",
                        ),
                        class_="d-flex justify-content-between align-items-center mb-2",
                    ),
                    shiny_ui.input_select(
                        "rename_ds_select",
                        "Dataset to Rename Column In",
                        choices={n: n for n in names},
                        selected=selected_rename_ds,
                    ),
                    shiny_ui.output_ui("rename_cols_controls_ui"),
                    StagedList.action_buttons(
                        add_id="btn_add_rename",
                        clear_id="btn_clear_renames",
                        pop_id="btn_pop_rename",
                        add_label="Stage Rename",
                    ),
                    shiny_ui.output_ui("staged_renames_ui"),
                    shiny_ui.hr(class_="my-2"),
                    shiny_ui.input_text(
                        "merge_output_name",
                        "Merged Dataset Name (Optional)",
                        value=curr_out_name,
                        placeholder="Defaults to combined dataset names",
                    ),
                    shiny_ui.input_action_button(
                        "btn_execute_merge",
                        "Merge & Register Dataset",
                        icon=Icon.lucide("check", size=16),
                        class_="btn-primary w-100 mt-2",
                    ),
                ),
            ),
            shiny_ui.card(
                shiny_ui.card_header(
                    shiny_ui.span(
                        Icon.lucide("table", size=16),
                        " Schema Alignment & Live Preview",
                        class_="fw-semibold",
                    )
                ),
                shiny_ui.card_body(
                    shiny_ui.output_ui("merge_diagnostics_ui"),
                    shiny_ui.output_data_frame("merge_preview_grid"),
                ),
            ),
            col_widths=(5, 7),
        )

    @bind_output(render.ui, "rename_cols_controls_ui")
    def render_rename_cols_controls_ui(self) -> TagChild:
        """Renders source and target column selectors for the dataset chosen in `rename_ds_select`.

        Returns:
            Two-column UI row with `rename_src_col` and `rename_dst_col` inputs.
        """
        ds_name = str(self.val("rename_ds_select") or "")
        ds = self.state.datasets.get(ds_name) if ds_name else None
        if ds is None:
            return None

        src_choices = {
            col: f"{col} ({ds.variables[col].__class__.__name__})" if col in ds.variables else col
            for col in ds.data.columns
        }

        # Populate target column suggestions from primary dataset + all registered dataset columns
        active_list = self._active_merge_datasets()
        primary_cols = list(active_list[0].data.columns) if active_list else []
        other_cols = [c for d in self.state.datasets.items.values() for c in d.data.columns if c not in primary_cols]
        target_suggestions = list(dict.fromkeys([*primary_cols, *other_cols]))
        dst_choices = {c: c for c in target_suggestions}

        # Default source column to the first unaligned match if available
        auto_matches = [m for m in self.compute_auto_alignments() if m.dataset_name == ds.name]
        default_src = auto_matches[0].old_col if auto_matches else (next(iter(src_choices.keys()), ""))
        default_dst = auto_matches[0].new_col if auto_matches else default_src

        return shiny_ui.layout_columns(
            shiny_ui.input_select(
                "rename_src_col",
                "Source Column",
                choices=src_choices,
                selected=default_src,
            ),
            shiny_ui.input_selectize(
                "rename_dst_col",
                "Target Column Name",
                choices=dst_choices,
                selected=default_dst,
                options={"create": True, "placeholder": "Select or type new name..."},
            ),
            col_widths=(6, 6),
            fill=False,
        )

    @bind_output(render.ui, "staged_renames_ui")
    def render_staged_renames_ui(self) -> TagChild:
        """Renders auto-alignment suggestions, staged rename badges, and the in-place apply button.

        Returns:
            UI fragment summarizing staged column renames.
        """
        auto_suggestions = self.compute_auto_alignments()
        items = self.staged_renames

        children: list[TagChild] = []
        if auto_suggestions:
            preview_pairs = ", ".join(f"{r.old_col} → {r.new_col}" for r in auto_suggestions[:3])
            more = f" (+{len(auto_suggestions) - 3} more)" if len(auto_suggestions) > 3 else ""
            children.append(
                shiny_ui.div(
                    Icon.lucide("sparkles", size=13),
                    f" Detected {len(auto_suggestions)} alignable column(s): ",
                    shiny_ui.code(f"{preview_pairs}{more}"),
                    class_="alert alert-info py-1 px-2 small mb-2",
                )
            )

        badges = self._staged_renames.render_badges(
            lambda idx, r: shiny_ui.span(
                f"{r.dataset_name}: {r.old_col} → {r.new_col}",
                class_="badge bg-primary-subtle text-primary-emphasis border border-primary-subtle",
            ),
            empty_text="No column renames staged.",
        )
        children.append(badges)

        if items:
            children.append(
                shiny_ui.input_action_button(
                    "btn_apply_renames_in_place",
                    "Apply Renames Directly to Source Dataset(s)",
                    icon=Icon.lucide("check", size=13),
                    class_="btn-sm btn-outline-primary w-100 mt-1",
                )
            )

        return shiny_ui.div(*children)

    @bind_effect("btn_add_rename")
    def _on_add_rename(self) -> None:
        ds_name = str(self.val("rename_ds_select") or "").strip()
        src_col = str(self.val("rename_src_col") or "").strip()
        dst_col = str(self.val("rename_dst_col") or "").strip()
        if not ds_name or not src_col or not dst_col:
            self.notify("Specify dataset, source column, and target column name.", type_="warning")
            return
        if src_col == dst_col:
            self.notify(f"Column '{src_col}' already has that name.", type_="warning")
            return
        # Replace any prior staged rename for the same (dataset_name, old_col)
        updated = [r for r in self.staged_renames if not (r.dataset_name == ds_name and r.old_col == src_col)]
        updated.append(StagedRename(dataset_name=ds_name, old_col=src_col, new_col=dst_col))
        self.staged_renames = updated

    @bind_effect("btn_pop_rename")
    def _on_pop_rename(self) -> None:
        self._staged_renames.pop()

    @bind_effect("btn_clear_renames")
    def _on_clear_renames(self) -> None:
        self._staged_renames.clear()

    @bind_effect("btn_auto_align_cols")
    def _on_auto_align_cols(self) -> None:
        suggestions = self.compute_auto_alignments()
        if not suggestions:
            self.notify("All matching columns across selected datasets are already aligned.", type_="message")
            return
        updated = list(self.staged_renames)
        for s in suggestions:
            if s not in updated:
                updated.append(s)
        self.staged_renames = updated
        self.notify(f"Auto-staged {len(suggestions)} column alignment(s).", type_="message")

    @bind_effect("btn_apply_renames_in_place")
    def _on_apply_renames_in_place(self) -> None:
        rules = self.staged_renames
        if not rules:
            return
        affected_names = list(dict.fromkeys(r.dataset_name for r in rules))
        applied_count = 0
        active_before = self.state.datasets.active_key
        for ds_name in affected_names:
            ds = self.state.datasets.get(ds_name)
            if ds is None:
                continue
            renamed_ds = self._apply_renames_to_dataset(ds, rules)
            self.state.datasets.register(ds_name, renamed_ds, set_active=(ds_name == active_before))
            applied_count += 1
        if active_before:
            self.state.reset_dataset_filters()
        self._staged_renames.clear()
        self.notify(
            f"Applied {len(rules)} column rename(s) across {applied_count} dataset(s).",
            type_="message",
        )

    @bind_output(render.ui, "merge_diagnostics_ui")
    def render_merge_diagnostics_ui(self) -> TagChild:
        """Renders live schema alignment diagnostics (row counts, overlapping IDs, shared/added columns, distances).

        Returns:
            Diagnostic badge bar for the live-staged merge result.
        """
        merged, meta, err = self.staged_merge()
        if err or merged is None:
            return shiny_ui.div(
                shiny_ui.span(
                    Icon.lucide("circle-alert", size=14),
                    f" {err or 'Invalid merge configuration.'}",
                    class_="badge bg-warning-subtle text-warning-emphasis p-2 w-100 mb-2",
                )
            )

        dist_n = int(meta.get("dist_samples", 0))
        dist_badge = (
            shiny_ui.span(
                Icon.lucide("dna", size=12),
                f" Distances: {dist_n:,} / {meta['output_rows']:,} isolates",
                class_="badge bg-success-subtle text-success-emphasis",
            )
            if dist_n > 0
            else shiny_ui.span(
                "Distances: None",
                class_="badge bg-secondary-subtle text-secondary-emphasis",
            )
        )
        renamed_n = int(meta.get("renamed_cols", 0))
        rename_badge = (
            shiny_ui.span(
                f"{renamed_n} staged rename(s)",
                class_="badge bg-warning-subtle text-warning-emphasis",
            )
            if renamed_n > 0
            else None
        )

        return shiny_ui.div(
            shiny_ui.div(
                shiny_ui.span(
                    f"{meta['output_rows']:,} merged rows (from {meta['input_rows']:,})",
                    class_="badge bg-primary-subtle text-primary-emphasis",
                ),
                shiny_ui.span(
                    f"{meta['overlap_ids']:,} overlapping IDs",
                    class_="badge bg-info-subtle text-info-emphasis",
                ),
                shiny_ui.span(
                    f"ID: {meta['id_alignment']}",
                    class_="badge bg-secondary-subtle text-secondary-emphasis",
                ),
                shiny_ui.span(
                    f"{meta['shared_cols']} shared cols, +{meta['added_cols']} extra cols",
                    class_="badge bg-success-subtle text-success-emphasis",
                ),
                rename_badge,
                dist_badge,
                class_="d-flex flex-wrap gap-1 mb-2",
            ),
            shiny_ui.div(
                f"Previewing first {min(len(merged), 200):,} rows of '{meta['default_name']}' "
                f"({merged.data.width} columns, {len(merged.variables)} registered Variables).",
                class_="text-muted small mb-2",
            ),
        )

    @bind_datagrid("merge_preview_grid")
    def render_merge_preview_grid(self) -> Any:
        """Renders a 200-row DataGrid preview of the live-staged merged `Dataset`.

        Returns:
            `render.DataGrid` wrapping the first 200 rows of the merged dataset.
        """
        merged, _, err = self.staged_merge()
        req(merged is not None and err is None)
        assert merged is not None
        return render.DataGrid(
            merged.data.head(200),
            filters=False,
            summary=True,
            height="420px",
            width="100%",
        )

    @bind_effect("btn_execute_merge")
    def _on_execute_merge(self) -> None:
        merged, _, err = self.staged_merge()
        if err or merged is None:
            self.notify(err or "Cannot execute invalid merge.", type_="error")
            return
        self.state.register_dataset(merged, name=merged.name)
        self.notify(f"Registered merged dataset '{merged.name}' ({len(merged):,} isolates, {merged.data.width} cols).")
