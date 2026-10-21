"""Epidemiological UpSet visualizer for multi-trait co-occurrence and cross-stratum/formulation target sharing."""

from collections.abc import Sequence
from typing import Any, Literal

import altair as alt
import numpy as np
import polars as pl

from seroepi.estimators.base import BaseEstimator
from seroepi.estimators.formulation import FormulationComparison
from seroepi.estimators.intervals import wilson_interval
from seroepi.plotting.base import SingleStratumPlotter
from seroepi.variables import Target


class UpSetPlotter(SingleStratumPlotter):
    """Epidemiological UpSet visualizer supporting cross-stratum/formulation target sharing and multi-trait co-occurrence.

    Supports two distinct set-intersection modes:
    1. `'stratum_sharing'`: Groups targets by the exact subset of strata (in a `BaseEstimator`) or
       formulations (in a `FormulationComparison`) in which they occur, highlighting globally shared
       ("core") targets versus region- or cohort-specific targets alongside their combined prevalence.
    2. `'trait_cooccurrence'`: Decomposes compound/multi-criterion `Target` categories (e.g.,
       `K_locus + O_locus` or multi-drug resistance profiles) into constituent traits on the Y-axis
       and co-occurring trait combinations on the X-axis.

    Args:
        mode: `'auto'` (detects `'trait_cooccurrence'` for multi-criterion targets and `'stratum_sharing'` otherwise), `'stratum_sharing'`, or `'trait_cooccurrence'`.
        intersection_metric: Top bar chart metric: `'prevalence'` (combined prevalence with CI/CrI error bars) or `'count'` (number of targets or isolate events).
        max_intersections: Maximum number of intersection columns to display on the X-axis.
        x_limit: Maximum number of strata or constituent trait sets (rows) to display before pooling the tail.
        top_n: Maximum number of ranked targets or combinations to evaluate.
        y_order: Optional explicit ordering for Y-axis sets or targets.
        bar_color: Hex color for marginal bars and active intersection glyphs.
        ci_color: Hex color for confidence/credible interval error bars.
        inactive_color: Hex color for inactive matrix grid dots.
        zebra_color: Hex color for alternating horizontal matrix bands.
        glyph_size: Marker size for matrix dots.
        line_width: Stroke width for vertical intersection connecting rules.
        x_label_angle: Rotation angle for X-axis intersection labels.
        other_label: Label for pooled tail strata when `x_limit` is exceeded.
        min_event: Minimum isolate event count for a target to be considered present in a stratum.
        min_prevalence: Minimum prevalence fraction (0.0 to 1.0) for a target to be considered present in a stratum.
        show_error_bars: Whether to display CI/CrI error bars on the top and right marginal bar panels.
        show_zebra_bands: Whether to render alternating horizontal zebra bands in the intersection matrix.
        width: Width of the central matrix and top bar panel in pixels.
        height: Height of the central matrix and right bar panel in pixels.
        top_height: Height of the top intersection bar panel in pixels.
        side_width: Width of the right set-size bar panel in pixels.
        spacing: Pixel spacing between concatenated UpSet panels.
        font: Font family for chart text.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for axis and legend labels.
        format_: Numeric format string for prevalence percentages.

    Attributes:
        mode: Active intersection mode (`"auto"`, `"stratum_sharing"`, or `"trait_cooccurrence"`).
        intersection_metric: Metric displayed on the top bar panel (`"prevalence"` or `"count"`).
        max_intersections: Maximum number of intersection columns rendered on the X-axis.
        x_limit: Maximum number of strata or constituent trait rows rendered on the Y-axis.
        y_order: Explicit ordering sequence for Y-axis sets or targets, or `None`.
        bar_color: Hex fill color for marginal bars and active intersection dots.
        ci_color: Hex stroke color for uncertainty interval rules.
        inactive_color: Hex fill color for inactive matrix grid dots in light mode.
        zebra_color: Hex fill color for alternating horizontal matrix bands in light mode.
        glyph_size: Area size in square pixels for matrix dot glyphs.
        line_width: Stroke width in pixels for vertical intersection connecting rules.
        other_label: Label used for pooled tail strata when `x_limit` is exceeded.
        min_event: Minimum isolate event count threshold for stratum presence.
        min_prevalence: Minimum prevalence fraction threshold for stratum presence.
        show_zebra_bands: Whether alternating horizontal zebra bands are rendered in the matrix.
        top_height: Height in pixels of the top intersection bar panel.
        side_width: Width in pixels of the right marginal set-size bar panel.

    Examples:
        >>> from seroepi.plotting.upset import UpSetPlotter
        >>> plotter = UpSetPlotter(mode="stratum_sharing", intersection_metric="prevalence")
        >>> chart = plotter.with_theme("dark").plot(
        ...     stratified_estimator,
        ...     max_intersections=12,
        ...     title="Cross-Region K-Locus Sharing",
        ... )
        >>> UpSetPlotter.save(chart, "upset_sharing.svg")
    """

    def __init__(
        self,
        mode: Literal["auto", "stratum_sharing", "trait_cooccurrence"] = "auto",
        intersection_metric: Literal["prevalence", "count"] = "prevalence",
        max_intersections: int | None = 15,
        x_limit: int | None = 8,
        top_n: int | None = 20,
        y_order: Sequence[str] | None = None,
        bar_color: str = "#4c78a8",
        ci_color: str = "#1b406b",
        inactive_color: str = "#e2e8f0",
        zebra_color: str = "#f8fafc",
        glyph_size: int = 160,
        line_width: float = 2.0,
        x_label_angle: int = -45,
        other_label: str = "Other",
        min_event: int = 1,
        min_prevalence: float = 0.0,
        show_error_bars: bool = True,
        show_zebra_bands: bool = True,
        width: int = 450,
        height: int = 350,
        top_height: int = 120,
        side_width: int = 180,
        spacing: int = 6,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = None,
    ) -> None:
        """Initializes the epidemiological UpSet visualizer.

        Args:
            mode: `'auto'` (detects `'trait_cooccurrence'` for multi-criterion targets and `'stratum_sharing'` otherwise), `'stratum_sharing'`, or `'trait_cooccurrence'`.
            intersection_metric: Top bar chart metric: `'prevalence'` (combined prevalence with CI/CrI error bars) or `'count'` (number of targets or isolate events).
            max_intersections: Maximum number of intersection columns to display on the X-axis.
            x_limit: Maximum number of strata or constituent trait sets (rows) to display before pooling the tail.
            top_n: Maximum number of ranked targets or combinations to evaluate.
            y_order: Optional explicit ordering for Y-axis sets or targets.
            bar_color: Hex color for marginal bars and active intersection glyphs.
            ci_color: Hex color for confidence/credible interval error bars.
            inactive_color: Hex color for inactive matrix grid dots.
            zebra_color: Hex color for alternating horizontal matrix bands.
            glyph_size: Marker size for matrix dots.
            line_width: Stroke width for vertical intersection connecting rules.
            x_label_angle: Rotation angle for X-axis intersection labels.
            other_label: Label for pooled tail strata when `x_limit` is exceeded.
            min_event: Minimum isolate event count for a target to be considered present in a stratum.
            min_prevalence: Minimum prevalence fraction (0.0 to 1.0) for a target to be considered present in a stratum.
            show_error_bars: Whether to display CI/CrI error bars on the top and right marginal bar panels.
            show_zebra_bands: Whether to render alternating horizontal zebra bands in the intersection matrix.
            width: Width of the central matrix and top bar panel in pixels.
            height: Height of the central matrix and right bar panel in pixels.
            top_height: Height of the top intersection bar panel in pixels.
            side_width: Width of the right set-size bar panel in pixels.
            spacing: Pixel spacing between concatenated UpSet panels.
            font: Font family for chart text.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for axis and legend labels.
            format_: Numeric format string for prevalence percentages.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            x_label_angle=x_label_angle,
            top_n=top_n,
            format_=format_,
            show_error_bars=show_error_bars,
            spacing=spacing,
        )
        self.mode = mode
        self.intersection_metric = intersection_metric
        self.max_intersections = max_intersections
        self.x_limit = x_limit
        self.y_order = y_order
        self.bar_color = bar_color
        self.ci_color = ci_color
        self.inactive_color = inactive_color
        self.zebra_color = zebra_color
        self.glyph_size = glyph_size
        self.line_width = line_width
        self.other_label = other_label
        self.min_event = min_event
        self.min_prevalence = min_prevalence
        self.show_zebra_bands = show_zebra_bands
        self.top_height = top_height
        self.side_width = side_width

    @staticmethod
    def is_multi_trait_estimator(estimator: BaseEstimator) -> bool:
        """Determines whether `estimator` was fitted on a multi-criterion or compound multi-token `Target`.

        Inspects `estimator.aggregator_.target` for multiple `Target` criteria first, and falls back
        to checking whether target category strings in `estimator.estimates_` contain compound
        delimiters (`"+"`, `";"`, or internal whitespace).

        Args:
            estimator: Fitted `BaseEstimator` instance to inspect.

        Returns:
            `True` if the estimator's target categories represent compound multi-trait combinations.
        """
        agg = getattr(estimator, "aggregator_", None)
        if agg is not None and getattr(agg, "target", None) is not None:
            try:
                tgt = Target.coerce(agg.target)
                if len(tgt.criteria) >= 2:
                    return True
            except Exception:
                pass
        if estimator.estimates_ is None or len(estimator.estimates_) == 0:
            return False
        t_col = estimator.target_col
        sample_vals = estimator.estimates_[t_col].drop_nulls().cast(pl.String).head(50).to_list()
        return any(("+" in v or ";" in v or " " in v.strip()) for v in sample_vals)

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Checks whether `estimator` is compatible with UpSet visualization.

        Args:
            estimator: Candidate `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators currently available.

        Returns:
            `True` if `estimator` is fitted and is either single-stratum (`stratum_sharing`) or multi-trait (`trait_cooccurrence`).
        """
        if not (estimator.is_fitted_ and estimator.estimates_ is not None):
            return False
        return estimator.is_single_stratum or cls.is_multi_trait_estimator(estimator)

    @staticmethod
    def _format_intersection_label(
        sig: tuple[str, ...],
        total_sets: int,
        count: int,
        used_labels: set[str],
    ) -> str:
        """Formats a concise, unique X-axis label for a set intersection signature."""
        if len(sig) == total_sets and total_sets > 1:
            base = f"All {total_sets} ({count})"
        elif len(sig) == 1:
            base = f"{sig[0]} only ({count})"
        elif len(sig) == 2:
            base = f"{sig[0]} + {sig[1]} ({count})"
        else:
            base = f"{sig[0]} + {sig[1]} +{len(sig) - 2} ({count})"

        label = base
        suffix = 2
        while label in used_labels:
            label = f"{base} #{suffix}"
            suffix += 1
        used_labels.add(label)
        return label

    @staticmethod
    def _summarize_target_subset(
        estimator: BaseEstimator,
        subset_targets: Sequence[str],
        global_summary_df: pl.DataFrame,
        draw_lookup: tuple[dict[str, int], np.ndarray] | None,
    ) -> dict[str, float]:
        """Computes pooled event, n, prevalence, and model-faithful CI/CrI bounds for a subset of targets."""
        target_col = estimator.target_col
        est_col = estimator.estimate.column
        sub_df = global_summary_df.filter(pl.col(target_col).cast(pl.String).is_in([str(t) for t in subset_targets]))
        if len(sub_df) == 0:
            return {"event": 0.0, "n": 0.0, est_col: 0.0, "lower": 0.0, "upper": 0.0}

        n_val = UpSetPlotter.series_max(sub_df["n"])
        raw_event = float(sub_df["event"].cast(pl.Float64).sum() or 0.0)
        event_val = min(raw_event, n_val) if n_val > 0.0 else raw_event

        if draw_lookup is not None:
            idx_map, draws_mat = draw_lookup
            col_indices = [idx_map[str(t)] for t in subset_targets if str(t) in idx_map]
            if col_indices:
                sub_draws = np.clip(np.sum(draws_mat[:, col_indices], axis=1), 0.0, 1.0)
                mean_p, low_p, up_p = estimator._summarize_draws(sub_draws[:, None], estimator.alpha)
                return {
                    "event": event_val,
                    "n": n_val,
                    est_col: float(mean_p[0]),
                    "lower": float(low_p[0]),
                    "upper": float(up_p[0]),
                }

        k_np = np.asarray([event_val], dtype=float)
        n_np = np.asarray([max(1.0, n_val)], dtype=float)
        compute_ci = getattr(estimator, "_compute_ci", None)
        if callable(compute_ci):
            p_arr, low_arr, up_arr = compute_ci(k_np, n_np)
        else:
            p_arr, low_arr, up_arr = wilson_interval(k_np, n_np, estimator.alpha)
        raw_est_sum = float(sub_df[est_col].cast(pl.Float64).sum() or 0.0)
        est_sum = min(1.0, max(0.0, raw_est_sum))
        return {
            "event": event_val,
            "n": n_val,
            est_col: est_sum if n_val > 0 else float(p_arr[0]),
            "lower": float(low_arr[0]) if n_val > 0 else 0.0,
            "upper": float(up_arr[0]) if n_val > 0 else 0.0,
        }

    def plot(
        self,
        subject: BaseEstimator | FormulationComparison,
        *,
        mode: Literal["auto", "stratum_sharing", "trait_cooccurrence"] | None = None,
        intersection_metric: Literal["prevalence", "count"] | None = None,
        max_intersections: int | None = None,
        x_limit: int | None = None,
        top_n: int | None = None,
        y_order: Sequence[str] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        show_error_bars: bool | None = None,
    ) -> alt.TopLevelMixin:
        """Renders the coordinated UpSet visualization for a `BaseEstimator` or `FormulationComparison`.

        Args:
            subject: Fitted `BaseEstimator` (for stratum sharing or trait co-occurrence) or `FormulationComparison` (for cross-formulation overlap).
            mode: Optional mode override (`"auto"`, `"stratum_sharing"`, or `"trait_cooccurrence"`).
            intersection_metric: Optional top bar panel metric override (`"prevalence"` or `"count"`).
            max_intersections: Optional maximum number of X-axis intersection columns overriding `self.max_intersections`.
            x_limit: Optional maximum number of Y-axis strata or constituent trait rows overriding `self.x_limit`.
            top_n: Optional maximum number of ranked targets or combinations overriding `self.top_n`.
            y_order: Optional explicit target or set ordering sequence overriding `self.y_order`.
            title: Custom overall chart title.
            subtitle: Custom overall chart subtitle.
            show_error_bars: Optional boolean overriding `self.show_error_bars` for marginal CI/CrI rules.

        Returns:
            Configured Altair `VConcatChart` combining the top intersection bar panel, central matrix, and right set-size bar panel.

        Raises:
            TypeError: If `subject` is neither a `BaseEstimator` nor a `FormulationComparison`.
            RuntimeError: If `subject` has not been fitted yet.
            ValueError: If `subject` is an unstratified single-trait estimator that cannot form set intersections.
        """
        eff_mode = self.mode if mode is None else mode
        eff_metric = self.intersection_metric if intersection_metric is None else intersection_metric
        eff_max_int = self.max_intersections if max_intersections is None else max_intersections
        eff_x_limit = self.x_limit if x_limit is None else x_limit
        eff_top_n = self.top_n if top_n is None else top_n
        error_bars_flag = self.show_error_bars if show_error_bars is None else show_error_bars

        if isinstance(subject, FormulationComparison):
            subject.check_is_fitted()
            return self._plot_comparison_sharing(
                subject,
                eff_metric=eff_metric,
                eff_max_int=eff_max_int,
                title=title,
                subtitle=subtitle,
            )

        if not isinstance(subject, BaseEstimator):
            raise TypeError(
                f"Expected a fitted BaseEstimator or FormulationComparison instance, got {type(subject).__name__}."
            )
        subject.check_is_fitted()
        if subject.estimates_ is None:
            raise ValueError("Estimator estimates_ DataFrame is None.")

        resolved_mode: Literal["stratum_sharing", "trait_cooccurrence"]
        if eff_mode == "auto":
            if self.is_multi_trait_estimator(subject):
                resolved_mode = "trait_cooccurrence"
            elif subject.n_strata >= 1:
                resolved_mode = "stratum_sharing"
            else:
                raise ValueError(
                    "UpSetPlotter requires either a stratified estimator (for 'stratum_sharing') "
                    "or a multi-criterion/multi-token Target (for 'trait_cooccurrence')."
                )
        elif eff_mode == "stratum_sharing" and subject.n_strata == 0:
            if self.is_multi_trait_estimator(subject):
                resolved_mode = "trait_cooccurrence"
            else:
                raise ValueError(
                    "UpSetPlotter in 'stratum_sharing' mode requires an estimator with at least 1 stratum."
                )
        else:
            resolved_mode = eff_mode

        if resolved_mode == "trait_cooccurrence":
            return self._plot_trait_cooccurrence(
                subject,
                eff_metric=eff_metric,
                eff_max_int=eff_max_int,
                eff_x_limit=eff_x_limit,
                eff_top_n=eff_top_n,
                error_bars_flag=error_bars_flag,
                title=title,
                subtitle=subtitle,
            )

        return self._plot_estimator_stratum_sharing(
            subject,
            eff_metric=eff_metric,
            eff_max_int=eff_max_int,
            eff_x_limit=eff_x_limit,
            eff_top_n=eff_top_n,
            y_order=y_order,
            error_bars_flag=error_bars_flag,
            title=title,
            subtitle=subtitle,
        )

    def _plot_estimator_stratum_sharing(
        self,
        estimator: BaseEstimator,
        *,
        eff_metric: Literal["prevalence", "count"],
        eff_max_int: int | None,
        eff_x_limit: int | None,
        eff_top_n: int | None,
        y_order: Sequence[str] | None,
        error_bars_flag: bool,
        title: str | None,
        subtitle: str | None,
    ) -> alt.TopLevelMixin:
        ctx = self.extract_context(estimator)
        stratum_col, stratum_title = ctx.stratum_col, ctx.stratum_title
        target_col, target_title = ctx.target_col, ctx.target_title
        est_col = ctx.est_col
        fmt = ctx.format
        ci_prefix = ctx.ci_prefix

        effective_targets = self.resolve_targets(estimator, y_order or self.y_order, eff_top_n)
        global_summary_df = estimator.summarize_by_target(targets=effective_targets)
        draw_res = estimator.target_posterior_draws()
        draw_lookup: tuple[dict[str, int], np.ndarray] | None = None
        if draw_res is not None:
            draw_targets, draws_mat = draw_res
            draw_lookup = ({str(t): i for i, t in enumerate(draw_targets)}, draws_mat)

        set_df, set_order = estimator.summarize_by_stratum(
            targets=effective_targets,
            x_limit=eff_x_limit,
            other_label=self.other_label,
        )
        matrix_df, _ = estimator.lump_strata(
            targets=effective_targets,
            x_limit=eff_x_limit,
            other_label=self.other_label,
        )

        matrix_df = matrix_df.with_columns(
            active=(
                (pl.col("event").fill_null(0) >= self.min_event)
                & (pl.col(est_col).fill_null(0.0) >= self.min_prevalence)
            )
            if "event" in matrix_df.columns
            else (pl.col(est_col).fill_null(0.0) >= self.min_prevalence)
        )

        active_pairs = matrix_df.filter(pl.col("active")).select(
            pl.col(target_col).cast(pl.String).alias("_t"),
            pl.col(stratum_col).cast(pl.String).alias("_s"),
        )
        target_to_sets: dict[str, set[str]] = {str(t): set() for t in effective_targets}
        for row in active_pairs.iter_rows(named=True):
            if row["_t"] in target_to_sets:
                target_to_sets[row["_t"]].add(row["_s"])

        sig_groups: dict[tuple[str, ...], list[str]] = {}
        for t_name in effective_targets:
            present = target_to_sets.get(str(t_name), set())
            sig = tuple(s for s in set_order if s in present)
            if sig:
                sig_groups.setdefault(sig, []).append(str(t_name))

        used_labels: set[str] = set()
        int_records: list[dict[str, Any]] = []
        for sig, t_list in sig_groups.items():
            stats_dict = self._summarize_target_subset(estimator, t_list, global_summary_df, draw_lookup)
            preview = ", ".join(t_list[:5]) + (f" (+{len(t_list) - 5})" if len(t_list) > 5 else "")
            int_records.append(
                {
                    "_sig": sig,
                    "strata_pattern": " + ".join(sig),
                    "degree": len(sig),
                    "target_count": len(t_list),
                    "targets_list": preview,
                    "event": stats_dict["event"],
                    "n": stats_dict["n"],
                    est_col: stats_dict[est_col],
                    "lower": stats_dict["lower"],
                    "upper": stats_dict["upper"],
                }
            )

        sort_key = est_col if eff_metric == "prevalence" else "target_count"
        int_records.sort(key=lambda r: (float(r[sort_key]), int(r["degree"]), int(r["target_count"])), reverse=True)
        if eff_max_int is not None and eff_max_int > 0:
            int_records = int_records[:eff_max_int]

        for rec in int_records:
            rec["intersection_id"] = self._format_intersection_label(
                rec["_sig"],
                len(set_order),
                int(rec["target_count"]),
                used_labels,
            )

        if not int_records:
            int_records = [
                {
                    "_sig": (),
                    "intersection_id": "None (0)",
                    "strata_pattern": "None",
                    "degree": 0,
                    "target_count": 0,
                    "targets_list": "None",
                    "event": 0.0,
                    "n": 0.0,
                    est_col: 0.0,
                    "lower": 0.0,
                    "upper": 0.0,
                }
            ]

        intersection_order = [str(r["intersection_id"]) for r in int_records]
        top_df = pl.DataFrame([{k: v for k, v in r.items() if k != "_sig"} for r in int_records])

        grid_rows: list[dict[str, Any]] = []
        line_rows: list[dict[str, Any]] = []
        set_idx_map = {s: idx for idx, s in enumerate(set_order)}

        for rec in int_records:
            int_id = str(rec["intersection_id"])
            sig_set = set(rec["_sig"])
            for s_name in set_order:
                grid_rows.append(
                    {
                        "intersection_id": int_id,
                        stratum_col: s_name,
                        "active": s_name in sig_set,
                        "strata_pattern": rec["strata_pattern"],
                        "degree": rec["degree"],
                        "target_count": rec["target_count"],
                        "targets_list": rec["targets_list"],
                        est_col: rec[est_col],
                    }
                )
            if len(rec["_sig"]) >= 2:
                ordered_sig = sorted(rec["_sig"], key=lambda s: set_idx_map.get(s, 9999))
                line_rows.append(
                    {
                        "intersection_id": int_id,
                        "y_start": ordered_sig[0],
                        "y_end": ordered_sig[-1],
                    }
                )

        grid_df = pl.DataFrame(grid_rows)
        lines_df = (
            pl.DataFrame(line_rows) if line_rows else pl.DataFrame({"intersection_id": [], "y_start": [], "y_end": []})
        )

        top_y_col = est_col if eff_metric == "prevalence" else "target_count"
        top_y_title = (
            f"Shared {target_title} Prevalence" if eff_metric == "prevalence" else f"Shared {target_title} Count"
        )
        top_y_fmt = fmt if eff_metric == "prevalence" else ",.0f"

        top_tooltips: list[alt.Tooltip] = [
            alt.Tooltip("strata_pattern:N", title=f"Shared {stratum_title} Set"),
            alt.Tooltip("degree:Q", title=f"Number of {stratum_title}s"),
            alt.Tooltip("target_count:Q", title=f"{target_title} Count"),
            alt.Tooltip("targets_list:N", title=f"Shared {target_title}s"),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Combined Global Prevalence"),
            alt.Tooltip("lower:Q", format=fmt, title=f"Lower {ci_prefix}"),
            alt.Tooltip("upper:Q", format=fmt, title=f"Upper {ci_prefix}"),
            alt.Tooltip("event:Q", format=",.0f", title="Global Events"),
        ]

        right_tooltips = self.build_tooltips(
            set_df,
            dimensions=[(stratum_col, stratum_title)],
            est_col=est_col,
            est_title=f"Top-{len(effective_targets)} {stratum_title} Coverage",
            fmt=fmt,
            ci_prefix=ci_prefix,
            event_title="Stratum Events",
        )

        resolved_title = title or f"UpSet Target Sharing: {target_title} Across {stratum_title}"
        resolved_sub = subtitle or (
            f"Top {len(effective_targets)} {target_title}s grouped by presence across {len(set_order)} {stratum_title} sets "
            f"(min_event={self.min_event}, min_prev={self.min_prevalence:{fmt}})"
        )

        return self._assemble_upset_panels(
            top_df=top_df,
            grid_df=grid_df,
            lines_df=lines_df,
            right_df=set_df,
            x_col="intersection_id",
            x_title=f"{stratum_title} Intersection ({target_title} count)",
            x_order=intersection_order,
            y_col=stratum_col,
            y_title=stratum_title,
            y_order=set_order,
            top_y_col=top_y_col,
            top_y_title=top_y_title,
            top_y_fmt=top_y_fmt,
            top_has_ci=(eff_metric == "prevalence" and error_bars_flag),
            top_tooltips=top_tooltips,
            right_x_col=est_col,
            right_x_title=f"{stratum_title} Coverage",
            right_x_fmt=fmt,
            right_has_ci=error_bars_flag,
            right_tooltips=right_tooltips,
            title=resolved_title,
            subtitle=resolved_sub,
        )

    def _plot_comparison_sharing(
        self,
        comparison: FormulationComparison,
        *,
        eff_metric: Literal["prevalence", "count"],
        eff_max_int: int | None,
        title: str | None,
        subtitle: str | None,
    ) -> alt.TopLevelMixin:
        first_form = next(iter(comparison.formulations_.values()))
        est = first_form.estimator_
        target_col = est.target_col
        target_title = est.target_title
        est_col = est.estimate.column
        fmt = self.format_ or est.estimate.format

        set_order = list(comparison.formulations_.keys())
        rankings = comparison.rankings_

        selected_rows = rankings.filter(pl.col("in_formulation")).select(
            pl.col(target_col).cast(pl.String).alias("_t"),
            pl.col("formulation").cast(pl.String).alias("_f"),
        )
        all_selected_targets = selected_rows["_t"].unique(maintain_order=True).to_list()
        target_to_forms: dict[str, set[str]] = {str(t): set() for t in all_selected_targets}
        for row in selected_rows.iter_rows(named=True):
            target_to_forms.setdefault(row["_t"], set()).add(row["_f"])

        mean_prev_df = rankings.group_by(pl.col(target_col).cast(pl.String).alias("_t")).agg(
            pl.col(est_col).mean().alias("_mean_est")
        )
        prev_map = dict(zip(mean_prev_df["_t"].to_list(), mean_prev_df["_mean_est"].to_list(), strict=False))

        sig_groups: dict[tuple[str, ...], list[str]] = {}
        for t_name in all_selected_targets:
            present = target_to_forms.get(str(t_name), set())
            sig = tuple(f for f in set_order if f in present)
            if sig:
                sig_groups.setdefault(sig, []).append(str(t_name))

        used_labels: set[str] = set()
        int_records: list[dict[str, Any]] = []
        for sig, t_list in sig_groups.items():
            comb_prev = float(np.clip(sum(prev_map.get(t, 0.0) for t in t_list), 0.0, 1.0))
            preview = ", ".join(t_list[:5]) + (f" (+{len(t_list) - 5})" if len(t_list) > 5 else "")
            int_records.append(
                {
                    "_sig": sig,
                    "strata_pattern": " + ".join(sig),
                    "degree": len(sig),
                    "target_count": len(t_list),
                    "targets_list": preview,
                    est_col: comb_prev,
                }
            )

        sort_key = est_col if eff_metric == "prevalence" else "target_count"
        int_records.sort(key=lambda r: (float(r[sort_key]), int(r["degree"]), int(r["target_count"])), reverse=True)
        if eff_max_int is not None and eff_max_int > 0:
            int_records = int_records[:eff_max_int]

        for rec in int_records:
            rec["intersection_id"] = self._format_intersection_label(
                rec["_sig"],
                len(set_order),
                int(rec["target_count"]),
                used_labels,
            )

        intersection_order = [str(r["intersection_id"]) for r in int_records]
        top_df = pl.DataFrame([{k: v for k, v in r.items() if k != "_sig"} for r in int_records])

        grid_rows: list[dict[str, Any]] = []
        line_rows: list[dict[str, Any]] = []
        set_idx_map = {s: idx for idx, s in enumerate(set_order)}

        for rec in int_records:
            int_id = str(rec["intersection_id"])
            sig_set = set(rec["_sig"])
            for f_name in set_order:
                grid_rows.append(
                    {
                        "intersection_id": int_id,
                        "formulation": f_name,
                        "active": f_name in sig_set,
                        "strata_pattern": rec["strata_pattern"],
                        "degree": rec["degree"],
                        "target_count": rec["target_count"],
                        "targets_list": rec["targets_list"],
                        est_col: rec[est_col],
                    }
                )
            if len(rec["_sig"]) >= 2:
                ordered_sig = sorted(rec["_sig"], key=lambda s: set_idx_map.get(s, 9999))
                line_rows.append(
                    {
                        "intersection_id": int_id,
                        "y_start": ordered_sig[0],
                        "y_end": ordered_sig[-1],
                    }
                )

        grid_df = pl.DataFrame(grid_rows)
        lines_df = (
            pl.DataFrame(line_rows) if line_rows else pl.DataFrame({"intersection_id": [], "y_start": [], "y_end": []})
        )

        right_rows: list[dict[str, Any]] = []
        for f_name, form_obj in comparison.formulations_.items():
            raw_cov = float(form_obj.rankings_.filter(pl.col("in_formulation"))[est_col].cast(pl.Float64).sum() or 0.0)
            f_cov = min(1.0, max(0.0, raw_cov))
            right_rows.append(
                {
                    "formulation": f_name,
                    est_col: f_cov,
                    "valency": form_obj.valency,
                    "selected": ", ".join(form_obj.selected_targets_[:6]),
                }
            )
        right_df = pl.DataFrame(right_rows)

        top_y_col = est_col if eff_metric == "prevalence" else "target_count"
        top_y_title = (
            f"Shared {target_title} Prevalence" if eff_metric == "prevalence" else f"Shared {target_title} Count"
        )
        top_y_fmt = fmt if eff_metric == "prevalence" else ",.0f"

        top_tooltips = [
            alt.Tooltip("strata_pattern:N", title="Shared Formulations"),
            alt.Tooltip("degree:Q", title="Formulations in Intersection"),
            alt.Tooltip("target_count:Q", title=f"{target_title} Count"),
            alt.Tooltip("targets_list:N", title=f"Shared {target_title}s"),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Mean Combined Prevalence"),
        ]
        right_tooltips = [
            alt.Tooltip("formulation:N", title="Cohort / Formulation"),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Formulation Coverage"),
            alt.Tooltip("valency:Q", title="Valency (K)"),
            alt.Tooltip("selected:N", title="Selected Targets"),
        ]

        resolved_title = title or f"UpSet Formulation Overlap: Top-{comparison.valency} {target_title}s"
        resolved_sub = subtitle or (
            f"Core vs. cohort-specific {target_title}s across {len(set_order)} compared formulations"
        )

        return self._assemble_upset_panels(
            top_df=top_df,
            grid_df=grid_df,
            lines_df=lines_df,
            right_df=right_df,
            x_col="intersection_id",
            x_title=f"Formulation Intersection ({target_title} count)",
            x_order=intersection_order,
            y_col="formulation",
            y_title="Cohort / Formulation",
            y_order=set_order,
            top_y_col=top_y_col,
            top_y_title=top_y_title,
            top_y_fmt=top_y_fmt,
            top_has_ci=False,
            top_tooltips=top_tooltips,
            right_x_col=est_col,
            right_x_title="Formulation Coverage",
            right_x_fmt=fmt,
            right_has_ci=False,
            right_tooltips=right_tooltips,
            title=resolved_title,
            subtitle=resolved_sub,
        )

    def _plot_trait_cooccurrence(
        self,
        estimator: BaseEstimator,
        *,
        eff_metric: Literal["prevalence", "count"],
        eff_max_int: int | None,
        eff_x_limit: int | None,
        eff_top_n: int | None,
        error_bars_flag: bool,
        title: str | None,
        subtitle: str | None,
    ) -> alt.TopLevelMixin:
        target_col = estimator.target_col
        target_title = estimator.target_title
        est_col = estimator.estimate.column
        fmt = self.format_ or estimator.estimate.format
        ci_prefix = estimator.estimate.ci_prefix

        sep = " "
        agg = getattr(estimator, "aggregator_", None)
        if agg is not None and getattr(agg, "target", None) is not None:
            try:
                coerced_tgt = Target.coerce(agg.target)
                if coerced_tgt.sep:
                    sep = coerced_tgt.sep
            except Exception:
                pass

        global_summary_df = estimator.summarize_by_target()
        draw_res = estimator.target_posterior_draws()
        draw_lookup: tuple[dict[str, int], np.ndarray] | None = None
        if draw_res is not None:
            draw_targets, draws_mat = draw_res
            draw_lookup = ({str(t): i for i, t in enumerate(draw_targets)}, draws_mat)

        cap_combos = eff_max_int if eff_max_int is not None else (eff_top_n or 15)
        combo_df = global_summary_df.head(cap_combos)
        combo_names = combo_df[target_col].cast(pl.String).to_list()

        def _split_tokens(raw_val: str) -> list[str]:
            if sep and sep in raw_val:
                parts = [p.strip() for p in raw_val.split(sep) if p.strip()]
                if len(parts) >= 2:
                    return parts
            for alt_sep in (";", "+", " "):
                if alt_sep in raw_val:
                    parts = [p.strip() for p in raw_val.split(alt_sep) if p.strip()]
                    if len(parts) >= 2:
                        return parts
            return [raw_val.strip()]

        combo_tokens: dict[str, list[str]] = {c: _split_tokens(c) for c in combo_names}
        all_cats = global_summary_df[target_col].cast(pl.String).to_list()
        token_to_combos: dict[str, list[str]] = {}
        for cat in all_cats:
            for tok in _split_tokens(cat):
                token_to_combos.setdefault(tok, []).append(cat)

        candidate_tokens = list(dict.fromkeys(tok for c in combo_names for tok in combo_tokens[c]))
        token_records: list[dict[str, Any]] = []
        for tok in candidate_tokens:
            matching_cats = token_to_combos.get(tok, [tok])
            stats_dict = self._summarize_target_subset(estimator, matching_cats, global_summary_df, draw_lookup)
            token_records.append(
                {
                    "trait_token": tok,
                    "event": stats_dict["event"],
                    "n": stats_dict["n"],
                    est_col: stats_dict[est_col],
                    "lower": stats_dict["lower"],
                    "upper": stats_dict["upper"],
                }
            )

        token_records.sort(key=lambda r: float(r[est_col]), reverse=True)
        if eff_x_limit is not None and eff_x_limit > 0:
            token_records = token_records[:eff_x_limit]

        trait_order = [str(r["trait_token"]) for r in token_records]
        trait_set = set(trait_order)
        right_df = pl.DataFrame(token_records)

        grid_rows: list[dict[str, Any]] = []
        line_rows: list[dict[str, Any]] = []
        trait_idx_map = {t: idx for idx, t in enumerate(trait_order)}

        for row in combo_df.iter_rows(named=True):
            c_name = str(row[target_col])
            active_toks = [t for t in combo_tokens.get(c_name, []) if t in trait_set]
            active_tok_set = set(active_toks)
            for t_name in trait_order:
                grid_rows.append(
                    {
                        target_col: c_name,
                        "trait_token": t_name,
                        "active": t_name in active_tok_set,
                        est_col: row[est_col],
                        "event": row.get("event", 0.0),
                        "n": row.get("n", 0.0),
                    }
                )
            if len(active_toks) >= 2:
                ordered_toks = sorted(active_toks, key=lambda t: trait_idx_map.get(t, 9999))
                line_rows.append(
                    {
                        target_col: c_name,
                        "y_start": ordered_toks[0],
                        "y_end": ordered_toks[-1],
                    }
                )

        grid_df = pl.DataFrame(grid_rows)
        lines_df = pl.DataFrame(line_rows) if line_rows else pl.DataFrame({target_col: [], "y_start": [], "y_end": []})

        top_y_col = est_col if eff_metric == "prevalence" else "event"
        top_y_title = f"{target_title} Prevalence" if eff_metric == "prevalence" else "Isolate Events"
        top_y_fmt = fmt if eff_metric == "prevalence" else ",.0f"

        top_tooltips = self.build_tooltips(
            combo_df,
            dimensions=[(target_col, f"{target_title} Combination")],
            est_col=est_col,
            est_title="Combination Prevalence",
            fmt=fmt,
            ci_prefix=ci_prefix,
            event_title="Combination Events",
        )
        right_tooltips = self.build_tooltips(
            right_df,
            dimensions=[("trait_token", "Constituent Trait")],
            est_col=est_col,
            est_title="Marginal Trait Prevalence",
            fmt=fmt,
            ci_prefix=ci_prefix,
            event_title="Total Trait Events",
        )

        resolved_title = title or f"UpSet Co-occurrence: {target_title} Combinations"
        resolved_sub = subtitle or (
            f"Top {len(combo_names)} co-occurring {target_title} combinations decomposed across {len(trait_order)} constituent traits"
        )

        return self._assemble_upset_panels(
            top_df=combo_df,
            grid_df=grid_df,
            lines_df=lines_df,
            right_df=right_df,
            x_col=target_col,
            x_title=f"{target_title} Combination",
            x_order=combo_names,
            y_col="trait_token",
            y_title="Constituent Trait",
            y_order=trait_order,
            top_y_col=top_y_col,
            top_y_title=top_y_title,
            top_y_fmt=top_y_fmt,
            top_has_ci=(eff_metric == "prevalence" and error_bars_flag and "lower" in combo_df.columns),
            top_tooltips=top_tooltips,
            right_x_col=est_col,
            right_x_title="Marginal Trait Prevalence",
            right_x_fmt=fmt,
            right_has_ci=(error_bars_flag and "lower" in right_df.columns),
            right_tooltips=right_tooltips,
            title=resolved_title,
            subtitle=resolved_sub,
        )

    def _assemble_upset_panels(
        self,
        *,
        top_df: pl.DataFrame,
        grid_df: pl.DataFrame,
        lines_df: pl.DataFrame,
        right_df: pl.DataFrame,
        x_col: str,
        x_title: str,
        x_order: list[str],
        y_col: str,
        y_title: str,
        y_order: list[str],
        top_y_col: str,
        top_y_title: str,
        top_y_fmt: str,
        top_has_ci: bool,
        top_tooltips: list[alt.Tooltip],
        right_x_col: str,
        right_x_title: str,
        right_x_fmt: str,
        right_has_ci: bool,
        right_tooltips: list[alt.Tooltip],
        title: str,
        subtitle: str,
    ) -> alt.TopLevelMixin:
        eff_ci_color = self.resolve_ci_color(self.ci_color)

        # 1. Top Intersection Bar Panel
        top_layers: list[alt.Chart] = [
            alt.Chart(top_df)
            .encode(
                x=alt.X(f"{x_col}:N", sort=x_order, axis=None),
                y=alt.Y(f"{top_y_col}:Q", title=top_y_title, axis=alt.Axis(format=top_y_fmt)),
                tooltip=top_tooltips,
            )
            .mark_bar(color=self.bar_color)
        ]
        if top_has_ci and "lower" in top_df.columns and "upper" in top_df.columns:
            top_layers.extend(
                self.build_error_bar_layers(
                    top_df,
                    cat_col=x_col,
                    cat_sort=x_order,
                    orientation="vertical",
                    color=eff_ci_color,
                    est_col=top_y_col,
                    stroke_width=1.5,
                    cap_style="tick",
                    cap_size=10,
                )
            )
        top_panel = alt.layer(*top_layers).properties(width=self.width, height=self.top_height)

        # 2. Central Intersection Matrix Panel
        zebra_records = [{"y_band": y_name} for idx, y_name in enumerate(y_order) if idx % 2 == 1]
        zebra_df = pl.DataFrame(zebra_records) if zebra_records else pl.DataFrame({"y_band": []})
        matrix_layers: list[alt.Chart] = []

        if self.show_zebra_bands and len(zebra_df) > 0:
            eff_zebra = (
                "rgba(255, 255, 255, 0.04)"
                if (self.theme == "dark" and self.zebra_color == "#f8fafc")
                else self.zebra_color
            )
            matrix_layers.append(
                alt.Chart(zebra_df)
                .encode(y=alt.Y("y_band:N", sort=y_order, axis=None))
                .mark_rect(opacity=0.5, color=eff_zebra),
            )

        eff_inactive = (
            "rgba(255, 255, 255, 0.14)"
            if (self.theme == "dark" and self.inactive_color == "#e2e8f0")
            else self.inactive_color
        )
        matrix_layers.append(
            alt.Chart(grid_df)
            .encode(
                x=alt.X(
                    f"{x_col}:N",
                    sort=x_order,
                    axis=alt.Axis(title=x_title, labelAngle=self.x_label_angle),
                ),
                y=alt.Y(
                    f"{y_col}:N",
                    sort=y_order,
                    axis=alt.Axis(title=y_title),
                ),
            )
            .mark_circle(size=self.glyph_size, color=eff_inactive),
        )

        if len(lines_df) > 0:
            matrix_layers.append(
                alt.Chart(lines_df)
                .encode(
                    x=alt.X(f"{x_col}:N", sort=x_order),
                    y=alt.Y("y_start:N", sort=y_order),
                    y2=alt.Y2("y_end:N"),
                )
                .mark_rule(color=self.bar_color, strokeWidth=self.line_width),
            )

        grid_tooltips: list[alt.Tooltip] = [
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{y_col}:N", title=y_title),
        ]
        if "targets_list" in grid_df.columns:
            grid_tooltips.append(alt.Tooltip("targets_list:N", title="Shared Targets"))
        if "target_count" in grid_df.columns:
            grid_tooltips.append(alt.Tooltip("target_count:Q", title="Target Count"))

        matrix_layers.append(
            alt.Chart(grid_df.filter(pl.col("active")))
            .encode(
                x=alt.X(f"{x_col}:N", sort=x_order),
                y=alt.Y(f"{y_col}:N", sort=y_order),
                tooltip=grid_tooltips,
            )
            .mark_circle(size=self.glyph_size, color=self.bar_color),
        )
        matrix_panel = alt.layer(*matrix_layers).properties(width=self.width, height=self.height)

        # 3. Right Marginal Set-Size Bar Panel
        right_layers: list[alt.Chart] = [
            alt.Chart(right_df)
            .encode(
                y=alt.Y(f"{y_col}:N", sort=y_order, axis=None),
                x=alt.X(f"{right_x_col}:Q", title=right_x_title, axis=alt.Axis(format=right_x_fmt)),
                tooltip=right_tooltips,
            )
            .mark_bar(color=self.bar_color)
        ]
        if right_has_ci and "lower" in right_df.columns and "upper" in right_df.columns:
            right_layers.extend(
                self.build_error_bar_layers(
                    right_df,
                    cat_col=y_col,
                    cat_sort=y_order,
                    orientation="horizontal",
                    color=eff_ci_color,
                    est_col=right_x_col,
                    stroke_width=1.5,
                    cap_style="tick",
                    cap_size=10,
                )
            )
        right_panel = alt.layer(*right_layers).properties(width=self.side_width, height=self.height)

        bottom_row = alt.hconcat(matrix_panel, right_panel, spacing=self.spacing).resolve_scale(y="shared")
        full_chart = (
            alt.vconcat(top_panel, bottom_row, spacing=self.spacing)
            .resolve_scale(x="shared")
            .properties(
                title=alt.TitleParams(
                    text=title,
                    subtitle=subtitle,
                    anchor="middle",
                ),
            )
        )
        return self.configure_chart(full_chart)
