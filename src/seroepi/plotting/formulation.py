"""Altair visualizers for Formulation design, LOSO stability, rank shifts, and differential prevalence."""

from typing import Literal

import altair as alt
import numpy as np
import polars as pl

from seroepi.estimators.formulation import Formulation, FormulationComparison
from seroepi.plotting.base import BasePlotter


class FormulationCoveragePlotter(BasePlotter):
    """Pareto marginal prevalence and per-stratum cumulative coverage visualizer for a fitted Formulation.

    Args:
        view_mode: `'pareto'` (global marginal bars + cumulative curve) or `'stratum_trajectories'` (per-stratum step-by-step cumulative coverage curves).
        top_n: Maximum number of top-ranked target categories to display.
        max_strata: Maximum number of strata to display individually in `'stratum_trajectories'` mode before pooling the tail into `'Other'`.
        threshold: Optional horizontal cumulative coverage target line (e.g., 0.80 for 80% coverage).
        show_cumulative: Whether to overlay the global cumulative coverage curve (`cum_estimate`).
        show_valency_cutoff: Whether to draw a vertical rule at the K-valent formulation boundary.
        show_error_bars: Whether to display marginal error bars (`'pareto'`) or cumulative CI ribbons (`'stratum_trajectories'`).
        ci_opacity: Fill opacity for cumulative coverage credible/confidence ribbons in `'stratum_trajectories'` mode.
        color_scheme: Vega color scheme for stratum trajectories in `'stratum_trajectories'` mode.
        selected_sig_color: Bar hex color for targets in the formulation that are FDR-significant.
        selected_nonsig_color: Bar hex color for targets in the formulation that are not FDR-significant.
        excluded_color: Bar hex color for targets outside the formulation.
        cumulative_color: Line and point hex color for the global cumulative coverage curve.
        threshold_color: Rule hex color for the target coverage threshold.
        point_size: Marker size for cumulative coverage points.
        stroke_width: Stroke width for the cumulative coverage line.
        x_label_angle: Rotation angle for X-axis target labels.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart text.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for axis and legend labels.
        format_: Numeric format string for prevalence and coverage percentages.

    Attributes:
        view_mode: Active chart layout mode (`"pareto"` or `"stratum_trajectories"`).
        max_strata: Maximum number of individual strata displayed in `"stratum_trajectories"` mode.
        threshold: Horizontal cumulative coverage target line (`0.0` to `1.0`), or `None`.
        show_cumulative: Whether the global cumulative coverage curve is overlaid.
        show_valency_cutoff: Whether a vertical dashed rule marks the K-valent formulation cutoff.
        ci_opacity: Fill opacity for cumulative coverage uncertainty ribbons in `"stratum_trajectories"` mode.
        selected_sig_color: Hex fill color for FDR-significant targets included in the formulation.
        selected_nonsig_color: Hex fill color for non-significant targets included in the formulation.
        excluded_color: Hex fill color for targets outside the K-valent formulation.
        cumulative_color: Hex stroke/fill color for the global cumulative coverage curve and markers.
        threshold_color: Hex stroke color for the horizontal coverage threshold rule.
        point_size: Marker area size in square pixels for cumulative coverage nodes.

    Examples:
        >>> from seroepi.plotting.formulation import FormulationCoveragePlotter
        >>> plotter = FormulationCoveragePlotter(view_mode="pareto", threshold=0.80)
        >>> chart = plotter.with_theme("dark").plot(
        ...     formulation,
        ...     top_n=20,
        ...     title="20-Valent K-Locus Pareto Coverage",
        ... )
        >>> FormulationCoveragePlotter.save(chart, "formulation_coverage.svg")
    """

    def __init__(
        self,
        view_mode: Literal["pareto", "stratum_trajectories"] = "pareto",
        top_n: int | None = 20,
        max_strata: int | None = 8,
        threshold: float | None = 0.80,
        show_cumulative: bool = True,
        show_valency_cutoff: bool = True,
        show_error_bars: bool = True,
        ci_opacity: float = 0.14,
        color_scheme: str = "tableau10",
        selected_sig_color: str = "#2563eb",
        selected_nonsig_color: str = "#60a5fa",
        excluded_color: str = "#94a3b8",
        cumulative_color: str = "#e15759",
        threshold_color: str = "#10b981",
        point_size: int = 55,
        stroke_width: float = 2.5,
        x_label_angle: int = -45,
        width: int = 620,
        height: int = 390,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = ".1%",
    ) -> None:
        """Initializes the formulation Pareto and cumulative trajectory visualizer.

        Args:
            view_mode: `'pareto'` (global marginal bars + cumulative curve) or `'stratum_trajectories'` (per-stratum step-by-step cumulative coverage curves).
            top_n: Maximum number of top-ranked target categories to display.
            max_strata: Maximum number of strata to display individually in `'stratum_trajectories'` mode before pooling the tail into `'Other'`.
            threshold: Optional horizontal cumulative coverage target line (e.g., 0.80 for 80% coverage).
            show_cumulative: Whether to overlay the global cumulative coverage curve (`cum_estimate`).
            show_valency_cutoff: Whether to draw a vertical rule at the K-valent formulation boundary.
            show_error_bars: Whether to display marginal error bars (`'pareto'`) or cumulative CI ribbons (`'stratum_trajectories'`).
            ci_opacity: Fill opacity for cumulative coverage credible/confidence ribbons in `'stratum_trajectories'` mode.
            color_scheme: Vega color scheme for stratum trajectories in `'stratum_trajectories'` mode.
            selected_sig_color: Bar hex color for targets in the formulation that are FDR-significant.
            selected_nonsig_color: Bar hex color for targets in the formulation that are not FDR-significant.
            excluded_color: Bar hex color for targets outside the formulation.
            cumulative_color: Line and point hex color for the global cumulative coverage curve.
            threshold_color: Rule hex color for the target coverage threshold.
            point_size: Marker size for cumulative coverage points.
            stroke_width: Stroke width for the cumulative coverage line.
            x_label_angle: Rotation angle for X-axis target labels.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart text.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for axis and legend labels.
            format_: Numeric format string for prevalence and coverage percentages.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            x_label_angle=x_label_angle,
            color_scheme=color_scheme,
            top_n=top_n,
            format_=format_,
            show_error_bars=show_error_bars,
            stroke_width=stroke_width,
        )
        self.view_mode = view_mode
        self.max_strata = max_strata
        self.threshold = threshold
        self.show_cumulative = show_cumulative
        self.show_valency_cutoff = show_valency_cutoff
        self.ci_opacity = ci_opacity
        self.selected_sig_color = selected_sig_color
        self.selected_nonsig_color = selected_nonsig_color
        self.excluded_color = excluded_color
        self.cumulative_color = cumulative_color
        self.threshold_color = threshold_color
        self.point_size = point_size

    def plot(
        self,
        formulation: Formulation,
        *,
        view_mode: Literal["pareto", "stratum_trajectories"] | None = None,
        top_n: int | None = None,
        max_strata: int | None = None,
        threshold: float | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        """Renders the Pareto marginal prevalence or per-stratum cumulative trajectory chart for `formulation`.

        Args:
            formulation: Fitted `Formulation` instance containing ranked targets and coverage trajectories.
            view_mode: Optional mode override (`"pareto"` or `"stratum_trajectories"`).
            top_n: Optional maximum number of top-ranked targets overriding `self.top_n`.
            max_strata: Optional maximum number of individual strata overriding `self.max_strata`.
            threshold: Optional horizontal cumulative coverage target line overriding `self.threshold`.
            title: Custom chart title.
            subtitle: Custom chart subtitle.

        Returns:
            Configured Altair `LayerChart` displaying marginal bars, cumulative curves, and cutoff rules.

        Raises:
            TypeError: If `formulation` is not a `Formulation` instance.
            RuntimeError: If `formulation` has not been fitted yet.
        """
        if not isinstance(formulation, Formulation):
            raise TypeError(f"Expected a fitted Formulation instance, got {type(formulation).__name__}.")
        formulation.check_is_fitted()

        mode = self.view_mode if view_mode is None else view_mode
        eff_top_n = self.top_n if top_n is None else top_n
        eff_max_strata = self.max_strata if max_strata is None else max_strata
        eff_threshold = self.threshold if threshold is None else threshold

        if mode == "stratum_trajectories":
            if formulation.aggregator_.n_strata < 1:
                fallback_sub = (
                    subtitle
                    if subtitle is not None
                    else (
                        f"{formulation.valency}-Valent ({formulation.weighting} weighting) · "
                        "Unstratified estimator (showing global Pareto view)"
                    )
                )
                return self._plot_pareto(
                    formulation,
                    eff_top_n=eff_top_n,
                    eff_threshold=eff_threshold,
                    title=title,
                    subtitle=fallback_sub,
                )
            return self._plot_stratum_trajectories(
                formulation,
                eff_top_n=eff_top_n,
                eff_max_strata=eff_max_strata,
                eff_threshold=eff_threshold,
                title=title,
                subtitle=subtitle,
            )

        return self._plot_pareto(
            formulation,
            eff_top_n=eff_top_n,
            eff_threshold=eff_threshold,
            title=title,
            subtitle=subtitle,
        )

    def _plot_pareto(
        self,
        formulation: Formulation,
        *,
        eff_top_n: int | None,
        eff_threshold: float | None,
        title: str | None,
        subtitle: str | None,
    ) -> alt.TopLevelMixin:
        est = formulation.estimator_
        target_col = est.target_col
        target_title = est.target_title
        est_col = est.estimate.column
        fmt = self.format_ or est.estimate.format

        df = formulation.rankings_
        if eff_top_n is not None and eff_top_n > 0:
            df = df.head(eff_top_n)

        df = df.with_columns(
            status=pl.when(pl.col("in_formulation") & pl.col("significant"))
            .then(pl.lit("In Formulation (FDR Sig.)"))
            .when(pl.col("in_formulation"))
            .then(pl.lit("In Formulation (Non-Sig.)"))
            .otherwise(pl.lit("Excluded"))
        )

        ordered_targets = df[target_col].cast(pl.String).to_list()
        status_domain = [
            "In Formulation (FDR Sig.)",
            "In Formulation (Non-Sig.)",
            "Excluded",
        ]
        status_range = [
            self.selected_sig_color,
            self.selected_nonsig_color,
            self.excluded_color,
        ]

        tooltips: list[alt.Tooltip] = [
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip(f"{target_col}:N", title=target_title),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Marginal Prevalence"),
            alt.Tooltip("cum_estimate:Q", format=fmt, title="Cumulative Coverage"),
            alt.Tooltip("event:Q", format=",.1f", title="Events"),
            alt.Tooltip("n:Q", format=",.0f", title="Total (n)"),
            alt.Tooltip("q_value:Q", format=".4f", title="FDR q-value"),
            alt.Tooltip("status:N", title="Status"),
        ]
        if "lower" in df.columns and "upper" in df.columns:
            tooltips.extend(
                [
                    alt.Tooltip("lower:Q", format=fmt, title="Lower Bound"),
                    alt.Tooltip("upper:Q", format=fmt, title="Upper Bound"),
                ]
            )

        base = alt.Chart(df).encode(
            x=alt.X(
                f"{target_col}:N",
                sort=ordered_targets,
                title=target_title,
                axis=alt.Axis(labelAngle=self.x_label_angle),
            )
        )

        bars = base.encode(
            y=alt.Y(f"{est_col}:Q", title="Prevalence / Cumulative Coverage", axis=alt.Axis(format=fmt)),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=status_domain, range=status_range),
                title="Formulation Status",
            ),
            tooltip=tooltips,
        ).mark_bar(opacity=0.88)

        layers: list[alt.Chart] = [bars]

        if self.show_error_bars and "lower" in df.columns and "upper" in df.columns:
            layers.extend(
                self.build_error_bar_layers(
                    df,
                    cat_col=target_col,
                    cat_sort=ordered_targets,
                    orientation="vertical",
                    color=self.resolve_ci_color(None),
                    lower_col="lower",
                    upper_col="upper",
                    est_col=est_col,
                    stroke_width=1.2,
                )
            )

        if self.show_cumulative and "cum_estimate" in df.columns:
            cum_line = base.encode(
                y=alt.Y("cum_estimate:Q"),
                order=alt.Order("rank:Q"),
            ).mark_line(color=self.cumulative_color, strokeWidth=self.stroke_width)
            cum_pts = base.encode(
                y=alt.Y("cum_estimate:Q"),
                order=alt.Order("rank:Q"),
                tooltip=tooltips,
            ).mark_circle(color=self.cumulative_color, size=self.point_size)
            layers.extend([cum_line, cum_pts])

        if self.show_valency_cutoff and 1 <= formulation.valency <= len(ordered_targets):
            cutoff_target = ordered_targets[formulation.valency - 1]
            cutoff_df = pl.DataFrame({target_col: [cutoff_target]})
            cutoff_rule = (
                alt.Chart(cutoff_df)
                .encode(x=alt.X(f"{target_col}:N", sort=ordered_targets))
                .mark_rule(color=self.palette.text_secondary, strokeDash=[4, 4], strokeWidth=1.5)
            )
            layers.append(cutoff_rule)

        if eff_threshold is not None and 0.0 < eff_threshold <= 1.0:
            th_df = pl.DataFrame({"th": [float(eff_threshold)]})
            th_rule = (
                alt.Chart(th_df)
                .encode(y=alt.Y("th:Q"))
                .mark_rule(color=self.threshold_color, strokeDash=[6, 4], strokeWidth=1.5)
            )
            layers.append(th_rule)

        k_cov = float(df.filter(pl.col("in_formulation"))[est_col].sum()) if len(df) > 0 else 0.0
        k_cov = min(1.0, max(0.0, k_cov))
        default_sub = (
            f"{formulation.valency}-Valent ({formulation.weighting} weighting) · "
            f"Est. Coverage: {k_cov:{fmt}} · FDR α={formulation.alpha} (τ={formulation.min_prevalence:{fmt}})"
        )

        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"Formulation Coverage & Pareto Ranking ({target_title})",
                subtitle=subtitle or default_sub,
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)

    def _plot_stratum_trajectories(
        self,
        formulation: Formulation,
        *,
        eff_top_n: int | None,
        eff_max_strata: int | None,
        eff_threshold: float | None,
        title: str | None,
        subtitle: str | None,
    ) -> alt.TopLevelMixin:
        est = formulation.estimator_
        target_col = est.target_col
        target_title = est.target_title
        est_col = est.estimate.column
        fmt = self.format_ or est.estimate.format

        cum_df, ordered_targets = formulation.stratum_trajectories(
            top_n=eff_top_n,
            max_strata=eff_max_strata,
        )

        strata_cols = list(est.strata_cols)
        if len(strata_cols) == 1:
            stratum_col = strata_cols[0]
            stratum_title = est.strata_titles[0]
            cum_df = cum_df.with_columns(pl.col(stratum_col).cast(pl.String))
        else:
            stratum_col = "_stratum_label"
            stratum_title = " · ".join(est.strata_titles)
            cum_df = cum_df.with_columns(
                pl.concat_str([pl.col(c).cast(pl.String) for c in strata_cols], separator=" · ").alias(stratum_col)
            )

        stratum_order = cum_df.select(stratum_col).unique(maintain_order=True)[stratum_col].to_list()
        legend_sel = alt.selection_point(fields=[stratum_col], bind="legend")

        tooltips: list[alt.Tooltip] = [
            alt.Tooltip(f"{stratum_col}:N", title=stratum_title),
            alt.Tooltip("rank:Q", title="Global Rank"),
            alt.Tooltip(f"{target_col}:N", title=target_title),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Stratum Prevalence"),
            alt.Tooltip("cum_estimate:Q", format=fmt, title="Stratum Cumulative Coverage"),
            alt.Tooltip("cum_event:Q", format=",.1f", title="Cumulative Events"),
            alt.Tooltip("n:Q", format=",.0f", title="Stratum Total (n)"),
        ]
        has_ci = "cum_lower" in cum_df.columns and "cum_upper" in cum_df.columns
        if has_ci:
            tooltips.extend(
                [
                    alt.Tooltip("cum_lower:Q", format=fmt, title="Cumulative Lower Bound"),
                    alt.Tooltip("cum_upper:Q", format=fmt, title="Cumulative Upper Bound"),
                ]
            )

        base = alt.Chart(cum_df).encode(
            x=alt.X(
                f"{target_col}:N",
                sort=ordered_targets,
                title=f"{target_title} (Global Formulation Order)",
                axis=alt.Axis(labelAngle=self.x_label_angle),
            ),
            color=alt.Color(
                f"{stratum_col}:N",
                sort=stratum_order,
                scale=alt.Scale(scheme=self.color_scheme),
                title=stratum_title,
            ),
            order=alt.Order("rank:Q"),
        )

        layers: list[alt.Chart] = []

        if self.show_error_bars and has_ci:
            band = base.encode(
                y=alt.Y("cum_lower:Q", title="Cumulative Coverage", axis=alt.Axis(format=fmt)),
                y2=alt.Y2("cum_upper:Q"),
                opacity=alt.condition(legend_sel, alt.value(self.ci_opacity), alt.value(0.03)),
            ).mark_area()
            layers.append(band)

        lines = base.encode(
            y=alt.Y("cum_estimate:Q", title="Cumulative Coverage", axis=alt.Axis(format=fmt)),
            opacity=alt.condition(legend_sel, alt.value(0.92), alt.value(0.15)),
        ).mark_line(strokeWidth=self.stroke_width)

        pts = (
            base.encode(
                y=alt.Y("cum_estimate:Q"),
                opacity=alt.condition(legend_sel, alt.value(0.95), alt.value(0.15)),
                tooltip=tooltips,
            )
            .mark_circle(size=self.point_size)
            .add_params(legend_sel)
        )
        layers.extend([lines, pts])

        if self.show_cumulative:
            global_df = formulation.rankings_.filter(pl.col(target_col).cast(pl.String).is_in(ordered_targets))
            global_tooltips = [
                alt.Tooltip("rank:Q", title="Global Rank"),
                alt.Tooltip(f"{target_col}:N", title=target_title),
                alt.Tooltip(f"{est_col}:Q", format=fmt, title=f"Global Prevalence ({formulation.weighting})"),
                alt.Tooltip("cum_estimate:Q", format=fmt, title=f"Global Cumulative ({formulation.weighting})"),
            ]
            global_base = alt.Chart(global_df).encode(
                x=alt.X(f"{target_col}:N", sort=ordered_targets),
                order=alt.Order("rank:Q"),
            )
            global_line = global_base.encode(
                y=alt.Y("cum_estimate:Q"),
            ).mark_line(
                color=self.cumulative_color,
                strokeWidth=self.stroke_width + 0.8,
                strokeDash=[6, 3],
            )
            global_pts = global_base.encode(
                y=alt.Y("cum_estimate:Q"),
                tooltip=global_tooltips,
            ).mark_point(
                shape="diamond",
                filled=True,
                color=self.cumulative_color,
                size=self.point_size + 15,
            )
            layers.extend([global_line, global_pts])

        if self.show_valency_cutoff and 1 <= formulation.valency <= len(ordered_targets):
            cutoff_target = ordered_targets[formulation.valency - 1]
            cutoff_df = pl.DataFrame({target_col: [cutoff_target]})
            cutoff_rule = (
                alt.Chart(cutoff_df)
                .encode(x=alt.X(f"{target_col}:N", sort=ordered_targets))
                .mark_rule(color=self.palette.text_secondary, strokeDash=[4, 4], strokeWidth=1.5)
            )
            layers.append(cutoff_rule)

        if eff_threshold is not None and 0.0 < eff_threshold <= 1.0:
            th_df = pl.DataFrame({"th": [float(eff_threshold)]})
            th_rule = (
                alt.Chart(th_df)
                .encode(y=alt.Y("th:Q"))
                .mark_rule(color=self.threshold_color, strokeDash=[6, 4], strokeWidth=1.5)
            )
            layers.append(th_rule)

        total_strata = len(formulation.score_by_stratum())
        pooled_note = (
            f"Top {len(stratum_order) - 1} of {total_strata} strata + Other"
            if ("Other" in stratum_order and total_strata > len(stratum_order))
            else f"{len(stratum_order)} strata"
        )
        default_sub = (
            f"{formulation.valency}-Valent ({formulation.weighting} global order) · "
            f"{pooled_note} · Click legend to isolate stratum"
        )

        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"Per-Stratum Cumulative Coverage Trajectories ({target_title} by {stratum_title})",
                subtitle=subtitle or default_sub,
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)


class FormulationStabilityPlotter(BasePlotter):
    """Visualizer for Leave-One-Stratum-Out (LOSO) target rank stability and per-stratum coverage.

    Args:
        view_mode: `'target_stability'` (LOSO rank range & selection frequency) or `'stratum_coverage'` (per-stratum observed/predicted/held-out coverage).
        top_n: Maximum number of ranked targets or strata to display.
        show_valency_cutoff: Whether to draw a vertical rule at the K-valent rank boundary in `'target_stability'` mode.
        threshold: Optional reference coverage threshold in `'stratum_coverage'` mode.
        color_scheme: Vega color scheme for LOSO selection frequency or coverage series.
        point_size: Marker size for mean rank or stratum coverage points.
        stroke_width: Stroke width for rank range bars or dumbbell connectors.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart text.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for axis and legend labels.
        format_: Numeric format string for percentages.

    Attributes:
        view_mode: Active chart layout mode (`"target_stability"` or `"stratum_coverage"`).
        show_valency_cutoff: Whether a vertical dashed rule marks the K-valent rank boundary in `"target_stability"` mode.
        threshold: Reference coverage threshold rule in `"stratum_coverage"` mode, or `None`.
        point_size: Marker area size in square pixels for mean rank or stratum coverage dots.

    Examples:
        >>> from seroepi.plotting.formulation import FormulationStabilityPlotter
        >>> plotter = FormulationStabilityPlotter(view_mode="target_stability", top_n=15)
        >>> chart = plotter.with_theme("dark").plot(
        ...     formulation,
        ...     title="LOSO K-Locus Rank Stability",
        ... )
        >>> FormulationStabilityPlotter.save(chart, "loso_stability.svg")
    """

    def __init__(
        self,
        view_mode: Literal["target_stability", "stratum_coverage"] = "target_stability",
        top_n: int | None = 20,
        show_valency_cutoff: bool = True,
        threshold: float | None = 0.80,
        color_scheme: str = "viridis",
        point_size: int = 75,
        stroke_width: float = 2.5,
        width: int = 620,
        height: int = 400,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = ".1%",
    ) -> None:
        """Initializes the LOSO stability and stratum coverage generalization visualizer.

        Args:
            view_mode: `'target_stability'` (LOSO rank range & selection frequency) or `'stratum_coverage'` (per-stratum observed/predicted/held-out coverage).
            top_n: Maximum number of ranked targets or strata to display.
            show_valency_cutoff: Whether to draw a vertical rule at the K-valent rank boundary in `'target_stability'` mode.
            threshold: Optional reference coverage threshold in `'stratum_coverage'` mode.
            color_scheme: Vega color scheme for LOSO selection frequency or coverage series.
            point_size: Marker size for mean rank or stratum coverage points.
            stroke_width: Stroke width for rank range bars or dumbbell connectors.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart text.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for axis and legend labels.
            format_: Numeric format string for percentages.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            color_scheme=color_scheme,
            top_n=top_n,
            format_=format_,
            stroke_width=stroke_width,
        )
        self.view_mode: Literal["target_stability", "stratum_coverage"] = view_mode
        self.show_valency_cutoff = show_valency_cutoff
        self.threshold = threshold
        self.point_size = point_size

    def plot(
        self,
        formulation: Formulation,
        *,
        view_mode: Literal["target_stability", "stratum_coverage"] | None = None,
        top_n: int | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        """Renders either the LOSO target stability chart or the per-stratum coverage comparison.

        Args:
            formulation: Fitted `Formulation` instance with LOSO cross-validation metrics and stratum scores.
            view_mode: Optional mode override (`"target_stability"` or `"stratum_coverage"`).
            top_n: Optional maximum number of targets or strata overriding `self.top_n`.
            title: Custom chart title.
            subtitle: Custom chart subtitle.

        Returns:
            Configured Altair `Chart` or `LayerChart` depicting LOSO rank stability or stratum coverage generalization.

        Raises:
            TypeError: If `formulation` is not a `Formulation` instance.
            RuntimeError: If `formulation` has not been fitted yet.
        """
        if not isinstance(formulation, Formulation):
            raise TypeError(f"Expected a fitted Formulation instance, got {type(formulation).__name__}.")
        formulation.check_is_fitted()

        mode = view_mode or self.view_mode
        eff_top_n = self.top_n if top_n is None else top_n

        if mode == "stratum_coverage":
            return self._plot_stratum_coverage(formulation, eff_top_n, title=title, subtitle=subtitle)
        return self._plot_target_stability(formulation, eff_top_n, title=title, subtitle=subtitle)

    def _plot_target_stability(
        self,
        formulation: Formulation,
        eff_top_n: int | None,
        *,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        est = formulation.estimator_
        target_col = est.target_col
        target_title = est.target_title
        fmt = self.format_ or ".1%"

        if len(formulation.stability_metrics_) == 0:
            # Graceful fallback when unstratified or cv=None
            df = formulation.rankings_
            if eff_top_n is not None and eff_top_n > 0:
                df = df.head(eff_top_n)
            ordered = df[target_col].cast(pl.String).to_list()
            chart = (
                alt.Chart(df)
                .encode(
                    y=alt.Y(f"{target_col}:N", sort=ordered, title=target_title),
                    x=alt.X("rank:Q", title="Global Rank (1 = Highest Prevalence)"),
                    color=alt.Color("in_formulation:N", title="In Formulation"),
                    tooltip=[
                        alt.Tooltip(f"{target_col}:N", title=target_title),
                        alt.Tooltip("rank:Q", title="Global Rank"),
                        alt.Tooltip(f"{est.estimate.column}:Q", format=fmt, title="Prevalence"),
                    ],
                )
                .mark_circle(size=self.point_size)
                .properties(
                    width=self.width,
                    height=self.height,
                    title=alt.TitleParams(
                        text=title or f"Target Ranking ({target_title})",
                        subtitle=subtitle or "LOSO cross-validation requires a stratified estimator with cv='loso'",
                        anchor="middle",
                    ),
                )
            )
            return self.configure_chart(chart)

        df = formulation.stability_metrics_.join(
            formulation.rankings_.select([target_col, est.estimate.column, "in_formulation", "q_value"]),
            on=target_col,
            how="left",
        ).sort("baseline_rank")
        if eff_top_n is not None and eff_top_n > 0:
            df = df.head(eff_top_n)

        ordered = df[target_col].cast(pl.String).to_list()
        tooltips = [
            alt.Tooltip(f"{target_col}:N", title=target_title),
            alt.Tooltip("baseline_rank:Q", title="Full-Dataset Rank"),
            alt.Tooltip("mean_cv_rank:Q", format=".2f", title="Mean LOSO Rank"),
            alt.Tooltip("min_cv_rank:Q", title="Best LOSO Rank"),
            alt.Tooltip("max_cv_rank:Q", title="Worst LOSO Rank"),
            alt.Tooltip("rank_std:Q", format=".2f", title="LOSO Rank SD"),
            alt.Tooltip("selection_frequency:Q", format=fmt, title="Top-K Selection Freq."),
            alt.Tooltip(f"{est.estimate.column}:Q", format=fmt, title="Marginal Prevalence"),
        ]

        base = alt.Chart(df).encode(
            y=alt.Y(f"{target_col}:N", sort=ordered, title=target_title),
        )

        range_rule = base.encode(
            x=alt.X("min_cv_rank:Q", title="LOSO Cross-Validation Rank (Min – Mean – Max)"),
            x2=alt.X2("max_cv_rank:Q"),
            color=alt.Color(
                "selection_frequency:Q",
                scale=alt.Scale(scheme=self.color_scheme, domain=[0.0, 1.0]),
                title="Top-K Selection Freq.",
                legend=alt.Legend(format=fmt),
            ),
            tooltip=tooltips,
        ).mark_rule(strokeWidth=self.stroke_width)

        mean_pts = base.encode(
            x=alt.X("mean_cv_rank:Q"),
            color=alt.Color(
                "selection_frequency:Q",
                scale=alt.Scale(scheme=self.color_scheme, domain=[0.0, 1.0]),
            ),
            tooltip=tooltips,
        ).mark_circle(size=self.point_size)

        baseline_pts = base.encode(
            x=alt.X("baseline_rank:Q"),
            tooltip=tooltips,
        ).mark_tick(color=self.palette.text_primary, size=12, thickness=2.0)

        layers: list[alt.Chart] = [range_rule, mean_pts, baseline_pts]

        if self.show_valency_cutoff:
            cutoff_df = pl.DataFrame({"cutoff": [float(formulation.valency) + 0.5]})
            cutoff_rule = (
                alt.Chart(cutoff_df)
                .encode(x=alt.X("cutoff:Q"))
                .mark_rule(color="#e15759", strokeDash=[5, 4], strokeWidth=1.5)
            )
            layers.append(cutoff_rule)

        n_folds = len(formulation.fold_summary_)
        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"LOSO Target Rank Stability ({target_title})",
                subtitle=subtitle
                or f"Rank range and Top-{formulation.valency} selection frequency across {n_folds} held-out strata",
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)

    def _plot_stratum_coverage(
        self,
        formulation: Formulation,
        eff_top_n: int | None,
        *,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        fmt = self.format_ or ".1%"
        cov_df = formulation.score_by_stratum()
        strata_cols = list(formulation.aggregator_.strata_cols)

        if not strata_cols:
            cov_df = cov_df.with_columns(stratum=pl.lit("All Isolates"))
        elif len(strata_cols) == 1:
            cov_df = cov_df.with_columns(stratum=pl.col(strata_cols[0]).cast(pl.String))
        else:
            cov_df = cov_df.with_columns(
                stratum=pl.concat_str([pl.col(c).cast(pl.String) for c in strata_cols], separator=" · ")
            )

        if len(formulation.fold_summary_) > 0:
            cov_df = cov_df.join(
                formulation.fold_summary_.select(
                    [
                        pl.col("holdout_stratum").alias("stratum"),
                        pl.col("heldout_predicted_coverage"),
                        pl.col("jaccard_similarity"),
                    ]
                ),
                on="stratum",
                how="left",
            )

        cov_df = cov_df.sort("predicted_coverage", descending=True)
        if eff_top_n is not None and eff_top_n > 0:
            cov_df = cov_df.head(eff_top_n)

        ordered_strata = cov_df["stratum"].to_list()
        records: list[dict[str, object]] = []
        for row in cov_df.to_dicts():
            s_name = str(row["stratum"])
            n_iso = int(row.get("n", 0) or 0)
            jacc = row.get("jaccard_similarity")
            records.append(
                {
                    "stratum": s_name,
                    "metric": "Observed Coverage",
                    "coverage": float(row["observed_coverage"]),
                    "n": n_iso,
                    "jaccard": float(jacc) if jacc is not None else None,
                }
            )
            records.append(
                {
                    "stratum": s_name,
                    "metric": "Predicted Coverage (In-Sample)",
                    "coverage": float(row["predicted_coverage"]),
                    "n": n_iso,
                    "jaccard": float(jacc) if jacc is not None else None,
                }
            )
            if row.get("heldout_predicted_coverage") is not None:
                records.append(
                    {
                        "stratum": s_name,
                        "metric": "LOSO Held-Out Coverage",
                        "coverage": float(row["heldout_predicted_coverage"]),
                        "n": n_iso,
                        "jaccard": float(jacc) if jacc is not None else None,
                    }
                )

        long_df = pl.DataFrame(records)
        base = alt.Chart(long_df).encode(
            y=alt.Y("stratum:N", sort=ordered_strata, title="Stratum"),
        )

        span_df = cov_df.with_columns(
            min_cov=pl.min_horizontal("observed_coverage", "predicted_coverage"),
            max_cov=pl.max_horizontal("observed_coverage", "predicted_coverage"),
        )
        connector = (
            alt.Chart(span_df)
            .encode(
                y=alt.Y("stratum:N", sort=ordered_strata),
                x=alt.X("min_cov:Q", title="Formulation Coverage", axis=alt.Axis(format=fmt)),
                x2=alt.X2("max_cov:Q"),
            )
            .mark_rule(color=self.palette.domain_color, strokeWidth=self.stroke_width)
        )

        pts = base.encode(
            x=alt.X("coverage:Q", title="Formulation Coverage", axis=alt.Axis(format=fmt)),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=[
                        "Observed Coverage",
                        "Predicted Coverage (In-Sample)",
                        "LOSO Held-Out Coverage",
                    ],
                    range=["#2563eb", "#10b981", "#f59e0b"],
                ),
                title="Coverage Metric",
            ),
            tooltip=[
                alt.Tooltip("stratum:N", title="Stratum"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("coverage:Q", format=fmt, title="Coverage"),
                alt.Tooltip("n:Q", format=",.0f", title="Isolates (n)"),
                alt.Tooltip("jaccard:Q", format=".2f", title="LOSO Jaccard Overlap"),
            ],
        ).mark_circle(size=self.point_size)

        layers: list[alt.Chart] = [connector, pts]
        if self.threshold is not None and 0.0 < self.threshold <= 1.0:
            th_df = pl.DataFrame({"th": [float(self.threshold)]})
            layers.append(
                alt.Chart(th_df).encode(x=alt.X("th:Q")).mark_rule(color="#e15759", strokeDash=[5, 4], strokeWidth=1.5)
            )

        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"Stratum Coverage Generalization ({formulation.display_name})",
                subtitle=subtitle or "Observed vs. model-predicted and LOSO held-out formulation coverage",
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)


class DifferentialPrevalencePlotter(BasePlotter):
    """Volcano and Forest plot visualizer for pairwise cohort or model differential prevalence.

    Args:
        view_mode: `'volcano'` (effect size vs. -log10 FDR q-value) or `'forest'` (ranked horizontal Δ prevalence intervals).
        x_metric: Effect size column for volcano X-axis (`'delta_prevalence'` or `'log2_fold_change'`).
        top_n: Maximum number of ranked targets to display in `'forest'` mode or label in `'volcano'` mode.
        show_labels: Whether to render text labels for significant targets in `'volcano'` mode.
        show_error_bars: Whether to render uncertainty intervals in `'forest'` mode.
        enriched_b_color: Hex color for targets significantly enriched in Cohort/Model B (Δ > 0).
        enriched_a_color: Hex color for targets significantly enriched in Cohort/Model A (Δ < 0).
        nonsig_color: Hex color for non-significant targets.
        point_size: Base marker size for points.
        stroke_width: Stroke width for forest intervals and reference lines.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart text.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for axis and legend labels.
        format_: Numeric format string for prevalence differences.

    Attributes:
        view_mode: Active chart layout mode (`"volcano"` or `"forest"`).
        x_metric: Effect size column used on the volcano X-axis (`"delta_prevalence"` or `"log2_fold_change"`).
        show_labels: Whether text annotations are rendered for FDR-significant targets in `"volcano"` mode.
        enriched_b_color: Hex fill/stroke color for targets significantly enriched in Cohort/Model B.
        enriched_a_color: Hex fill/stroke color for targets significantly enriched in Cohort/Model A.
        nonsig_color: Hex fill/stroke color for non-significant targets.
        point_size: Base marker area size in square pixels for volcano and forest dots.

    Examples:
        >>> from seroepi.plotting.formulation import DifferentialPrevalencePlotter
        >>> plotter = DifferentialPrevalencePlotter(view_mode="volcano", top_n=15)
        >>> chart = plotter.with_theme("dark").plot(
        ...     comparison,
        ...     label_a="Neonatal",
        ...     label_b="Adult",
        ...     title="Differential K-Locus Prevalence: Adult vs. Neonatal",
        ... )
        >>> DifferentialPrevalencePlotter.save(chart, "differential_volcano.svg")
    """

    def __init__(
        self,
        view_mode: Literal["volcano", "forest"] = "volcano",
        x_metric: Literal["delta_prevalence", "log2_fold_change"] = "delta_prevalence",
        top_n: int | None = 20,
        show_labels: bool = True,
        show_error_bars: bool = True,
        enriched_b_color: str = "#dc2626",
        enriched_a_color: str = "#2563eb",
        nonsig_color: str = "#94a3b8",
        point_size: int = 80,
        stroke_width: float = 1.8,
        width: int = 620,
        height: int = 410,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = "+.1%",
    ) -> None:
        """Initializes the pairwise differential prevalence Volcano and Forest visualizer.

        Args:
            view_mode: `'volcano'` (effect size vs. -log10 FDR q-value) or `'forest'` (ranked horizontal Δ prevalence intervals).
            x_metric: Effect size column for volcano X-axis (`'delta_prevalence'` or `'log2_fold_change'`).
            top_n: Maximum number of ranked targets to display in `'forest'` mode or label in `'volcano'` mode.
            show_labels: Whether to render text labels for significant targets in `'volcano'` mode.
            show_error_bars: Whether to render uncertainty intervals in `'forest'` mode.
            enriched_b_color: Hex color for targets significantly enriched in Cohort/Model B (Δ > 0).
            enriched_a_color: Hex color for targets significantly enriched in Cohort/Model A (Δ < 0).
            nonsig_color: Hex color for non-significant targets.
            point_size: Base marker size for points.
            stroke_width: Stroke width for forest intervals and reference lines.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart text.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for axis and legend labels.
            format_: Numeric format string for prevalence differences.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            top_n=top_n,
            format_=format_,
            show_error_bars=show_error_bars,
            stroke_width=stroke_width,
        )
        self.view_mode: Literal["volcano", "forest"] = view_mode
        self.x_metric: Literal["delta_prevalence", "log2_fold_change"] = x_metric
        self.show_labels = show_labels
        self.enriched_b_color = enriched_b_color
        self.enriched_a_color = enriched_a_color
        self.nonsig_color = nonsig_color
        self.point_size = point_size

    def plot(
        self,
        comparison: FormulationComparison,
        *,
        label_a: str | None = None,
        label_b: str | None = None,
        view_mode: Literal["volcano", "forest"] | None = None,
        top_n: int | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        """Renders the Volcano or Forest differential prevalence chart for `(label_a, label_b)`.

        Args:
            comparison: Fitted `FormulationComparison` instance containing pairwise contrasts.
            label_a: Baseline cohort/model key in `comparison.formulations_` (defaults to the first formulation).
            label_b: Comparison cohort/model key in `comparison.formulations_` (defaults to the second formulation).
            view_mode: Optional mode override (`"volcano"` or `"forest"`).
            top_n: Optional maximum number of targets to display or annotate overriding `self.top_n`.
            title: Custom chart title.
            subtitle: Custom chart subtitle.

        Returns:
            Configured Altair `LayerChart` depicting differential prevalence effect sizes, significance thresholds, and intervals.

        Raises:
            TypeError: If `comparison` is not a `FormulationComparison` instance.
            RuntimeError: If `comparison` has not been fitted yet.
        """
        if not isinstance(comparison, FormulationComparison):
            raise TypeError(f"Expected a fitted FormulationComparison instance, got {type(comparison).__name__}.")
        comparison.check_is_fitted()

        names = list(comparison.formulations_.keys())
        eff_a = label_a if (label_a and label_a in comparison.formulations_) else names[0]
        eff_b = (
            label_b
            if (label_b and label_b in comparison.formulations_ and label_b != eff_a)
            else (names[1] if len(names) > 1 else names[0])
        )
        df = comparison.contrast(eff_a, eff_b)
        first_form = comparison.formulations_[eff_a]
        target_col = first_form.estimator_.target_col
        target_title = first_form.estimator_.target_title
        fmt = self.format_ or "+.1%"

        q_clipped = np.clip(df["q_value"].to_numpy().astype(float), 1e-10, 1.0)
        neg_log10_q = -np.log10(q_clipped)

        dir_b = f"Enriched in {eff_b} (FDR < {comparison.alpha})"
        dir_a = f"Enriched in {eff_a} (FDR < {comparison.alpha})"
        dir_ns = "Not Significant"

        df = df.with_columns(
            neg_log10_q=pl.Series(neg_log10_q, dtype=pl.Float64),
            abs_delta=pl.col("delta_prevalence").abs(),
            total_events=pl.col("event_a") + pl.col("event_b"),
            direction=pl.when(pl.col("significant") & (pl.col("delta_prevalence") > 0))
            .then(pl.lit(dir_b))
            .when(pl.col("significant") & (pl.col("delta_prevalence") < 0))
            .then(pl.lit(dir_a))
            .otherwise(pl.lit(dir_ns)),
        )

        color_scale = alt.Scale(
            domain=[dir_b, dir_a, dir_ns],
            range=[self.enriched_b_color, self.enriched_a_color, self.nonsig_color],
        )

        tooltips = [
            alt.Tooltip(f"{target_col}:N", title=target_title),
            alt.Tooltip("estimate_a:Q", format=".1%", title=f"Prevalence ({eff_a})"),
            alt.Tooltip("estimate_b:Q", format=".1%", title=f"Prevalence ({eff_b})"),
            alt.Tooltip("delta_prevalence:Q", format=fmt, title=f"Δ ({eff_b} - {eff_a})"),
            alt.Tooltip("delta_lower:Q", format=fmt, title="Δ Lower Bound"),
            alt.Tooltip("delta_upper:Q", format=fmt, title="Δ Upper Bound"),
            alt.Tooltip("log2_fold_change:Q", format="+.2f", title="log2 Fold Change"),
            alt.Tooltip("p_or_lfsr:Q", format=".4f", title="p-value / LFSR"),
            alt.Tooltip("q_value:Q", format=".4f", title="FDR q-value"),
            alt.Tooltip("direction:N", title="Call"),
        ]

        mode = view_mode or self.view_mode
        eff_top_n = self.top_n if top_n is None else top_n
        method_str = str(df["test_method"][0]) if len(df) > 0 else "fdr"
        method_display = "Bayesian LFSR" if method_str == "bayesian_lfsr" else "Fisher's Exact + BH"

        if mode == "forest":
            df_forest = df.sort(["significant", "abs_delta"], descending=[True, True])
            if eff_top_n is not None and eff_top_n > 0:
                df_forest = df_forest.head(eff_top_n)
            ordered = df_forest[target_col].cast(pl.String).to_list()

            base = alt.Chart(df_forest).encode(
                y=alt.Y(f"{target_col}:N", sort=ordered, title=target_title),
            )
            layers: list[alt.Chart] = []

            # Zero and null_margin reference lines
            zero_df = pl.DataFrame({"x": [0.0]})
            layers.append(
                alt.Chart(zero_df)
                .encode(x=alt.X("x:Q"))
                .mark_rule(color=self.palette.text_secondary, strokeDash=[4, 4], strokeWidth=1.2)
            )
            if comparison.null_margin > 0.0:
                margin_df = pl.DataFrame({"x": [-comparison.null_margin, comparison.null_margin]})
                layers.append(
                    alt.Chart(margin_df)
                    .encode(x=alt.X("x:Q"))
                    .mark_rule(color="#f59e0b", strokeDash=[3, 3], strokeWidth=1.2)
                )

            if self.show_error_bars:
                layers.append(
                    base.encode(
                        x=alt.X("delta_lower:Q", title=f"Δ Prevalence ({eff_b} − {eff_a})", axis=alt.Axis(format=fmt)),
                        x2=alt.X2("delta_upper:Q"),
                        color=alt.Color("direction:N", scale=color_scale, title="Differential Call"),
                        tooltip=tooltips,
                    ).mark_rule(strokeWidth=self.stroke_width)
                )

            pts = base.encode(
                x=alt.X("delta_prevalence:Q", title=f"Δ Prevalence ({eff_b} − {eff_a})", axis=alt.Axis(format=fmt)),
                color=alt.Color("direction:N", scale=color_scale, title="Differential Call"),
                tooltip=tooltips,
            ).mark_circle(size=self.point_size)
            layers.append(pts)

            chart = alt.layer(*layers).properties(
                width=self.width,
                height=self.height,
                title=alt.TitleParams(
                    text=title or f"Differential Prevalence Forest Plot: {eff_b} vs. {eff_a}",
                    subtitle=subtitle
                    or f"{method_display} · FDR α={comparison.alpha} · Null margin δ={comparison.null_margin:.1%}",
                    anchor="middle",
                ),
            )
            return self.configure_chart(chart)

        # Volcano mode
        x_col = self.x_metric
        x_title = (
            f"Δ Prevalence ({eff_b} − {eff_a})"
            if x_col == "delta_prevalence"
            else f"log2 Fold Change ({eff_b} / {eff_a})"
        )
        x_fmt = fmt if x_col == "delta_prevalence" else "+.2f"

        pts = (
            alt.Chart(df)
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_title, axis=alt.Axis(format=x_fmt)),
                y=alt.Y("neg_log10_q:Q", title="-log10(FDR q-value)"),
                color=alt.Color("direction:N", scale=color_scale, title="Differential Call"),
                size=alt.Size(
                    "total_events:Q",
                    scale=alt.Scale(range=[max(35, self.point_size // 2), self.point_size * 2.5]),
                    title="Total Events",
                ),
                tooltip=tooltips,
            )
            .mark_circle(opacity=0.88)
        )

        alpha_line_df = pl.DataFrame({"y": [-float(np.log10(comparison.alpha))]})
        alpha_rule = (
            alt.Chart(alpha_line_df)
            .encode(y=alt.Y("y:Q"))
            .mark_rule(color="#e15759", strokeDash=[5, 4], strokeWidth=1.4)
        )

        zero_df = pl.DataFrame({"x": [0.0]})
        zero_rule = (
            alt.Chart(zero_df)
            .encode(x=alt.X("x:Q"))
            .mark_rule(color=self.palette.domain_color, strokeDash=[3, 3], strokeWidth=1.2)
        )

        layers = [zero_rule, alpha_rule, pts]
        if x_col == "delta_prevalence" and comparison.null_margin > 0.0:
            margin_df = pl.DataFrame({"x": [-comparison.null_margin, comparison.null_margin]})
            layers.append(
                alt.Chart(margin_df)
                .encode(x=alt.X("x:Q"))
                .mark_rule(color="#f59e0b", strokeDash=[4, 4], strokeWidth=1.2)
            )

        if self.show_labels:
            label_df = df.filter(pl.col("significant")).sort("q_value")
            if eff_top_n is not None and eff_top_n > 0:
                label_df = label_df.head(eff_top_n)
            if len(label_df) > 0:
                text_layer = (
                    alt.Chart(label_df)
                    .encode(
                        x=alt.X(f"{x_col}:Q"),
                        y=alt.Y("neg_log10_q:Q"),
                        text=alt.Text(f"{target_col}:N"),
                    )
                    .mark_text(dy=-10, fontSize=self.label_font_size, color=self.palette.text_primary)
                )
                layers.append(text_layer)

        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"Differential Prevalence Volcano: {eff_b} vs. {eff_a}",
                subtitle=subtitle
                or f"{method_display} · FDR α={comparison.alpha} · Null margin δ={comparison.null_margin:.1%}",
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)


class RankShiftPlotter(BasePlotter):
    """Bump chart / slopegraph visualizer of target rank trajectories across formulations or cohorts.

    Args:
        top_n: Include any target reaching at least rank `top_n` in any compared formulation.
        highlight_shifts_only: Dim targets that do not cross the top-valency boundary or shift by >= 2 ranks.
        show_valency_cutoff: Whether to draw a horizontal dashed rule at the K-valent inclusion cutoff.
        color_scheme: Vega categorical color scheme for target trajectories.
        point_size: Marker size for rank nodes.
        stroke_width: Line thickness for rank trajectories.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart text.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for axis and legend labels.
        format_: Numeric format string for prevalence tooltips.

    Attributes:
        highlight_shifts_only: Whether trajectories that do not cross the valency cutoff or shift by `>= 2` ranks are dimmed.
        show_valency_cutoff: Whether a horizontal dashed rule marks the K-valent inclusion cutoff.
        point_size: Marker area size in square pixels for rank nodes.

    Examples:
        >>> from seroepi.plotting.formulation import RankShiftPlotter
        >>> plotter = RankShiftPlotter(top_n=15, highlight_shifts_only=True)
        >>> chart = plotter.with_theme("dark").plot(
        ...     comparison,
        ...     title="K-Locus Rank Shifts Across Cohorts",
        ... )
        >>> RankShiftPlotter.save(chart, "rank_shifts.svg")
    """

    def __init__(
        self,
        top_n: int | None = 15,
        highlight_shifts_only: bool = False,
        show_valency_cutoff: bool = True,
        color_scheme: str = "tableau10",
        point_size: int = 85,
        stroke_width: float = 2.4,
        width: int = 620,
        height: int = 410,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = ".1%",
    ) -> None:
        """Initializes the bump chart / slopegraph rank trajectory visualizer.

        Args:
            top_n: Include any target reaching at least rank `top_n` in any compared formulation.
            highlight_shifts_only: Dim targets that do not cross the top-valency boundary or shift by >= 2 ranks.
            show_valency_cutoff: Whether to draw a horizontal dashed rule at the K-valent inclusion cutoff.
            color_scheme: Vega categorical color scheme for target trajectories.
            point_size: Marker size for rank nodes.
            stroke_width: Line thickness for rank trajectories.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart text.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for axis and legend labels.
            format_: Numeric format string for prevalence tooltips.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            color_scheme=color_scheme,
            top_n=top_n,
            format_=format_,
            stroke_width=stroke_width,
        )
        self.highlight_shifts_only = highlight_shifts_only
        self.show_valency_cutoff = show_valency_cutoff
        self.point_size = point_size

    def plot(
        self,
        comparison: FormulationComparison,
        *,
        top_n: int | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        """Renders the bump chart / slopegraph of target rankings across compared models or cohorts.

        Args:
            comparison: Fitted `FormulationComparison` instance containing multi-formulation rankings and rank shifts.
            top_n: Optional rank cutoff overriding `self.top_n` (`min_rank <= top_n`).
            title: Custom chart title.
            subtitle: Custom chart subtitle.

        Returns:
            Configured Altair `LayerChart` depicting connected target rank trajectories and the K-valent cutoff rule.

        Raises:
            TypeError: If `comparison` is not a `FormulationComparison` instance.
            RuntimeError: If `comparison` has not been fitted yet.
        """
        if not isinstance(comparison, FormulationComparison):
            raise TypeError(f"Expected a fitted FormulationComparison instance, got {type(comparison).__name__}.")
        comparison.check_is_fitted()

        names = list(comparison.formulations_.keys())
        first_form = comparison.formulations_[names[0]]
        target_col = first_form.estimator_.target_col
        target_title = first_form.estimator_.target_title
        est_col = first_form.estimator_.estimate.column
        fmt = self.format_ or ".1%"

        eff_top_n = self.top_n if top_n is None else top_n
        shifts_df = comparison.rank_shifts_
        if eff_top_n is not None and eff_top_n > 0:
            shifts_df = shifts_df.filter(pl.col("min_rank") <= eff_top_n)

        keep_targets = shifts_df[target_col].cast(pl.String).to_list()
        long_df = (
            comparison.rankings_.filter(pl.col(target_col).cast(pl.String).is_in(keep_targets))
            .join(
                shifts_df.select([target_col, "min_rank", "max_rank", "rank_range"]),
                on=target_col,
                how="left",
            )
            .with_columns(
                crosses_valency=(pl.col("min_rank") <= comparison.valency) & (pl.col("max_rank") > comparison.valency),
            )
            .with_columns(
                is_highlighted=pl.col("crosses_valency") | (pl.col("rank_range") >= 2),
            )
        )

        selection = alt.selection_point(fields=[target_col], bind="legend")
        if self.highlight_shifts_only:
            opacity_expr = alt.condition(
                selection & (alt.datum.is_highlighted == True),  # noqa: E712
                alt.value(0.95),
                alt.value(0.18),
            )
        else:
            opacity_expr = alt.condition(selection, alt.value(0.92), alt.value(0.15))

        tooltips = [
            alt.Tooltip("formulation:N", title="Cohort / Formulation"),
            alt.Tooltip(f"{target_col}:N", title=target_title),
            alt.Tooltip("rank:Q", title="Rank"),
            alt.Tooltip(f"{est_col}:Q", format=fmt, title="Prevalence"),
            alt.Tooltip("in_formulation:N", title=f"In Top-{comparison.valency}"),
            alt.Tooltip("rank_range:Q", title="Max Rank Shift"),
        ]

        base = alt.Chart(long_df).encode(
            x=alt.X("formulation:N", sort=names, title="Cohort / Formulation", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "rank:Q",
                scale=alt.Scale(reverse=True),
                title="Target Rank (1 = Highest)",
                axis=alt.Axis(tickMinStep=1),
            ),
            color=alt.Color(
                f"{target_col}:N",
                sort=keep_targets,
                scale=alt.Scale(scheme=self.color_scheme),
                title=target_title,
            ),
        )

        lines = base.encode(
            detail=alt.Detail(f"{target_col}:N"),
            opacity=opacity_expr,
        ).mark_line(strokeWidth=self.stroke_width)

        pts = (
            base.encode(
                opacity=opacity_expr,
                tooltip=tooltips,
            )
            .mark_circle(size=self.point_size)
            .add_params(selection)
        )

        layers: list[alt.Chart] = [lines, pts]

        if self.show_valency_cutoff:
            cutoff_df = pl.DataFrame({"y": [float(comparison.valency) + 0.5]})
            layers.append(
                alt.Chart(cutoff_df)
                .encode(y=alt.Y("y:Q", scale=alt.Scale(reverse=True)))
                .mark_rule(color="#e15759", strokeDash=[5, 4], strokeWidth=1.5)
            )

        chart = alt.layer(*layers).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=title or f"Target Rank Shifts Across Formulations ({target_title})",
                subtitle=subtitle
                or f"Top-{eff_top_n or 'All'} trajectories across {len(names)} cohorts/models · Dashed line = Top-{comparison.valency} cutoff",
                anchor="middle",
            ),
        )
        return self.configure_chart(chart)
