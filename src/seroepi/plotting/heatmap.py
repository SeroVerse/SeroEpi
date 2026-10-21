"""Stateful epidemiological heatmap visualizer for single-stratum fitted estimators."""

from collections.abc import Sequence

import altair as alt

from seroepi.estimators.base import BaseEstimator
from seroepi.plotting.base import SingleStratumPlotter


class HeatmapPlotter(SingleStratumPlotter):
    """Stateful epidemiological heatmap visualizer for single-stratum fitted estimators.

    Renders a 2D target-by-stratum matrix of prevalence estimates with subtle tile
    borders, continuous sequential shading, contrast-adaptive overlaid event counts
    (or formatted estimates), automatic tail-stratum lumping (`estimator.lump_strata`),
    and rich uncertainty tooltips.

    Args:
        x_limit: Maximum number of X-axis strata columns to display before pooling the tail into `other_label` (`None` for all).
        top_n: Maximum number of top-ranked Y-axis target categories to display (`None` for all).
        y_order: Optional explicit sequence of target categories defining Y-axis row ordering.
        color_scheme: Vega sequential color scheme for tile fill shading (defaults to `"blues"`).
        opacity: Fill opacity (`0.0` to `1.0`) for heatmap tiles.
        stroke_color: Border stroke color separating adjacent heatmap tiles.
        stroke_width: Border stroke width in pixels for heatmap tiles.
        text_color: Explicit text color for cell annotations (`None` for automatic white/dark contrast switching).
        cell_font_size: Font size in pixels for overlaid cell count or estimate annotations.
        show_y_labels: Whether to render Y-axis target category labels and ticks.
        x_label_angle: Rotation angle in degrees for X-axis stratum labels.
        other_label: Category label assigned to pooled tail strata when `x_limit` is exceeded.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart typography.
        title_font_size: Font size in pixels for the chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for axis and legend labels.
        show_error_bars: Whether to include uncertainty interval bounds in hover tooltips.

    Attributes:
        x_limit: Maximum number of X-axis strata columns before pooling into `other_label`.
        y_order: Explicit sequence of target categories defining Y-axis ordering, or `None`.
        opacity: Fill opacity for heatmap tiles.
        stroke_color: Border stroke color for heatmap tiles.
        text_color: Explicit cell annotation text color, or `None` for auto-contrast.
        cell_font_size: Font size in pixels for cell text annotations.
        show_y_labels: Whether Y-axis labels and ticks are displayed.
        other_label: Display label for pooled tail strata.

    Examples:
        >>> from seroepi.plotting.heatmap import HeatmapPlotter
        >>> plotter = HeatmapPlotter(top_n=15, x_limit=8, color_scheme="blues")
        >>> chart = plotter.with_theme("dark").plot(
        ...     stratified_estimator,
        ...     title="K-Locus Prevalence by Region",
        ... )
        >>> HeatmapPlotter.save(chart, "region_heatmap.svg")
    """

    def __init__(
        self,
        x_limit: int | None = None,
        top_n: int | None = None,
        y_order: Sequence[str] | None = None,
        color_scheme: str = "blues",
        opacity: float = 0.85,
        stroke_color: str = "white",
        stroke_width: float = 1.0,
        text_color: str | None = None,
        cell_font_size: int = 10,
        show_y_labels: bool = True,
        x_label_angle: int = -45,
        other_label: str = "Other",
        width: int = 300,
        height: int = 450,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        show_error_bars: bool = True,
    ) -> None:
        """Initializes the single-stratum epidemiological heatmap visualizer.

        Args:
            x_limit: Maximum number of X-axis strata columns to display before pooling the tail into `other_label` (`None` for all).
            top_n: Maximum number of top-ranked Y-axis target categories to display (`None` for all).
            y_order: Optional explicit sequence of target categories defining Y-axis row ordering.
            color_scheme: Vega sequential color scheme for tile fill shading (defaults to `"blues"`).
            opacity: Fill opacity (`0.0` to `1.0`) for heatmap tiles.
            stroke_color: Border stroke color separating adjacent heatmap tiles.
            stroke_width: Border stroke width in pixels for heatmap tiles.
            text_color: Explicit text color for cell annotations (`None` for automatic white/dark contrast switching).
            cell_font_size: Font size in pixels for overlaid cell count or estimate annotations.
            show_y_labels: Whether to render Y-axis target category labels and ticks.
            x_label_angle: Rotation angle in degrees for X-axis stratum labels.
            other_label: Category label assigned to pooled tail strata when `x_limit` is exceeded.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart typography.
            title_font_size: Font size in pixels for the chart title.
            subtitle_font_size: Font size in pixels for the chart subtitle.
            label_font_size: Font size in pixels for axis and legend labels.
            show_error_bars: Whether to include uncertainty interval bounds in hover tooltips.
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            color_scheme=color_scheme,
            x_label_angle=x_label_angle,
            top_n=top_n,
            show_error_bars=show_error_bars,
            stroke_width=stroke_width,
        )
        self.x_limit = x_limit
        self.y_order = y_order
        self.opacity = opacity
        self.stroke_color = stroke_color
        self.text_color = text_color
        self.cell_font_size = cell_font_size
        self.show_y_labels = show_y_labels
        self.other_label = other_label

    def plot(
        self,
        estimator: BaseEstimator,
        *,
        x_limit: int | None = None,
        y_order: Sequence[str] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> alt.TopLevelMixin:
        """Renders a 2D target-by-stratum prevalence heatmap with overlaid event count annotations.

        Args:
            estimator: Fitted single-stratum `BaseEstimator` instance.
            x_limit: Optional maximum number of X-axis strata columns overriding `self.x_limit`.
            y_order: Optional explicit target category ordering overriding `self.y_order`.
            title: Custom chart title overriding the default `"{target_title} Prevalence by {stratum_title}"`.
            subtitle: Custom chart subtitle overriding the automatic ordering and lumping summary.

        Returns:
            Configured Altair `LayerChart` combining shaded prevalence tiles and text annotations.

        Raises:
            TypeError: If `estimator` is not a `BaseEstimator` instance.
            RuntimeError: If `estimator` has not been fitted yet.
            ValueError: If `estimator` does not have exactly 1 stratification dimension.
        """
        ctx = self.extract_context(estimator)
        assert estimator.estimates_ is not None

        # 1. Resolve Y-axis ordering and lump X-axis strata via the estimator
        effective_y_order = self.resolve_targets(estimator, y_order or self.y_order, self.top_n)
        effective_x_limit = self.x_limit if x_limit is None else x_limit
        total_strata = estimator.estimates_[ctx.stratum_col].n_unique()

        plot_df, effective_x_order = estimator.lump_strata(
            targets=effective_y_order,
            x_limit=effective_x_limit,
            other_label=self.other_label,
        )

        # 2. Resolve Color Scale Domain
        max_est = self.series_max(plot_df[ctx.est_col])
        if max_est <= 0.0:
            max_est = 1.0

        # 3. Build Encodings & Tooltips
        y_axis = alt.Axis(title=ctx.target_title, labels=True, ticks=True) if self.show_y_labels else None
        y_encoding = alt.Y(f"{ctx.target_col}:N", sort=effective_y_order, axis=y_axis)
        x_axis = alt.Axis(title=ctx.stratum_title, labelAngle=self.x_label_angle, labels=True, ticks=True)
        x_encoding = alt.X(f"{ctx.stratum_col}:N", sort=effective_x_order, axis=x_axis)

        tooltips = self.build_tooltips(
            plot_df,
            dimensions=[(ctx.target_col, ctx.target_title), (ctx.stratum_col, ctx.stratum_title)],
            est_col=ctx.est_col,
            est_title=ctx.est_title,
            fmt=ctx.format,
            ci_prefix=ctx.ci_prefix,
            lower_col=estimator.estimate.lower,
            upper_col=estimator.estimate.upper,
            include_intervals=estimator.estimate.has_interval,
        )

        # 4. Rect (Tile) Layer
        rect = (
            alt.Chart(plot_df)
            .encode(
                x=x_encoding,
                y=y_encoding,
                color=alt.Color(
                    f"{ctx.est_col}:Q",
                    scale=alt.Scale(scheme=self.color_scheme, domain=[0, max_est]),
                    legend=alt.Legend(title=ctx.est_title, format=ctx.format),
                ),
                tooltip=tooltips,
            )
            .mark_rect(
                opacity=self.opacity,
                stroke=self.stroke_color,
                strokeWidth=self.stroke_width,
            )
        )

        # 5. Text Annotation Layer (Visual number of events)
        if self.text_color is not None:
            text_color_encoding = alt.value(self.text_color)
        else:
            text_color_encoding = alt.condition(
                alt.datum[ctx.est_col] > (max_est * 0.5),
                alt.value("white"),
                alt.value("#1e293b"),
            )

        text_mark = (
            alt.Chart(plot_df)
            .encode(
                x=alt.X(f"{ctx.stratum_col}:N", sort=effective_x_order),
                y=alt.Y(f"{ctx.target_col}:N", sort=effective_y_order),
                text=alt.Text("event:Q", format=",.0f")
                if "event" in plot_df.columns
                else alt.Text(f"{ctx.est_col}:Q", format=ctx.format),
                color=text_color_encoding,
                tooltip=tooltips,
            )
            .mark_text(
                baseline="middle",
                fontSize=self.cell_font_size,
            )
        )

        # 6. Titles and Subtitles
        resolved_title = title or f"{ctx.target_title} Prevalence by {ctx.stratum_title}"
        if subtitle is not None:
            resolved_subtitle = subtitle
        else:
            sub_parts: list[str] = [f"X-axis ordered descending by total {ctx.stratum_title} prevalence"]
            if effective_x_limit is not None and total_strata > effective_x_limit:
                sub_parts.append(f"Top {effective_x_limit - 1} categories; remaining binned into '{self.other_label}'")
            if estimator.estimate.has_interval and estimator.estimate.interval_description:
                sub_parts.append(estimator.estimate.interval_description)
            resolved_subtitle = " | ".join(sub_parts)

        compound = alt.layer(rect, text_mark).properties(
            width=self.width,
            height=self.height,
            title=alt.TitleParams(
                text=resolved_title,
                subtitle=resolved_subtitle,
                anchor="middle",
            ),
        )
        return self.configure_chart(compound)
