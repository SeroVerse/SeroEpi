"""Stateful back-to-back pyramid visualizer for comparing two fitted prevalence estimators."""

from collections.abc import Sequence
from typing import Literal

import altair as alt
import polars as pl

from seroepi.estimators.base import BaseEstimator
from seroepi.plotting.base import UnstratifiedPlotter


class PyramidPlotter(UnstratifiedPlotter):
    """Stateful prevalence back-to-back pyramid visualizer comparing fitted estimators.

    Constructs a symmetrical three-panel horizontal concatenation (`left` reversed
    bar panel, centered category label spine, and `right` bar panel) with a shared
    Y-scale and synchronized X-axis domain. Ideal for contrasting unadjusted vs.
    cluster-adjusted estimates, frequentist vs. Bayesian models, or two isolated
    epidemiological cohorts/strata.

    Args:
        top_n: Maximum number of top categories to display (defaults to 20; None for all).
        y_order: Optional explicit sequence of categories defining Y-axis ordering.
        left_color: Fill color for left bars (defaults to slate blue).
        left_ci_color: Color for left uncertainty interval rules and point markers.
        right_color: Fill color for right bars (defaults to orange).
        right_ci_color: Color for right uncertainty interval rules and point markers.
        width: Width in pixels for each side panel.
        center_width: Width in pixels for the center category label strip.
        height: Height in pixels for the plot.
        spacing: Horizontal spacing between pyramid panels in pixels.
        font: Font family for typography across the chart.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for category and axis labels.
        show_error_bars: Whether to display uncertainty intervals and point markers.
        stroke_width: Stroke width for uncertainty interval rules.
        theme: Active color theme mode (`"light"` or `"dark"`).

    Attributes:
        y_order: Explicit sequence of categories defining Y-axis ordering, or `None`.
        left_color: Hex fill color for the left panel bars.
        left_ci_color: Hex color for the left panel uncertainty intervals.
        right_color: Hex fill color for the right panel bars.
        right_ci_color: Hex color for the right panel uncertainty intervals.
        center_width: Width in pixels for the center category label strip.
        last_y_order_: Resolved list of Y-axis target categories from the most recent `.plot()` call.

    Examples:
        >>> from seroepi.plotting.pyramid import PyramidPlotter
        >>> plotter = PyramidPlotter(top_n=15, left_color="#4c78a8", right_color="#f58518")
        >>> chart = plotter.with_theme("dark").plot(
        ...     unadjusted_est,
        ...     adjusted_est,
        ...     left_label="Unadjusted (Wilson)",
        ...     right_label="Cluster-Adjusted (Bayesian)",
        ... )
        >>> PyramidPlotter.save(chart, "pyramid_comparison.svg")
    """

    def __init__(
        self,
        top_n: int | None = 20,
        y_order: Sequence[str] | None = None,
        left_color: str = "#4c78a8",
        left_ci_color: str = "#1b406b",
        right_color: str = "#f58518",
        right_ci_color: str = "#994207",
        width: int = 260,
        center_width: int = 60,
        height: int = 450,
        spacing: int = 6,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        show_error_bars: bool = True,
        stroke_width: int = 2,
        theme: Literal["light", "dark"] = "light",
    ) -> None:
        """Initializes the prevalence visualizer with custom layout and aesthetics.

        Args:
            top_n: Maximum number of top categories to display (defaults to 20; None for all).
            y_order: Optional explicit sequence of categories defining Y-axis ordering.
            left_color: Fill color for left bars (defaults to slate blue).
            left_ci_color: Color for left uncertainty interval rules and point markers.
            right_color: Fill color for right bars (defaults to orange).
            right_ci_color: Color for right uncertainty interval rules and point markers.
            width: Width in pixels for each side panel.
            center_width: Width in pixels for the center category label strip.
            height: Height in pixels for the plot.
            spacing: Horizontal spacing between pyramid panels in pixels.
            font: Font family for typography across the chart.
            title_font_size: Font size for chart title.
            subtitle_font_size: Font size for chart subtitle.
            label_font_size: Font size for category and axis labels.
            show_error_bars: Whether to display uncertainty intervals and point markers.
            stroke_width: Stroke width for uncertainty interval rules.
            theme: Active color theme mode (`"light"` or `"dark"`).
        """
        super().__init__(
            width=width,
            height=height,
            font=font,
            title_font_size=title_font_size,
            subtitle_font_size=subtitle_font_size,
            label_font_size=label_font_size,
            top_n=top_n,
            show_error_bars=show_error_bars,
            spacing=spacing,
            stroke_width=stroke_width,
            theme=theme,
        )
        self.y_order = y_order
        self.left_color = left_color
        self.left_ci_color = left_ci_color
        self.right_color = right_color
        self.right_ci_color = right_ci_color
        self.center_width = center_width
        self.last_y_order_: list[str] | None = None

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Checks whether at least two fitted estimators are available for back-to-back comparison.

        Args:
            estimator: Primary fitted `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators available for comparison.

        Returns:
            `True` if `estimator` is fitted, has non-null `estimates_`, and `n_estimators >= 2`.
        """
        return bool(estimator.is_fitted_ and estimator.estimates_ is not None and n_estimators >= 2)

    def plot(
        self,
        left: BaseEstimator,
        right: BaseEstimator,
        *,
        left_label: str | None = None,
        right_label: str | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        show_error_bars: bool | None = None,
    ) -> alt.TopLevelMixin:
        """Constructs a back-to-back pyramid comparison chart between two fitted estimators.

        Args:
            left: First fitted `BaseEstimator` rendered on the left panel (e.g., unadjusted estimates).
            right: Second fitted `BaseEstimator` rendered on the right panel and used for default ranking.
            left_label: Custom panel title for the left side (inferred via `contrast_label_with` if omitted).
            right_label: Custom panel title for the right side (inferred via `contrast_label_with` if omitted).
            title: Custom overall chart title.
            subtitle: Custom overall chart subtitle.
            show_error_bars: Whether to display uncertainty intervals and point markers (overrides instance setting).

        Returns:
            Configured Altair `HConcatChart` compound specification with a shared Y-scale.

        Raises:
            TypeError: If either `left` or `right` is not a `BaseEstimator` instance.
            RuntimeError: If either `left` or `right` has not been fitted yet.
            ValueError: If either `left` or `right` contains duplicate target categories (i.e., is stratified).
        """
        self.validate_unstratified(left, "left")
        self.validate_unstratified(right, "right")
        assert left.estimates_ is not None
        assert right.estimates_ is not None

        resolved_left_label, resolved_right_label = left.contrast_label_with(right, left_label, right_label)

        y_col = right.target_col
        y_title = right.target_title
        x_col_left = left.estimate.column
        x_col_right = right.estimate.column
        fmt_left = left.estimate.format
        fmt_right = right.estimate.format
        eff_show_error_bars = self.show_error_bars if show_error_bars is None else show_error_bars

        # 1. Determine Top-Heavy Y Ordering (descending by right estimate by default)
        effective_order = self.resolve_targets(right, self.y_order, self.top_n)
        self.last_y_order_ = effective_order

        # 2. Filter datasets to effective categories
        left_df = left.estimates_.filter(pl.col(y_col).is_in(effective_order))
        right_df = right.estimates_.filter(pl.col(y_col).is_in(effective_order))

        # 3. Calculate Symmetrical X Domain
        left_max = (
            self.series_max(left_df[left.estimate.upper])
            if eff_show_error_bars
            and left.estimate.has_interval
            and left.estimate.upper
            and left.estimate.upper in left_df.columns
            else self.series_max(left_df[x_col_left])
        )
        right_max = (
            self.series_max(right_df[right.estimate.upper])
            if eff_show_error_bars
            and right.estimate.has_interval
            and right.estimate.upper
            and right.estimate.upper in right_df.columns
            else self.series_max(right_df[x_col_right])
        )
        max_val = max(left_max, right_max) * 1.08
        if max_val <= 0.0:
            max_val = 1.0

        # 4. Build Left Chart (Reversed X scale)
        left_tooltips = self.build_tooltips(
            left_df,
            dimensions=[(y_col, y_title)],
            est_col=x_col_left,
            est_title=resolved_left_label,
            fmt=fmt_left,
            ci_prefix=left.estimate.ci_prefix,
            lower_col=left.estimate.lower,
            upper_col=left.estimate.upper,
            include_intervals=left.estimate.has_interval,
        )
        left_layers: list[alt.Chart] = [
            alt.Chart(left_df)
            .encode(
                y=alt.Y(f"{y_col}:N", sort=effective_order, axis=None),
                x=alt.X(
                    f"{x_col_left}:Q",
                    title=resolved_left_label,
                    scale=alt.Scale(domain=[0, max_val], reverse=True),
                    axis=alt.Axis(format=fmt_left),
                ),
                tooltip=left_tooltips,
            )
            .mark_bar(color=self.left_color, opacity=0.75)
        ]
        if eff_show_error_bars and left.estimate.has_interval and left.estimate.lower and left.estimate.upper:
            left_layers.extend(
                self.build_error_bar_layers(
                    left_df,
                    cat_col=y_col,
                    cat_sort=effective_order,
                    orientation="horizontal",
                    color=self.resolve_ci_color(self.left_ci_color, "#1b406b"),
                    lower_col=left.estimate.lower,
                    upper_col=left.estimate.upper,
                    est_col=x_col_left,
                    stroke_width=self.stroke_width,
                    cap_style="point",
                    cap_size=25,
                )
            )

        chart_left = alt.layer(*left_layers).properties(
            title=alt.TitleParams(resolved_left_label, anchor="middle"),
            width=self.width,
            height=self.height,
        )

        # 5. Build Center Column (Category Labels)
        chart_center = (
            alt.Chart(right_df)
            .encode(
                y=alt.Y(f"{y_col}:N", sort=effective_order, axis=None),
                text=alt.Text(f"{y_col}:N"),
            )
            .mark_text(fontWeight="bold", fontSize=self.label_font_size + 1, color=self.palette.text_primary)
            .properties(
                title=alt.TitleParams(y_title, anchor="middle"),
                width=self.center_width,
                height=self.height,
            )
        )

        # 6. Build Right Chart (Standard X scale)
        right_tooltips = self.build_tooltips(
            right_df,
            dimensions=[(y_col, y_title)],
            est_col=x_col_right,
            est_title=resolved_right_label,
            fmt=fmt_right,
            ci_prefix=right.estimate.ci_prefix,
            lower_col=right.estimate.lower,
            upper_col=right.estimate.upper,
            include_intervals=right.estimate.has_interval,
        )
        right_layers: list[alt.Chart] = [
            alt.Chart(right_df)
            .encode(
                y=alt.Y(f"{y_col}:N", sort=effective_order, axis=None),
                x=alt.X(
                    f"{x_col_right}:Q",
                    title=resolved_right_label,
                    scale=alt.Scale(domain=[0, max_val]),
                    axis=alt.Axis(format=fmt_right),
                ),
                tooltip=right_tooltips,
            )
            .mark_bar(color=self.right_color, opacity=0.75)
        ]
        if eff_show_error_bars and right.estimate.has_interval and right.estimate.lower and right.estimate.upper:
            right_layers.extend(
                self.build_error_bar_layers(
                    right_df,
                    cat_col=y_col,
                    cat_sort=effective_order,
                    orientation="horizontal",
                    color=self.resolve_ci_color(self.right_ci_color, "#994207"),
                    lower_col=right.estimate.lower,
                    upper_col=right.estimate.upper,
                    est_col=x_col_right,
                    stroke_width=self.stroke_width,
                    cap_style="point",
                    cap_size=25,
                )
            )

        chart_right = alt.layer(*right_layers).properties(
            title=alt.TitleParams(resolved_right_label, anchor="middle"),
            width=self.width,
            height=self.height,
        )

        # 7. Dynamic Titles & Subtitles
        resolved_title = title or f"{y_title} Estimates: {resolved_left_label} vs {resolved_right_label}"
        if subtitle is not None:
            resolved_subtitle = subtitle
        else:
            ordering_text = f"Ordered descending by {resolved_right_label}"
            left_interval = left.estimate.interval_description
            right_interval = right.estimate.interval_description

            if left_interval and right_interval:
                if left_interval == right_interval:
                    resolved_subtitle = f"{ordering_text} with {left_interval}"
                else:
                    resolved_subtitle = f"{ordering_text} | Left: {left_interval} | Right: {right_interval}"
            elif left_interval:
                resolved_subtitle = f"{ordering_text} (Left: {left_interval})"
            elif right_interval:
                resolved_subtitle = f"{ordering_text} (Right: {right_interval})"
            else:
                resolved_subtitle = ordering_text

        # 8. Combine Panels
        compound = (
            alt.hconcat(chart_left, chart_center, chart_right, spacing=self.spacing)
            .resolve_scale(y="shared")
            .properties(
                title=alt.TitleParams(
                    text=resolved_title,
                    subtitle=resolved_subtitle,
                    anchor="middle",
                )
            )
        )
        return self.configure_chart(compound)
