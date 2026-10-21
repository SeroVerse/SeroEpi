"""Single-estimator unstratified horizontal bar & uncertainty interval visualizer."""

from collections.abc import Sequence
from typing import Literal

import altair as alt
import polars as pl

from seroepi.estimators.base import BaseEstimator
from seroepi.plotting.base import UnstratifiedPlotter


class BarPlotter(UnstratifiedPlotter):
    """Stateful horizontal bar and error-bar visualizer for unstratified prevalence estimates.

    Renders ranked horizontal prevalence bars with optional confidence or credible
    interval rules and point markers for an unstratified (or single-stratum-filtered)
    `BaseEstimator`. Automatically scales the X-axis domain to accommodate upper
    uncertainty bounds and populates rich hover tooltips with event counts, sample
    sizes, and interval bounds.

    Args:
        top_n: Maximum number of top target categories to display (defaults to 20; None for all).
        y_order: Optional explicit sequence of target categories defining Y-axis ordering.
        bar_color: Hex fill color for horizontal prevalence bars.
        ci_color: Hex color for uncertainty interval rules and point markers.
        width: Chart width in pixels.
        height: Chart height in pixels.
        font: Font family for typography across the chart.
        title_font_size: Font size for chart title.
        subtitle_font_size: Font size for chart subtitle.
        label_font_size: Font size for category and axis labels.
        show_error_bars: Whether to display uncertainty intervals and point markers.
        stroke_width: Stroke width for uncertainty interval rules.
        theme: Active color theme mode (`"light"` or `"dark"`).

    Attributes:
        y_order: Explicit sequence of target categories defining Y-axis ordering, or `None`.
        bar_color: Hex fill color for horizontal prevalence bars.
        ci_color: Hex color for uncertainty interval rules and point markers.

    Examples:
        >>> from seroepi.plotting.bar import BarPlotter
        >>> plotter = BarPlotter(top_n=15, bar_color="#4c78a8")
        >>> chart = plotter.with_theme("dark").plot(
        ...     estimator,
        ...     title="Top 15 K-Locus Prevalence",
        ... )
        >>> BarPlotter.save(chart, "k_locus_bar.svg")
    """

    def __init__(
        self,
        top_n: int | None = 20,
        y_order: Sequence[str] | None = None,
        bar_color: str = "#4c78a8",
        ci_color: str = "#1b406b",
        width: int = 480,
        height: int = 420,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        show_error_bars: bool = True,
        stroke_width: float = 1.8,
        theme: Literal["light", "dark"] = "light",
    ) -> None:
        """Initializes the horizontal prevalence bar visualizer.

        Args:
            top_n: Maximum number of top target categories to display (defaults to 20; None for all).
            y_order: Optional explicit sequence of target categories defining Y-axis ordering.
            bar_color: Hex fill color for horizontal prevalence bars.
            ci_color: Hex color for uncertainty interval rules and point markers.
            width: Chart width in pixels.
            height: Chart height in pixels.
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
            stroke_width=stroke_width,
            theme=theme,
        )
        self.y_order = y_order
        self.bar_color = bar_color
        self.ci_color = ci_color

    def plot(
        self,
        estimator: BaseEstimator,
        *,
        title: str | None = None,
        subtitle: str | None = None,
        show_error_bars: bool | None = None,
    ) -> alt.TopLevelMixin:
        """Renders a horizontal bar chart with optional uncertainty intervals for an unstratified estimator.

        Args:
            estimator: Fitted unstratified (or stratum-isolated/pooled) `BaseEstimator`.
            title: Custom chart title overriding the default `"{target_title} Prevalence"`.
            subtitle: Custom chart subtitle overriding the automatic estimator summary.
            show_error_bars: Overrides instance `show_error_bars` setting if provided.

        Returns:
            Configured Altair `LayerChart` containing horizontal bars and optional error bars.

        Raises:
            TypeError: If `estimator` is not a `BaseEstimator` instance.
            RuntimeError: If `estimator` has not been fitted yet.
            ValueError: If `estimator.estimates_` contains duplicate target categories (i.e., is stratified).
        """
        self.validate_unstratified(estimator)
        assert estimator.estimates_ is not None

        y_col = estimator.target_col
        y_title = estimator.target_title
        x_col = estimator.estimate.column
        x_title = estimator.estimate.title
        fmt = self.format_ or estimator.estimate.format
        eff_show_error_bars = self.show_error_bars if show_error_bars is None else show_error_bars

        effective_order = self.resolve_targets(estimator, self.y_order, self.top_n)
        df = estimator.estimates_.filter(pl.col(y_col).is_in(effective_order))

        max_bound = (
            self.series_max(df[estimator.estimate.upper])
            if eff_show_error_bars
            and estimator.estimate.has_interval
            and estimator.estimate.upper
            and estimator.estimate.upper in df.columns
            else self.series_max(df[x_col])
        )
        max_val = max_bound * 1.08 if max_bound > 0.0 else 1.0

        tooltips = self.build_tooltips(
            df,
            dimensions=[(y_col, y_title)],
            est_col=x_col,
            est_title=x_title,
            fmt=fmt,
            ci_prefix=estimator.estimate.ci_prefix,
            lower_col=estimator.estimate.lower,
            upper_col=estimator.estimate.upper,
            include_intervals=estimator.estimate.has_interval,
        )

        layers: list[alt.Chart] = [
            alt.Chart(df)
            .encode(
                y=alt.Y(f"{y_col}:N", sort=effective_order, title=y_title),
                x=alt.X(
                    f"{x_col}:Q",
                    title=x_title,
                    scale=alt.Scale(domain=[0, max_val]),
                    axis=alt.Axis(format=fmt),
                ),
                tooltip=tooltips,
            )
            .mark_bar(color=self.bar_color, opacity=0.8)
        ]

        if (
            eff_show_error_bars
            and estimator.estimate.has_interval
            and estimator.estimate.lower
            and estimator.estimate.upper
        ):
            layers.extend(
                self.build_error_bar_layers(
                    df,
                    cat_col=y_col,
                    cat_sort=effective_order,
                    orientation="horizontal",
                    color=self.resolve_ci_color(self.ci_color),
                    lower_col=estimator.estimate.lower,
                    upper_col=estimator.estimate.upper,
                    est_col=x_col,
                    stroke_width=self.stroke_width,
                    cap_style="point",
                    cap_size=25,
                )
            )

        resolved_title = title or f"{y_title} Prevalence"
        stratum_val = estimator.infer_single_stratum_value()
        adj_str = f"Adjusted for {estimator.adjust_for_title}" if estimator.adjust_for_ is not None else "Unadjusted"
        default_sub_parts = [estimator.display_name, adj_str, estimator.describe_interval()]
        if stratum_val:
            default_sub_parts.insert(0, f"{estimator.single_stratum_title}: {stratum_val}")
        resolved_subtitle = subtitle if subtitle is not None else " • ".join(default_sub_parts)

        chart = alt.layer(*layers).properties(
            title=alt.TitleParams(
                text=resolved_title,
                subtitle=resolved_subtitle,
                anchor="start",
            ),
            width=self.width,
            height=self.height,
        )
        return self.configure_chart(chart)
