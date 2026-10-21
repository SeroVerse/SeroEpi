"""Stateful visualization engine and epidemiological visualizers powered by Altair."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import altair as alt
import polars as pl

from seroepi.estimators.base import BaseEstimator


@dataclass(frozen=True, slots=True)
class PlotThemePalette:
    """Color palette specification for light and dark UI themes.

    Defines coordinated foreground typography, gridline, axis domain, uncertainty
    interval, and canvas background colors for Altair epidemiological charts.

    Args:
        text_primary: Primary hex or CSS color for titles and axis headers.
        text_secondary: Secondary hex or CSS color for subtitles, tick labels, and legend labels.
        grid_color: CSS color (typically translucent RGBA) for background grid rules.
        domain_color: CSS color for axis domain baselines and tick marks.
        error_bar_color: Default hex color for confidence or credible interval rules and caps.
        background: Canvas background fill color (defaults to `"transparent"`).

    Attributes:
        text_primary: Primary hex or CSS color for titles and axis headers.
        text_secondary: Secondary hex or CSS color for subtitles, tick labels, and legend labels.
        grid_color: CSS color for background grid rules.
        domain_color: CSS color for axis domain baselines and tick marks.
        error_bar_color: Default hex color for confidence or credible interval rules and caps.
        background: Canvas background fill color.
    """

    text_primary: str
    text_secondary: str
    grid_color: str
    domain_color: str
    error_bar_color: str
    background: str = "transparent"


THEME_PALETTES: dict[Literal["light", "dark"], PlotThemePalette] = {
    "light": PlotThemePalette(
        text_primary="#0f172a",
        text_secondary="#475569",
        grid_color="rgba(15, 23, 42, 0.08)",
        domain_color="rgba(15, 23, 42, 0.25)",
        error_bar_color="#1e293b",
    ),
    "dark": PlotThemePalette(
        text_primary="#f8fafc",
        text_secondary="#94a3b8",
        grid_color="rgba(255, 255, 255, 0.08)",
        domain_color="rgba(255, 255, 255, 0.25)",
        error_bar_color="#e2e8f0",
    ),
}


@dataclass(frozen=True, slots=True)
class SingleStratumContext:
    """Pre-computed column names, display titles, and format metadata extracted from a single-stratum estimator.

    Provides a unified, immutable lookup structure used by `SingleStratumPlotter`
    subclasses (`HeatmapPlotter`, `ChoroplethPlotter`, `UpSetPlotter`) to bind
    target, stratum, and prevalence estimate columns consistently across Altair encodings.

    Args:
        target_col: Name of the target category column in `estimator.estimates_`.
        target_title: Human-readable display title for the target variable.
        stratum_col: Name of the single stratification column in `estimator.estimates_`.
        stratum_title: Human-readable display title for the stratification variable.
        est_col: Name of the point-estimate column (e.g., `"prevalence"`).
        est_title: Human-readable display title for the point estimate.
        format: D3/Vega numeric format string for displaying prevalence values (e.g., `".1%"`).
        ci_prefix: Short label prefix for uncertainty interval bounds (e.g., `"95% CI"` or `"95% CrI"`).

    Attributes:
        target_col: Name of the target category column in `estimator.estimates_`.
        target_title: Human-readable display title for the target variable.
        stratum_col: Name of the single stratification column in `estimator.estimates_`.
        stratum_title: Human-readable display title for the stratification variable.
        est_col: Name of the point-estimate column.
        est_title: Human-readable display title for the point estimate.
        format: D3/Vega numeric format string for displaying prevalence values.
        ci_prefix: Short label prefix for uncertainty interval bounds.
    """

    target_col: str
    target_title: str
    stratum_col: str
    stratum_title: str
    est_col: str
    est_title: str
    format: str
    ci_prefix: str


class BasePlotter(ABC):
    """Abstract base class for stateful epidemiological visualizers.

    Encapsulates shared visual aesthetics (typography, canvas dimensions, color
    schemes, uncertainty interval toggles, and light/dark theme palettes) alongside
    reusable Altair layer builders, target ranking resolvers, and export utilities.

    Args:
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family applied to titles, subtitles, axis labels, and legends.
        title_font_size: Font size in pixels for the main chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for axis ticks, axis titles, and legend labels.
        color_scheme: Default Vega categorical or sequential color scheme name.
        x_label_angle: Rotation angle in degrees for X-axis tick labels.
        top_n: Maximum number of top-ranked target categories to display (`None` for all).
        format_: Optional D3 numeric format string overriding the estimator's default format.
        show_error_bars: Whether to render confidence or credible interval layers when available.
        spacing: Pixel spacing between panels in multi-panel concatenated charts.
        stroke_width: Default stroke width in pixels for lines, borders, or interval rules.
        theme: Active UI color theme mode (`"light"` or `"dark"`).

    Attributes:
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family applied across chart typography.
        title_font_size: Font size in pixels for the main chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for axis and legend labels.
        color_scheme: Default Vega color scheme name.
        x_label_angle: Rotation angle in degrees for X-axis tick labels.
        top_n: Maximum number of top-ranked target categories to display.
        format_: Optional D3 numeric format string override.
        show_error_bars: Whether to render uncertainty interval layers when available.
        spacing: Pixel spacing between panels in compound charts.
        stroke_width: Default stroke width in pixels for rules and borders.
        theme: Active UI color theme mode (`"light"` or `"dark"`).
    """

    def __init__(
        self,
        width: int = 400,
        height: int = 350,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        color_scheme: str = "tableau10",
        x_label_angle: int = -45,
        top_n: int | None = 20,
        format_: str | None = None,
        show_error_bars: bool = True,
        spacing: int = 6,
        stroke_width: float = 1.0,
        theme: Literal["light", "dark"] = "light",
    ) -> None:
        """Initializes shared visual aesthetics and layout configuration for the plotter.

        Args:
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family applied to titles, subtitles, axis labels, and legends.
            title_font_size: Font size in pixels for the main chart title.
            subtitle_font_size: Font size in pixels for the chart subtitle.
            label_font_size: Font size in pixels for axis ticks, axis titles, and legend labels.
            color_scheme: Default Vega categorical or sequential color scheme name.
            x_label_angle: Rotation angle in degrees for X-axis tick labels.
            top_n: Maximum number of top-ranked target categories to display (`None` for all).
            format_: Optional D3 numeric format string overriding the estimator's default format.
            show_error_bars: Whether to render confidence or credible interval layers when available.
            spacing: Pixel spacing between panels in multi-panel concatenated charts.
            stroke_width: Default stroke width in pixels for lines, borders, or interval rules.
            theme: Active UI color theme mode (`"light"` or `"dark"`).
        """
        self.width = width
        self.height = height
        self.font = font
        self.title_font_size = title_font_size
        self.subtitle_font_size = subtitle_font_size
        self.label_font_size = label_font_size
        self.color_scheme = color_scheme
        self.x_label_angle = x_label_angle
        self.top_n = top_n
        self.format_ = format_
        self.show_error_bars = show_error_bars
        self.spacing = spacing
        self.stroke_width = stroke_width
        self.theme: Literal["light", "dark"] = "dark" if theme == "dark" else "light"

    @property
    def palette(self) -> PlotThemePalette:
        """Active theme color palette corresponding to `self.theme`.

        Returns:
            The `PlotThemePalette` instance for the current `"light"` or `"dark"` theme.
        """
        return THEME_PALETTES.get(self.theme, THEME_PALETTES["light"])

    def with_theme(self, theme: Literal["light", "dark"]) -> Self:
        """Fluently updates the active UI color theme on this plotter instance.

        Args:
            theme: Target UI color theme mode (`"light"` or `"dark"`).

        Returns:
            The mutated plotter instance (`self`) for method chaining.
        """
        self.theme = "dark" if theme == "dark" else "light"
        return self

    def resolve_ci_color(self, explicit_color: str | None, default_light_hex: str = "#1b406b") -> str:
        """Resolves an uncertainty interval stroke color that remains legible in both light and dark modes.

        When `explicit_color` is `None`, or when `self.theme == "dark"` and `explicit_color`
        still matches the default dark-navy light-theme hex (`default_light_hex`), this method
        substitutes `self.palette.error_bar_color` so error bars remain high-contrast on dark canvases.

        Args:
            explicit_color: User-configured or instance-level hex color string, or `None`.
            default_light_hex: Default light-mode hex color to replace automatically in dark mode.

        Returns:
            Resolved CSS/hex color string for rendering uncertainty interval marks.
        """
        if explicit_color is None:
            return self.palette.error_bar_color
        if self.theme == "dark" and explicit_color.lower() == default_light_hex.lower():
            return self.palette.error_bar_color
        return explicit_color

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> alt.TopLevelMixin:
        """Renders the primary Altair chart visualization.

        Args:
            *args: Positional arguments specific to the concrete plotter subclass (e.g., fitted estimators).
            **kwargs: Keyword overrides specific to the concrete plotter subclass (e.g., `title`, `subtitle`).

        Returns:
            Configured top-level Altair chart specification (`alt.Chart`, `alt.LayerChart`, or compound chart).
        """

    def resolve_targets(
        self,
        estimator: BaseEstimator,
        y_order: Sequence[str] | None = None,
        top_n: int | None = None,
    ) -> list[str]:
        """Resolves the ordered list of target categories to display from explicit ordering or estimator rankings.

        Args:
            estimator: Fitted `BaseEstimator` providing ranked target categories when `y_order` is omitted.
            y_order: Optional explicit sequence of target category names defining display order.
            top_n: Optional maximum number of targets to retain (falls back to `self.top_n` when `None`).

        Returns:
            Ordered list of target category strings truncated to the effective `top_n` limit.
        """
        effective_top_n = self.top_n if top_n is None else top_n
        if y_order is not None:
            ordered = [str(t) for t in y_order]
            if effective_top_n is not None and effective_top_n > 0:
                return ordered[:effective_top_n]
            return ordered
        return estimator.ranked_targets(effective_top_n)

    @staticmethod
    def build_tooltips(
        df: pl.DataFrame,
        *,
        dimensions: Sequence[tuple[str, str]],
        est_col: str,
        est_title: str,
        fmt: str,
        ci_prefix: str = "CI",
        lower_col: str | None = "lower",
        upper_col: str | None = "upper",
        event_title: str = "Events",
        include_counts: bool = True,
        include_intervals: bool = True,
    ) -> list[alt.Tooltip]:
        """Constructs standardized Altair hover tooltips for epidemiological prevalence charts.

        Args:
            df: Source Polars `DataFrame` inspected for optional count (`"event"`, `"n"`) and bound columns.
            dimensions: Sequence of `(column_name, display_title)` pairs for nominal stratification/target fields.
            est_col: Quantitative point-estimate column name in `df`.
            est_title: Display title for the point estimate in the tooltip.
            fmt: D3 numeric format string applied to the point estimate and interval bounds.
            ci_prefix: Label prefix for uncertainty interval bounds (e.g., `"95% CI"` or `"95% CrI"`).
            lower_col: Column name for the lower uncertainty bound, or `None` if unavailable.
            upper_col: Column name for the upper uncertainty bound, or `None` if unavailable.
            event_title: Display title for the `"event"` count column when present.
            include_counts: Whether to append `"event"` and `"n"` sample size tooltips if present in `df`.
            include_intervals: Whether to append lower and upper uncertainty bound tooltips if present in `df`.

        Returns:
            Ordered list of `alt.Tooltip` encodings ready to bind to an Altair mark.
        """
        tips: list[alt.Tooltip] = [alt.Tooltip(f"{col}:N", title=title) for col, title in dimensions]
        tips.append(alt.Tooltip(f"{est_col}:Q", format=fmt, title=est_title))
        if include_counts:
            if "event" in df.columns:
                tips.append(alt.Tooltip("event:Q", format=",.0f", title=event_title))
            if "n" in df.columns:
                tips.append(alt.Tooltip("n:Q", format=",.0f", title="Total (n)"))
        if include_intervals and lower_col and upper_col and lower_col in df.columns and upper_col in df.columns:
            tips.append(alt.Tooltip(f"{lower_col}:Q", format=fmt, title=f"{ci_prefix} Lower"))
            tips.append(alt.Tooltip(f"{upper_col}:Q", format=fmt, title=f"{ci_prefix} Upper"))
        return tips

    @staticmethod
    def build_error_bar_layers(
        df: pl.DataFrame,
        *,
        cat_col: str,
        cat_sort: Sequence[str],
        orientation: Literal["horizontal", "vertical"],
        color: str | None = None,
        lower_col: str = "lower",
        upper_col: str = "upper",
        est_col: str | None = None,
        stroke_width: float = 1.5,
        cap_style: Literal["tick", "point", "none"] = "tick",
        cap_size: int = 10,
    ) -> list[alt.Chart]:
        """Constructs Altair rule and tick/point layers representing uncertainty intervals.

        Args:
            df: Source Polars `DataFrame` containing category and interval bound columns.
            cat_col: Nominal category column name for positioning each error bar.
            cat_sort: Explicit ordering sequence of category values along the categorical axis.
            orientation: Direction of the interval rules (`"horizontal"` along X or `"vertical"` along Y).
            color: Hex or CSS stroke/fill color for the interval rule and caps (defaults to light palette error bar color).
            lower_col: Column name for the lower uncertainty bound in `df`.
            upper_col: Column name for the upper uncertainty bound in `df`.
            est_col: Optional point-estimate column name required when `cap_style="point"`.
            stroke_width: Stroke thickness in pixels for the interval rules and tick caps.
            cap_style: Visual marker style at interval endpoints or center (`"tick"`, `"point"`, or `"none"`).
            cap_size: Size in pixels for tick caps or center point markers.

        Returns:
            List of `alt.Chart` layers (empty if `lower_col` or `upper_col` is absent from `df`).
        """
        if lower_col not in df.columns or upper_col not in df.columns:
            return []
        eff_color = color if color is not None else THEME_PALETTES["light"].error_bar_color
        sort_list = list(cat_sort)
        base = alt.Chart(df)
        layers: list[alt.Chart] = []
        if orientation == "vertical":
            x_enc = alt.X(f"{cat_col}:N", sort=sort_list)
            layers.append(
                base.encode(x=x_enc, y=alt.Y(f"{lower_col}:Q"), y2=alt.Y2(f"{upper_col}:Q")).mark_rule(
                    color=eff_color, strokeWidth=stroke_width
                )
            )
            if cap_style == "tick":
                layers.extend(
                    [
                        base.encode(x=x_enc, y=alt.Y(f"{lower_col}:Q")).mark_tick(
                            color=eff_color, size=cap_size, thickness=stroke_width
                        ),
                        base.encode(x=x_enc, y=alt.Y(f"{upper_col}:Q")).mark_tick(
                            color=eff_color, size=cap_size, thickness=stroke_width
                        ),
                    ]
                )
            elif cap_style == "point" and est_col:
                layers.append(
                    base.encode(x=x_enc, y=alt.Y(f"{est_col}:Q")).mark_point(
                        color=eff_color, filled=True, size=cap_size
                    )
                )
        else:
            y_enc = alt.Y(f"{cat_col}:N", sort=sort_list)
            layers.append(
                base.encode(y=y_enc, x=alt.X(f"{lower_col}:Q"), x2=alt.X2(f"{upper_col}:Q")).mark_rule(
                    color=eff_color, strokeWidth=stroke_width
                )
            )
            if cap_style == "tick":
                layers.extend(
                    [
                        base.encode(y=y_enc, x=alt.X(f"{lower_col}:Q")).mark_tick(
                            color=eff_color, size=cap_size, thickness=stroke_width
                        ),
                        base.encode(y=y_enc, x=alt.X(f"{upper_col}:Q")).mark_tick(
                            color=eff_color, size=cap_size, thickness=stroke_width
                        ),
                    ]
                )
            elif cap_style == "point" and est_col:
                layers.append(
                    base.encode(y=y_enc, x=alt.X(f"{est_col}:Q")).mark_point(
                        color=eff_color, filled=True, size=cap_size
                    )
                )
        return layers

    def configure_chart(self, chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
        """Applies consistent typography, transparent canvas, and theme-aware palette styling to an Altair chart.

        Args:
            chart: Unconfigured top-level Altair chart (`alt.Chart`, `alt.LayerChart`, `alt.HConcatChart`, or `alt.VConcatChart`).

        Returns:
            The Altair chart with global `.configure(...)`, `.configure_view(...)`, `.configure_title(...)`,
            `.configure_axis(...)`, and `.configure_legend(...)` properties applied.
        """
        pal = self.palette
        return (
            chart.configure(background=pal.background)
            .configure_view(stroke=None)
            .configure_title(
                font=self.font,
                fontSize=self.title_font_size,
                color=pal.text_primary,
                subtitleFont=self.font,
                subtitleFontSize=self.subtitle_font_size,
                subtitleColor=pal.text_secondary,
            )
            .configure_axis(
                labelFont=self.font,
                labelFontSize=self.label_font_size,
                labelColor=pal.text_secondary,
                titleFont=self.font,
                titleFontSize=self.label_font_size,
                titleColor=pal.text_primary,
                gridColor=pal.grid_color,
                domainColor=pal.domain_color,
                tickColor=pal.domain_color,
            )
            .configure_legend(
                labelFont=self.font,
                labelFontSize=self.label_font_size,
                labelColor=pal.text_secondary,
                titleFont=self.font,
                titleFontSize=self.label_font_size,
                titleColor=pal.text_primary,
            )
        )

    @staticmethod
    def series_max(series: pl.Series) -> float:
        """Safely extracts the maximum scalar value as a `float` from a numeric Polars `Series`.

        Args:
            series: Numeric Polars `Series` whose maximum value is queried.

        Returns:
            Maximum numeric value as a Python `float`, or `0.0` if the series is empty or all-null.
        """
        val = series.max()
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    @staticmethod
    def fit_to_container(
        chart: alt.TopLevelMixin,
        width: float | None = None,
        height: float | None = None,
    ) -> alt.TopLevelMixin:
        """Adapts an Altair chart's dimensions to responsively fill its parent web container.

        Args:
            chart: Top-level Altair chart specification to resize.
            width: Optional explicit container width in pixels (`None` for responsive `"container"` sizing).
            height: Optional explicit container height in pixels (`None` to preserve or adapt chart height).

        Returns:
            Modified Altair chart configured for container-responsive rendering.
        """
        eff_w = float(width) if isinstance(width, (int, float)) and width > 150 else 920.0
        eff_h = float(height) if isinstance(height, (int, float)) and height > 150 else 640.0

        if isinstance(chart, (alt.Chart, alt.LayerChart)):
            if getattr(chart, "projection", alt.Undefined) is not alt.Undefined:
                target_w = max(int(eff_w - 110), 420)
                target_h = max(int(eff_h - 160), 320)
                old_w = float(chart.width) if isinstance(chart.width, (int, float)) and chart.width > 0 else 680.0
                old_h = float(chart.height) if isinstance(chart.height, (int, float)) and chart.height > 0 else 440.0
                scale_ratio = min(target_w / old_w, target_h / old_h)
                c = chart.copy()
                c.width = target_w
                c.height = target_h
                proj_dict = c.projection.to_dict()
                proj_dict["clipExtent"] = [[0, 0], [target_w, target_h]]
                if isinstance(proj_dict.get("scale"), (int, float)):
                    proj_dict["scale"] = round(float(proj_dict["scale"]) * scale_ratio, 0)
                c.projection = alt.Projection(**proj_dict)
                if getattr(c, "params", alt.Undefined) is not alt.Undefined:
                    for p in c.params:
                        pname = p.name.to_json().strip('"') if hasattr(p.name, "to_json") else str(p.name)
                        if pname.startswith("map_zoom") and isinstance(getattr(p, "value", None), (int, float)):
                            p.value = round(float(p.value) * scale_ratio, 0)
                return c

            return chart.properties(
                width="container",
                height="container",
                autosize=alt.AutoSizeParams(type="fit", contains="padding"),
            )

        if isinstance(chart, alt.HConcatChart):
            avail_w = max(int(eff_w - 70), 420)
            avail_h = max(int(eff_h - 120), 320)
            c = chart.copy()
            subs = list(c.hconcat)
            widths = [
                float(getattr(s, "width", 200)) if isinstance(getattr(s, "width", None), (int, float)) else 200.0
                for s in subs
            ]
            fixed_w = sum(w for w in widths if w <= 90)
            flex_orig = sum(w for w in widths if w > 90)
            spacing = float(c.spacing) if isinstance(getattr(c, "spacing", None), (int, float)) else 6.0
            rem_w = max(avail_w - fixed_w - spacing * max(len(subs) - 1, 0), 200.0)
            for s, w in zip(subs, widths, strict=False):
                if w > 90 and flex_orig > 0:
                    s.width = max(int(round(rem_w * (w / flex_orig))), 150)
                s.height = avail_h
            return c

        if isinstance(chart, alt.VConcatChart):
            avail_w = max(int(eff_w - 130), 420)
            avail_h = max(int(eff_h - 145), 340)
            c = chart.copy()
            rows = list(c.vconcat)

            def _row_h(r: Any) -> float:
                if isinstance(r, alt.HConcatChart) and list(r.hconcat):
                    h0 = getattr(list(r.hconcat)[0], "height", None)
                    return float(h0) if isinstance(h0, (int, float)) else 200.0
                rh = getattr(r, "height", None)
                return float(rh) if isinstance(rh, (int, float)) else 120.0

            def _row_w(r: Any) -> float:
                if isinstance(r, alt.HConcatChart) and list(r.hconcat):
                    return sum(
                        float(getattr(s, "width", 200))
                        if isinstance(getattr(s, "width", None), (int, float))
                        else 200.0
                        for s in r.hconcat
                    )
                rw = getattr(r, "width", None)
                return float(rw) if isinstance(rw, (int, float)) else 400.0

            orig_hs = [_row_h(r) for r in rows]
            total_orig_h = sum(orig_hs) or 400.0
            max_orig_w = max((_row_w(r) for r in rows), default=500.0) or 500.0
            w_ratio = avail_w / max_orig_w
            h_ratio = avail_h / total_orig_h

            for r in rows:
                if isinstance(r, alt.HConcatChart):
                    for s in r.hconcat:
                        sw = getattr(s, "width", None)
                        sh = getattr(s, "height", None)
                        if isinstance(sw, (int, float)):
                            s.width = max(int(round(float(sw) * w_ratio)), 80)
                        if isinstance(sh, (int, float)):
                            s.height = max(int(round(float(sh) * h_ratio)), 80)
                else:
                    rw = getattr(r, "width", None)
                    rh = getattr(r, "height", None)
                    if isinstance(rw, (int, float)):
                        r.width = max(int(round(float(rw) * w_ratio)), 120)
                    if isinstance(rh, (int, float)):
                        r.height = max(int(round(float(rh) * h_ratio)), 80)
            return c

        return chart

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Determines whether this visualizer class is compatible with the given fitted estimator state.

        Args:
            estimator: Candidate `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators currently available in the workspace or comparison.

        Returns:
            `True` if `estimator` is fitted, has non-null `estimates_`, and satisfies the plotter's structural requirements.
        """
        return bool(estimator.is_fitted_ and estimator.estimates_ is not None)


class SingleStratumPlotter(BasePlotter):
    """Abstract base class for visualizers operating on 2D target-by-stratum matrices.

    Enforces that the input `BaseEstimator` was fitted with exactly one stratification
    dimension (`estimator.is_single_stratum`) and provides `extract_context` to unpack
    target, stratum, and prevalence metadata into a `SingleStratumContext`.

    Args:
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family applied to titles, subtitles, axis labels, and legends.
        title_font_size: Font size in pixels for the main chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for axis ticks, axis titles, and legend labels.
        color_scheme: Default Vega categorical or sequential color scheme name.
        x_label_angle: Rotation angle in degrees for X-axis tick labels.
        top_n: Maximum number of top-ranked target categories to display (`None` for all).
        format_: Optional D3 numeric format string overriding the estimator's default format.
        show_error_bars: Whether to render confidence or credible interval layers when available.
        spacing: Pixel spacing between panels in multi-panel concatenated charts.
        stroke_width: Default stroke width in pixels for lines, borders, or interval rules.
        theme: Active UI color theme mode (`"light"` or `"dark"`).
    """

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Checks whether `estimator` is fitted and stratified by exactly one stratum dimension.

        Args:
            estimator: Candidate `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators currently available.

        Returns:
            `True` if `estimator` is fitted and `estimator.is_single_stratum` is `True`.
        """
        return super().can_plot(estimator, n_estimators=n_estimators) and estimator.is_single_stratum

    def extract_context(
        self,
        estimator: BaseEstimator,
        *,
        override_format: str | None = None,
    ) -> SingleStratumContext:
        """Validates the single-stratum invariant and extracts column and formatting metadata.

        Args:
            estimator: Fitted `BaseEstimator` expected to have exactly one stratification variable.
            override_format: Optional D3 format string overriding both `self.format_` and `estimator.estimate.format`.

        Returns:
            Populated `SingleStratumContext` containing target, stratum, and estimate column names and titles.

        Raises:
            TypeError: If `estimator` is not a `BaseEstimator` instance.
            RuntimeError: If `estimator` has not been fitted yet.
            ValueError: If `estimator.estimates_` is `None` or `estimator` does not have exactly 1 stratum.
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(
                f"Expected a fitted BaseEstimator instance, got {type(estimator).__name__}. "
                "Fit an estimator to an Aggregator first."
            )
        estimator.check_is_fitted()
        if estimator.estimates_ is None:
            raise ValueError("Estimator estimates_ DataFrame is None.")

        if not estimator.is_single_stratum:
            if estimator.n_strata == 0:
                raise ValueError(
                    f"{self.__class__.__name__} strictly requires an estimator fitted with exactly 1 stratum "
                    f"(got 0 strata). To plot unstratified estimates, use PyramidPlotter."
                )
            raise ValueError(
                f"{self.__class__.__name__} strictly requires an estimator fitted with exactly 1 stratum, "
                f"got {estimator.n_strata}: {list(estimator.strata_cols)}. "
                "Filter or aggregate your dataset with exactly one stratum before plotting."
            )

        return SingleStratumContext(
            target_col=estimator.target_col,
            target_title=estimator.target_title,
            stratum_col=estimator.single_stratum_col,
            stratum_title=estimator.single_stratum_title,
            est_col=estimator.estimate.column,
            est_title=estimator.estimate.title,
            format=override_format or self.format_ or estimator.estimate.format,
            ci_prefix=estimator.estimate.ci_prefix,
        )


class UnstratifiedPlotter(BasePlotter):
    """Abstract base class for visualizers operating on 1D unstratified target categories.

    Enforces that each target category appears at most once in `estimator.estimates_`
    (either from an unstratified estimator or after isolating a single stratum via
    `.filter()` or `.split()`).

    Args:
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family applied to titles, subtitles, axis labels, and legends.
        title_font_size: Font size in pixels for the main chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for axis ticks, axis titles, and legend labels.
        color_scheme: Default Vega categorical or sequential color scheme name.
        x_label_angle: Rotation angle in degrees for X-axis tick labels.
        top_n: Maximum number of top-ranked target categories to display (`None` for all).
        format_: Optional D3 numeric format string overriding the estimator's default format.
        show_error_bars: Whether to render confidence or credible interval layers when available.
        spacing: Pixel spacing between panels in multi-panel concatenated charts.
        stroke_width: Default stroke width in pixels for lines, borders, or interval rules.
        theme: Active UI color theme mode (`"light"` or `"dark"`).
    """

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Checks whether `estimator` is fitted and contains unstratified (or single-value isolated) estimates.

        Args:
            estimator: Candidate `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators currently available.

        Returns:
            `True` if `estimator` is fitted and does not contain multiple active strata (`not estimator.is_single_stratum`).
        """
        return super().can_plot(estimator, n_estimators=n_estimators) and not estimator.is_single_stratum

    def validate_unstratified(self, estimator: BaseEstimator, side_name: str = "Estimator") -> None:
        """Validates that `estimator` is fitted and contains no duplicate target categories.

        Args:
            estimator: Fitted `BaseEstimator` to validate for 1D target uniqueness.
            side_name: Descriptive label for the estimator slot used in error messages (e.g., `"left"` or `"right"`).

        Raises:
            TypeError: If `estimator` is not a `BaseEstimator` instance.
            RuntimeError: If `estimator` has not been fitted yet.
            ValueError: If `estimator.estimates_` is `None` or contains duplicate values in `estimator.target_col`.
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(f"Expected a fitted BaseEstimator instance, got {type(estimator).__name__}.")
        estimator.check_is_fitted()
        if estimator.estimates_ is None:
            raise ValueError("Estimator estimates_ DataFrame is None.")

        y_col = estimator.target_col
        if estimator.estimates_[y_col].is_duplicated().any():
            num_rows = len(estimator.estimates_)
            num_unique = estimator.estimates_[y_col].n_unique()
            strata_info = f" (detected strata: {list(estimator.strata_cols)})" if estimator.strata_cols else ""
            raise ValueError(
                f"{self.__class__.__name__} received estimates with duplicate '{y_col}' categories on {side_name} "
                f"({num_rows} rows for {num_unique} unique categories){strata_info}. "
                f"This visualizer requires exactly one row per category. "
                f"If your estimator is stratified, isolate a single stratum using "
                f".filter() (e.g. est.filter(country='AUS')) or .split() before plotting."
            )
