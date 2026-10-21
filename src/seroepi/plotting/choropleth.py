"""Stateful Mapbox-style choropleth visualizer for single-stratum Spatial estimators."""

import math
from collections.abc import Sequence
from typing import Literal

import altair as alt
import polars as pl
import shapely

from seroepi.country import CountryData
from seroepi.estimators.base import BaseEstimator
from seroepi.plotting.base import SingleStratumPlotter
from seroepi.variables import Spatial


class ChoroplethPlotter(SingleStratumPlotter):
    """Mapbox-style interactive choropleth visualizer for single-stratum spatial estimates.

    Requires an estimator fitted with a single `Spatial` stratum at `COUNTRY`, `REGION`, or
    `CONTINENT` resolution (treating `UNKNOWN` as `COUNTRY`). Automatically dissolves polygons
    from `CountryData` to match the spatial resolution, auto-scales the initial viewport to the
    bounding box of sampled strata, provides interactive zoom/pan controls, and binds a target
    selector defaulting to overall coverage (`summarize_by_stratum`) alongside individual targets.

    Args:
        top_n: Maximum number of top-ranked target categories to include in the interactive dropdown (`None` for all).
        targets: Optional explicit sequence of target categories to include instead of top-ranked targets.
        total_label: Display label for the pooled multi-target coverage option in the dropdown selector.
        summary_top_k: Number of top positive targets per spatial stratum summarized in the hover tooltip.
        projection: Vega-Lite geographic projection type (`"mercator"`, `"equalEarth"`, or `"naturalEarth1"`).
        zoomable: Whether to bind interactive zoom, longitude, and latitude range sliders.
        auto_scale: Whether to automatically center and scale the initial viewport to the sampled polygons' bounding box.
        padding: Multiplicative bounding-box expansion factor used when `auto_scale` is enabled.
        color_scheme: Vega sequential color scheme for choropleth shading.
        ocean_color: Hex fill color for the background ocean sphere layer in light mode.
        land_color: Hex fill color for unsampled background land polygons in light mode.
        border_color: Hex stroke color for unsampled background polygon borders in light mode.
        basemap_stroke_width: Stroke width in pixels for unsampled background polygon borders.
        stroke_color: Hex stroke color for active sampled choropleth polygon borders in light mode.
        stroke_width: Stroke width in pixels for active sampled choropleth polygon borders.
        opacity: Fill opacity (`0.0` to `1.0`) for active sampled choropleth polygons.
        width: Chart canvas width in pixels.
        height: Chart canvas height in pixels.
        font: Font family for chart typography.
        title_font_size: Font size in pixels for the chart title.
        subtitle_font_size: Font size in pixels for the chart subtitle.
        label_font_size: Font size in pixels for legend and control labels.
        format_: Optional D3 numeric format string overriding the estimator's default format.
        show_error_bars: Whether to include confidence or credible interval bounds in polygon hover tooltips.

    Attributes:
        targets: Explicit sequence of target categories to include, or `None`.
        total_label: Display label for the pooled multi-target coverage option.
        summary_top_k: Number of top positive targets per spatial stratum shown in tooltips.
        projection: Geographic projection type (`"mercator"`, `"equalEarth"`, or `"naturalEarth1"`).
        zoomable: Whether interactive zoom and pan range sliders are bound.
        auto_scale: Whether the initial viewport auto-scales to sampled polygon bounds.
        padding: Bounding-box padding multiplier for `auto_scale`.
        ocean_color: Background ocean fill color in light mode.
        land_color: Unsampled land polygon fill color in light mode.
        border_color: Unsampled polygon border stroke color in light mode.
        basemap_stroke_width: Stroke width for unsampled basemap borders.
        stroke_color: Active choropleth polygon border stroke color in light mode.
        opacity: Fill opacity for active choropleth polygons.

    Examples:
        >>> from seroepi.plotting.choropleth import ChoroplethPlotter
        >>> plotter = ChoroplethPlotter(top_n=10, projection="equalEarth", color_scheme="tealblues")
        >>> chart = plotter.with_theme("dark").plot(
        ...     spatial_estimator,
        ...     title="Global K-Locus Coverage by Country",
        ... )
        >>> ChoroplethPlotter.save(chart, "choropleth_map.svg")
    """

    def __init__(
        self,
        top_n: int | None = 15,
        targets: Sequence[str] | None = None,
        total_label: str = "Total Coverage",
        summary_top_k: int = 5,
        projection: Literal["mercator", "equalEarth", "naturalEarth1"] = "mercator",
        zoomable: bool = True,
        auto_scale: bool = True,
        padding: float = 1.35,
        color_scheme: str = "tealblues",
        ocean_color: str = "#e8eff7",
        land_color: str = "#f8fafc",
        border_color: str = "#cbd5e1",
        basemap_stroke_width: float = 0.6,
        stroke_color: str = "#ffffff",
        stroke_width: float = 1.2,
        opacity: float = 0.92,
        width: int = 680,
        height: int = 440,
        font: str = "sans-serif",
        title_font_size: int = 14,
        subtitle_font_size: int = 11,
        label_font_size: int = 10,
        format_: str | None = None,
        show_error_bars: bool = True,
    ) -> None:
        """Initializes the interactive spatial choropleth visualizer.

        Args:
            top_n: Maximum number of top-ranked target categories to include in the interactive dropdown (`None` for all).
            targets: Optional explicit sequence of target categories to include instead of top-ranked targets.
            total_label: Display label for the pooled multi-target coverage option in the dropdown selector.
            summary_top_k: Number of top positive targets per spatial stratum summarized in the hover tooltip.
            projection: Vega-Lite geographic projection type (`"mercator"`, `"equalEarth"`, or `"naturalEarth1"`).
            zoomable: Whether to bind interactive zoom, longitude, and latitude range sliders.
            auto_scale: Whether to automatically center and scale the initial viewport to the sampled polygons' bounding box.
            padding: Multiplicative bounding-box expansion factor used when `auto_scale` is enabled.
            color_scheme: Vega sequential color scheme for choropleth shading.
            ocean_color: Hex fill color for the background ocean sphere layer in light mode.
            land_color: Hex fill color for unsampled background land polygons in light mode.
            border_color: Hex stroke color for unsampled background polygon borders in light mode.
            basemap_stroke_width: Stroke width in pixels for unsampled background polygon borders.
            stroke_color: Hex stroke color for active sampled choropleth polygon borders in light mode.
            stroke_width: Stroke width in pixels for active sampled choropleth polygon borders.
            opacity: Fill opacity (`0.0` to `1.0`) for active sampled choropleth polygons.
            width: Chart canvas width in pixels.
            height: Chart canvas height in pixels.
            font: Font family for chart typography.
            title_font_size: Font size in pixels for the chart title.
            subtitle_font_size: Font size in pixels for the chart subtitle.
            label_font_size: Font size in pixels for legend and control labels.
            format_: Optional D3 numeric format string overriding the estimator's default format.
            show_error_bars: Whether to include confidence or credible interval bounds in polygon hover tooltips.
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
            show_error_bars=show_error_bars,
            stroke_width=stroke_width,
        )
        self.targets = targets
        self.total_label = total_label
        self.summary_top_k = summary_top_k
        self.projection = projection
        self.zoomable = zoomable
        self.auto_scale = auto_scale
        self.padding = padding
        self.ocean_color = ocean_color
        self.land_color = land_color
        self.border_color = border_color
        self.basemap_stroke_width = basemap_stroke_width
        self.stroke_color = stroke_color
        self.opacity = opacity

    @classmethod
    def can_plot(cls, estimator: BaseEstimator, *, n_estimators: int = 1) -> bool:
        """Checks whether `estimator` is fitted with a single `Spatial` stratum at polygon resolution.

        Args:
            estimator: Candidate `BaseEstimator` instance to inspect.
            n_estimators: Total number of fitted estimators currently available.

        Returns:
            `True` if `estimator` has a single `Spatial` stratum with `COUNTRY`, `REGION`, `CONTINENT`, or `UNKNOWN` resolution.
        """
        if not super().can_plot(estimator, n_estimators=n_estimators):
            return False
        stratum = estimator.strata_[0]
        if not isinstance(stratum, Spatial):
            return False
        return stratum.resolution in {
            Spatial.Resolution.COUNTRY,
            Spatial.Resolution.REGION,
            Spatial.Resolution.CONTINENT,
            Spatial.Resolution.UNKNOWN,
        }

    @staticmethod
    def _resolve_spatial_key(
        raw_val: object,
        eff_res: Spatial.Resolution,
        cd: CountryData,
        geom_map: dict[str, shapely.Geometry],
    ) -> str | None:
        """Resolves a stratum string to a canonical polygon key at the target spatial resolution."""
        if raw_val is None:
            return None
        val = str(raw_val).strip()
        if not val:
            return None
        if val in geom_map:
            return val

        iso = cd.resolve_iso3(val)
        if iso is None:
            return None
        if eff_res == Spatial.Resolution.COUNTRY:
            return iso if iso in geom_map else None
        if eff_res == Spatial.Resolution.REGION:
            sub = str(cd.lookup(iso, "subregion")).strip()
            return sub if sub in geom_map else None
        if eff_res == Spatial.Resolution.CONTINENT:
            reg = str(cd.lookup(iso, "region")).strip()
            return reg if reg in geom_map else None
        return None

    def _compute_initial_view(
        self,
        sampled_geoms: Sequence[shapely.Geometry],
    ) -> tuple[float, float, float]:
        """Computes initial (center_lon, center_lat, scale) from sampled polygon bounds."""
        min_global_scale = round(self.width / (2.0 * math.pi), 0)
        if not self.auto_scale or not sampled_geoms:
            return 0.0, 15.0, min_global_scale

        min_lon, min_lat, max_lon, max_lat = shapely.total_bounds(list(sampled_geoms))
        # Clamp Mercator latitudes to avoid polar infinity
        min_lat_clamped = max(float(min_lat), -72.0)
        max_lat_clamped = min(float(max_lat), 75.0)

        center_lon = round((float(min_lon) + float(max_lon)) / 2.0, 1)
        center_lat = round((min_lat_clamped + max_lat_clamped) / 2.0, 1)

        lon_span = max(float(max_lon) - float(min_lon), 8.0) * self.padding
        lat_span = max(max_lat_clamped - min_lat_clamped, 8.0) * self.padding

        scale_x = self.width / math.radians(min(lon_span, 360.0))
        scale_y = self.height / math.radians(min(lat_span, 160.0))
        scale = round(max(min(scale_x, scale_y, 2500.0), min_global_scale), 0)
        return center_lon, center_lat, scale

    @staticmethod
    def _build_top_targets_map(
        estimates_df: pl.DataFrame,
        *,
        stratum_col: str,
        target_col: str,
        est_col: str,
        fmt: str,
        top_k: int,
    ) -> dict[str, str]:
        """Builds a per-stratum summary string of top targets (e.g. 'KL102 (24.1%), KL2 (18.5%)')."""
        summary_map: dict[str, str] = {}
        partitions = estimates_df.partition_by(stratum_col, as_dict=True)
        for (raw_stratum,), grp in partitions.items():
            s_key = str(raw_stratum)
            pos = grp.filter(pl.col(est_col) > 0).sort(est_col, descending=True)
            if "event" in pos.columns:
                pos = pos.filter(pl.col("event") > 0)
            if len(pos) == 0:
                summary_map[s_key] = "None"
                continue
            top_rows = pos.head(top_k)
            parts = [
                f"{t} ({float(e):{fmt}})"
                for t, e in zip(top_rows[target_col].to_list(), top_rows[est_col].to_list(), strict=True)
            ]
            summary_map[s_key] = ", ".join(parts)
        return summary_map

    def plot(
        self,
        estimator: BaseEstimator,
        *,
        targets: Sequence[str] | None = None,
        top_n: int | None = None,
        zoomable: bool | None = None,
        auto_scale: bool | None = None,
        country_data: CountryData | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        param_suffix: str = "",
    ) -> alt.TopLevelMixin:
        """Renders an interactive geographic choropleth map for a single-stratum spatial estimator.

        Args:
            estimator: Fitted `BaseEstimator` stratified by a single `Spatial` variable.
            targets: Optional explicit sequence of target categories overriding `self.targets`.
            top_n: Optional maximum number of top targets overriding `self.top_n`.
            zoomable: Optional boolean overriding `self.zoomable` for zoom/pan range controls.
            auto_scale: Optional boolean overriding `self.auto_scale` for initial bounding-box fitting.
            country_data: Optional custom `CountryData` instance (defaults to `CountryData.get_instance()`).
            title: Custom chart title overriding the default geographic header.
            subtitle: Custom chart subtitle overriding the automatic resolution and interval summary.
            param_suffix: Optional string suffix appended to Vega-Lite parameter names to prevent collisions in multi-map layouts.

        Returns:
            Configured Altair `LayerChart` projecting ocean, basemap, and interactive choropleth layers.

        Raises:
            TypeError: If `estimator` is not a `BaseEstimator` or its single stratum is not a `Spatial` variable.
            RuntimeError: If `estimator` has not been fitted yet.
            ValueError: If `estimator` does not have exactly 1 stratum or no strata match polygons in `CountryData`.
        """
        ctx = self.extract_context(estimator)
        assert estimator.estimates_ is not None

        stratum_var = estimator.strata_[0]
        if not isinstance(stratum_var, Spatial):
            raise TypeError(
                f"ChoroplethPlotter requires the single stratum to be a Spatial variable, "
                f"got {type(stratum_var).__name__} ('{stratum_var.column}')."
            )

        cd = country_data or CountryData.get_instance()
        eff_res = (
            Spatial.Resolution.COUNTRY
            if stratum_var.resolution == Spatial.Resolution.UNKNOWN
            else stratum_var.resolution
        )
        features, geom_map, name_map = cd.spatial_features(eff_res)

        eff_top_n = self.top_n if top_n is None else top_n
        eff_zoomable = self.zoomable if zoomable is None else zoomable
        eff_auto_scale = self.auto_scale if auto_scale is None else auto_scale

        resolved_targets = self.resolve_targets(estimator, targets or self.targets, eff_top_n)
        multi_target = estimator.is_compositional and len(resolved_targets) > 1

        # Build per-stratum top-K target breakdown string
        top_targets_by_stratum = self._build_top_targets_map(
            estimator.estimates_,
            stratum_col=ctx.stratum_col,
            target_col=ctx.target_col,
            est_col=ctx.est_col,
            fmt=ctx.format,
            top_k=self.summary_top_k,
        )

        # Assemble plot DataFrame (Total Coverage + individual targets when compositional)
        target_rows = estimator.estimates_.filter(pl.col(ctx.target_col).is_in(resolved_targets))
        if multi_target:
            total_df, _ = estimator.summarize_by_stratum(targets=resolved_targets)
            total_df = total_df.with_columns(pl.lit(self.total_label).alias(ctx.target_col))
            common_cols = [c for c in total_df.columns if c in target_rows.columns]
            plot_df = pl.concat(
                [total_df.select(common_cols), target_rows.select(common_cols)],
                how="diagonal_relaxed",
            )
            target_options = [self.total_label, *resolved_targets]
        else:
            plot_df = target_rows
            target_options = list(resolved_targets)

        # Resolve spatial polygon IDs and human-readable location names
        raw_strata = plot_df[ctx.stratum_col].to_list()
        spatial_ids = [self._resolve_spatial_key(v, eff_res, cd, geom_map) for v in raw_strata]
        location_names = [
            name_map.get(sid, str(raw)) if sid else str(raw) for sid, raw in zip(spatial_ids, raw_strata, strict=True)
        ]
        top_summaries = [top_targets_by_stratum.get(str(raw), "") for raw in raw_strata]

        plot_df = plot_df.with_columns(
            pl.Series("_spatial_id", spatial_ids, dtype=pl.String),
            pl.Series("location_name", location_names, dtype=pl.String),
            pl.Series("top_targets", top_summaries, dtype=pl.String),
        ).filter(pl.col("_spatial_id").is_not_null())

        if len(plot_df) == 0:
            raise ValueError(
                f"No strata in column '{ctx.stratum_col}' could be matched to {eff_res.value} polygons in CountryData."
            )

        matched_ids = plot_df["_spatial_id"].unique().to_list()
        sampled_geoms = [geom_map[sid] for sid in matched_ids if sid in geom_map]

        prev_auto = self.auto_scale
        self.auto_scale = eff_auto_scale
        try:
            init_lon, init_lat, init_scale = self._compute_initial_view(sampled_geoms)
        finally:
            self.auto_scale = prev_auto

        # Projection & interactive zoom/pan parameters
        min_scale = max(round(self.width / (2.0 * math.pi), 0), 60.0)
        max_scale = max(init_scale * 5.0, 2000.0)
        params: list[alt.Parameter] = []

        if eff_zoomable:
            zoom_name = f"map_zoom{param_suffix}"
            lon_name = f"map_lon{param_suffix}"
            lat_name = f"map_lat{param_suffix}"
            zoom_param = alt.param(
                name=zoom_name,
                value=init_scale,
                bind=alt.binding_range(min=min_scale, max=max_scale, step=10, name="Zoom: "),
            )
            lon_param = alt.param(
                name=lon_name,
                value=init_lon,
                bind=alt.binding_range(min=-180, max=180, step=1, name="Longitude: "),
            )
            lat_param = alt.param(
                name=lat_name,
                value=init_lat,
                bind=alt.binding_range(min=-70, max=75, step=1, name="Latitude: "),
            )
            params.extend([zoom_param, lon_param, lat_param])
            proj_kwargs = {
                "type": self.projection,
                "scale": zoom_param,
                "center": alt.expr(f"[{lon_name}, {lat_name}]"),
                "clipExtent": [[0, 0], [self.width, self.height]],
            }
        else:
            proj_kwargs = {
                "type": self.projection,
                "scale": init_scale,
                "center": [init_lon, init_lat],
                "clipExtent": [[0, 0], [self.width, self.height]],
            }

        eff_ocean = "#0f172a" if (self.theme == "dark" and self.ocean_color == "#e8eff7") else self.ocean_color
        eff_land = "#1e293b" if (self.theme == "dark" and self.land_color == "#f8fafc") else self.land_color
        eff_border = "#334155" if (self.theme == "dark" and self.border_color == "#cbd5e1") else self.border_color
        eff_stroke = "#0f172a" if (self.theme == "dark" and self.stroke_color == "#ffffff") else self.stroke_color

        # Ocean sphere layer + Basemap layer (surrounding countries / regions / continents in Mapbox Positron/Dark style)
        ocean_layer = alt.Chart(alt.Data(values=[{"type": "Sphere"}])).mark_geoshape(
            fill=eff_ocean,
            stroke=eff_border,
            strokeWidth=1.0,
        )

        basemap_features = (
            [f for f in features if f.get("properties", {}).get("area", 1e12) >= 3e9]
            if eff_res == Spatial.Resolution.COUNTRY
            else features
        )
        basemap = (
            alt.Chart(alt.Data(values=basemap_features))
            .mark_geoshape(
                fill=eff_land,
                stroke=eff_border,
                strokeWidth=self.basemap_stroke_width,
            )
            .encode(
                tooltip=[alt.Tooltip("properties.name:N", title=ctx.stratum_title)],
            )
        )

        # Tooltips for active choropleth layer
        dimensions: list[tuple[str, str]] = [("location_name", ctx.stratum_title)]
        if eff_res == Spatial.Resolution.COUNTRY:
            dimensions.append(("_spatial_id", "ISO3"))
        dimensions.append((ctx.target_col, ctx.target_title))

        tooltips = self.build_tooltips(
            plot_df,
            dimensions=dimensions,
            est_col=ctx.est_col,
            est_title=ctx.est_title,
            fmt=ctx.format,
            ci_prefix=ctx.ci_prefix,
            lower_col=estimator.estimate.lower,
            upper_col=estimator.estimate.upper,
            include_intervals=self.show_error_bars and estimator.estimate.has_interval,
        )
        if estimator.is_compositional:
            tooltips.append(alt.Tooltip("top_targets:N", title=f"Top {ctx.target_title}s"))

        geo_lookup = alt.LookupData(
            data=alt.Data(values=features),
            key="id",
            fields=["type", "geometry", "properties"],
        )

        data_chart = alt.Chart(plot_df)
        if multi_target:
            target_param = alt.selection_point(
                name=f"target_select{param_suffix}",
                fields=[ctx.target_col],
                bind=alt.binding_select(options=target_options, name=f"{ctx.target_title}: "),
                value=self.total_label,
            )
            params.insert(0, target_param)
            data_chart = data_chart.transform_filter(target_param)

        choropleth_layer = (
            data_chart.transform_lookup(lookup="_spatial_id", from_=geo_lookup)
            .mark_geoshape(
                stroke=eff_stroke,
                strokeWidth=self.stroke_width,
                opacity=self.opacity,
            )
            .encode(
                color=alt.Color(
                    f"{ctx.est_col}:Q",
                    scale=alt.Scale(scheme=self.color_scheme, domainMin=0.0),
                    legend=alt.Legend(title=ctx.est_title, format=ctx.format),
                ),
                tooltip=tooltips,
            )
        )

        # Titles and subtitles
        resolved_title = title or f"{ctx.target_title} {ctx.est_title} by {ctx.stratum_title}"
        if subtitle is not None:
            resolved_subtitle = subtitle
        else:
            sub_parts: list[str] = [f"Spatial resolution: {eff_res.value.title()}"]
            if multi_target:
                sub_parts.append(f"Interactive breakdown across top {len(resolved_targets)} {ctx.target_title} targets")
            if estimator.estimate.has_interval and estimator.estimate.interval_description:
                sub_parts.append(estimator.estimate.interval_description)
            resolved_subtitle = " | ".join(sub_parts)

        layered = (
            alt.layer(ocean_layer, basemap, choropleth_layer)
            .project(**proj_kwargs)
            .properties(
                width=self.width,
                height=self.height,
                title=alt.TitleParams(
                    text=resolved_title,
                    subtitle=resolved_subtitle,
                    anchor="middle",
                ),
            )
        )
        if params:
            layered = layered.add_params(*params)

        return self.configure_chart(layered).configure_view(stroke=eff_border, strokeWidth=1.0)
