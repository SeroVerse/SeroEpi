"""Object-oriented geographic data management, country dimensions, and reverse geocoding."""

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import polars as pl
import pyproj
import shapely
import shapely.ops

from seroepi.models import DataModel
from seroepi.variables import (
    Count,
    Estimate,
    Identifier,
    Metric,
    Spatial,
    Temporal,
    Variable,
)


@dataclass(frozen=True, slots=True)
class CountryData(DataModel):
    """Container for global geographic boundaries, country metadata, and reverse geocoding.

    Backed by the static ``src/seroepi/assets/countries.joblib`` package asset and cached in
    memory on first access via ``CountryData.load()``.

    Attributes:
        data: Polars ``DataFrame`` containing one row per country with ISO codes, UN regions,
            World Bank indicators, aliases, and WKB boundary geometries.
        variables: Mapping of column names to semantic ``Variable`` descriptors.
        name: Logical name of the country dataset (``"country_data"``).

    Examples:
        >>> from seroepi.country import CountryData
        >>> cd = CountryData.load()
        >>> cd.resolve_iso3("United Kingdom")
        'GBR'
        >>> cd.lookup("GBR", "continent")
        'Europe'
    """

    ASSET_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "assets" / "countries.joblib"

    VARIABLES: ClassVar[Mapping[str, Variable]] = {
        "iso3": Identifier(column="iso3"),
        "iso2": Variable(column="iso2"),
        "numeric": Variable(column="numeric"),
        "name": Spatial(column="name", title="country", resolution=Spatial.Resolution.COUNTRY),
        "official_name": Variable(column="official_name"),
        "region": Spatial(column="region", resolution=Spatial.Resolution.CONTINENT),
        "subregion": Spatial(column="subregion", resolution=Spatial.Resolution.REGION),
        "continent": Spatial(column="continent", resolution=Spatial.Resolution.CONTINENT),
        "centroid_lat": Spatial(column="centroid_lat", resolution=Spatial.Resolution.POINT),
        "centroid_lon": Spatial(column="centroid_lon", resolution=Spatial.Resolution.POINT),
        "population": Count(column="population"),
        "life_expectancy": Estimate(column="life_expectancy", title="life_expectancy", bounded=False),
        "life_expectancy_year": Temporal(column="life_expectancy_year", resolution=Temporal.Resolution.YEAR),
        "income_level": Variable(column="income_level"),
        "income_level_code": Variable(column="income_level_code"),
        "lending_type": Variable(column="lending_type"),
        "capital_city": Spatial(column="capital_city", resolution=Spatial.Resolution.CITY),
        "aliases": Variable(column="aliases"),
        "geometry": Spatial(column="geometry"),
        "geometry_area": Metric(column="geometry_area"),
    }

    _instance: ClassVar["CountryData | None"] = None
    _spatial_features_cache: ClassVar[
        dict[tuple[Spatial.Resolution, bool], tuple[list[dict[str, Any]], dict[str, shapely.Geometry], dict[str, str]]]
    ] = {}

    # In-memory spatial and alias caches (slotted in frozen dataclass)
    _aliases_df: pl.DataFrame | None = field(default=None, repr=False, compare=False)
    _tree_4326: shapely.STRtree | None = field(default=None, repr=False, compare=False)
    _tree_3857: shapely.STRtree | None = field(default=None, repr=False, compare=False)
    _transformer_3857: pyproj.Transformer | None = field(default=None, repr=False, compare=False)
    _country_iso3s: list[str] | None = field(default=None, repr=False, compare=False)
    _country_areas: list[float] | None = field(default=None, repr=False, compare=False)
    _geoms_4326: list[shapely.Geometry] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, source: Path | str | bytes | io.BytesIO | None = None) -> Self:
        """Loads ``CountryData`` from the bundled package asset or a custom Joblib source.

        When ``source`` is ``None``, returns the process-wide in-memory singleton loaded from
        ``ASSET_PATH``.

        Args:
            source: Optional custom path, bytes, or ``BytesIO`` buffer. Defaults to ``ASSET_PATH``.

        Returns:
            The loaded ``CountryData`` instance.

        Raises:
            FileNotFoundError: If the target Joblib file does not exist.
        """
        if source is None:
            if not isinstance(cls._instance, cls):
                cls._instance = super().load(cls.ASSET_PATH)
                cls._spatial_features_cache.clear()
            return cls._instance

        instance = super().load(source)
        cls._instance = instance
        cls._spatial_features_cache.clear()
        return instance

    @property
    def aliases_df(self) -> pl.DataFrame:
        """Returns an exploded two-column Polars ``DataFrame`` mapping uppercase ``alias`` to ``iso3``.

        Returns:
            Polars ``DataFrame`` with columns ``["alias", "iso3"]``.
        """
        if not isinstance(self._aliases_df, pl.DataFrame):
            df = (
                self.data.select(["iso3", "aliases"])
                .explode("aliases", empty_as_null=True)
                .filter(pl.col("aliases").is_not_null())
                .select([pl.col("aliases").str.to_uppercase().alias("alias"), pl.col("iso3")])
                .unique(subset=["alias"], keep="first")
            )
            object.__setattr__(self, "_aliases_df", df)
            return df
        if len(self._aliases_df) != self._aliases_df["alias"].n_unique():
            df = self._aliases_df.unique(subset=["alias"], keep="first")
            object.__setattr__(self, "_aliases_df", df)
            return df
        return self._aliases_df

    @property
    def tree(self) -> shapely.STRtree:
        """Returns the spatial ``STRtree`` index in EPSG:4326 for accelerated spatial queries.

        Returns:
            C-accelerated Shapely ``STRtree`` over country boundary geometries.
        """
        t, _, _, _ = self._get_spatial_index()
        return t

    def lookup(self, iso3: str, column: str) -> Any:
        """Returns the value of ``column`` for a given ISO3 country code directly from ``self.data``.

        Args:
            iso3: Three-letter ISO 3166-1 alpha-3 country code.
            column: Name of the metadata column in ``self.data``.

        Returns:
            Scalar cell value for the requested country and column.

        Raises:
            KeyError: If ``iso3`` is not found in ``self.data``.
        """
        match = self.data.filter(pl.col("iso3") == iso3.strip().upper())
        if len(match) == 0:
            raise KeyError(f"ISO3 code '{iso3}' not found in CountryData.")
        return match[column][0]

    def resolve_iso3(self, country: str | None) -> str | None:
        """Resolves an arbitrary country name, code, translation, or abbreviation to strict ISO3.

        Args:
            country: Raw country name, ISO2/ISO3 code, or compound locality string.

        Returns:
            Three-letter uppercase ISO3 code, or ``None`` if unrecognized.
        """
        if not country or not isinstance(country, str):
            return None
        clean = country.strip().upper()
        if not clean:
            return None
        adf = self.aliases_df
        match = adf.filter(pl.col("alias") == clean)
        if len(match) > 0:
            return str(match["iso3"][0])

        for sep in (" - ", " / ", " : ", "-", ":", "/"):
            if sep in country:
                for part in country.split(sep):
                    p_clean = part.strip().upper()
                    if p_clean:
                        part_match = adf.filter(pl.col("alias") == p_clean)
                        if len(part_match) > 0:
                            return str(part_match["iso3"][0])
        return None

    def resolve_point(self, lat: float, lon: float, fallback_nearest_km: float = 25.0) -> str | None:
        """Reverse geocodes a single ``(lat, lon)`` coordinate pair to an ISO3 country code.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            fallback_nearest_km: Maximum coastal fallback distance in kilometers.

        Returns:
            Resolved three-letter ISO3 code, or ``None`` if no country polygon matches.
        """
        if lat is None or lon is None:
            return None
        df = pl.DataFrame({"lat": [float(lat)], "lon": [float(lon)]})
        res = self.reverse_geocode(
            df, lat_col="lat", lon_col="lon", output_col="iso3", fallback_nearest_km=fallback_nearest_km
        )
        val = res["iso3"][0]
        if val is None or str(val) in ("None", "nan", "NaN", ""):
            return None
        return str(val)

    def _get_spatial_index(self) -> tuple[shapely.STRtree, list[str], list[float], list[shapely.Geometry]]:
        """Returns or builds the C-accelerated Shapely STRtree in EPSG:4326."""
        if (
            self._tree_4326 is None
            or self._country_iso3s is None
            or self._country_areas is None
            or self._geoms_4326 is None
        ):
            iso3s = self.data["iso3"].to_list()
            areas = self.data["geometry_area"].to_list()
            geoms = shapely.from_wkb(self.data["geometry"].to_list())
            tree = shapely.STRtree(geoms)
            object.__setattr__(self, "_country_iso3s", iso3s)
            object.__setattr__(self, "_country_areas", areas)
            object.__setattr__(self, "_geoms_4326", geoms)
            object.__setattr__(self, "_tree_4326", tree)
            return tree, iso3s, areas, geoms
        return self._tree_4326, self._country_iso3s, self._country_areas, self._geoms_4326

    def spatial_features(
        self,
        resolution: Spatial.Resolution = Spatial.Resolution.COUNTRY,
        *,
        exclude_antarctica: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, shapely.Geometry], dict[str, str]]:
        """Compiles GeoJSON features, Shapely geometries, and display names for a polygon ``Spatial.Resolution``.

        Supports ``COUNTRY`` (ISO3 polygons), ``REGION`` (dissolved UN M49 subregion polygons), and
        ``CONTINENT`` (dissolved macro-continent polygons). Treats ``UNKNOWN`` as ``COUNTRY``.

        Args:
            resolution: Polygon spatial resolution (``COUNTRY``, ``REGION``, ``CONTINENT``, or ``UNKNOWN``).
            exclude_antarctica: Whether to omit Antarctica from the returned features.

        Returns:
            A 3-tuple ``(features, geom_map, name_map)`` of GeoJSON feature dicts, Shapely geometries,
            and human-readable display names keyed by feature ID.

        Raises:
            ValueError: If ``resolution`` is not a polygon-compatible resolution.
        """
        eff_res = Spatial.Resolution.COUNTRY if resolution == Spatial.Resolution.UNKNOWN else resolution
        if eff_res not in (Spatial.Resolution.COUNTRY, Spatial.Resolution.REGION, Spatial.Resolution.CONTINENT):
            raise ValueError(
                f"Polygon geometries are only supported for COUNTRY, REGION, and CONTINENT spatial resolutions, "
                f"got {resolution!r}."
            )

        cache_key = (eff_res, exclude_antarctica)
        if cache_key in self._spatial_features_cache:
            return self._spatial_features_cache[cache_key]

        df = self.data
        if exclude_antarctica:
            df = df.filter((pl.col("iso3") != "ATA") & (pl.col("region") != "Antarctic"))

        features: list[dict[str, Any]] = []
        geom_map: dict[str, shapely.Geometry] = {}
        name_map: dict[str, str] = {}

        if eff_res == Spatial.Resolution.COUNTRY:
            iso_list = df["iso3"].to_list()
            name_list = df["name"].to_list()
            area_list = df["geometry_area"].to_list()
            geoms = shapely.from_wkb(df["geometry"].to_list())
            for iso3, name, area, geom in zip(iso_list, name_list, area_list, geoms, strict=True):
                if geom is None:
                    continue
                key = str(iso3)
                disp = str(name) if name else key
                geom_map[key] = geom
                name_map[key] = disp
                features.append(
                    {
                        "type": "Feature",
                        "id": key,
                        "geometry": json.loads(shapely.to_geojson(geom)),
                        "properties": {"id": key, "name": disp, "area": float(area)},
                    }
                )
        else:
            group_col = "subregion" if eff_res == Spatial.Resolution.REGION else "region"
            partitions = df.filter(pl.col(group_col).is_not_null() & (pl.col(group_col) != "")).partition_by(
                group_col, as_dict=True
            )
            for (raw_key,), grp_df in partitions.items():
                key = str(raw_key).strip()
                if not key:
                    continue
                large_df = grp_df.filter(pl.col("geometry_area") >= 3e9)
                target_df = large_df if len(large_df) > 0 else grp_df
                grp_geoms = [g for g in shapely.from_wkb(target_df["geometry"].to_list()) if g is not None]
                if not grp_geoms:
                    continue
                dissolved = shapely.union_all(grp_geoms)
                if isinstance(dissolved, shapely.Polygon):
                    dissolved = shapely.Polygon(dissolved.exterior)
                elif isinstance(dissolved, shapely.MultiPolygon):
                    dissolved = shapely.union_all([shapely.Polygon(p.exterior) for p in dissolved.geoms])
                geom_map[key] = dissolved
                name_map[key] = key
                features.append(
                    {
                        "type": "Feature",
                        "id": key,
                        "geometry": json.loads(shapely.to_geojson(dissolved)),
                        "properties": {"id": key, "name": key},
                    }
                )

        result = (features, geom_map, name_map)
        self._spatial_features_cache[cache_key] = result
        return result

    def reverse_geocode(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        output_col: str = "country",
        fallback_nearest_km: float = 25.0,
    ) -> pl.DataFrame:
        """Performs vectorized reverse geocoding via GEOS C-accelerated ``STRtree`` indexing.

        Uses area-ascending priority so small micro-states take precedence over enclosing
        continental polygons, with optional nearest fallback for coastal coordinates.

        Args:
            data: Input Polars ``DataFrame`` or ``LazyFrame`` containing coordinate columns.
            lat_col: Column name for decimal latitude.
            lon_col: Column name for decimal longitude.
            output_col: Name of the output column receiving resolved ISO3 country codes.
            fallback_nearest_km: Maximum distance in kilometers for coastal nearest-polygon fallback.

        Returns:
            A Polars ``DataFrame`` with ``output_col`` joined.

        Raises:
            KeyError: If ``lat_col`` or ``lon_col`` is missing from ``data``.
        """
        df = data.collect() if isinstance(data, pl.LazyFrame) else data
        if lat_col not in df.columns or lon_col not in df.columns:
            raise KeyError(f"Columns '{lat_col}' and '{lon_col}' must exist in DataFrame: {df.columns}")

        coords = (
            df.select([pl.col(lat_col).cast(pl.Float64), pl.col(lon_col).cast(pl.Float64)])
            .filter(pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null())
            .unique()
        )
        if coords.height == 0:
            return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(output_col))

        tree, iso3s, areas, geoms = self._get_spatial_index()
        pts = shapely.points(coords[lon_col].to_numpy(), coords[lat_col].to_numpy())

        # Query STRtree in EPSG:4326
        pt_indices, geom_indices = tree.query(pts, predicate="intersects")

        # Prioritize smallest area if a point intersects overlapping polygons (e.g. microstates)
        matched: dict[int, tuple[int, float]] = {}
        for p_idx, g_idx in zip(pt_indices, geom_indices, strict=False):
            area = areas[g_idx]
            if p_idx not in matched or area < matched[p_idx][1]:
                matched[p_idx] = (g_idx, area)

        # Coastal nearest fallback in planar metric CRS EPSG:3857 for coordinates just offshore
        unmatched_pt_indices = [i for i in range(len(pts)) if i not in matched]
        if unmatched_pt_indices and fallback_nearest_km > 0:
            if self._tree_3857 is None or self._transformer_3857 is None:
                transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                geoms_3857 = [shapely.ops.transform(transformer.transform, g) for g in geoms]
                tree_3857 = shapely.STRtree(geoms_3857)
                object.__setattr__(self, "_transformer_3857", transformer)
                object.__setattr__(self, "_tree_3857", tree_3857)

            max_meters = fallback_nearest_km * 1000.0
            unmatched_lons = coords[lon_col].to_numpy()[unmatched_pt_indices]
            unmatched_lats = coords[lat_col].to_numpy()[unmatched_pt_indices]
            assert self._transformer_3857 is not None
            assert self._tree_3857 is not None
            proj_x, proj_y = self._transformer_3857.transform(unmatched_lons, unmatched_lats)
            unmatched_pts_3857 = shapely.points(proj_x, proj_y)

            res = self._tree_3857.query_nearest(unmatched_pts_3857, max_distance=max_meters)
            if len(res) == 2 and len(res[0]) > 0:
                for u_idx, g_idx in zip(res[0], res[1], strict=False):
                    orig_idx = unmatched_pt_indices[u_idx]
                    area = areas[g_idx]
                    if orig_idx not in matched or area < matched[orig_idx][1]:
                        matched[orig_idx] = (g_idx, area)

        result_iso = [iso3s[matched[i][0]] if i in matched else None for i in range(len(pts))]

        lookup_df = pl.DataFrame(
            {
                lat_col: coords[lat_col],
                lon_col: coords[lon_col],
                output_col: pl.Series(output_col, result_iso, dtype=pl.Utf8),
            }
        )

        return df.join(lookup_df, on=[lat_col, lon_col], how="left")
