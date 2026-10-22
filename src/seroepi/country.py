"""Object-oriented geographic data management, country dimensions, and reverse geocoding.

Zero-payload by default. All country polygons, ISO codes, UN regions, and aliases
are fetched and compiled on demand into a single ~/.seroepi/data/countries.joblib
file via CountryData.build().
"""

import asyncio
import io
import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import polars as pl
import pyogrio
import pyproj
import shapely
import shapely.ops
import yaml
from shapely.geometry import Point

from seroepi.client.world_bank import WorldBankClient
from seroepi.models import LoadedData
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
class CountryRecord:
    """Strongly-typed metadata and geometry record for a sovereign or dependent country.

    Attributes:
        iso3: Three-letter uppercase ISO 3166-1 alpha-3 country code (or ``"KOS"`` for Kosovo).
        geometry: Serialized Well-Known Binary (WKB) polygon or multipolygon boundary in EPSG:4326.
        geometry_area: Planar surface area in square meters computed in EPSG:3857 projection.
        iso2: Two-letter uppercase ISO 3166-1 alpha-2 country code.
        numeric: Three-digit ISO 3166-1 numeric code string.
        name: Common English display name of the country.
        official_name: Formal diplomatic or official English name of the country.
        region: Macro-geographic continent or UN region name.
        subregion: UN M49 intermediate or subregion name.
        continent: Macro-continental grouping aligned with ``region``.
        centroid_lat: Representative geographic centroid latitude in decimal degrees.
        centroid_lon: Representative geographic centroid longitude in decimal degrees.
        population: Estimated national population count.
        life_expectancy: Latest World Bank life expectancy at birth in years, or ``None``.
        life_expectancy_year: Reference year string for ``life_expectancy``.
        income_level: World Bank income classification label.
        income_level_code: World Bank income classification code.
        lending_type: World Bank lending category.
        capital_city: Name of the national capital city.
        aliases: Sorted list of uppercase multilingual names, translations, and abbreviations.
    """

    iso3: str
    geometry: bytes
    geometry_area: float
    iso2: str = ""
    numeric: str = ""
    name: str = ""
    official_name: str = ""
    region: str = ""
    subregion: str = ""
    continent: str = ""
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    population: int = 0
    life_expectancy: float | None = None
    life_expectancy_year: str = ""
    income_level: str = ""
    income_level_code: str = ""
    lending_type: str = ""
    capital_city: str = ""
    aliases: list[str] = field(default_factory=list)

    @staticmethod
    def extract_aliases(
        c: dict[str, Any],
        iso3: str,
        iso2: str,
        opensanctions_data: dict[str, Any],
    ) -> list[str]:
        """Extracts multilingual translations, official names, and OpenSanctions aliases for a country.

        Args:
            c: Raw ``mledoze/countries`` dictionary entry for the country.
            iso3: Three-letter uppercase ISO3 code.
            iso2: Two-letter uppercase ISO2 code.
            opensanctions_data: Parsed OpenSanctions country alias mapping.

        Returns:
            Sorted list of unique uppercase country name aliases.
        """
        aliases: set[str] = {iso3.upper()}
        if iso2:
            aliases.add(iso2.upper())

        name_dict = c.get("name", {})
        for name_key in ("common", "official"):
            val = str(name_dict.get(name_key, "")).strip().upper()
            if val:
                aliases.add(val)

        for alt in c.get("altSpellings", []):
            clean = str(alt).strip().upper()
            if clean:
                aliases.add(clean)

        for trans in c.get("translations", {}).values():
            if isinstance(trans, dict):
                for k in ("common", "official"):
                    val = str(trans.get(k, "")).strip().upper()
                    if val:
                        aliases.add(val)

        for n in name_dict.get("native", {}).values():
            if isinstance(n, dict):
                for k in ("common", "official"):
                    val = str(n.get(k, "")).strip().upper()
                    if val:
                        aliases.add(val)

        # OpenSanctions aliases by ISO2 and ISO3
        for key in (iso2, iso3):
            if key and key in opensanctions_data and isinstance(opensanctions_data[key], list):
                for item in opensanctions_data[key]:
                    clean = str(item).strip().upper()
                    if clean:
                        aliases.add(clean)

        # OpenSanctions sub-regions (e.g. GB-ENG, GB-SCT for GBR)
        if iso2:
            prefix = f"{iso2}-"
            for os_key, os_items in opensanctions_data.items():
                if os_key.startswith(prefix) and isinstance(os_items, list):
                    for item in os_items:
                        clean = str(item).strip().upper()
                        if clean:
                            aliases.add(clean)

        # Special handling for Kosovo (XK / XKX in OpenSanctions)
        if iso3 == "KOS":
            for xk_key in ("XK", "XKX"):
                if xk_key in opensanctions_data and isinstance(opensanctions_data[xk_key], list):
                    for item in opensanctions_data[xk_key]:
                        clean = str(item).strip().upper()
                        if clean:
                            aliases.add(clean)

        return sorted(aliases)

    @classmethod
    def from_sources(
        cls,
        c: dict[str, Any],
        geom_map: dict[str, Any],
        wb_profiles: dict[str, dict[str, Any]],
        opensanctions_data: dict[str, Any],
        transformer: pyproj.Transformer,
    ) -> "CountryRecord | None":
        """Compiles a single ``CountryRecord`` from raw mledoze, Natural Earth, and World Bank sources.

        Args:
            c: Raw ``mledoze/countries`` dictionary entry.
            geom_map: Mapping from ISO3 codes to Shapely polygon geometries.
            wb_profiles: Mapping from ISO3 codes to World Bank socioeconomic indicator dictionaries.
            opensanctions_data: Parsed OpenSanctions alias dictionary.
            transformer: PyProj coordinate transformer from ``EPSG:4326`` to ``EPSG:3857``.

        Returns:
            A populated ``CountryRecord`` instance, or ``None`` if the record has no valid 3-letter ISO3 code.
        """
        iso3 = str(c.get("cca3", "")).strip().upper()
        if iso3 == "UNK":
            iso3 = "KOS"
        if not iso3 or len(iso3) != 3:
            return None

        iso2 = str(c.get("cca2", "")).strip().upper()
        numeric = str(c.get("ccn3", "")).strip()
        name = str(c.get("name", {}).get("common", "")).strip()
        official_name = str(c.get("name", {}).get("official", "")).strip()
        region = str(c.get("region", "")).strip()
        subregion = str(c.get("subregion", "")).strip()
        latlng = c.get("latlng", [0.0, 0.0])
        lat = float(latlng[0]) if len(latlng) > 0 else 0.0
        lon = float(latlng[1]) if len(latlng) > 1 else 0.0
        population = int(c.get("population", 0))

        # Dynamic World Bank profile attributes
        wb = wb_profiles.get(iso3, {})
        life_exp = wb.get("life_expectancy")
        life_exp_val = float(life_exp) if life_exp is not None else None
        life_exp_yr = str(wb.get("life_expectancy_year") or "")
        income_lvl = str(wb.get("income_level") or "")
        income_code = str(wb.get("income_level_code") or "")
        lending_type = str(wb.get("lending_type") or "")
        capital = str(wb.get("capital_city") or (c.get("capital", [""])[0] if c.get("capital") else ""))

        aliases = cls.extract_aliases(c, iso3, iso2, opensanctions_data)

        geom = geom_map.get(iso3)
        if geom is None:
            # Fallback to centroid buffered by ~0.15 degrees (~16 km) for microstates not in Natural Earth
            geom = Point(lon, lat).buffer(0.15)

        area = float(shapely.ops.transform(transformer.transform, geom).area)
        wkb_bytes = shapely.to_wkb(geom)

        return cls(
            iso3=iso3,
            iso2=iso2,
            numeric=numeric,
            name=name,
            official_name=official_name,
            region=region,
            subregion=subregion,
            continent=region,
            centroid_lat=lat,
            centroid_lon=lon,
            population=population,
            life_expectancy=life_exp_val,
            life_expectancy_year=life_exp_yr,
            income_level=income_lvl,
            income_level_code=income_code,
            lending_type=lending_type,
            capital_city=capital,
            aliases=aliases,
            geometry=wkb_bytes,
            geometry_area=area,
        )


@dataclass(frozen=True, slots=True)
class CountryData(LoadedData):
    """Container for global geographic boundaries, country metadata, and reverse geocoding.

    Zero-payload by default. Assets are fetched and compiled on demand via ``build()``
    and saved locally to ``~/.seroepi/data/countries.joblib``.

    Attributes:
        data: Polars ``DataFrame`` containing one row per country with ISO codes, UN regions,
            World Bank indicators, aliases, and WKB boundary geometries.
        variables: Mapping of column names to semantic ``Variable`` descriptors.
        name: Logical name of the country dataset (``"country_data"``).

    Examples:
        >>> from seroepi.country import CountryData
        >>> cd = CountryData.ensure()
        >>> cd.resolve_iso3("United Kingdom")
        'GBR'
        >>> cd.lookup("GBR", "continent")
        'Europe'
    """

    FILENAME: ClassVar[str] = "countries.joblib"
    DIR_ENV_VAR: ClassVar[str] = "SEROEPI_DATA_DIR"
    DEFAULT_SUBDIR: ClassVar[str] = "data"

    DEFAULT_NE_COUNTRIES_URL: ClassVar[str] = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    )
    DEFAULT_NE_TINY_COUNTRIES_URL: ClassVar[str] = (
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_tiny_countries.zip"
    )
    DEFAULT_MLEDOZE_URL: ClassVar[str] = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"
    DEFAULT_OPENSANCTIONS_URL: ClassVar[str] = (
        "https://raw.githubusercontent.com/opensanctions/countrynames/main/countrynames/data.yaml"
    )

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

    _data_path: Path | None = field(default=None, repr=False, compare=False)

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
        """Loads ``CountryData`` from local Joblib storage and caches the singleton instance.

        Args:
            source: Optional custom path, bytes, or ``BytesIO`` buffer. Defaults to ``get_default_path()``.

        Returns:
            The loaded ``CountryData`` instance.

        Raises:
            FileNotFoundError: If the local Joblib file does not exist.
        """
        instance = super().load(source)
        if isinstance(source, (str, Path)):
            object.__setattr__(instance, "_data_path", Path(source))
        elif source is None:
            object.__setattr__(instance, "_data_path", cls.get_default_path())
        cls._instance = instance
        cls._spatial_features_cache.clear()
        return instance

    @classmethod
    def get_instance(cls, path: Path | str | None = None) -> "CountryData":
        """Returns the cached singleton ``CountryData`` instance, loading from disk if needed.

        Args:
            path: Optional custom path to ``countries.joblib``.

        Returns:
            The singleton ``CountryData`` instance.

        Raises:
            FileNotFoundError: If the asset has not been compiled via ``CountryData.build()``.
        """
        target = Path(path) if path else cls.get_default_path()
        if cls._instance is None or cls._instance._data_path != target:
            if cls.is_built(target):
                instance = cls.load(target)
                object.__setattr__(instance, "_data_path", target)
                cls._instance = instance
                return instance
            return cls.build(out_path=target)
        return cls._instance

    @classmethod
    def build(
        cls,
        out_path: Path | str | None = None,
        force: bool = False,
        ne_countries_url: str = DEFAULT_NE_COUNTRIES_URL,
        ne_tiny_countries_url: str = DEFAULT_NE_TINY_COUNTRIES_URL,
        mledoze_url: str = DEFAULT_MLEDOZE_URL,
        opensanctions_url: str = DEFAULT_OPENSANCTIONS_URL,
        **kwargs: Any,
    ) -> Self:
        """Downloads and compiles the ``CountryData`` container into ``out_path``.

        Args:
            out_path: Optional output file path. Defaults to ``~/.seroepi/data/countries.joblib``.
            force: Whether to rebuild even if ``out_path`` already exists.
            ne_countries_url: URL for Natural Earth 110m Admin-0 countries shapefile zip.
            ne_tiny_countries_url: URL for Natural Earth 110m tiny countries shapefile zip.
            mledoze_url: URL for ``mledoze/countries`` JSON metadata.
            opensanctions_url: URL for OpenSanctions country name aliases YAML.
            **kwargs: Additional keyword arguments forwarded to ``compile()``.

        Returns:
            The compiled and persisted ``CountryData`` instance.
        """
        target = Path(out_path) if out_path else cls.get_default_path()
        instance = super().build(
            out_path=target,
            force=force,
            ne_countries_url=ne_countries_url,
            ne_tiny_countries_url=ne_tiny_countries_url,
            mledoze_url=mledoze_url,
            opensanctions_url=opensanctions_url,
            **kwargs,
        )
        object.__setattr__(instance, "_data_path", target)
        cls._instance = instance
        cls._spatial_features_cache.clear()
        return instance

    @classmethod
    def clear(cls, path: Path | str | None = None) -> None:
        """Deletes the compiled ``countries.joblib`` file from disk and resets all singleton caches.

        Args:
            path: Optional custom path to remove. Defaults to ``get_default_path()``.
        """
        super().clear(path)
        cls._instance = None
        cls._spatial_features_cache.clear()

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

    @staticmethod
    async def _fetch_sources(
        ne_countries_url: str,
        ne_tiny_countries_url: str,
        mledoze_url: str,
        opensanctions_url: str,
    ) -> tuple[bytes, bytes, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
        """Concurrently fetches Natural Earth shapefiles, mledoze JSON, OpenSanctions YAML, and World Bank profiles."""
        import aiohttp

        headers = {"User-Agent": "SeroEpi/1.0"}
        async with aiohttp.ClientSession(headers=headers) as session:

            async def fetch_bytes(url: str) -> bytes:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to fetch {url}: HTTP {response.status}")
                    return await response.read()

            wb_client = WorldBankClient()
            try:
                ne_c_bytes, ne_t_bytes, ml_bytes, os_bytes, wb_profiles = await asyncio.gather(
                    fetch_bytes(ne_countries_url),
                    fetch_bytes(ne_tiny_countries_url),
                    fetch_bytes(mledoze_url),
                    fetch_bytes(opensanctions_url),
                    wb_client.get_country_profiles(),
                )
            finally:
                await wb_client.close()

            mledoze_data = json.loads(ml_bytes.decode("utf-8"))
            opensanctions_data = yaml.safe_load(os_bytes.decode("utf-8"))
            return ne_c_bytes, ne_t_bytes, mledoze_data, opensanctions_data, wb_profiles

    @staticmethod
    def _extract_geometries(ne_countries_bytes: bytes, ne_tiny_bytes: bytes) -> dict[str, Any]:
        """Extracts sovereign and microstate country geometries from Natural Earth shapefiles."""
        geom_map: dict[str, Any] = {}

        # 1. Read Natural Earth sovereign shapefile
        _, table_countries = pyogrio.read_arrow(io.BytesIO(ne_countries_bytes))
        adm0_list = table_countries["ADM0_A3"].to_pylist() if "ADM0_A3" in table_countries.column_names else []
        iso_list = table_countries["ISO_A3"].to_pylist() if "ISO_A3" in table_countries.column_names else []
        wkb_list = table_countries["wkb_geometry"].to_pylist()
        geoms = shapely.from_wkb(wkb_list)

        for i in range(table_countries.num_rows):
            if geoms[i] is None:
                continue
            for candidate in (
                str(iso_list[i]).strip().upper() if iso_list else "",
                str(adm0_list[i]).strip().upper() if adm0_list else "",
            ):
                if candidate and candidate not in ("-99", "", "NONE") and candidate not in geom_map:
                    geom_map[candidate] = geoms[i]

        # 2. Enrich microstates with Natural Earth tiny countries
        _, table_tiny = pyogrio.read_arrow(io.BytesIO(ne_tiny_bytes))
        adm0_tiny = table_tiny["ADM0_A3"].to_pylist() if "ADM0_A3" in table_tiny.column_names else []
        sov_tiny = table_tiny["SOV_A3"].to_pylist() if "SOV_A3" in table_tiny.column_names else []
        wkb_tiny = table_tiny["wkb_geometry"].to_pylist()
        geoms_tiny = shapely.from_wkb(wkb_tiny)

        for i in range(table_tiny.num_rows):
            iso = str(adm0_tiny[i] if adm0_tiny else (sov_tiny[i] if sov_tiny else "")).strip().upper()
            if iso and iso not in ("-99", "", "NONE") and geoms_tiny[i] is not None and iso not in geom_map:
                geom_map[iso] = geoms_tiny[i].buffer(0.15)

        return geom_map

    @classmethod
    def compile(
        cls,
        ne_countries_url: str = DEFAULT_NE_COUNTRIES_URL,
        ne_tiny_countries_url: str = DEFAULT_NE_TINY_COUNTRIES_URL,
        mledoze_url: str = DEFAULT_MLEDOZE_URL,
        opensanctions_url: str = DEFAULT_OPENSANCTIONS_URL,
    ) -> Self:
        """Fetches remote data, compiles ``CountryRecord`` instances, and builds ``CountryData`` with pre-warmed caches.

        Args:
            ne_countries_url: URL for Natural Earth 110m Admin-0 countries shapefile zip.
            ne_tiny_countries_url: URL for Natural Earth 110m tiny countries shapefile zip.
            mledoze_url: URL for ``mledoze/countries`` JSON metadata.
            opensanctions_url: URL for OpenSanctions country name aliases YAML.

        Returns:
            Compiled ``CountryData`` instance with pre-built ``STRtree`` and ``aliases_df``.
        """

        def _run_fetch() -> tuple[bytes, bytes, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
            return asyncio.run(
                cls._fetch_sources(
                    ne_countries_url=ne_countries_url,
                    ne_tiny_countries_url=ne_tiny_countries_url,
                    mledoze_url=mledoze_url,
                    opensanctions_url=opensanctions_url,
                )
            )

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            with ThreadPoolExecutor(max_workers=1) as pool:
                ne_c_bytes, ne_t_bytes, mledoze_data, opensanctions_data, wb_profiles = pool.submit(_run_fetch).result()
        else:
            ne_c_bytes, ne_t_bytes, mledoze_data, opensanctions_data, wb_profiles = _run_fetch()
        geom_map = cls._extract_geometries(ne_c_bytes, ne_t_bytes)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        records: list[CountryRecord] = []
        seen_iso3: set[str] = set()

        for c in mledoze_data:
            rec = CountryRecord.from_sources(c, geom_map, wb_profiles, opensanctions_data, transformer)
            if rec is not None and rec.iso3 not in seen_iso3:
                seen_iso3.add(rec.iso3)
                records.append(rec)

        df = pl.DataFrame([asdict(r) for r in records])

        # Pre-warm spatial STRtree index & aliases DataFrame for 0ms cold-start query latency
        iso3s = df["iso3"].to_list()
        areas = df["geometry_area"].to_list()
        geoms = shapely.from_wkb(df["geometry"].to_list())
        tree_4326 = shapely.STRtree(geoms)

        aliases_df = (
            df.select(["iso3", "aliases"])
            .explode("aliases", empty_as_null=True)
            .filter(pl.col("aliases").is_not_null())
            .select([pl.col("aliases").str.to_uppercase().alias("alias"), pl.col("iso3")])
            .unique(subset=["alias"], keep="first")
        )

        return cls(
            data=df,
            variables=dict(cls.VARIABLES),
            name="country_data",
            _aliases_df=aliases_df,
            _tree_4326=tree_4326,
            _country_iso3s=iso3s,
            _country_areas=areas,
            _geoms_4326=geoms,
        )
