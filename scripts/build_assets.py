#!/usr/bin/env python3
"""Compile static SeroEpi package assets (`countries.joblib` and `klebnet_neonatal_sepsis.joblib`).

Moves the build-time data compilation pipelines (Natural Earth shapefiles, mledoze/countries,
OpenSanctions aliases, World Bank indicators, and Pathogenwatch KlebNET benchmark collection)
out of the core runtime package into `scripts/build_assets.py`.
"""

import argparse
import asyncio
import io
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl
import pyogrio
import pyproj
import shapely
import shapely.ops
import yaml
from shapely.geometry import Point

from seroepi.client.base import Client
from seroepi.client.pathogenwatch import PathogenwatchClient
from seroepi.country import CountryData
from seroepi.dataset import Dataset

ASSETS_DIR = Path(__file__).resolve().parents[1] / "src" / "seroepi" / "assets"
COUNTRIES_ASSET = ASSETS_DIR / "countries.joblib"
KLEBNET_ASSET = ASSETS_DIR / "klebnet_neonatal_sepsis.joblib"
KLEBNET_COLLECTION_UUID = "myp3PKqcM1tiqWVmFcEVHn"

DEFAULT_NE_COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
DEFAULT_NE_TINY_COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_tiny_countries.zip"
DEFAULT_MLEDOZE_URL = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"
DEFAULT_OPENSANCTIONS_URL = "https://raw.githubusercontent.com/opensanctions/countrynames/main/countrynames/data.yaml"


class WorldBankClient(Client):
    """Build-time client for querying the World Bank Open Data API v2."""

    DEFAULT_BASE_URL: str = "https://api.worldbank.org/v2"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = "SeroEpi-WorldBankClient/0.1",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(base_url=base_url, user_agent=user_agent, timeout=timeout)

    async def get_countries(self, page_size: int = 500) -> list[dict[str, Any]]:
        data = await self.get(f"country?format=json&per_page={page_size}")
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return data[1]
        return []

    async def get_indicator(
        self,
        indicator: str = "SP.DYN.LE00.IN",
        mrnev: int = 1,
        page_size: int = 500,
    ) -> list[dict[str, Any]]:
        data = await self.get(f"country/all/indicator/{indicator}?format=json&mrnev={mrnev}&per_page={page_size}")
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return data[1]
        return []

    async def get_country_profiles(self, indicator: str = "SP.DYN.LE00.IN") -> dict[str, dict[str, Any]]:
        raw_countries, raw_indicators = await asyncio.gather(
            self.get_countries(),
            self.get_indicator(indicator=indicator),
        )
        profiles: dict[str, dict[str, Any]] = {}
        for c in raw_countries:
            iso3 = str(c.get("id", "")).strip().upper()
            if not iso3 or len(iso3) != 3:
                continue
            inc = c.get("incomeLevel") or {}
            lend = c.get("lendingType") or {}
            reg = c.get("region") or {}
            if str(reg.get("id", "")).strip() == "NA":
                continue
            profiles[iso3] = {
                "iso3": iso3,
                "name": str(c.get("name", "")).strip(),
                "capital_city": str(c.get("capitalCity", "")).strip(),
                "income_level": str(inc.get("value", "")).strip(),
                "income_level_code": str(inc.get("id", "")).strip(),
                "lending_type": str(lend.get("value", "")).strip(),
                "world_bank_region": str(reg.get("value", "")).strip(),
                "life_expectancy": None,
                "life_expectancy_year": "",
            }

        for obs in raw_indicators:
            iso3 = str(obs.get("countryiso3code", "")).strip().upper()
            val = obs.get("value")
            date = str(obs.get("date", "")).strip()
            if not iso3 or val is None:
                continue
            entry = profiles.setdefault(
                iso3,
                {
                    "iso3": iso3,
                    "name": str((obs.get("country") or {}).get("value", "")).strip(),
                    "capital_city": "",
                    "income_level": "",
                    "income_level_code": "",
                    "lending_type": "",
                    "world_bank_region": "",
                    "life_expectancy": None,
                    "life_expectancy_year": "",
                },
            )
            if entry["life_expectancy"] is None:
                entry["life_expectancy"] = float(val)
                entry["life_expectancy_year"] = date

        return profiles


@dataclass(frozen=True, slots=True)
class CountryRecord:
    """Strongly-typed metadata and geometry record for a sovereign or dependent country."""

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

        for key in (iso2, iso3):
            if key and key in opensanctions_data and isinstance(opensanctions_data[key], list):
                for item in opensanctions_data[key]:
                    clean = str(item).strip().upper()
                    if clean:
                        aliases.add(clean)

        if iso2:
            prefix = f"{iso2}-"
            for os_key, os_items in opensanctions_data.items():
                if os_key.startswith(prefix) and isinstance(os_items, list):
                    for item in os_items:
                        clean = str(item).strip().upper()
                        if clean:
                            aliases.add(clean)

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


async def _fetch_country_sources() -> tuple[
    bytes, bytes, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]
]:
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
                fetch_bytes(DEFAULT_NE_COUNTRIES_URL),
                fetch_bytes(DEFAULT_NE_TINY_COUNTRIES_URL),
                fetch_bytes(DEFAULT_MLEDOZE_URL),
                fetch_bytes(DEFAULT_OPENSANCTIONS_URL),
                wb_client.get_country_profiles(),
            )
        finally:
            await wb_client.close()

        mledoze_data = json.loads(ml_bytes.decode("utf-8"))
        opensanctions_data = yaml.safe_load(os_bytes.decode("utf-8"))
        return ne_c_bytes, ne_t_bytes, mledoze_data, opensanctions_data, wb_profiles


def _extract_geometries(ne_countries_bytes: bytes, ne_tiny_bytes: bytes) -> dict[str, Any]:
    geom_map: dict[str, Any] = {}
    _, table_countries = pyogrio.read_arrow(io.BytesIO(ne_countries_bytes))
    adm0_list = table_countries["ADM0_A3"].to_pylist() if "ADM0_A3" in table_countries.column_names else []
    iso_list = table_countries["ISO_A3"].to_pylist() if "ISO_A3" in table_countries.column_names else []
    geoms = shapely.from_wkb(table_countries["wkb_geometry"].to_pylist())

    for i in range(table_countries.num_rows):
        if geoms[i] is None:
            continue
        for candidate in (
            str(iso_list[i]).strip().upper() if iso_list else "",
            str(adm0_list[i]).strip().upper() if adm0_list else "",
        ):
            if candidate and candidate not in ("-99", "", "NONE") and candidate not in geom_map:
                geom_map[candidate] = geoms[i]

    _, table_tiny = pyogrio.read_arrow(io.BytesIO(ne_tiny_bytes))
    adm0_tiny = table_tiny["ADM0_A3"].to_pylist() if "ADM0_A3" in table_tiny.column_names else []
    sov_tiny = table_tiny["SOV_A3"].to_pylist() if "SOV_A3" in table_tiny.column_names else []
    geoms_tiny = shapely.from_wkb(table_tiny["wkb_geometry"].to_pylist())

    for i in range(table_tiny.num_rows):
        iso = str(adm0_tiny[i] if adm0_tiny else (sov_tiny[i] if sov_tiny else "")).strip().upper()
        if iso and iso not in ("-99", "", "NONE") and geoms_tiny[i] is not None and iso not in geom_map:
            geom_map[iso] = geoms_tiny[i].buffer(0.15)

    return geom_map


def build_country_data(out_path: Path = COUNTRIES_ASSET) -> CountryData:
    """Compiles and saves `CountryData` to `out_path`."""
    ne_c_bytes, ne_t_bytes, mledoze_data, opensanctions_data, wb_profiles = asyncio.run(_fetch_country_sources())
    geom_map = _extract_geometries(ne_c_bytes, ne_t_bytes)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    records: list[CountryRecord] = []
    seen_iso3: set[str] = set()
    for c in mledoze_data:
        rec = CountryRecord.from_sources(c, geom_map, wb_profiles, opensanctions_data, transformer)
        if rec is not None and rec.iso3 not in seen_iso3:
            seen_iso3.add(rec.iso3)
            records.append(rec)

    df = pl.DataFrame([asdict(r) for r in records])
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

    cd = CountryData(
        data=df,
        variables=dict(CountryData.VARIABLES),
        name="country_data",
        _aliases_df=aliases_df,
        _tree_4326=tree_4326,
        _country_iso3s=iso3s,
        _country_areas=areas,
        _geoms_4326=geoms,
    )
    cd.save(out_path)
    print(f"Saved CountryData ({len(cd):,} countries) -> {out_path}")
    return cd


def build_klebnet_dataset(api_key: str, out_path: Path = KLEBNET_ASSET) -> Dataset:
    """Fetches and saves the KlebNET Neonatal Sepsis benchmark `Dataset` to `out_path`."""
    with PathogenwatchClient(api_key=api_key) as client:
        ds = client.fetch_collection_dataset_sync(
            uuid=KLEBNET_COLLECTION_UUID,
            with_distances=True,
            with_metadata=True,
            dataset_name="klebnet_neonatal_sepsis",
        )
    ds.save(out_path)
    print(f"Saved Dataset '{ds.name}' ({len(ds):,} isolates, {len(ds.variables)} variables) -> {out_path}")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static SeroEpi package assets.")
    parser.add_argument("--countries", action="store_true", help="Rebuild countries.joblib")
    parser.add_argument("--klebnet", action="store_true", help="Rebuild klebnet_neonatal_sepsis.joblib")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PATHOGENWATCH_API_KEY", ""),
        help="Pathogenwatch API key for --klebnet",
    )
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    run_all = not args.countries and not args.klebnet

    if args.countries or run_all:
        build_country_data(COUNTRIES_ASSET)
    if args.klebnet or run_all:
        if not args.api_key:
            raise SystemExit("Error: --api-key is required when building klebnet_neonatal_sepsis.joblib.")
        build_klebnet_dataset(args.api_key, KLEBNET_ASSET)


if __name__ == "__main__":
    main()
