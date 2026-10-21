"""Asynchronous and synchronous client for World Bank Open Data API."""

import asyncio
from typing import Any

from seroepi.client.base import Client


class WorldBankClient(Client):
    """Client for querying the World Bank Open Data API v2.

    Provides access to country demographics, income classifications, lending types,
    and historical/recent development indicators (e.g. Life Expectancy ``SP.DYN.LE00.IN``).

    Args:
        base_url: Base endpoint for the World Bank API.
        user_agent: Custom User-Agent header value.
        timeout: Request timeout in seconds.

    Attributes:
        DEFAULT_BASE_URL: Default World Bank REST API v2 base URL.

    Examples:
        >>> from seroepi.client.world_bank import WorldBankClient
        >>> client = WorldBankClient()
        >>> client.base_url
        'https://api.worldbank.org/v2'
    """

    DEFAULT_BASE_URL: str = "https://api.worldbank.org/v2"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        user_agent: str = "SeroEpi-WorldBankClient/0.1",
        timeout: float = 30.0,
    ) -> None:
        """Initializes the World Bank API client.

        Args:
            base_url: Base endpoint for the World Bank API.
            user_agent: Custom User-Agent header value.
            timeout: Request timeout in seconds.
        """
        super().__init__(base_url=base_url, user_agent=user_agent, timeout=timeout)

    async def get_countries(self, page_size: int = 500) -> list[dict[str, Any]]:
        """Retrieves global country metadata including income levels and regions.

        Args:
            page_size: Number of records per API page.

        Returns:
            List of raw country records containing incomeLevel, lendingType, and capitalCity.
        """
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
        """Retrieves country observations for a specific World Bank indicator.

        Args:
            indicator: Indicator code (defaults to Life Expectancy at birth SP.DYN.LE00.IN).
            mrnev: Most Recent Non-Empty Value count (1 returns latest available value).
            page_size: Number of records per API page.

        Returns:
            List of observation records containing countryiso3code, date, and value.
        """
        data = await self.get(f"country/all/indicator/{indicator}?format=json&mrnev={mrnev}&per_page={page_size}")
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            return data[1]
        return []

    async def get_country_profiles(self, indicator: str = "SP.DYN.LE00.IN") -> dict[str, dict[str, Any]]:
        """Concurrently fetches country classifications and indicator values.

        Args:
            indicator: Indicator code to attach to profiles (defaults to Life Expectancy).

        Returns:
            Dictionary keyed by uppercase ISO3 country code containing income level,
            capital city, lending type, and latest indicator value.
        """
        countries_task = self.get_countries()
        indicator_task = self.get_indicator(indicator=indicator)

        raw_countries, raw_indicators = await asyncio.gather(countries_task, indicator_task)

        profiles: dict[str, dict[str, Any]] = {}

        for c in raw_countries:
            iso3 = str(c.get("id", "")).strip().upper()
            if not iso3:
                continue

            raw_income = c.get("incomeLevel")
            income_dict = raw_income if isinstance(raw_income, dict) else {}

            raw_lending = c.get("lendingType")
            lending_dict = raw_lending if isinstance(raw_lending, dict) else {}

            raw_region = c.get("region")
            region_dict = raw_region if isinstance(raw_region, dict) else {}

            profiles[iso3] = {
                "iso3": iso3,
                "iso2": str(c.get("iso2Code", "")).strip().upper(),
                "name": str(c.get("name", "")).strip(),
                "capital_city": str(c.get("capitalCity", "")).strip(),
                "income_level": str(income_dict.get("value", "")).strip(),
                "income_level_code": str(income_dict.get("id", "")).strip().upper(),
                "lending_type": str(lending_dict.get("value", "")).strip(),
                "world_bank_region": str(region_dict.get("value", "")).strip(),
                "life_expectancy": None,
                "life_expectancy_year": None,
            }

        for ind in raw_indicators:
            iso3 = str(ind.get("countryiso3code", "")).strip().upper()
            if not iso3:
                continue

            val = ind.get("value")
            numeric_val = float(val) if val is not None else None
            date_str = str(ind.get("date", "")).strip()

            if iso3 in profiles:
                profiles[iso3]["life_expectancy"] = numeric_val
                profiles[iso3]["life_expectancy_year"] = date_str
            else:
                profiles[iso3] = {
                    "iso3": iso3,
                    "iso2": "",
                    "name": "",
                    "capital_city": "",
                    "income_level": "",
                    "income_level_code": "",
                    "lending_type": "",
                    "world_bank_region": "",
                    "life_expectancy": numeric_val,
                    "life_expectancy_year": date_str,
                }

        return profiles

    def get_countries_sync(self, page_size: int = 500) -> list[dict[str, Any]]:
        """Synchronously retrieves global country metadata.

        Args:
            page_size: Number of records per API page.

        Returns:
            List of raw country metadata dictionaries.
        """
        return self.run_sync(self.get_countries(page_size))

    def get_indicator_sync(
        self,
        indicator: str = "SP.DYN.LE00.IN",
        mrnev: int = 1,
        page_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Synchronously retrieves country observations for a World Bank indicator.

        Args:
            indicator: World Bank indicator code.
            mrnev: Most Recent Non-Empty Value count.
            page_size: Number of records per API page.

        Returns:
            List of indicator observation dictionaries.
        """
        return self.run_sync(self.get_indicator(indicator, mrnev, page_size))

    def get_country_profiles_sync(self, indicator: str = "SP.DYN.LE00.IN") -> dict[str, dict[str, Any]]:
        """Synchronously fetches combined country classifications and latest indicator values.

        Args:
            indicator: World Bank indicator code to attach to profiles.

        Returns:
            Dictionary keyed by uppercase ISO3 code containing socioeconomic metadata.
        """
        return self.run_sync(self.get_country_profiles(indicator))
