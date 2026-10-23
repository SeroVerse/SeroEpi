"""Pathogenwatch REST API client and collection models."""

import asyncio
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from shiny_kit.progress import current_progress

from seroepi.client.base import Client
from seroepi.dataset import Dataset
from seroepi.io.dataset.base import DatasetParser, PathogenWatchFile
from seroepi.io.dataset.kaptive import PathogenWatchKaptiveParser
from seroepi.io.dataset.kleborate import KleborateParser
from seroepi.io.dataset.pathogenwatch import (
    PathogenWatchCoreParser,
    PathogenWatchLINCodeParser,
    PathogenWatchMLSTParser,
    PathogenWatchPlasmidFinderParser,
)
from seroepi.io.dist import PathogenwatchDistanceParser

_ = (
    KleborateParser,
    PathogenWatchKaptiveParser,
    PathogenWatchCoreParser,
    PathogenWatchMLSTParser,
    PathogenWatchLINCodeParser,
    PathogenWatchPlasmidFinderParser,
)


@dataclass(frozen=True, slots=True)
class PathogenwatchCollection:
    """Dataclass representing a Pathogenwatch collection metadata object.

    Attributes:
        uuid: Short UUID identifier of the collection.
        name: Human-readable title of the collection.
        id: Optional numeric or string internal Pathogenwatch collection ID.
        description: Optional collection description text.
        size: Total number of genomes in the collection, or ``None``.
        organism_id: Taxonomic organism identifier.
        organism_name: Human-readable species/organism name.
        owner: Ownership tag (e.g., ``"me"`` for user-owned collections).
        access: Visibility/access level (e.g., ``"PUBLIC"`` or ``"PRIVATE"``).
        binned: Whether the collection is binned/archived.
        created_at: ISO timestamp string when the collection was created.
    """

    uuid: str
    name: str
    id: int | str | None = None
    description: str | None = None
    size: int | None = None
    organism_id: str | None = None
    organism_name: str | None = None
    owner: str | None = None
    access: str | None = None
    binned: bool | None = None
    created_at: str | None = None

    ORGANISM_TAXID_MAP: ClassVar[dict[str, str]] = {
        "197": "Campylobacter jejuni",
        "210": "Helicobacter pylori",
        "287": "Pseudomonas aeruginosa",
        "470": "Acinetobacter baumannii",
        "485": "Neisseria gonorrhoeae",
        "487": "Neisseria meningitidis",
        "562": "Escherichia coli",
        "573": "Klebsiella pneumoniae",
        "590": "Salmonella",
        "620": "Shigella",
        "666": "Vibrio cholerae",
        "670": "Vibrio parahaemolyticus",
        "1280": "Staphylococcus aureus",
        "1311": "Streptococcus agalactiae",
        "1313": "Streptococcus pneumoniae",
        "1314": "Streptococcus pyogenes",
        "1336": "Streptococcus equi",
        "1351": "Enterococcus faecalis",
        "1352": "Enterococcus faecium",
        "1639": "Listeria monocytogenes",
        "1646": "Renibacterium salmoninarum",
        "1773": "Mycobacterium tuberculosis",
        "28901": "Salmonella enterica",
        "64320": "Zika virus",
        "90370": "Salmonella enterica serovar Typhi",
        "2697049": "SARS-CoV-2",
        "3390273": "Klebsiella pneumoniae species complex",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathogenwatchCollection":
        """Instantiates a ``PathogenwatchCollection`` from a raw API dictionary payload.

        Args:
            data: Raw dictionary returned by the Pathogenwatch ``collections/list`` endpoint.

        Returns:
            A populated ``PathogenwatchCollection`` instance.
        """
        uuid_val = str(data.get("uuid") or data.get("id") or "")
        name_val = str(data.get("name") or "")
        size_val = data.get("size")
        if size_val is None:
            size_val = data.get("genomeCount") if data.get("genomeCount") is not None else data.get("genome_count")
        org_id_raw = data.get("organismId") or data.get("organism_id")
        org_id = str(org_id_raw).strip() if org_id_raw is not None and str(org_id_raw).strip() else None
        org_name_raw = data.get("organismName") or data.get("organism_name")
        org_name = str(org_name_raw).strip() if org_name_raw is not None and str(org_name_raw).strip() else None
        if org_name is None and org_id is not None:
            org_name = cls.ORGANISM_TAXID_MAP.get(org_id)

        return cls(
            uuid=uuid_val,
            name=name_val,
            id=data.get("id"),
            description=data.get("description"),
            size=int(size_val) if size_val is not None else None,
            organism_id=org_id,
            organism_name=org_name,
            owner=data.get("owner"),
            access=data.get("access"),
            binned=data.get("binned"),
            created_at=data.get("createdAt") or data.get("created_at"),
        )

    @property
    def organism_label(self) -> str | None:
        """Returns the resolved human-readable species/organism label for grouping, or ``None``.

        Returns:
            Species name (e.g., ``"Klebsiella pneumoniae"``), ``"Organism (<id>)"``, or ``None``.
        """
        if self.organism_name and self.organism_name.strip():
            return self.organism_name.strip()
        if self.organism_id and self.organism_id.strip():
            oid = self.organism_id.strip()
            return self.ORGANISM_TAXID_MAP.get(oid, f"Organism ({oid})")
        return None

    @property
    def genome_count(self) -> int | None:
        """Returns the collection genome count.

        Returns:
            Number of genomes in the collection, or ``None`` if unknown.
        """
        return self.size

    @property
    def is_owned(self) -> bool:
        """Returns whether this collection belongs to the authenticated user (``"me"``).

        Returns:
            ``True`` if ``self.owner == "me"``, otherwise ``False``.
        """
        return self.owner == "me"

    @property
    def display_label(self) -> str:
        """Returns a human-readable selector label including genome count when available.

        Returns:
            Formatted string ``"<name> (n=<size>)"`` or ``"<name>"``.
        """
        title = self.name.strip() or self.uuid
        if self.size is not None:
            return f"{title} (n={self.size:,})"
        return title

    @staticmethod
    def group_choices(collections: Sequence["PathogenwatchCollection"]) -> dict[str, Any]:
        """Builds grouped selectize choices organized by species/organism (with owned collections first).

        When ``organism_id`` or ``organism_name`` is populated on collections, groups choices by
        ``organism_label`` (e.g., ``"Klebsiella pneumoniae"``, ``"Acinetobacter baumannii"``),
        sorting user-owned collections to the top of each organism group with a ``"★ "`` prefix.
        Collections without organism metadata fall back to ``"My Collections"`` and ``"Public / Shared"``.

        Args:
            collections: Sequence of ``PathogenwatchCollection`` records to partition.

        Returns:
            Nested dictionary mapping group headers to ``{uuid: display_label}`` dicts.
        """
        valid = [c for c in collections if c.uuid and (c.size is None or c.size > 0)]
        by_organism: dict[str, list[PathogenwatchCollection]] = {}
        untyped_owned: dict[str, str] = {}
        untyped_shared: dict[str, str] = {}

        for c in valid:
            org_label = c.organism_label
            if org_label is not None:
                by_organism.setdefault(org_label, []).append(c)
            elif c.is_owned:
                untyped_owned[c.uuid] = c.display_label
            else:
                untyped_shared[c.uuid] = c.display_label

        grouped: dict[str, Any] = {}
        sorted_org_keys = sorted(
            by_organism.keys(),
            key=lambda k: (not any(c.is_owned for c in by_organism[k]), -len(by_organism[k]), k),
        )
        for org_key in sorted_org_keys:
            items = sorted(by_organism[org_key], key=lambda c: (not c.is_owned, c.name.lower()))
            grouped[org_key] = {c.uuid: f"★ {c.display_label}" if c.is_owned else c.display_label for c in items}

        if untyped_owned:
            grouped["My Collections"] = untyped_owned
        if untyped_shared:
            grouped["Public / Shared"] = untyped_shared
        return grouped


class PathogenwatchClient(Client):
    """Asynchronous and synchronous client for interacting with the Pathogenwatch REST API.

    Args:
        api_key: Required Pathogenwatch API key string.
        base_url: Base URL for the Pathogenwatch API. Defaults to ``"https://pathogen.watch/api"``.
        timeout: Request timeout in seconds. Defaults to ``30.0``.

    Attributes:
        base_url: Resolved Pathogenwatch REST API base endpoint ending in ``/api``.

    Examples:
        >>> from seroepi.client.pathogenwatch import PathogenwatchClient
        >>> client = PathogenwatchClient(api_key="your-api-key", base_url="https://pathogen.watch/api")
        >>> client.base_url
        'https://pathogen.watch/api'
    """

    DEFAULT_BASE_URL: ClassVar[str] = "https://pathogen.watch/api"

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initializes the Pathogenwatch API client.

        Args:
            api_key: Required Pathogenwatch API key string.
            base_url: Base URL for the Pathogenwatch API.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If ``api_key`` is missing or blank.
        """
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty Pathogenwatch API key is required.")
        resolved_url = base_url.rstrip("/")
        if not resolved_url.endswith("/api"):
            resolved_url = f"{resolved_url}/api"

        super().__init__(
            base_url=resolved_url,
            api_key_header="X-API-Key",
            api_key=clean_key,
            timeout=timeout,
        )

    async def get_collections(
        self,
        owner: str | None = None,
        exclude: str | None = None,
        binned: bool | None = None,
        limit: int | None = None,
    ) -> list[PathogenwatchCollection]:
        """Fetches collections available to the authenticated account or public collections.

        Args:
            owner: Filter by ownership (e.g. 'me', 'me,shared', 'shared').
            exclude: Exclude collections of a given type ('PUBLIC' or 'PRIVATE').
            binned: Whether to include binned collections.
            limit: Maximum number of collections to return.

        Returns:
            List of PathogenwatchCollection instances.
        """
        params: dict[str, str] = {}
        if owner is not None:
            params["owner"] = owner
        if exclude is not None:
            params["exclude"] = exclude
        if binned is not None:
            params["binned"] = "true" if binned else "false"
        if limit is not None:
            params["limit"] = str(limit)

        res = await self.get("collections/list", params=params or None)
        if isinstance(res, list):
            return [PathogenwatchCollection.from_dict(c) for c in res]
        return []

    async def get_collection(self, uuid: str) -> dict[str, Any]:
        """Fetches complete metadata and analysis jobs for a specific collection.

        Args:
            uuid: Short UUID identifier of the collection.

        Returns:
            Dictionary containing collection details, downloads, and tree configurations.
        """
        res = await self.get("collections/details", params={"uuid": uuid})
        return res if isinstance(res, dict) else {}

    async def get_collection_genomes(
        self,
        uuid: str,
        share_id: str | None = None,
        page: int = 1,
        all_pages: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetches genome analysis records for a collection.

        Args:
            uuid: Short UUID of the collection.
            share_id: Optional secure ID used with share links.
            page: 1-based page number (defaults to 1).
            all_pages: If True, automatically paginates through all pages until complete.

        Returns:
            List of raw genome record dictionaries.
        """
        all_records: list[dict[str, Any]] = []
        current_page = page

        while True:
            params: dict[str, str] = {"uuid": uuid, "page": str(current_page)}
            if share_id:
                params["shareId"] = share_id

            res = await self.get("collections/genomes", params=params)
            if not isinstance(res, dict):
                break

            records = res.get("genomes") or res.get("genome") or []
            all_records.extend(records)

            if not all_pages or not res.get("hasMore") or not records:
                break

            current_page += 1

        return all_records

    async def download_distance_matrix(
        self,
        uuid: str,
        matrix_type: str = "difference",
    ) -> str:
        """Downloads pairwise genomic distance matrix CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            matrix_type: Distance metric, either 'difference' or 'score'. Defaults to 'difference'.

        Returns:
            Raw CSV text representing the symmetric distance matrix.
        """
        res = await self.get(f"downloads/matrix/{matrix_type}", params={"uuid": uuid})
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from matrix endpoint, got {type(res)}")
        return res

    async def download_analysis(
        self,
        uuid: str,
        download_type: str,
        job: str | None = None,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads discrete analysis CSV results (e.g. Kleborate, Kaptive) for a collection.

        Args:
            uuid: Short UUID of the collection.
            download_type: Analysis type name (e.g. 'kleborate', 'kaptive').
            job: The specific job ID. If None, auto-resolved from collection downloads.
            genome_ids: Specific genome IDs to include. If None, auto-resolved from collection.
            prefix: Optional filename prefix.

        Returns:
            Raw CSV string.
        """
        if job is None:
            details = await self.get_collection(uuid)
            downloads = details.get("downloads", [])
            for d in downloads:
                if d.get("name", "").lower() == download_type.lower():
                    job = d.get("job")
                    break
            if not job:
                raise ValueError(
                    f"Analysis download '{download_type}' not found in collection '{uuid}'. "
                    f"Available downloads: {[d.get('name') for d in downloads]}"
                )

        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {"job": job}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            f"downloads/{download_type.lower()}",
            params=params,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/{download_type}, got {type(res)}")
        return res

    async def download_kleborate(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads Kleborate analysis CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs to include.
            prefix: Optional filename prefix.

        Returns:
            Raw Kleborate CSV string.
        """
        return await self.download_analysis(
            uuid=uuid,
            download_type="kleborate",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    async def download_kaptive(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads Kaptive analysis CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs to include.
            prefix: Optional filename prefix.

        Returns:
            Raw Kaptive CSV string.
        """
        return await self.download_analysis(
            uuid=uuid,
            download_type="kaptive",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    async def download_metadata(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads isolate metadata CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs to include.
            prefix: Optional filename prefix.

        Returns:
            Raw metadata CSV string.

        Raises:
            ValueError: If no genomes are found or the API response is not CSV text.
        """
        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            "downloads/metadata",
            params=params or None,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/metadata, got {type(res)}")
        return res

    async def download_summary(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads summary CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs to include.
            prefix: Optional filename prefix.

        Returns:
            Raw summary CSV string.

        Raises:
            ValueError: If no genomes are found or the API response is not CSV text.
        """
        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            "downloads/summary",
            params=params or None,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/summary, got {type(res)}")
        return res

    async def download_collection_files(
        self,
        uuid: str,
        output_dir: str | Path,
        analysis_types: Sequence[str] = ("kleborate", "kaptive"),
        include_metadata: bool = True,
        include_matrix: bool = True,
        matrix_type: str = "difference",
        prefix: str | None = None,
    ) -> dict[str, Path]:
        """Downloads discrete collection files to an output directory on disk.

        Args:
            uuid: Short UUID of the collection.
            output_dir: Directory where downloaded files should be stored.
            analysis_types: Sequence of analysis downloads to fetch (e.g. 'kleborate', 'kaptive').
            include_metadata: Whether to download metadata CSV.
            include_matrix: Whether to download distance matrix CSV.
            matrix_type: 'difference' or 'score'.
            prefix: Optional filename prefix.

        Returns:
            Dictionary mapping asset kind to Path on disk.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        # Concurrently resolve collection details, isolate list, and optional distance matrix
        async def _try_matrix_raw() -> str | None:
            if not include_matrix:
                return None
            return await self.download_distance_matrix(uuid=uuid, matrix_type=matrix_type)

        details, records, matrix_csv = await asyncio.gather(
            self.get_collection(uuid),
            self.get_collection_genomes(uuid, all_pages=True),
            _try_matrix_raw(),
        )
        genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")
        available_downloads = {d.get("name", "").lower(): d.get("job") for d in details.get("downloads", [])}

        tasks = []
        task_names = []

        for at in analysis_types:
            at_lower = at.lower()
            if at_lower in available_downloads:
                tasks.append(
                    self.download_analysis(
                        uuid=uuid,
                        download_type=at_lower,
                        job=available_downloads[at_lower],
                        genome_ids=genome_ids,
                        prefix=prefix,
                    )
                )
                task_names.append(at_lower)

        if include_metadata:
            tasks.append(self.download_metadata(uuid=uuid, genome_ids=genome_ids, prefix=prefix))
            task_names.append("metadata")

        results = list(await asyncio.gather(*tasks)) if tasks else []
        if include_matrix and matrix_csv is not None:
            task_names.append(f"{matrix_type}_matrix")
            results.append(matrix_csv)

        async def _write_csv(name: str, csv_text: str) -> tuple[str, Path]:
            fpath = out / f"{name}.csv"
            await asyncio.to_thread(fpath.write_text, csv_text, encoding="utf-8")
            return name, fpath

        written = await asyncio.gather(*(_write_csv(n, txt) for n, txt in zip(task_names, results, strict=True)))
        for name, fpath in written:
            paths[name] = fpath

        return paths

    @staticmethod
    def default_analysis_types(downloads: Sequence[dict[str, Any]]) -> tuple[str, ...]:
        """Selects the default genotype analysis types from a collection's ``downloads`` list.

        Prioritizes ``"kleborate"``, then ``"kaptive"``, and otherwise falls back to the first
        available analysis in ``downloads``.

        Args:
            downloads: Sequence of download dictionaries from ``get_collection(uuid)["downloads"]``.

        Returns:
            Tuple of default analysis names (e.g., ``("kleborate",)`` or ``("kaptive",)``).
        """
        names = [
            str(d["name"]).strip().lower()
            for d in downloads
            if isinstance(d, dict) and d.get("name") and str(d["name"]).strip()
        ]
        if "kleborate" in names:
            return ("kleborate",)
        if "kaptive" in names:
            return ("kaptive",)
        if names:
            return (names[0],)
        return ()

    async def fetch_collection_dataset(
        self,
        uuid: str,
        with_distances: bool = True,
        with_metadata: bool = True,
        matrix_type: str = "difference",
        dataset_name: str | None = None,
        strict_distances: bool = False,
        analysis_types: Sequence[str] | str | None = None,
    ) -> Dataset:
        """Downloads collection files in-memory and compiles a unified SeroEpi Dataset.

        Args:
            uuid: Short UUID of the collection.
            with_distances: Whether to download and attach pairwise GenomicDistances.
            with_metadata: Whether to download and enrich with spatiotemporal metadata.
            matrix_type: Distance matrix type ('difference' or 'score').
            dataset_name: Optional custom Dataset name. Defaults to collection name.
            strict_distances: If True, raise if the distance matrix endpoint fails on the
                Pathogenwatch server. If False (default), emit a RuntimeWarning and return
                the dataset without distances when Pathogenwatch cannot generate the matrix.
            analysis_types: Optional analysis name or sequence of analysis names to fetch and enrich
                (e.g., ``("kleborate", "kaptive")`` or ``("kaptive",)``). When ``None``, automatically
                selects ``"kleborate"`` if available, otherwise ``"kaptive"``.

        Returns:
            A unified Dataset with variables, metadata, and attached GenomicDistances.
        """
        prog = current_progress()
        prog.report(
            0.05,
            message="Fetching Pathogenwatch collection",
            detail=f"Metadata and isolate list for {uuid}...",
            force=True,
        )

        # Start distance matrix request immediately alongside collection details and isolate pagination
        # since downloads/matrix/{matrix_type} only requires `uuid`.
        async def _try_matrix() -> tuple[str | None, Exception | None]:
            if not with_distances:
                return None, None
            try:
                return await self.download_distance_matrix(uuid=uuid, matrix_type=matrix_type), None
            except Exception as exc:
                return None, exc

        matrix_task = asyncio.create_task(_try_matrix())
        try:
            details, records = await asyncio.gather(
                self.get_collection(uuid),
                self.get_collection_genomes(uuid, all_pages=True),
            )
        except Exception:
            matrix_task.cancel()
            raise

        resolved_name = dataset_name or details.get("name") or f"Pathogenwatch {uuid}"
        genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            matrix_task.cancel()
            raise ValueError(f"Collection '{resolved_name}' ({uuid}) contains no genomes.")
        raw_downloads = details.get("downloads", [])
        available_downloads = {
            str(d.get("name", "")).strip().lower(): d.get("job")
            for d in raw_downloads
            if isinstance(d, dict) and d.get("name")
        }

        if analysis_types is None:
            requested_analyses = [a for a in ("kleborate", "kaptive") if a in available_downloads][:1]
        elif isinstance(analysis_types, str):
            clean_a = analysis_types.strip().lower()
            requested_analyses = [clean_a] if clean_a and clean_a in available_downloads else []
        else:
            requested_analyses = []
            for a in analysis_types:
                clean_a = str(a).strip().lower()
                if clean_a and clean_a in available_downloads and clean_a not in requested_analyses:
                    requested_analyses.append(clean_a)

        prog.report(
            0.25,
            message=f"Downloading '{resolved_name}' ({len(genome_ids):,} isolates)",
            detail="Requesting genotypes, metadata, and distance matrix...",
            force=True,
        )
        completed_downloads = 0
        total_downloads = max(1, len(requested_analyses)) + int(with_metadata) + int(with_distances)

        def _bump_download(stage_label: str) -> None:
            nonlocal completed_downloads
            completed_downloads += 1
            frac = 0.25 + 0.55 * (completed_downloads / max(1, total_downloads))
            prog.report(frac, message=f"Downloading '{resolved_name}'", detail=stage_label, force=True)

        async def _fetch_single_analysis(kind: str) -> tuple[str, str]:
            text = await self.download_analysis(
                uuid=uuid,
                download_type=kind,
                job=available_downloads[kind],
                genome_ids=genome_ids,
            )
            _bump_download(f"Genotype calls ({kind.capitalize()}) downloaded")
            return kind, text

        async def _fetch_genotypes() -> list[tuple[str, str]]:
            if not requested_analyses:
                _bump_download("No genotype export required")
                return []
            return list(await asyncio.gather(*(_fetch_single_analysis(k) for k in requested_analyses)))

        async def _fetch_meta() -> str | None:
            if with_metadata:
                res = await self.download_metadata(uuid=uuid, genome_ids=genome_ids)
                _bump_download("Spatiotemporal metadata downloaded")
                return res
            return None

        trees_list = details.get("trees") or []
        fallback_newick = (
            str(trees_list[0].get("newick", "")).strip()
            if (isinstance(trees_list, list) and trees_list and isinstance(trees_list[0], dict))
            else ""
        )

        async def _fetch_distances() -> str | None:
            res, exc = await matrix_task
            if not with_distances:
                return None
            if exc is not None:
                if strict_distances:
                    raise exc
                if fallback_newick:
                    _bump_download("Using collection Newick tree for pairwise distances")
                    return None
                warnings.warn(
                    f"Could not download '{matrix_type}' distance matrix for Pathogenwatch collection "
                    f"'{uuid}' ({exc}); proceeding without pairwise distances.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _bump_download("Distance matrix unavailable (skipped)")
                return None
            _bump_download("Pairwise distance matrix downloaded")
            return res

        geno_items, meta_csv, dist_csv = await asyncio.gather(
            _fetch_genotypes(),
            _fetch_meta(),
            _fetch_distances(),
        )

        if not geno_items and meta_csv is None:
            raise ValueError(f"Collection '{uuid}' has no genotype or metadata exports available.")

        prog.report(
            0.85,
            message=f"Compiling '{resolved_name}'",
            detail="Parsing genotypes, metadata, and distances concurrently...",
            force=True,
        )

        def _parse_single_geno(kind: str, csv_text: str, name: str | None = None) -> Dataset:
            parser_cls = PathogenWatchFile.parser_for_download(kind)
            if parser_cls is not None:
                return parser_cls.parse(csv_text, dataset_name=name)
            return DatasetParser.parse(csv_text, id_column="Genome Name", dataset_name=name)

        def _parse_primary() -> tuple[Dataset, list[Dataset]]:
            if geno_items:
                first_kind, first_csv = geno_items[0]
                primary_ds = _parse_single_geno(first_kind, first_csv, name=resolved_name)
                extra_geno_ds = [_parse_single_geno(k, txt) for k, txt in geno_items[1:]]
                return primary_ds, extra_geno_ds
            assert meta_csv is not None
            return DatasetParser.parse(meta_csv, id_column="Genome Name", dataset_name=resolved_name), []

        def _parse_meta() -> Dataset | None:
            if meta_csv is not None and geno_items:
                return DatasetParser.parse(meta_csv, id_column="Genome Name")
            return None

        def _parse_dist() -> Any:
            if dist_csv is not None:
                return PathogenwatchDistanceParser.parse(dist_csv)
            if with_distances and fallback_newick:
                uuid_to_name = {
                    str(r["uuid"]): str(r["name"])
                    for r in records
                    if isinstance(r, dict) and "uuid" in r and "name" in r
                }
                return PathogenwatchDistanceParser.from_newick(fallback_newick, label_map=uuid_to_name)
            return None

        (ds, extra_geno_datasets), meta_ds, dist = await asyncio.gather(
            asyncio.to_thread(_parse_primary),
            asyncio.to_thread(_parse_meta),
            asyncio.to_thread(_parse_dist),
        )

        def _assemble(
            base_ds: Dataset,
            aux_genos: list[Dataset],
            aux_meta: Dataset | None,
            parsed_dist: Any,
        ) -> Dataset:
            out_ds = base_ds
            for extra_ds in aux_genos:
                join_col = (
                    "Genome ID" if "Genome ID" in out_ds.data.columns and "Genome ID" in extra_ds.data.columns else None
                )
                out_ds = out_ds.enrich(extra_ds, on=join_col)
            if aux_meta is not None:
                join_col = (
                    "Genome ID" if "Genome ID" in out_ds.data.columns and "Genome ID" in aux_meta.data.columns else None
                )
                out_ds = out_ds.enrich(aux_meta, on=join_col)
            if parsed_dist is not None:
                out_ds = out_ds.with_distances(parsed_dist)
            return out_ds

        if extra_geno_datasets or meta_ds is not None or dist is not None:
            prog.report(
                0.95,
                message=f"Compiling '{resolved_name}'",
                detail="Attaching metadata and genomic distances...",
                force=True,
            )
            ds = await asyncio.to_thread(_assemble, ds, extra_geno_datasets, meta_ds, dist)

        prog.report(1.0, message=f"Loaded '{resolved_name}'", detail=f"{len(ds):,} isolates ready", force=True)
        return ds

    # Synchronous wrappers
    def get_collections_sync(
        self,
        owner: str | None = None,
        exclude: str | None = None,
        binned: bool | None = None,
        limit: int | None = None,
    ) -> list[PathogenwatchCollection]:
        """Synchronously fetches collections available to the authenticated account or public collections.

        Args:
            owner: Filter by ownership (e.g. ``"me"``, ``"me,shared"``).
            exclude: Exclude collections of a given access type.
            binned: Whether to include binned collections.
            limit: Maximum number of collections to return.

        Returns:
            List of ``PathogenwatchCollection`` instances.
        """
        return self.run_sync(self.get_collections(owner=owner, exclude=exclude, binned=binned, limit=limit))

    def get_collection_sync(self, uuid: str) -> dict[str, Any]:
        """Synchronously fetches complete metadata and analysis jobs for a collection.

        Args:
            uuid: Short UUID identifier of the collection.

        Returns:
            Dictionary of collection details.
        """
        return self.run_sync(self.get_collection(uuid))

    def get_collection_genomes_sync(
        self,
        uuid: str,
        share_id: str | None = None,
        page: int = 1,
        all_pages: bool = True,
    ) -> list[dict[str, Any]]:
        """Synchronously fetches genome analysis records for a collection.

        Args:
            uuid: Short UUID of the collection.
            share_id: Optional share token.
            page: Starting 1-based page number.
            all_pages: Whether to paginate through all pages automatically.

        Returns:
            List of raw genome record dictionaries.
        """
        return self.run_sync(self.get_collection_genomes(uuid, share_id=share_id, page=page, all_pages=all_pages))

    def download_distance_matrix_sync(self, uuid: str, matrix_type: str = "difference") -> str:
        """Synchronously downloads the pairwise genomic distance matrix CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            matrix_type: Distance metric (``"difference"`` or ``"score"``).

        Returns:
            Raw CSV text representing the symmetric distance matrix.
        """
        return self.run_sync(self.download_distance_matrix(uuid, matrix_type=matrix_type))

    def download_analysis_sync(
        self,
        uuid: str,
        download_type: str,
        job: str | None = None,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads discrete analysis CSV results for a collection.

        Args:
            uuid: Short UUID of the collection.
            download_type: Analysis type name (e.g. ``"kleborate"``, ``"kaptive"``).
            job: Optional specific job ID.
            genome_ids: Optional subset of genome IDs.
            prefix: Optional filename prefix.

        Returns:
            Raw analysis CSV string.
        """
        return self.run_sync(
            self.download_analysis(
                uuid=uuid,
                download_type=download_type,
                job=job,
                genome_ids=genome_ids,
                prefix=prefix,
            )
        )

    def download_kleborate_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads Kleborate analysis CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs.
            prefix: Optional filename prefix.

        Returns:
            Raw Kleborate CSV string.
        """
        return self.run_sync(self.download_kleborate(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_kaptive_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads Kaptive analysis CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs.
            prefix: Optional filename prefix.

        Returns:
            Raw Kaptive CSV string.
        """
        return self.run_sync(self.download_kaptive(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_metadata_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads isolate metadata CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs.
            prefix: Optional filename prefix.

        Returns:
            Raw metadata CSV string.
        """
        return self.run_sync(self.download_metadata(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_summary_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads summary CSV for a collection.

        Args:
            uuid: Short UUID of the collection.
            genome_ids: Optional subset of genome IDs.
            prefix: Optional filename prefix.

        Returns:
            Raw summary CSV string.
        """
        return self.run_sync(self.download_summary(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_collection_files_sync(
        self,
        uuid: str,
        output_dir: str | Path,
        analysis_types: Sequence[str] = ("kleborate", "kaptive"),
        include_metadata: bool = True,
        include_matrix: bool = True,
        matrix_type: str = "difference",
        prefix: str | None = None,
    ) -> dict[str, Path]:
        """Synchronously downloads collection CSV files to disk.

        Args:
            uuid: Short UUID of the collection.
            output_dir: Target directory path on disk.
            analysis_types: Sequence of analysis downloads to fetch.
            include_metadata: Whether to download metadata CSV.
            include_matrix: Whether to download distance matrix CSV.
            matrix_type: ``"difference"`` or ``"score"``.
            prefix: Optional filename prefix.

        Returns:
            Dictionary mapping asset kind to written ``Path`` on disk.
        """
        return self.run_sync(
            self.download_collection_files(
                uuid=uuid,
                output_dir=output_dir,
                analysis_types=analysis_types,
                include_metadata=include_metadata,
                include_matrix=include_matrix,
                matrix_type=matrix_type,
                prefix=prefix,
            )
        )

    def fetch_collection_dataset_sync(
        self,
        uuid: str,
        with_distances: bool = True,
        with_metadata: bool = True,
        matrix_type: str = "difference",
        dataset_name: str | None = None,
        strict_distances: bool = False,
        analysis_types: Sequence[str] | str | None = None,
    ) -> Dataset:
        """Synchronously fetches collection files and compiles a unified ``Dataset``.

        Args:
            uuid: Short UUID of the collection.
            with_distances: Whether to download and attach pairwise ``GenomicDistances``.
            with_metadata: Whether to download and enrich with spatiotemporal metadata.
            matrix_type: Distance matrix type (``"difference"`` or ``"score"``).
            dataset_name: Optional custom ``Dataset`` name.
            strict_distances: Whether to raise an exception if distance matrix generation fails.
            analysis_types: Optional analysis name or sequence of analysis names to fetch and enrich.

        Returns:
            Compiled ``Dataset`` instance.
        """
        return self.run_sync(
            self.fetch_collection_dataset(
                uuid=uuid,
                with_distances=with_distances,
                with_metadata=with_metadata,
                matrix_type=matrix_type,
                dataset_name=dataset_name,
                strict_distances=strict_distances,
                analysis_types=analysis_types,
            )
        )
