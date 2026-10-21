"""Pathogenwatch REST API client and collection models."""

import asyncio
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shiny_kit.progress import current_progress

from seroepi.client.base import Client
from seroepi.client.settings import IOSettings, get_io_settings
from seroepi.dataset import Dataset
from seroepi.io.dataset.base import DatasetParser
from seroepi.io.dataset.kleborate import KleborateParser
from seroepi.io.dist import PathogenwatchDistanceParser


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

        return cls(
            uuid=uuid_val,
            name=name_val,
            id=data.get("id"),
            description=data.get("description"),
            size=int(size_val) if size_val is not None else None,
            organism_id=data.get("organismId") or data.get("organism_id"),
            organism_name=data.get("organismName") or data.get("organism_name"),
            owner=data.get("owner"),
            access=data.get("access"),
            binned=data.get("binned"),
            created_at=data.get("createdAt") or data.get("created_at"),
        )

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
        """Builds grouped selectize choices (``"My Collections"`` and ``"Public / Shared"``) keyed by UUID.

        Args:
            collections: Sequence of ``PathogenwatchCollection`` records to partition.

        Returns:
            Nested dictionary mapping group headers to ``{uuid: display_label}`` dicts.
        """
        owned = {
            c.uuid: c.display_label for c in collections if c.is_owned and c.uuid and (c.size is None or c.size > 0)
        }
        shared = {
            c.uuid: c.display_label for c in collections if not c.is_owned and c.uuid and (c.size is None or c.size > 0)
        }
        grouped: dict[str, Any] = {}
        if owned:
            grouped["My Collections"] = owned
        if shared:
            grouped["Public / Shared"] = shared
        return grouped


class PathogenwatchClient(Client):
    """Asynchronous and synchronous client for interacting with the Pathogenwatch REST API.

    Args:
        api_key: Optional Pathogenwatch API key string.
        base_url: Optional custom base URL for the Pathogenwatch API.
        timeout: Optional request timeout in seconds.
        settings: Optional custom ``IOSettings`` instance.

    Attributes:
        base_url: Resolved Pathogenwatch REST API base endpoint ending in ``/api``.

    Examples:
        >>> from seroepi.client.pathogenwatch import PathogenwatchClient
        >>> client = PathogenwatchClient(base_url="https://pathogen.watch/api")
        >>> client.base_url
        'https://pathogen.watch/api'
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        settings: IOSettings | None = None,
    ) -> None:
        """Initializes the Pathogenwatch API client.

        Args:
            api_key: Optional Pathogenwatch API key string.
                If omitted or blank, automatically loads from environment variables or ~/.seroepi/credentials.
            base_url: Optional custom base URL for the Pathogenwatch API.
            timeout: Optional request timeout in seconds.
            settings: Optional custom IOSettings instance.
        """
        cfg = settings or get_io_settings()
        clean_key = api_key.strip() if isinstance(api_key, str) else None
        resolved_key = clean_key if clean_key else cfg.pathogenwatch_api_key
        raw_url = base_url or cfg.pathogenwatch_base_url
        resolved_url = raw_url.rstrip("/")
        if not resolved_url.endswith("/api"):
            resolved_url = f"{resolved_url}/api"
        resolved_timeout = timeout if timeout is not None else cfg.default_timeout

        super().__init__(
            base_url=resolved_url,
            api_key_header="X-API-Key",
            api_key=resolved_key,
            timeout=resolved_timeout,
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

    async def fetch_collection_dataset(
        self,
        uuid: str,
        with_distances: bool = True,
        with_metadata: bool = True,
        matrix_type: str = "difference",
        dataset_name: str | None = None,
        strict_distances: bool = False,
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
        available_downloads = {d.get("name", "").lower(): d.get("job") for d in details.get("downloads", [])}

        prog.report(
            0.25,
            message=f"Downloading '{resolved_name}' ({len(genome_ids):,} isolates)",
            detail="Requesting genotypes, metadata, and distance matrix...",
            force=True,
        )
        completed_downloads = 0
        total_downloads = 1 + int(with_metadata) + int(with_distances)

        def _bump_download(stage_label: str) -> None:
            nonlocal completed_downloads
            completed_downloads += 1
            frac = 0.25 + 0.55 * (completed_downloads / max(1, total_downloads))
            prog.report(frac, message=f"Downloading '{resolved_name}'", detail=stage_label, force=True)

        # Concurrently fetch analysis, metadata, and await the already in-flight distance matrix
        async def _fetch_genotype() -> tuple[str | None, str | None]:
            if "kleborate" in available_downloads:
                text = await self.download_analysis(
                    uuid=uuid,
                    download_type="kleborate",
                    job=available_downloads["kleborate"],
                    genome_ids=genome_ids,
                )
                _bump_download("Genotype calls (Kleborate) downloaded")
                return "kleborate", text
            if "kaptive" in available_downloads:
                text = await self.download_analysis(
                    uuid=uuid,
                    download_type="kaptive",
                    job=available_downloads["kaptive"],
                    genome_ids=genome_ids,
                )
                _bump_download("Genotype calls (Kaptive) downloaded")
                return "kaptive", text
            _bump_download("No genotype export required")
            return None, None

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

        (geno_kind, geno_csv), meta_csv, dist_csv = await asyncio.gather(
            _fetch_genotype(),
            _fetch_meta(),
            _fetch_distances(),
        )

        if geno_csv is None and meta_csv is None:
            raise ValueError(f"Collection '{uuid}' has no genotype or metadata exports available.")

        prog.report(
            0.85,
            message=f"Compiling '{resolved_name}'",
            detail="Parsing genotypes, metadata, and distances concurrently...",
            force=True,
        )

        def _parse_primary() -> Dataset:
            if geno_csv is not None and geno_kind == "kleborate":
                return KleborateParser.parse(geno_csv, dataset_name=resolved_name)
            if geno_csv is not None:
                return DatasetParser.parse(geno_csv, id_column="Genome Name", dataset_name=resolved_name)
            assert meta_csv is not None
            return DatasetParser.parse(meta_csv, id_column="Genome Name", dataset_name=resolved_name)

        def _parse_meta() -> Dataset | None:
            if meta_csv is not None and geno_csv is not None:
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

        ds, meta_ds, dist = await asyncio.gather(
            asyncio.to_thread(_parse_primary),
            asyncio.to_thread(_parse_meta),
            asyncio.to_thread(_parse_dist),
        )

        def _assemble(base_ds: Dataset, aux_meta: Dataset | None, parsed_dist: Any) -> Dataset:
            out_ds = base_ds.enrich(aux_meta, on="Genome ID") if aux_meta is not None else base_ds
            if parsed_dist is not None:
                out_ds = out_ds.with_distances(parsed_dist)
            return out_ds

        if meta_ds is not None or dist is not None:
            prog.report(
                0.95,
                message=f"Compiling '{resolved_name}'",
                detail="Attaching metadata and genomic distances...",
                force=True,
            )
            ds = await asyncio.to_thread(_assemble, ds, meta_ds, dist)

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
    ) -> Dataset:
        """Synchronously fetches collection files and compiles a unified ``Dataset``.

        Args:
            uuid: Short UUID of the collection.
            with_distances: Whether to download and attach pairwise ``GenomicDistances``.
            with_metadata: Whether to download and enrich with spatiotemporal metadata.
            matrix_type: Distance matrix type (``"difference"`` or ``"score"``).
            dataset_name: Optional custom ``Dataset`` name.
            strict_distances: Whether to raise an exception if distance matrix generation fails.

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
            )
        )
