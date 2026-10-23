"""Multi-source dataset ingestion subcomponent (`DatasetLoader`)."""

import asyncio
import inspect
from pathlib import Path
from typing import Any, ClassVar

from htmltools import TagChild
from shiny import reactive, render
from shiny import ui as shiny_ui
from shiny_kit.component import Component
from shiny_kit.decorators import bind_effect, bind_output, bind_task, reactive_prop
from shiny_kit.ui import Icon

from seroepi.app.models import SessionState
from seroepi.client.pathogenwatch import PathogenwatchClient, PathogenwatchCollection
from seroepi.dataset import Dataset
from seroepi.dist import GenomicDistances
from seroepi.io.dataset.base import DatasetParser
from seroepi.io.dataset.kaptive import KaptiveParser, PathogenWatchKaptiveParser
from seroepi.io.dataset.kleborate import KleborateParser
from seroepi.io.dataset.pathogenwatch import (
    MLSTParser,
    PathogenWatchCoreParser,
    PathogenWatchLINCodeParser,
    PathogenWatchMLSTParser,
    PathogenWatchPlasmidFinderParser,
)
from seroepi.io.dist import (
    DistanceParser,
    NewickDistanceParser,
    PathogenwatchDistanceParser,
    Ska2DistanceParser,
    SquareDistanceParser,
)
from seroepi.variables import Identifier, Variable

# Ensure built-in concrete implementations are imported so __subclasses__() discovers them
_ = (
    KleborateParser,
    KaptiveParser,
    PathogenWatchKaptiveParser,
    MLSTParser,
    PathogenWatchCoreParser,
    PathogenWatchMLSTParser,
    PathogenWatchLINCodeParser,
    PathogenWatchPlasmidFinderParser,
    NewickDistanceParser,
    PathogenwatchDistanceParser,
    SquareDistanceParser,
    Ska2DistanceParser,
)


class DatasetLoader(Component[SessionState]):
    """Multi-source dataset ingestion (Example, Tabular Schema Upload, .joblib, Pathogenwatch).

    Discovers concrete subclasses of `DatasetParser` and `DistanceParser` dynamically via
    `__subclasses__()` and provides reactive ingestion workflows for each source.

    Attributes:
        peeked_columns (tuple[str, ...]): Column names peeked from the most recently uploaded tabular file.
        pw_collections (tuple[PathogenwatchCollection, ...]): Cached Pathogenwatch collections available to the user.
        pw_collection_details (dict[str, Any]): Cached details (`downloads`, `organismName`) for the selected collection.
    """

    DEFAULT_COLLECTION_UUID: ClassVar[str] = "myp3PKqcM1tiqWVmFcEVHn"

    peeked_columns: reactive_prop[tuple[str, ...]] = reactive_prop(())
    pw_collections: reactive_prop[tuple[PathogenwatchCollection, ...]] = reactive_prop(())
    pw_collection_details: reactive_prop[dict[str, Any]] = reactive_prop({})

    _PW_COLLECTION_CACHE: ClassVar[dict[str, tuple[PathogenwatchCollection, ...]]] = {}
    _PW_DETAILS_CACHE: ClassVar[dict[str, dict[str, Any]]] = {}

    @staticmethod
    def _class_display_label(cls: type[Any], strip_suffix: str = "") -> str:
        """Derives a clean human-readable display label from a class docstring or name."""
        doc = inspect.getdoc(cls)
        if doc:
            first_line = doc.splitlines()[0].strip().rstrip(".")
            if first_line:
                return first_line
        return cls.__name__.removesuffix(strip_suffix) if strip_suffix else cls.__name__

    @staticmethod
    def _discover_subclasses[T](base_cls: type[T]) -> dict[str, type[T]]:
        """Recursively discovers all concrete subclasses of `base_cls`."""
        found: dict[str, type[T]] = {}
        queue = list(base_cls.__subclasses__())
        while queue:
            sub = queue.pop(0)
            queue.extend(sub.__subclasses__())
            if not inspect.isabstract(sub) and not getattr(sub, "__abstractmethods__", None):
                found[sub.__name__] = sub
        return found

    @classmethod
    def dataset_parser_registry(cls) -> dict[str, type[DatasetParser]]:
        """Returns all concrete `DatasetParser` subclasses keyed by class name.

        Returns:
            Dictionary mapping subclass names to `DatasetParser` classes.
        """
        return cls._discover_subclasses(DatasetParser)

    @classmethod
    def distance_parser_registry(cls) -> dict[str, type[DistanceParser]]:
        """Returns all concrete user-facing `DistanceParser` subclasses keyed by class name.

        Returns:
            Dictionary mapping subclass names to `DistanceParser` classes.
        """
        return {
            name: sub
            for name, sub in cls._discover_subclasses(DistanceParser).items()
            if name != "PairwiseDistanceParser"
        }

    @classmethod
    async def fetch_cached_collections_async(
        cls,
        api_key: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[PathogenwatchCollection, ...]:
        """Asynchronously fetches Pathogenwatch collections once per API key and caches them in memory.

        Args:
            api_key: Required Pathogenwatch API key.
            force_refresh: If `True`, bypasses the in-memory cache and re-queries the API.

        Returns:
            Tuple of `PathogenwatchCollection` metadata records.

        Raises:
            ValueError: If `api_key` is missing or blank.
        """
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty Pathogenwatch API key is required.")
        if not force_refresh and clean_key in cls._PW_COLLECTION_CACHE:
            return cls._PW_COLLECTION_CACHE[clean_key]

        async with PathogenwatchClient(api_key=clean_key) as client:
            fetched = tuple(await client.get_collections())
        cls._PW_COLLECTION_CACHE[clean_key] = fetched
        return fetched

    @classmethod
    def fetch_cached_collections(
        cls,
        api_key: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[PathogenwatchCollection, ...]:
        """Fetches Pathogenwatch collections once per API key and caches them in memory.

        Args:
            api_key: Required Pathogenwatch API key.
            force_refresh: If `True`, bypasses the in-memory cache and re-queries the API.

        Returns:
            Tuple of `PathogenwatchCollection` metadata records.

        Raises:
            ValueError: If `api_key` is missing or blank.
        """
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty Pathogenwatch API key is required.")
        if not force_refresh and clean_key in cls._PW_COLLECTION_CACHE:
            return cls._PW_COLLECTION_CACHE[clean_key]

        with PathogenwatchClient(api_key=clean_key) as client:
            fetched = tuple(client.get_collections_sync())
        cls._PW_COLLECTION_CACHE[clean_key] = fetched
        return fetched

    @classmethod
    async def fetch_cached_collection_details_async(
        cls,
        uuid: str,
        api_key: str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Asynchronously fetches and caches `get_collection(uuid)` details (including available `downloads`).

        Args:
            uuid: Short UUID of the Pathogenwatch collection.
            api_key: Required Pathogenwatch API key.
            force_refresh: If `True`, bypasses the in-memory cache and re-queries the API.

        Returns:
            Collection details dictionary containing `downloads`, `analyses`, and `organismName`.

        Raises:
            ValueError: If `api_key` is missing or blank.
        """
        clean_uuid = uuid.strip()
        if not clean_uuid:
            return {}
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty Pathogenwatch API key is required.")
        cache_key = f"{clean_key}:{clean_uuid}"
        if not force_refresh and cache_key in cls._PW_DETAILS_CACHE:
            return cls._PW_DETAILS_CACHE[cache_key]

        async with PathogenwatchClient(api_key=clean_key) as client:
            details = await client.get_collection(clean_uuid)
        cls._PW_DETAILS_CACHE[cache_key] = details
        return details

    @classmethod
    def _render_ui(cls) -> Any:
        parser_choices = {
            **{name: cls._class_display_label(c, "Parser") for name, c in cls.dataset_parser_registry().items()},
            "generic": "Generic Tabular (Auto-infer Variables)",
        }
        dist_choices = {
            name: cls._class_display_label(c, "DistanceParser") for name, c in cls.distance_parser_registry().items()
        }

        return shiny_ui.navset_pill(
            shiny_ui.nav_panel(
                "Example",
                shiny_ui.div(
                    shiny_ui.div(
                        shiny_ui.div("KlebNET Neonatal Sepsis", class_="fw-semibold small"),
                        shiny_ui.div(
                            "981 K. pneumoniae neonatal sepsis isolates bundled with Kleborate, Kaptive, "
                            "LIN code, MLST & PlasmidFinder genotypes, metadata, and SNP distances.",
                            class_="text-muted small mt-1",
                        ),
                        class_="p-2 mb-3 rounded border bg-body-tertiary",
                    ),
                    shiny_ui.input_switch(
                        "example_with_distances",
                        "Include Pairwise Distance Matrix",
                        value=True,
                    ),
                    shiny_ui.input_action_button(
                        "btn_load_example",
                        "Load Example Dataset",
                        icon=Icon.lucide("database", size=15),
                        class_="btn-primary w-100 mt-2",
                    ),
                    class_="mt-3",
                ),
                icon=Icon.lucide("flask-conical", size=14),
            ),
            shiny_ui.nav_panel(
                "Upload",
                shiny_ui.div(
                    shiny_ui.input_file(
                        "upload_data_file",
                        "Genotype / Isolate Table",
                        accept=[".csv", ".tsv", ".txt", ".gz", ".parquet", ".xlsx"],
                    ),
                    shiny_ui.input_select(
                        "upload_parser_mode",
                        "Parser Schema",
                        choices=parser_choices,
                    ),
                    shiny_ui.output_ui("upload_id_col_ui"),
                    shiny_ui.hr(class_="my-2"),
                    shiny_ui.input_file(
                        "upload_meta_file",
                        "Auxiliary Metadata (Optional .enrich())",
                        accept=[".csv", ".tsv", ".txt", ".parquet", ".xlsx"],
                    ),
                    shiny_ui.input_file(
                        "upload_dist_file",
                        "Distance Matrix (Optional .with_distances())",
                        accept=[".csv", ".tsv", ".txt"],
                    ),
                    shiny_ui.input_select(
                        "upload_dist_flavour",
                        "Distance Format",
                        choices=dist_choices,
                    ),
                    shiny_ui.input_text(
                        "upload_dataset_name",
                        "Dataset Name (Optional)",
                        placeholder="Defaults to uploaded filename",
                    ),
                    shiny_ui.input_action_button(
                        "btn_load_upload",
                        "Parse & Register Dataset",
                        icon=Icon.lucide("folder", size=15),
                        class_="btn-primary w-100 mt-2",
                    ),
                    class_="mt-3",
                ),
                icon=Icon.lucide("folder", size=14),
            ),
            shiny_ui.nav_panel(
                ".joblib",
                shiny_ui.div(
                    shiny_ui.input_file(
                        "upload_joblib_file",
                        "Saved SeroEpi Dataset (.joblib)",
                        accept=[".joblib"],
                    ),
                    shiny_ui.input_action_button(
                        "btn_load_joblib",
                        "Load .joblib Dataset",
                        icon=Icon.lucide("database", size=15),
                        class_="btn-primary w-100 mt-2",
                    ),
                    class_="mt-3",
                ),
                icon=Icon.lucide("layers", size=14),
            ),
            shiny_ui.nav_panel(
                "Pathogenwatch",
                shiny_ui.div(
                    shiny_ui.div(
                        shiny_ui.div(
                            shiny_ui.input_password(
                                "pw_api_key",
                                "Pathogenwatch API Key",
                            ),
                            class_="flex-grow-1",
                        ),
                        shiny_ui.input_action_button(
                            "btn_refresh_pw_collections",
                            "Connect",
                            icon=Icon.lucide("cloud-sync", size=14),
                            class_="btn-outline-secondary btn-sm mb-3",
                        ),
                        class_="d-flex align-items-end gap-2",
                    ),
                    shiny_ui.output_ui("pw_authenticated_ui"),
                    class_="mt-3",
                ),
                icon=Icon.lucide("telescope", size=14),
            ),
        )

    @bind_effect("pw_api_key")
    async def _on_api_key_changed(self) -> None:
        """Fetches Pathogenwatch collections once a non-empty API key is entered, or clears state when blank."""
        api_key = str(self.val("pw_api_key") or "").strip()
        if not api_key:
            self.pw_collections = ()
            self.pw_collection_details = {}
            return
        try:
            self.pw_collections = await self.fetch_cached_collections_async(api_key=api_key, force_refresh=False)
        except Exception:
            self.pw_collections = ()

    @bind_effect("btn_refresh_pw_collections")
    async def _on_refresh_pw_collections(self) -> None:
        """Asynchronously refreshes the Pathogenwatch collection list using the provided API key."""
        api_key = str(self.val("pw_api_key") or "").strip()
        if not api_key:
            self.notify("Please enter a Pathogenwatch API key first.", type_="warning")
            return
        cols = await self.fetch_cached_collections_async(api_key=api_key, force_refresh=True)
        self.pw_collections = cols
        uuid = str(self.val("pw_collection_uuid") or self.DEFAULT_COLLECTION_UUID).strip()
        if uuid:
            try:
                self.pw_collection_details = await self.fetch_cached_collection_details_async(
                    uuid, api_key=api_key, force_refresh=True
                )
            except Exception:
                self.pw_collection_details = {}
        self.notify(f"Loaded {len(cols):,} Pathogenwatch collections.", type_="message")

    @bind_effect("pw_collection_uuid", "pw_api_key")
    async def _on_select_pw_collection(self) -> None:
        """Asynchronously fetches collection details (`downloads` and `organismName`) when a collection is selected."""
        api_key = str(self.val("pw_api_key") or "").strip()
        if not api_key:
            self.pw_collection_details = {}
            return
        uuid = str(self.val("pw_collection_uuid") or self.DEFAULT_COLLECTION_UUID).strip()
        if not uuid:
            self.pw_collection_details = {}
            return
        try:
            self.pw_collection_details = await self.fetch_cached_collection_details_async(
                uuid, api_key=api_key, force_refresh=False
            )
        except Exception:
            self.pw_collection_details = {}

    @bind_output(render.ui, "pw_authenticated_ui")
    def render_pw_authenticated_ui(self) -> TagChild:
        """Renders Pathogenwatch collection, analysis, and fetch controls only when an API key is provided."""
        api_key = str(self.val("pw_api_key") or "").strip()
        if not api_key:
            return shiny_ui.div(
                "Enter your Pathogenwatch API key above to browse collections and genotype analyses.",
                class_="text-muted small p-2 rounded border bg-body-tertiary",
            )
        return shiny_ui.div(
            shiny_ui.output_ui("pw_collection_select_ui"),
            shiny_ui.output_ui("pw_analysis_select_ui"),
            shiny_ui.input_text(
                "pw_dataset_name",
                "Custom Name (Optional)",
                placeholder="Defaults to collection title",
            ),
            shiny_ui.input_switch(
                "pw_with_metadata",
                "Fetch Spatiotemporal Metadata",
                value=True,
            ),
            shiny_ui.input_switch(
                "pw_with_distances",
                "Fetch Pairwise Distance Matrix",
                value=True,
            ),
            shiny_ui.input_select(
                "pw_matrix_type",
                "Distance Metric",
                choices={"difference": "SNP Difference", "score": "Score"},
            ),
            shiny_ui.input_action_button(
                "btn_load_pw",
                "Fetch from Pathogenwatch",
                icon=Icon.lucide("telescope", size=15),
                class_="btn-primary w-100 mt-2",
            ),
        )

    @bind_output(render.ui, "pw_collection_select_ui")
    def render_pw_collection_select_ui(self) -> TagChild:
        """Renders the grouped Pathogenwatch collection selectize dropdown.

        Returns:
            UI fragment containing the collection selector and collection count summary.
        """
        cols = self.pw_collections
        default_uuid = self.DEFAULT_COLLECTION_UUID
        with reactive.isolate():
            current_uuid = self.val("pw_collection_uuid")
        selected_uuid = current_uuid or default_uuid

        grouped_choices = PathogenwatchCollection.group_choices(cols)
        if not grouped_choices and default_uuid:
            grouped_choices = {"Public / Shared": {default_uuid: default_uuid}}

        n_owned = sum(1 for c in cols if c.is_owned)
        n_public = len(cols) - n_owned
        if cols:
            summary_str = (
                f"{len(cols):,} collections available ({n_owned:,} mine, {n_public:,} public/shared)"
                if n_owned > 0
                else f"{n_public:,} public collections available"
            )
        else:
            summary_str = "Select or paste a Pathogenwatch collection UUID"

        return shiny_ui.div(
            shiny_ui.input_selectize(
                "pw_collection_uuid",
                "Pathogenwatch Collection",
                choices=grouped_choices,
                selected=selected_uuid,
                options={
                    "placeholder": "Search collection name or paste UUID...",
                    "create": True,
                },
            ),
            shiny_ui.div(summary_str, class_="text-muted small mt-n2 mb-2"),
        )

    @bind_output(render.ui, "pw_analysis_select_ui")
    def render_pw_analysis_select_ui(self) -> TagChild:
        """Renders the multi-select Genotype Analyses selector populated from the selected collection's `downloads`.

        Returns:
            UI fragment containing the multi-select `pw_analysis_types` input and available analysis summary.
        """
        details = self.pw_collection_details
        raw_downloads = details.get("downloads", []) if isinstance(details, dict) else []
        choices: dict[str, str] = {}
        for d in raw_downloads:
            if isinstance(d, dict) and d.get("name"):
                name = str(d["name"]).strip().lower()
                label = str(d.get("label") or d["name"]).strip()
                if name:
                    choices[name] = label

        default_selected = list(PathogenwatchClient.default_analysis_types(raw_downloads))
        org_name = str(details.get("organismName") or "").strip() if isinstance(details, dict) else ""
        if choices:
            meta_prefix = f"{org_name} · " if org_name else ""
            caption = f"{meta_prefix}{len(choices)} analysis export{'s' if len(choices) != 1 else ''} available"
        else:
            caption = "Inspecting available collection analyses..."

        return shiny_ui.div(
            shiny_ui.input_selectize(
                "pw_analysis_types",
                "Genotype Analyses",
                choices=choices,
                selected=default_selected,
                multiple=True,
                options={
                    "placeholder": "Select one or more genotype analyses...",
                },
            ),
            shiny_ui.div(caption, class_="text-muted small mt-n2 mb-2"),
        )

    @bind_task("btn_load_example", register_to="datasets")
    def task_load_example(self) -> Dataset:
        """Loads the bundled KlebNET Neonatal Sepsis benchmark `Dataset`.

        Returns:
            Loaded `Dataset` instance.
        """
        with_dist = bool(self.val("example_with_distances", True))
        return Dataset.example(with_distances=with_dist)

    @bind_task("btn_load_upload", register_to="datasets")
    async def task_load_upload(self) -> Dataset:
        """Parses an uploaded primary genotype table, optional auxiliary metadata, and optional distance matrix.

        Returns:
            Parsed and enriched `Dataset` instance.

        Raises:
            ValueError: If no primary tabular file has been uploaded.
        """
        file_infos = self.val("upload_data_file")
        if not file_infos:
            raise ValueError("Please upload a primary Genotype / Isolate tabular file first.")

        primary_path = Path(file_infos[0]["datapath"])
        orig_name = Path(file_infos[0]["name"]).stem
        custom_name = self.val("upload_dataset_name", orig_name)
        parsers = self.dataset_parser_registry()
        mode = str(self.val("upload_parser_mode", next(iter(parsers.keys()), "generic")))
        id_col = self.val("upload_id_col")
        meta_infos = self.val("upload_meta_file")
        dist_infos = self.val("upload_dist_file")
        flavour = str(self.val("upload_dist_flavour", ""))

        def _parse_primary() -> Dataset:
            if mode in parsers:
                return parsers[mode].parse(primary_path, dataset_name=custom_name)
            return DatasetParser.parse(primary_path, id_column=id_col, dataset_name=custom_name)

        def _parse_dist() -> GenomicDistances | None:
            if not dist_infos:
                return None
            dist_path = Path(dist_infos[0]["datapath"])
            dist_parsers = self.distance_parser_registry()
            dist_cls = dist_parsers.get(flavour) or next(iter(dist_parsers.values()))
            return dist_cls.parse(dist_path)

        ds, dist = await asyncio.gather(
            asyncio.to_thread(_parse_primary),
            asyncio.to_thread(_parse_dist),
        )

        def _finalize(base_ds: Dataset) -> Dataset:
            out = base_ds
            if meta_infos:
                meta_path = Path(meta_infos[0]["datapath"])
                meta_peek = DatasetParser.peek(meta_path)
                meta_id = out.identifier.column if out.identifier.column in meta_peek.columns else None
                out = out.enrich(DatasetParser.parse(meta_path, id_column=meta_id))
            if dist is not None:
                out = out.with_distances(dist)
            return out

        if meta_infos or dist is not None:
            ds = await asyncio.to_thread(_finalize, ds)
        return ds

    @bind_task("btn_load_joblib", register_to="datasets")
    def task_load_joblib(self) -> Dataset:
        """Deserializes an uploaded `.joblib` `Dataset` file.

        Returns:
            Loaded `Dataset` instance.

        Raises:
            ValueError: If no `.joblib` file has been selected.
        """
        file_infos = self.val("upload_joblib_file")
        if not file_infos:
            raise ValueError("Please select a .joblib file to load.")
        return Dataset.load(Path(file_infos[0]["datapath"]))

    def _on_pw_loaded(self, ds: Dataset) -> str | None:
        with_dist = bool(self.val("pw_with_distances", True))
        if with_dist and ds.distances is None:
            return (
                f"Loaded '{ds.name}' ({len(ds):,} isolates) without pairwise distances "
                "(distance matrix unavailable on Pathogenwatch for this collection)."
            )
        return None

    @bind_task("btn_load_pw", register_to="datasets", on_success="_on_pw_loaded")
    async def task_load_pathogenwatch(self) -> Dataset:
        """Fetches a `Dataset` directly from the Pathogenwatch REST API for the selected collection UUID.

        Returns:
            Fetched `Dataset` instance.

        Raises:
            ValueError: If no API key or Pathogenwatch collection UUID is provided.
        """
        api_key = str(self.val("pw_api_key") or "").strip()
        if not api_key:
            raise ValueError("Please enter a Pathogenwatch API key first.")
        uuid = str(self.val("pw_collection_uuid", "")).strip()
        if not uuid:
            raise ValueError("Please select a Pathogenwatch collection.")
        custom_name = self.val("pw_dataset_name")
        if not custom_name:
            matched = next((c.name for c in self.pw_collections if c.uuid == uuid and c.name), None)
            custom_name = matched
        with_meta = bool(self.val("pw_with_metadata", True))
        with_dist = bool(self.val("pw_with_distances", True))
        matrix_type = self.val("pw_matrix_type", "difference")
        analysis_types = self.val("pw_analysis_types", None)

        async with PathogenwatchClient(api_key=api_key) as client:
            return await client.fetch_collection_dataset(
                uuid=uuid,
                with_distances=with_dist,
                with_metadata=with_meta,
                matrix_type=matrix_type,
                dataset_name=custom_name,
                analysis_types=analysis_types,
            )

    @bind_effect("upload_data_file", "upload_parser_mode")
    def _on_peek_upload(self) -> None:
        infos = self.val("upload_data_file")
        if not infos:
            self.peeked_columns = ()
            return
        try:
            self.peeked_columns = DatasetParser.peek(Path(infos[0]["datapath"])).columns
        except Exception:
            self.peeked_columns = ()

    @bind_output(render.ui, "upload_id_col_ui")
    def render_upload_id_col_ui(self) -> TagChild:
        """Renders the primary sample identifier column selector when generic tabular parsing is active.

        Returns:
            Select input for choosing the primary `Identifier` column, or `None`.
        """
        if self.val("upload_parser_mode") != "generic":
            return None
        cols = self.peeked_columns
        if not cols:
            return shiny_ui.p("Upload a file to select the primary Identifier column.", class_="text-muted small mt-1")
        guessed_vars = Variable.guess_many(cols)
        default_id = next((v.column for v in guessed_vars if isinstance(v, Identifier)), cols[0])
        return shiny_ui.input_select(
            "upload_id_col",
            "Primary Sample Identifier Column",
            choices={c: c for c in cols},
            selected=default_id,
        )
