"""Benchmark and example surveillance datasets managed by SeroEpi."""

import io
from abc import ABC
from pathlib import Path
from typing import ClassVar, Self

from seroepi.client.pathogenwatch import PathogenwatchClient
from seroepi.dataset import Dataset
from seroepi.models import LoadedData


class ExampleDataset(Dataset, LoadedData, ABC):
    """Abstract base class for Pathogenwatch example and benchmark datasets managed by SeroEpi.

    Attributes:
        DIR_ENV_VAR: Environment variable overriding the local examples cache directory.
        DEFAULT_SUBDIR: Subdirectory under ``~/.seroepi/`` used to store built ``.joblib`` files.
        COLLECTION_UUID: Pathogenwatch collection UUID for this benchmark dataset.
        FILENAME: Default ``.joblib`` filename under the examples directory.

    Examples:
        >>> from seroepi.examples import KlebnetNeonatalSepsis
        >>> ds = KlebnetNeonatalSepsis.ensure()
        >>> len(ds)
        981
    """

    __slots__ = ()

    DIR_ENV_VAR: ClassVar[str] = "SEROEPI_EXAMPLES_DIR"
    DEFAULT_SUBDIR: ClassVar[str] = "examples"
    COLLECTION_UUID: ClassVar[str]
    _INSTANCE_CACHE: ClassVar[dict[tuple[type["ExampleDataset"], Path, float, bool], "ExampleDataset"]] = {}

    @classmethod
    def compile(cls, client: PathogenwatchClient | None = None) -> Self:
        """Fetches the collection dataset from Pathogenwatch and constructs the ``ExampleDataset`` instance.

        Args:
            client: Optional pre-configured ``PathogenwatchClient``. A temporary client is created if ``None``.

        Returns:
            The compiled ``ExampleDataset`` instance.
        """
        owns_client = client is None
        pw_client = client or PathogenwatchClient()
        try:
            ds = pw_client.fetch_collection_dataset_sync(
                uuid=cls.COLLECTION_UUID,
                with_distances=True,
                with_metadata=True,
                dataset_name=cls.FILENAME.removesuffix(".joblib"),
            )
        finally:
            if owns_client:
                pw_client.close_sync()

        return cls(
            data=ds.data,
            variables=ds.variables,
            name=ds.name,
            distances=ds.distances,
        )

    @classmethod
    def clear(cls, path: Path | str | None = None) -> None:
        """Removes the local built asset file from disk and evicts in-memory cached instances.

        Args:
            path: Optional custom path to remove. Defaults to ``cls.get_default_path()``.
        """
        super().clear(path)
        stale_keys = [k for k in cls._INSTANCE_CACHE if k[0] is cls]
        for k in stale_keys:
            cls._INSTANCE_CACHE.pop(k, None)

    @classmethod
    def load(
        cls,
        source: str | Path | bytes | io.BytesIO | None = None,
        with_distances: bool = True,
    ) -> Self:
        """Loads the example dataset from local .joblib storage with process-wide in-memory caching.

        Args:
            source: Optional custom path or buffer to .joblib file.
            with_distances: Whether to retain the attached genomic distance matrix.

        Returns:
            A populated ExampleDataset instance.

        Raises:
            FileNotFoundError: If the asset has not been built via `cls.build()`.
        """
        if source is None or isinstance(source, (str, Path)):
            target = Path(source).resolve() if isinstance(source, (str, Path)) else cls.get_default_path().resolve()
            if source is None and not target.exists():
                cls.build(out_path=target)
            if target.exists():
                mtime = target.stat().st_mtime
                cache_key = (cls, target, mtime, with_distances)
                cached = cls._INSTANCE_CACHE.get(cache_key)
                if isinstance(cached, cls):
                    return cached

                full_key = (cls, target, mtime, True)
                full_ds = cls._INSTANCE_CACHE.get(full_key)
                if not isinstance(full_ds, cls):
                    full_ds = super().load(target)
                    cls._INSTANCE_CACHE[full_key] = full_ds

                if not with_distances and full_ds.distances is not None:
                    stripped = cls(
                        data=full_ds.data,
                        variables=full_ds.variables,
                        name=full_ds.name,
                        distances=None,
                    )
                    cls._INSTANCE_CACHE[cache_key] = stripped
                    return stripped
                return full_ds

        ds = super().load(source)
        if not with_distances and ds.distances is not None:
            return cls(
                data=ds.data,
                variables=ds.variables,
                name=ds.name,
                distances=None,
            )
        return ds


class KlebnetNeonatalSepsis(ExampleDataset):
    """Klebnet Neonatal Sepsis benchmark collection from Pathogenwatch.

    Contains 981 Klebsiella pneumoniae isolates collected from neonates across multiple
    hospital centers. Includes Kleborate genotyping (species, ST, virulence, resistance,
    capsule K/O serotypes), auxiliary clinical and geographic metadata, and a 981x981
    pairwise genomic difference matrix.

    Attributes:
        FILENAME: Default local cache filename (``"klebnet_neonatal_sepsis.joblib"``).
        COLLECTION_UUID: Pathogenwatch collection UUID (``"myp3PKqcM1tiqWVmFcEVHn"``).

    Examples:
        >>> from seroepi.examples import KlebnetNeonatalSepsis
        >>> ds = KlebnetNeonatalSepsis.ensure()
        >>> len(ds)
        981
    """

    __slots__ = ()

    FILENAME: ClassVar[str] = "klebnet_neonatal_sepsis.joblib"
    COLLECTION_UUID: ClassVar[str] = "myp3PKqcM1tiqWVmFcEVHn"
