"""Base abstractions for genotype parsers and asynchronous API clients."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import Any

import polars as pl


class BaseParser(ABC):
    """Abstract base parser using Polars for tabular and file I/O.

    Attributes:
        _READERS: Mapping of file extensions to Polars reader functions.
        _SEPARATORS: Mapping of delimited file extensions to default separator characters.
    """

    _READERS = {
        ".csv": pl.read_csv,
        ".tsv": pl.read_csv,
        ".txt": pl.read_csv,
        ".parquet": pl.read_parquet,
        ".json": pl.read_json,
        ".jsonl": pl.read_ndjson,
        ".xlsx": pl.read_excel,
        ".xls": pl.read_excel,
    }

    _SEPARATORS = {
        ".csv": ",",
        ".tsv": "\t",
    }

    @classmethod
    def _infer_separator(cls, source: Any, override: str | None = None, fallback: str = ",") -> str:
        if override is not None:
            return override
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower()
            return cls._SEPARATORS.get(suffix, fallback)
        return fallback

    @classmethod
    def _infer_reader(
        cls,
        source: Any,
        override: Callable[..., Any] | None = None,
        fallback: Callable[..., Any] = pl.read_csv,
    ) -> Callable[..., Any]:
        if override is not None:
            return override
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower()
            return cls._READERS.get(suffix, fallback)
        return fallback

    @staticmethod
    def _normalize_source(source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame) -> Any:
        """Normalizes various input source types for Polars CSV readers."""
        if isinstance(source, (bytes, Path)):
            return source
        if isinstance(source, str):
            if "\n" in source or "\r" in source or ("," in source and not Path(source).exists()):
                return StringIO(source)
            return Path(source)
        if isinstance(source, IOBase):
            if not source.seekable():
                content = source.read()
                return StringIO(content) if isinstance(content, str) else BytesIO(content)
            return source
        return source

    @classmethod
    @abstractmethod
    def schema(cls) -> pl.Schema:
        """Returns the fixed or derived Polars schema for this parser.

        Returns:
            Polars ``Schema`` mapping column names to data types.
        """
        ...

    @classmethod
    @abstractmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Parses the input source into a domain data structure.

        Args:
            source: File path, raw string/bytes, stream buffer, ``DataFrame``, or ``LazyFrame``.
            separator: Optional field delimiter override.
            reader: Optional Polars reader callable override.
            **kwargs: Additional keyword arguments forwarded to the underlying reader.

        Returns:
            Parsed domain object (e.g., ``Dataset`` or ``GenomicDistances``).
        """
        ...
