"""Dataset parsing."""

import gzip
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO, IOBase
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import polars as pl

from seroepi.dataset import Dataset
from seroepi.io.base import BaseParser
from seroepi.variables import (
    Identifier,
    Temporal,
    Trait,
    Variable,
)


@dataclass(frozen=True, slots=True)
class PeekResult:
    """Inspected file metadata and inferred parameters for ``DatasetParser.parse()``.

    Attributes:
        columns: Tuple of non-empty column names discovered in the source header.
        variables: Tuple of inferred or configured ``Variable`` instances aligned with ``columns``.
        separator: Detected or overridden field delimiter for text formats.
        reader: Detected or overridden Polars reader function.
        source: Normalized input source ready for consumption by ``DatasetParser.parse()``.
    """

    columns: tuple[str, ...]
    variables: tuple[Variable, ...]
    separator: str | None = None
    reader: Callable[..., Any] | None = None
    source: Any = None

    def as_kwargs(self) -> dict[str, Any]:
        """Returns all parsing arguments including normalized source as a dictionary.

        Returns:
            Dictionary suitable for keyword unpacking into ``DatasetParser.parse(**peek.as_kwargs())``.
        """
        kwargs: dict[str, Any] = {
            "source": self.source,
            "variables": self.variables,
        }
        if self.separator is not None:
            kwargs["separator"] = self.separator
        if self.reader is not None:
            kwargs["reader"] = self.reader
        return kwargs

    def keys(self) -> tuple[str, ...]:
        """Returns mapping keys for ``**peek`` unpacking into ``DatasetParser.parse(source, **peek)``.

        Returns:
            3-tuple ``("variables", "separator", "reader")``.
        """
        return "variables", "separator", "reader"

    def __getitem__(self, key: str) -> Any:
        """Accesses peek results by key for mapping protocol unpacking.

        Args:
            key: Attribute key (``"variables"``, ``"separator"``, ``"reader"``, or ``"source"``).

        Returns:
            Corresponding attribute value.

        Raises:
            KeyError: If ``key`` is not one of the supported mapping keys.
        """
        match key:
            case "variables":
                return self.variables
            case "separator":
                return self.separator
            case "reader":
                return self.reader
            case "source":
                return self.source
            case _:
                raise KeyError(key)


class DatasetParser(BaseParser):
    """Base class for parsing tabular data into SeroEpi ``Dataset`` instances.

    Subclasses declare a tuple of ``Variable`` instances mapping raw input column
    names to semantic variable logic. Exactly one variable must be an ``Identifier``.

    Attributes:
        variables: Tuple of ``Variable`` definitions declared by a concrete parser subclass.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.base import DatasetParser
        >>> ds = DatasetParser.parse(pl.DataFrame({"sample_id": ["s1", "s2"], "country": ["UK", "US"]}))
        >>> len(ds)
        2
    """

    variables: ClassVar[tuple[Variable, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validates that subclasses declare a non-empty ``variables`` tuple with exactly one ``Identifier``."""
        super().__init_subclass__(**kwargs)
        header_vars: tuple[Variable, ...] = getattr(cls, "header_variables", ())
        if (
            isinstance(cls.variables, tuple)
            and header_vars
            and not any(isinstance(v, Identifier) for v in cls.variables)
        ):
            cls.variables = (*header_vars, *cls.variables)
        if not cls.variables:
            raise TypeError(
                f"Class '{cls.__name__}' must define a non-empty class attribute "
                "'variables: ClassVar[tuple[Variable, ...]]'"
            )
        if not isinstance(cls.variables, tuple):
            raise TypeError(
                f"'{cls.__name__}.variables' must be a tuple of Variable instances, got {type(cls.variables).__name__}"
            )
        id_vars = [v for v in cls.variables if isinstance(v, Identifier)]
        if len(id_vars) != 1:
            raise ValueError(
                f"{cls.__name__} must define exactly one Identifier variable, found {len(id_vars)}: "
                f"{[v.column for v in id_vars]}"
            )
        dl_type = getattr(cls, "download_type", None)
        if isinstance(dl_type, str) and dl_type.strip():
            PathogenWatchFile._DOWNLOAD_REGISTRY[dl_type.strip().lower()] = cls

    @classmethod
    def _preprocess_dataframe(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Optional hook to transform or collapse the raw DataFrame prior to ``Dataset`` assembly.

        Args:
            df: Raw parsed ``pl.DataFrame``.

        Returns:
            Preprocessed ``pl.DataFrame`` with one row per isolate.
        """
        return df

    @classmethod
    def schema(cls) -> pl.Schema:
        """Derives the Polars schema directly from defined ``cls.variables``.

        Returns:
            Polars ``Schema`` mapping each variable's column name to its ``dtype``.
        """
        return pl.Schema({v.column: v.dtype for v in cls.variables})

    @staticmethod
    def _clean_mixed_dates(lf: pl.LazyFrame, date_col: str, res_col: str | None = None) -> pl.LazyFrame:
        """Standardizes mixed-format dates (YYYY, YYYY-MM, YYYY-MM-DD) into Polars Datetime."""
        s_col = pl.col(date_col).cast(pl.Utf8).str.strip_chars()
        s_len = s_col.str.len_bytes()

        res_expr = (
            pl.when(s_len == 4)
            .then(pl.lit(Temporal.Resolution.YEAR.value))
            .when((s_len >= 6) & (s_len <= 7) & s_col.str.contains(r"-|/"))
            .then(pl.lit(Temporal.Resolution.MONTH.value))
            .when(s_len >= 8)
            .then(pl.lit(Temporal.Resolution.DAY.value))
            .otherwise(pl.lit(Temporal.Resolution.UNKNOWN.value))
        )

        date_str_expr = (
            pl.when(res_expr == Temporal.Resolution.YEAR.value)
            .then(s_col + "-07-02")
            .when(res_expr == Temporal.Resolution.MONTH.value)
            .then(s_col + "-15")
            .otherwise(s_col)
        )

        parsed_date = date_str_expr.str.to_datetime(strict=False)
        exprs = [parsed_date.alias(date_col)]
        if res_col:
            final_res = (
                pl.when(parsed_date.is_null())
                .then(pl.lit(Temporal.Resolution.UNKNOWN.value))
                .otherwise(res_expr)
                .cast(pl.Categorical)
                .alias(res_col)
            )
            exprs.append(final_res)

        return lf.with_columns(exprs)

    @classmethod
    def _resolve_peek_variables(
        cls,
        columns: Sequence[str],
        id_column: str | None = None,
    ) -> tuple[Variable, ...]:
        """Resolves Variable instances for columns, honoring subclass variables if defined."""
        clean_columns = [c for c in columns if c.strip()]
        if not clean_columns:
            return ()

        if cls.variables:
            class_vars_by_col = {v.column: v for v in cls.variables}
            target_id = id_column
            has_id = target_id is not None or any(isinstance(v, Identifier) for v in cls.variables)
            inferred: list[Variable] = []
            for c in clean_columns:
                if target_id is not None and c == target_id:
                    inferred.append(Identifier(c))
                elif c in class_vars_by_col:
                    v = class_vars_by_col[c]
                    if isinstance(v, Identifier) and target_id is not None and c != target_id:
                        inferred.append(Variable(c))
                    else:
                        inferred.append(v)
                else:
                    var = Variable.guess(c)
                    if isinstance(var, Identifier) and has_id:
                        inferred.append(Variable(c))
                    else:
                        inferred.append(var)
            return tuple(inferred)

        return Variable.guess_many(clean_columns, id_column=id_column)

    @classmethod
    def peek(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        *,
        id_column: str | None = None,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
    ) -> PeekResult:
        """Inspects the header of a tabular source without consuming the data.

        Extracts column names, detects delimiter/reader, and generates typed Variable
        instances with strictly one Identifier.

        Args:
            source: File path, stream, raw bytes, DataFrame, or LazyFrame.
            id_column: Explicit column name to designate as primary Identifier.
            separator: Delimiter override.
            reader: Reader override.

        Returns:
            A PeekResult unpackable via **peek or directly passable to DatasetParser.parse(peek).
        """
        if isinstance(source, pl.DataFrame):
            cols = tuple(c for c in source.columns if c.strip())
            return PeekResult(
                columns=cols,
                variables=cls._resolve_peek_variables(cols, id_column=id_column),
                separator=separator,
                reader=reader,
                source=source,
            )

        if isinstance(source, pl.LazyFrame):
            cols = tuple(c for c in source.collect_schema().names() if c.strip())
            return PeekResult(
                columns=cols,
                variables=cls._resolve_peek_variables(cols, id_column=id_column),
                separator=separator,
                reader=reader,
                source=source,
            )

        norm_source = cls._normalize_source(source)
        suffixes = [s.lower() for s in norm_source.suffixes] if isinstance(norm_source, Path) else []

        # Parquet inspection
        if ".parquet" in suffixes or (isinstance(norm_source, bytes) and norm_source.startswith(b"PAR1")):
            pq_buf = BytesIO(norm_source) if isinstance(norm_source, bytes) else norm_source
            cols = tuple(c for c in pl.read_parquet_schema(pq_buf) if c.strip())
            return PeekResult(
                columns=cols,
                variables=cls._resolve_peek_variables(cols, id_column=id_column),
                separator=separator,
                reader=reader or pl.read_parquet,
                source=norm_source,
            )

        # Arrow / IPC inspection
        if (
            ".ipc" in suffixes
            or ".arrow" in suffixes
            or (isinstance(norm_source, bytes) and norm_source.startswith(b"ARROW1"))
        ):
            ipc_buf = BytesIO(norm_source) if isinstance(norm_source, bytes) else norm_source
            cols = tuple(c for c in pl.read_ipc_schema(ipc_buf) if c.strip())
            return PeekResult(
                columns=cols,
                variables=cls._resolve_peek_variables(cols, id_column=id_column),
                separator=separator,
                reader=reader or pl.read_ipc,
                source=norm_source,
            )

        # Excel inspection
        if ".xlsx" in suffixes or ".xls" in suffixes:
            excel_df = pl.read_excel(norm_source, read_options={"n_rows": 0})
            cols = tuple(c for c in excel_df.columns if c.strip())
            return PeekResult(
                columns=cols,
                variables=cls._resolve_peek_variables(cols, id_column=id_column),
                separator=separator,
                reader=reader or pl.read_excel,
                source=norm_source,
            )

        # Delimited text / gzip
        line = ""
        orig_pos: int | None = None
        if isinstance(norm_source, IOBase) and norm_source.seekable():
            orig_pos = norm_source.tell()

        if isinstance(norm_source, Path):
            if norm_source.name.lower().endswith(".gz"):
                with gzip.open(norm_source, "rt", encoding="utf-8-sig", errors="replace") as f:
                    line = f.readline()
            else:
                with open(norm_source, encoding="utf-8-sig", errors="replace") as f:
                    line = f.readline()
        elif isinstance(norm_source, bytes):
            if norm_source.startswith(b"\x1f\x8b"):
                with gzip.GzipFile(fileobj=BytesIO(norm_source)) as gz:
                    line = gz.readline().decode("utf-8-sig", errors="replace")
            else:
                line = norm_source.split(b"\n")[0].decode("utf-8-sig", errors="replace")
        elif isinstance(norm_source, IOBase):
            is_gz = False
            if hasattr(norm_source, "peek"):
                peek_bytes = norm_source.peek(2)[:2]
                is_gz = peek_bytes == b"\x1f\x8b"
            elif isinstance(norm_source, BytesIO):
                is_gz = norm_source.getvalue()[:2] == b"\x1f\x8b"

            if is_gz:
                with gzip.GzipFile(fileobj=norm_source) as gz:
                    line = gz.readline().decode("utf-8-sig", errors="replace")
            else:
                raw_line = norm_source.readline()
                line = raw_line.decode("utf-8-sig", errors="replace") if isinstance(raw_line, bytes) else raw_line

            if orig_pos is not None:
                norm_source.seek(orig_pos)

        # Strip BOM and newline chars
        line = line.lstrip("\ufeff").rstrip("\r\n")

        # Sniff delimiter
        if separator is not None:
            detected_sep = separator
        else:
            counts = {sep: line.count(sep) for sep in [",", "\t", ";", "|"]}
            if ".tsv" in suffixes and counts["\t"] > 0:
                detected_sep = "\t"
            elif ".csv" in suffixes and counts[","] > 0:
                detected_sep = ","
            elif counts["\t"] > counts[","] and counts["\t"] > 0:
                detected_sep = "\t"
            elif counts[","] > 0:
                detected_sep = ","
            elif any(c > 0 for c in counts.values()):
                detected_sep = max(counts.items(), key=lambda x: x[1])[0]
            else:
                detected_sep = "\t" if ".tsv" in suffixes else ","

        if line:
            header_df = pl.read_csv(norm_source, separator=detected_sep, n_rows=0)
            cols = tuple(c for c in header_df.columns if c.strip())
            if orig_pos is not None and isinstance(norm_source, IOBase):
                norm_source.seek(orig_pos)
        else:
            cols = ()

        inferred_vars = cls._resolve_peek_variables(cols, id_column=id_column)
        detected_reader = reader or pl.read_csv

        return PeekResult(
            columns=cols,
            variables=inferred_vars,
            separator=detected_sep,
            reader=detected_reader,
            source=norm_source,
        )

    @classmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame | PeekResult,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        variables: tuple[Variable, ...] | None = None,
        id_column: str | None = None,
        dataset_name: str | None = None,
        **kwargs: object,
    ) -> "Dataset":
        r"""Parses tabular genomics data into a SeroEpi Dataset with strongly typed variables.

        Args:
            source: File path, bytes, stream buffer, DataFrame, LazyFrame, or PeekResult.
            separator: Optional delimiter override.
            reader: Optional reader override.
            variables: Custom variables overriding or supplementing parser defaults.
            id_column: Optional column to designate as primary sample Identifier.
            dataset_name: Name for the resulting dataset container.
            **kwargs: Additional keyword arguments passed to underlying reader.

        Returns:
            A Dataset containing verbatim data and registered Variable mappings.
        """
        if isinstance(source, PeekResult):
            peek = source
            source = peek.source
            if separator is None:
                separator = peek.separator
            if reader is None:
                reader = peek.reader
            if variables is None:
                variables = peek.variables
        elif variables is None and not cls.variables:
            peek = cls.peek(source, id_column=id_column, separator=separator, reader=reader)
            source = peek.source
            if separator is None:
                separator = peek.separator
            if reader is None:
                reader = peek.reader
            variables = peek.variables

        effective_vars = tuple(v for v in (variables if variables is not None else cls.variables) if v.column.strip())

        if isinstance(source, pl.LazyFrame):
            df = source.collect()
        elif isinstance(source, pl.DataFrame):
            df = source
        else:
            norm_source = cls._normalize_source(source)
            if isinstance(norm_source, IOBase) and norm_source.seekable():
                norm_source.seek(0)
            reader_fn = cls._infer_reader(norm_source, reader)
            if reader_fn is pl.read_csv:
                schema_overrides = (
                    pl.Schema({v.column: v.dtype for v in effective_vars}) if variables is not None else cls.schema()
                )
                csv_kwargs: dict[str, Any] = {
                    "separator": cls._infer_separator(norm_source, separator),
                    "schema_overrides": schema_overrides,
                    "infer_schema_length": 10000,
                }
                csv_kwargs.update(kwargs)
                df = reader_fn(norm_source, **csv_kwargs)
            else:
                df = reader_fn(norm_source, **kwargs)

        blank_cols = [c for c in df.columns if not c.strip()]
        if blank_cols:
            df = df.drop(blank_cols)
        df = cls._preprocess_dataframe(df)

        # Validate strictly one Identifier column exists in data
        id_vars = [v for v in effective_vars if isinstance(v, Identifier)]
        if len(id_vars) != 1:
            raise ValueError(
                f"{cls.__name__} must define exactly one Identifier variable, found {len(id_vars)}: "
                f"{[v.column for v in id_vars]}"
            )
        id_var = id_vars[0]
        if id_var.column not in df.columns:
            raise ValueError(f"Expected Identifier column '{id_var.column}' not found in data columns: {df.columns}")

        # Assemble variables dict mapping verbatim column name -> Variable instance
        merged_vars: dict[str, Variable] = {v.column: v for v in effective_vars if v.column in df.columns}
        for c in df.columns:
            if c not in merged_vars:
                merged_vars[c] = Variable(c)

        return Dataset(
            data=df,
            name=dataset_name or str(uuid4()),
            variables=merged_vars,
        )


class PathogenWatchFile:
    """Mixin for Pathogenwatch CSV exports keyed by ``Genome Name`` and ``Genome ID``.

    Automatically prepends ``Identifier("Genome Name")`` and ``Variable("Genome ID")``
    to ``cls.variables`` on ``DatasetParser`` subclasses, registers ``download_type``
    for automatic ``PathogenwatchClient`` dispatch, and provides ``prefix_variables()``
    for multi-locus schemas such as Kaptive.

    Attributes:
        download_type: Optional Pathogenwatch analysis download identifier (e.g. ``"kleborate"``, ``"kaptive"``).
        join_column: Relational join column across Pathogenwatch tables (``"Genome ID"``).
        header_variables: Canonical ``(Identifier("Genome Name"), Variable("Genome ID"))`` tuple.
    """

    download_type: ClassVar[str | None] = None
    join_column: ClassVar[str] = "Genome ID"
    header_variables: ClassVar[tuple[Variable, ...]] = (
        Identifier("Genome Name"),
        Variable("Genome ID", dtype=pl.String(), title="Genome ID"),
    )
    _DOWNLOAD_REGISTRY: ClassVar[dict[str, type[DatasetParser]]] = {}

    @classmethod
    def parser_for_download(cls, download_type: str) -> type[DatasetParser] | None:
        """Returns the registered ``DatasetParser`` subclass for ``download_type``, or ``None``.

        Args:
            download_type: Pathogenwatch analysis download name (e.g. ``"kleborate"``, ``"kaptive"``, ``"mlst"``).

        Returns:
            Registered ``DatasetParser`` subclass, or ``None`` if unregistered.
        """
        return cls._DOWNLOAD_REGISTRY.get(download_type.strip().lower())

    @staticmethod
    def prefix_variables(
        variables: Sequence[Variable],
        prefix: str,
        trait_title: str | None = None,
    ) -> tuple[Variable, ...]:
        """Clones non-identifier ``Variable`` descriptors with a Pathogenwatch ``"<prefix>: "`` column prefix.

        Args:
            variables: Source ``Variable`` sequence (any ``Identifier`` entries are automatically skipped).
            prefix: Pathogenwatch column prefix (e.g. ``"K locus"``, ``"O locus"``, ``"OC locus"``).
            trait_title: Optional human-readable display title for ``Trait`` variables (e.g. ``"Capsule"``).

        Returns:
            Tuple of cloned ``Variable`` instances with prefixed column names and updated titles.
        """
        col_prefix = f"{prefix}: "
        title_prefix = f"{trait_title} " if trait_title else f"{prefix} "
        out: list[Variable] = []
        for var in variables:
            if isinstance(var, Identifier):
                continue
            new_col = f"{col_prefix}{var.column}"
            if isinstance(var, Trait) and trait_title is not None:
                out.append(var.copy(column=new_col, title=trait_title))
            else:
                out.append(var.copy(column=new_col, title=f"{title_prefix}{var.title}"))
        return tuple(out)
