"""Abstract base models and shared tabular interfaces for SeroEpi."""

import copy
import io
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self
from uuid import uuid4

import joblib
import polars as pl

from seroepi.variables import Variable


@dataclass(frozen=True, slots=True)
class DataModel(ABC):
    """Abstract base container wrapping a Polars DataFrame and semantic `Variable` mapping.

    Provides schema-synchronized `Variable` lookup (`var()`), row counting (`len()`), and
    compressed Joblib persistence (`save()` / `load()`).

    Attributes:
        data: Physical Polars `DataFrame` backing the container.
        variables: Mapping of physical column names to semantic `Variable` metadata instances.
        name: Human-readable or UUID identifier for the data container.
    """

    data: pl.DataFrame
    variables: Mapping[str, Variable]
    name: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        """Synchronizes variable dtypes with the physical DataFrame schema and strips blank column headers."""
        blank_cols = [c for c in self.data.columns if not c.strip()]
        if blank_cols:
            object.__setattr__(self, "data", self.data.drop(blank_cols))
        schema = self.data.schema
        updated_vars = {}
        for k, v in self.variables.items():
            if not k.strip() or not v.column.strip():
                continue
            if v.column in schema and v.dtype != schema[v.column]:
                updated_vars[k] = v.copy(dtype=schema[v.column])
            else:
                updated_vars[k] = v
        object.__setattr__(self, "variables", updated_vars)

    def __len__(self) -> int:
        """Returns the total number of rows in `self.data`."""
        return self.data.height

    def _lookup_variable(self, col: str) -> Variable:
        """Performs strict O(1) dictionary lookup by physical column name."""
        if not isinstance(col, str):
            raise TypeError(f"Column lookup requires a string column name, got {type(col).__name__}.")
        if col not in self.variables:
            raise KeyError(
                f"Variable '{col}' not found in {self.__class__.__name__} variables: {list(self.variables.keys())}"
            )
        return self.variables[col]

    def var(self, *names: Any) -> Any:
        """Retrieves one or more registered `Variable` instances from the container by column name.

        Args:
            *names: One or more physical column names, or a single sequence of column names.

        Returns:
            A single `Variable` if one column name was passed, or a tuple of `Variable` instances
            if multiple column names (or a sequence) were passed.

        Raises:
            ValueError: If no column names are provided.
            KeyError: If any requested column name is not registered in `self.variables`.
            TypeError: If any argument is not a string column name.
        """
        if not names:
            raise ValueError("At least one column name must be provided to var().")
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            return tuple(self._lookup_variable(i) for i in names[0])
        if len(names) == 1:
            return self._lookup_variable(names[0])
        return tuple(self._lookup_variable(n) for n in names)

    def save(self, path: str | Path | None = None, compress: int = 3) -> bytes | Path:
        """Serializes the `DataModel` instance into a compressed Joblib file or in-memory bytes.

        Args:
            path: Destination file path. If `None`, serializes to an in-memory `bytes` buffer.
            compress: Zlib compression level (`0` to `9`). Defaults to `3`.

        Returns:
            The resolved `Path` written to disk if `path` was provided, otherwise serialized `bytes`.
        """
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, dest, compress=compress)
            return dest
        buffer = io.BytesIO()
        joblib.dump(self, buffer, compress=compress)
        return buffer.getvalue()

    @classmethod
    def load(cls, source: str | Path | bytes | io.BytesIO) -> Self:
        """Loads a `DataModel` instance from a Joblib file, path, or bytes buffer.

        Args:
            source: Filesystem path to a `.joblib` file, raw `bytes`, or `io.BytesIO` buffer.

        Returns:
            The deserialized `DataModel` subclass instance.

        Raises:
            FileNotFoundError: If a file path is given but does not exist on disk.
            TypeError: If `source` is an unsupported type or the deserialized object is not an instance of `cls`.
        """
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Joblib file not found: {src_path}")
            obj = joblib.load(src_path)
        elif isinstance(source, bytes):
            obj = joblib.load(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            obj = joblib.load(source)
        else:
            raise TypeError(f"Unsupported source type for read_joblib: {type(source).__name__}")

        if not isinstance(obj, cls):
            raise TypeError(f"Expected deserialized object to be {cls.__name__}, got {type(obj).__name__}")
        return obj


class LoadedData(DataModel, ABC):
    """Abstract base class for persistent `DataModel` assets compiled and cached on disk.

    Subclasses define `FILENAME` and implement `compile()` to fetch, parse, and assemble reference
    or benchmark datasets, which `build()` atomically persists under `~/.seroepi/` (or `$SEROEPI_DATA_DIR`).

    Attributes:
        FILENAME: Default filename for the serialized `.joblib` asset inside `get_default_dir()`.
        DIR_ENV_VAR: Environment variable name overriding the default asset directory.
        DEFAULT_SUBDIR: Subdirectory under `~/.seroepi/` when `DIR_ENV_VAR` is unset.
    """

    __slots__ = ()

    FILENAME: ClassVar[str]
    DIR_ENV_VAR: ClassVar[str] = "SEROEPI_DATA_DIR"
    DEFAULT_SUBDIR: ClassVar[str] = "data"

    @classmethod
    def get_default_dir(cls) -> Path:
        """Returns the default directory for persistent assets.

        Returns:
            `Path` resolved from `os.environ[cls.DIR_ENV_VAR]` if set, otherwise `~/.seroepi/{DEFAULT_SUBDIR}`.
        """
        env_dir = os.environ.get(cls.DIR_ENV_VAR)
        if env_dir:
            return Path(env_dir)
        return Path.home() / ".seroepi" / cls.DEFAULT_SUBDIR

    @classmethod
    def get_default_path(cls) -> Path:
        """Returns the default local asset path (`{get_default_dir()}/{FILENAME}`).

        Returns:
            Full filesystem `Path` to the cached `.joblib` asset.
        """
        return cls.get_default_dir() / cls.FILENAME

    @classmethod
    def is_built(cls, path: Path | str | None = None) -> bool:
        """Checks whether the local asset exists on disk and is a non-empty file.

        Args:
            path: Optional custom path to check. Defaults to `cls.get_default_path()`.

        Returns:
            `True` if the target file exists and has size greater than zero bytes.
        """
        target = Path(path) if path else cls.get_default_path()
        return target.is_file() and target.stat().st_size > 0

    @classmethod
    def clear(cls, path: Path | str | None = None) -> None:
        """Removes the locally cached asset file from disk if present.

        Args:
            path: Optional custom path to remove. Defaults to `cls.get_default_path()`.
        """
        target = Path(path) if path else cls.get_default_path()
        if target.exists():
            target.unlink()

    @classmethod
    def load(cls, source: str | Path | bytes | io.BytesIO | None = None) -> Self:
        """Loads the persistent asset from disk or buffer into an in-memory instance.

        Args:
            source: Optional filesystem path, raw bytes, or `BytesIO` buffer.
                If `None`, resolves to `cls.get_default_path()`.

        Returns:
            The loaded instance of `cls`.

        Raises:
            FileNotFoundError: If the target path does not exist on disk.
        """
        target = (
            Path(source)
            if isinstance(source, (str, Path))
            else (source if source is not None else cls.get_default_path())
        )
        if isinstance(target, Path) and not target.exists():
            raise FileNotFoundError(
                f"{cls.__name__} asset not found at {target}. "
                f"Please call `{cls.__name__}.build()` to download and compile it."
            )
        return super().load(target)

    @classmethod
    @abstractmethod
    def compile(cls, *args: Any, **kwargs: Any) -> Self:
        """Downloads, parses, and constructs the in-memory `LoadedData` instance.

        Args:
            *args: Positional compilation arguments defined by subclasses.
            **kwargs: Keyword compilation arguments defined by subclasses.

        Returns:
            Newly compiled `LoadedData` subclass instance.
        """
        ...

    @classmethod
    def build(
        cls,
        out_path: Path | str | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Compiles and atomically persists the asset to disk if not already built.

        Args:
            out_path: Optional destination file path. Defaults to `cls.get_default_path()`.
            force: If `True`, forces recompilation even when `cls.is_built(out_path)` is `True`.
            **kwargs: Additional keyword arguments forwarded to `cls.compile(**kwargs)`.

        Returns:
            The compiled (or loaded) `LoadedData` subclass instance.
        """
        target = Path(out_path) if out_path else cls.get_default_path()
        if not force and cls.is_built(target):
            return cls.load(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        instance = cls.compile(**kwargs)

        tmp_path = target.with_suffix(f".tmp.{os.getpid()}.{uuid4().hex[:8]}")
        try:
            instance.save(tmp_path)
            tmp_path.replace(target)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return instance


class StratifiedResultMixin(ABC):
    """Shared mixin for fitted stratified tabular results (`Aggregator` and `BaseEstimator`).

    Provides uniform accessors for target/stratum/adjustment column names and titles,
    fitted-state guards (`check_is_fitted()`), Joblib persistence (`save()` / `load()`),
    and immutable subsetting/partitioning (`filter()` / `split()`).

    Attributes:
        is_fitted_: Whether `fit()` has completed on this instance.
        variables_: Mapping of output column names to semantic `Variable` instances.
        target_: Fitted target `Variable` (typically `Trait(column="target", ...)`).
        strata_: Tuple of stratification `Variable` instances used during aggregation.
        adjust_for_: Optional cluster-adjustment `Variable` used for effective sample counts.
        is_compositional_: Whether target categories are mutually exclusive compositional categories.
        is_padded_: Whether unobserved `(stratum, target)` cells were zero-padded or upsampled.
    """

    is_fitted_: bool = False
    variables_: dict[str, Variable]
    target_: Variable
    strata_: tuple[Variable, ...]
    adjust_for_: Variable | None = None
    is_compositional_: bool = False
    is_padded_: bool = False

    def check_is_fitted(self) -> None:
        """Validates that the instance has been fitted.

        Raises:
            RuntimeError: If `self.is_fitted_` is `False`.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using it."
            )

    @abstractmethod
    def _get_frame(self) -> pl.DataFrame:
        """Returns the primary fitted Polars DataFrame (`data_` or `estimates_`)."""
        ...

    @abstractmethod
    def _set_frame(self, df: pl.DataFrame) -> None:
        """Assigns the primary fitted Polars DataFrame on a filtered clone."""
        ...

    @property
    def is_compositional(self) -> bool:
        """Whether the fitted target represents mutually exclusive compositional categories."""
        self.check_is_fitted()
        return self.is_compositional_

    @property
    def is_padded(self) -> bool:
        """Whether unobserved stratum-target combinations were zero-padded or temporally upsampled."""
        self.check_is_fitted()
        return self.is_padded_

    @property
    def target_col(self) -> str:
        """Physical column name of the target variable in the fitted DataFrame."""
        self.check_is_fitted()
        return self.target_.column

    @property
    def target_title(self) -> str:
        """Human-readable display title of the target variable."""
        self.check_is_fitted()
        return self.target_.title

    @property
    def strata_cols(self) -> tuple[str, ...]:
        """Tuple of physical column names for all stratification variables."""
        self.check_is_fitted()
        return tuple(s.column for s in self.strata_)

    @property
    def strata_titles(self) -> tuple[str, ...]:
        """Tuple of human-readable display titles for all stratification variables."""
        self.check_is_fitted()
        return tuple(s.title for s in self.strata_)

    @property
    def adjust_for_col(self) -> str | None:
        """Physical column name of the cluster-adjustment variable, or `None` if unadjusted."""
        self.check_is_fitted()
        return self.adjust_for_.column if self.adjust_for_ is not None else None

    @property
    def adjust_for_title(self) -> str | None:
        """Human-readable display title of the cluster-adjustment variable, or `None` if unadjusted."""
        self.check_is_fitted()
        return self.adjust_for_.title if self.adjust_for_ is not None else None

    @property
    def n_strata(self) -> int:
        """Number of stratification dimensions."""
        self.check_is_fitted()
        return len(self.strata_)

    @property
    def is_stratified(self) -> bool:
        """Whether at least one stratification variable was used."""
        self.check_is_fitted()
        return len(self.strata_) > 0

    @property
    def is_single_stratum(self) -> bool:
        """Whether exactly one stratification variable was used."""
        self.check_is_fitted()
        return len(self.strata_) == 1

    @property
    def summary_label(self) -> str:
        """Human-readable summary of target, stratification, and cluster adjustment."""
        self.check_is_fitted()
        strata_part = f" by {', '.join(self.strata_titles)}" if self.strata_ else " (Unstratified)"
        adj_part = f" [Adj: {self.adjust_for_title}]" if self.adjust_for_ is not None else ""
        return f"{self.target_title}{strata_part}{adj_part}"

    def save(self, path: str | Path | None = None, compress: int = 3) -> bytes | Path:
        """Serializes the fitted result instance into a compressed Joblib file or bytes.

        Args:
            path: Destination file path. If `None`, returns serialized `bytes`.
            compress: Zlib compression level (`0` to `9`). Defaults to `3`.

        Returns:
            The destination `Path` if `path` was provided, otherwise `bytes`.
        """
        self.check_is_fitted()
        if path is not None:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, dest, compress=compress)
            return dest
        buffer = io.BytesIO()
        joblib.dump(self, buffer, compress=compress)
        return buffer.getvalue()

    @classmethod
    def load(cls, source: str | Path | bytes | io.BytesIO) -> Self:
        """Loads a fitted instance from a Joblib file, path, or bytes buffer.

        Args:
            source: Path to a `.joblib` file, raw `bytes`, or `io.BytesIO` buffer.

        Returns:
            The deserialized fitted instance of `cls`.

        Raises:
            FileNotFoundError: If a path is provided and does not exist.
            TypeError: If `source` has an unsupported type or deserializes to a different class.
        """
        if isinstance(source, (str, Path)):
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Joblib file not found: {src_path}")
            obj = joblib.load(src_path)
        elif isinstance(source, bytes):
            obj = joblib.load(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            obj = joblib.load(source)
        else:
            raise TypeError(f"Unsupported source type for load: {type(source).__name__}")

        if not isinstance(obj, cls):
            raise TypeError(f"Expected deserialized object to be {cls.__name__}, got {type(obj).__name__}")
        return obj

    @property
    def single_stratum_col(self) -> str:
        """Physical column name of the sole stratification variable when `is_single_stratum` is `True`.

        Raises:
            AttributeError: If `n_strata != 1`.
        """
        self.check_is_fitted()
        if not self.is_single_stratum:
            raise AttributeError(f"Expected exactly 1 stratum, found {self.n_strata}: {list(self.strata_cols)}")
        return self.strata_[0].column

    @property
    def single_stratum_title(self) -> str:
        """Human-readable display title of the sole stratification variable when `is_single_stratum` is `True`.

        Raises:
            AttributeError: If `n_strata != 1`.
        """
        self.check_is_fitted()
        if not self.is_single_stratum:
            raise AttributeError(f"Expected exactly 1 stratum, found {self.n_strata}: {list(self.strata_titles)}")
        return self.strata_[0].title

    def infer_single_stratum_value(self, data: pl.DataFrame | None = None) -> str | None:
        """Returns the unique stratum label if the single-stratum column contains exactly one non-null value.

        Args:
            data: Optional DataFrame to inspect. Defaults to the primary fitted DataFrame.

        Returns:
            String representation of the sole unique stratum value, or `None` if multiple/zero values exist.
        """
        self.check_is_fitted()
        df = self._get_frame() if data is None else data
        if self.is_single_stratum and self.single_stratum_col in df.columns:
            unique_vals = df[self.single_stratum_col].drop_nulls().unique().to_list()
            if len(unique_vals) == 1:
                return str(unique_vals[0])
        return None

    def _lookup_variable(self, col: str) -> Variable:
        """Performs strict O(1) dictionary lookup by physical column name."""
        if not isinstance(col, str):
            raise TypeError(f"Column lookup requires a string column name, got {type(col).__name__}.")
        if col not in self.variables_:
            raise KeyError(
                f"Variable '{col}' not found in {self.__class__.__name__} variables: {list(self.variables_.keys())}"
            )
        return self.variables_[col]

    def var(self, *names: Any) -> Any:
        """Retrieves one or more registered `Variable` instances from the fitted result by column name.

        Args:
            *names: One or more column names, or a sequence of column names.

        Returns:
            A single `Variable` or a tuple of `Variable` instances.

        Raises:
            ValueError: If no column names are provided.
            KeyError: If any column name is not registered in `self.variables_`.
        """
        self.check_is_fitted()
        if not names:
            raise ValueError("At least one column name must be provided to var().")
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            return tuple(self._lookup_variable(i) for i in names[0])
        if len(names) == 1:
            return self._lookup_variable(names[0])
        return tuple(self._lookup_variable(n) for n in names)

    def filter(self, *predicates: pl.Expr | bool, **kwargs: Any) -> Self:
        """Filters the fitted tabular data by Polars expressions or column equality keyword arguments.

        Args:
            *predicates: Polars boolean filter expressions (`pl.Expr`).
            **kwargs: Column equality constraints (`col_name=value`).

        Returns:
            A shallow clone of `self` with its primary fitted DataFrame filtered to matching rows.

        Raises:
            KeyError: If any keyword column name is not present in the fitted DataFrame.
            ValueError: If the filter predicates match zero rows.
        """
        self.check_is_fitted()
        df = self._get_frame()

        exprs: list[pl.Expr | bool] = list(predicates)
        for col_name, val in kwargs.items():
            if col_name not in df.columns:
                raise KeyError(f"Column '{col_name}' not found in {self.__class__.__name__} columns: {df.columns}")
            exprs.append(pl.col(col_name) == val)

        if not exprs:
            return self

        filtered_df = df.filter(*exprs)
        if len(filtered_df) == 0:
            raise ValueError(f"Filter produced an empty dataset (predicates={predicates}, kwargs={kwargs}).")

        cloned = copy.copy(self)
        cloned._set_frame(filtered_df)
        return cloned

    def split(self, by: str | Sequence[str] | None = None) -> dict[Any, Self]:
        """Partitions the fitted result by stratum column(s) into a dictionary of filtered instances.

        Args:
            by: Optional column name or sequence of column names to partition by.
                Defaults to all `self.strata_cols`.

        Returns:
            Dictionary mapping each unique stratum value (or tuple of values) to a filtered clone of `self`.

        Raises:
            ValueError: If `by` is `None` and the instance is unstratified.
            KeyError: If any specified column in `by` does not exist in the fitted DataFrame.
        """
        self.check_is_fitted()
        df = self._get_frame()

        if by is None:
            if not self.strata_:
                raise ValueError(
                    f"Cannot split {self.__class__.__name__}: no strata were defined during aggregation. "
                    "Specify 'by' column explicitly."
                )
            by_cols = [s.column for s in self.strata_]
        elif isinstance(by, str):
            by_cols = [by]
        else:
            by_cols = list(by)

        for col in by_cols:
            if col not in df.columns:
                raise KeyError(f"Split column '{col}' not found in {self.__class__.__name__} columns: {df.columns}")

        unique_groups = df.select(by_cols).unique(maintain_order=True)
        splits: dict[Any, Self] = {}
        for row in unique_groups.iter_rows(named=False):
            key = row[0] if len(row) == 1 else tuple(row)
            filter_kwargs = {col: val for col, val in zip(by_cols, row, strict=True)}
            splits[key] = self.filter(**filter_kwargs)

        return splits
