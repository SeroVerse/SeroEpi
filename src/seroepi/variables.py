"""Variable hierarchy, quality control expressions, and criteria generators for SeroEpi."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from enum import StrEnum, auto
from functools import total_ordering
from typing import Any, ClassVar, Literal, Self

import polars as pl


# Classes --------------------------------------------------------------------------------------------------------------
@total_ordering
class OrderedStrEnum(StrEnum):
    """String enum whose comparison ordering reflects member declaration order (`coarse -> fine`).

    Examples:
        >>> from seroepi.variables import Spatial
        >>> Spatial.Resolution.CONTINENT < Spatial.Resolution.COUNTRY
        True
    """

    @property
    def rank(self) -> int:
        """Zero-based hierarchy rank based on declaration order."""
        return list(self.__class__).index(self)

    def __lt__(self, other: object) -> bool:
        """Compares hierarchy ranks between two enum members of the same class."""
        if isinstance(other, self.__class__):
            return self.rank < other.rank
        return NotImplemented


DEFAULT_NEGATIVE_PATTERN: str = r"^(?i:-|0|false|f|none|absent|neg|negative|no|null|\s*)$"


@dataclass(frozen=True, slots=True)
class Variable:
    """Base semantic variable wrapping a physical Polars DataFrame column.

    Encapsulates physical column name, human-readable display title, Polars data type,
    and regex patterns for negative/absent indicators.

    Attributes:
        column: Physical column name in the backing `pl.DataFrame`.
        title: Human-readable display label (defaults to `column` when omitted).
        dtype: Expected Polars `DataType` of the column.
        negative_indicator: Optional regex pattern, string, or sequence of tokens treated as negative/absent calls.

    Examples:
        >>> from seroepi.variables import Variable
        >>> var = Variable.guess("Country")
        >>> type(var).__name__, var.column
        ('Spatial', 'Country')
    """

    column: str
    title: str = ""
    dtype: pl.DataType = field(default_factory=pl.String)
    negative_indicator: re.Pattern[str] | str | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not self.title:
            object.__setattr__(self, "title", self.column)

    def __str__(self) -> str:
        return self.column

    def copy(self, **changes: Any) -> Self:
        """Returns a modified copy of this variable with updated fields.

        Args:
            **changes: Field names and replacement values forwarded to `dataclasses.replace`.

        Returns:
            New `Variable` instance with the requested field updates.
        """
        return replace(self, **changes)

    @property
    def role(self) -> str:
        """Lowercase semantic role name derived from the concrete `Variable` subclass."""
        return self.__class__.__name__.lower()

    @property
    def is_compositional(self) -> bool:
        """Whether this variable represents a mutually exclusive compositional breakdown."""
        return False

    @classmethod
    def guess(cls, column: str) -> "Variable":
        """Infers a semantic `Variable` subclass from a column header via token lookup and typo recovery.

        Args:
            column: Physical column name from a tabular header.

        Returns:
            Inferred `Variable` subclass instance (`Identifier`, `Spatial`, `Temporal`, `Taxonomy`, etc.),
            or a base `Variable(column)` if no semantic token matches.
        """
        clean = column.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
        tokens = [t for t in clean.split("_") if t]

        # Pass 1: O(1) exact token match
        for token in tokens:
            if token in _KEYWORD_MAP:
                subclass, kwargs = _KEYWORD_MAP[token]
                return subclass(column, **kwargs)

        # Pass 2: secondary difflib fallback for human typos on tokens of length >= 5
        for token in tokens:
            if len(token) >= 5:
                if matches := get_close_matches(token, _FUZZY_CANDIDATES, n=1, cutoff=0.8):
                    subclass, kwargs = _KEYWORD_MAP[matches[0]]
                    return subclass(column, **kwargs)

        return cls(column)

    @classmethod
    def guess_many(
        cls,
        columns: Sequence[str],
        *,
        id_column: str | None = None,
    ) -> tuple["Variable", ...]:
        """Infers semantic Variable classes for a sequence of column names with exactly one Identifier.

        Ensures exactly one Identifier is present to uphold Dataset invariants:
        - If `id_column` is provided, that column is designated as the Identifier.
        - Otherwise, the first column matching an Identifier token is designated as
          the Identifier. If no column matches, the first column in the sequence
          (columns[0]) is used as the fallback Identifier.
        - Any subsequent column that would guess as an Identifier is demoted to a
          standard Variable.

        Args:
            columns: Sequence of column names from the data header.
            id_column: Explicit column name to designate as the Identifier.

        Returns:
            Tuple of Variable instances with strictly one Identifier.
        """
        if not columns:
            return ()

        if id_column is not None:
            if id_column not in columns:
                raise ValueError(f"Specified id_column '{id_column}' not found in columns: {list(columns)}")
            target_id_col = id_column
        else:
            target_id_col = columns[0]
            for c in columns:
                var = cls.guess(c)
                if isinstance(var, Identifier):
                    target_id_col = c
                    break

        assigned_id = False
        result: list[Variable] = []
        for c in columns:
            if c == target_id_col and not assigned_id:
                result.append(Identifier(c))
                assigned_id = True
            else:
                var = cls.guess(c)
                if isinstance(var, Identifier):
                    result.append(Variable(c))
                else:
                    result.append(var)

        return tuple(result)

    def to_expr(self) -> pl.Expr:
        """Returns a Polars expression referencing the physical column.

        Returns:
            `pl.col(self.column)` expression.
        """
        return pl.col(self.column)

    def negative_expr(self, indicator: re.Pattern[str] | str | Sequence[str] | None = None) -> pl.Expr:
        """Returns a boolean Polars expression identifying negative or absent observations.

        Args:
            indicator: Optional override for `self.negative_indicator` (regex, string, or token sequence).

        Returns:
            Boolean `pl.Expr` evaluating to `True` where the column is null or matches the negative indicator.
        """
        col = pl.col(self.column)
        if self.dtype == pl.Boolean():
            return col.is_null() | (~col)
        if self.dtype.is_numeric():
            return col.is_null() | (col <= 0)
        ind = indicator if indicator is not None else self.negative_indicator
        if ind is None:
            pat = DEFAULT_NEGATIVE_PATTERN
        elif isinstance(ind, re.Pattern):
            pat = ind.pattern
        elif isinstance(ind, (list, tuple, set)):
            pat = "|".join(f"(?:{re.escape(s)})" for s in ind)
        else:
            pat = str(ind)
        return col.is_null() | col.cast(pl.Utf8).str.strip_chars().str.contains(pat).fill_null(True)

    def match_expr(
        self,
        pattern: str | re.Pattern[str] = ".*",
        *,
        negative_indicator: re.Pattern[str] | str | Sequence[str] | None = None,
        equivalence: dict[str, str] | None = None,
    ) -> pl.Expr:
        """Returns a boolean Polars expression matching `pattern` while suppressing negative calls.

        Args:
            pattern: Regular expression string or compiled pattern to match positive calls against.
            negative_indicator: Optional override for `self.negative_indicator`.
            equivalence: Optional `{raw_value: canonical_value}` replacement map applied prior to matching.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for non-negative values matching `pattern`.
        """
        col = pl.col(self.column)
        if equivalence:
            col = col.replace(equivalence)

        is_neg = self.negative_expr(indicator=negative_indicator)
        pat_str = pattern.pattern if isinstance(pattern, re.Pattern) else str(pattern)

        if pat_str in (".*", "^.*$", ""):
            match_expr = col.is_not_null() & ~is_neg
        else:
            match_expr = col.cast(pl.Utf8).str.contains(pat_str).fill_null(False) & ~is_neg

        return match_expr.fill_null(False)

    def is_positive_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression checking if an observation is present and not negative.

        Returns:
            Boolean `pl.Expr` equivalent to `self.match_expr(".*")`.
        """
        return self.match_expr(".*")


@dataclass(frozen=True, slots=True)
class Identifier(Variable):
    """Primary key variable containing unique isolate or sample identifiers.

    Attributes:
        column: Physical column name of the sample identifier.
        title: Human-readable display label.
    """

    def check_uniqueness_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression verifying whether identifier values are unique.

        Returns:
            `pl.col(self.column).is_unique()` expression.
        """
        return pl.col(self.column).is_unique()


@dataclass(frozen=True, slots=True)
class QC(Variable):
    """Quality-control threshold variable generating boolean isolate filtering masks.

    Supports numeric lower/upper bounds (`min_value`, `max_value`), categorical allowlists
    (`allowed_values`), or custom Polars predicate expressions (`expr`).

    Attributes:
        column: Physical column name evaluated by the QC rule.
        title: Human-readable display label.
        min_value: Optional minimum inclusive numeric threshold (`column >= min_value`).
        max_value: Optional maximum inclusive numeric threshold (`column <= max_value`).
        allowed_values: Optional tuple of accepted string values (`column.is_in(allowed_values)`).
        expr: Optional custom Polars boolean filter expression overriding scalar bounds.
        dtype: Polars data type of the QC column (defaults to `pl.Float64`).

    Examples:
        >>> from seroepi.variables import QC
        >>> rule = QC.max("contig_count", 150, title="Max Contigs")
        >>> rule.describe()
        'contig_count <= 150'
    """

    min_value: float | None = None
    max_value: float | None = None
    allowed_values: tuple[str, ...] | None = None
    expr: pl.Expr | None = None
    dtype: pl.DataType = field(default_factory=pl.Float64)

    def __hash__(self) -> int:
        """Computes hash using scalar bounds and serialized Polars AST expression."""
        expr_key = self.expr.meta.serialize(format="json") if self.expr is not None else None
        return hash((self.column, self.min_value, self.max_value, self.allowed_values, expr_key))

    def __eq__(self, other: object) -> bool:
        """Compares QC instances by scalar thresholds and serialized Polars expression AST."""
        if not isinstance(other, QC):
            return False
        if (self.column, self.min_value, self.max_value, self.allowed_values, self.title, self.role) != (
            other.column,
            other.min_value,
            other.max_value,
            other.allowed_values,
            other.title,
            other.role,
        ):
            return False
        if (self.expr is None) != (other.expr is None):
            return False
        if self.expr is not None and other.expr is not None:
            return self.expr.meta.serialize(format="json") == other.expr.meta.serialize(format="json")
        return True

    @classmethod
    def min(cls, column: str, value: float, title: str = "") -> "QC":
        """Creates a minimum threshold QC rule (`column >= value`).

        Args:
            column: Physical column name.
            value: Minimum inclusive numeric threshold.
            title: Optional human-readable display label.

        Returns:
            Configured `QC` instance.
        """
        return cls(
            column,
            title=title,
            min_value=value,
            dtype=pl.Float64() if isinstance(value, float) else pl.Int64(),
        )

    @classmethod
    def max(cls, column: str, value: float, title: str = "") -> "QC":
        """Creates a maximum threshold QC rule (`column <= value`).

        Args:
            column: Physical column name.
            value: Maximum inclusive numeric threshold.
            title: Optional human-readable display label.

        Returns:
            Configured `QC` instance.
        """
        return cls(
            column,
            title=title,
            max_value=value,
            dtype=pl.Float64() if isinstance(value, float) else pl.Int64(),
        )

    @classmethod
    def between(cls, column: str, low: float, high: float, title: str = "") -> "QC":
        """Creates a bounded interval QC rule (`low <= column <= high`).

        Args:
            column: Physical column name.
            low: Minimum inclusive numeric bound.
            high: Maximum inclusive numeric bound.
            title: Optional human-readable display label.

        Returns:
            Configured `QC` instance.
        """
        return cls(
            column,
            title=title,
            min_value=low,
            max_value=high,
            dtype=pl.Float64(),
        )

    @classmethod
    def equals(cls, column: str, value: str, title: str = "") -> "QC":
        """Creates an exact categorical acceptance QC rule (`column == value`).

        Args:
            column: Physical column name.
            value: Required exact string value.
            title: Optional human-readable display label.

        Returns:
            Configured `QC` instance.
        """
        return cls(
            column,
            title=title,
            allowed_values=(value,),
            dtype=pl.String(),
        )

    @classmethod
    def one_of(cls, column: str, values: Sequence[str], title: str = "") -> "QC":
        """Creates a categorical set acceptance QC rule (`column in values`).

        Args:
            column: Physical column name.
            values: Sequence of allowed string values.
            title: Optional human-readable display label.

        Returns:
            Configured `QC` instance.
        """
        return cls(
            column,
            title=title,
            allowed_values=tuple(values),
            dtype=pl.String(),
        )

    def filter_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression applying the QC acceptance criteria.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for isolates passing this QC threshold.
        """
        if self.expr is not None:
            return self.expr
        col = pl.col(self.column)
        if self.min_value is not None and self.max_value is not None:
            return col.is_between(self.min_value, self.max_value)
        if self.min_value is not None:
            return col >= self.min_value
        if self.max_value is not None:
            return col <= self.max_value
        if self.allowed_values is not None:
            return col.is_in(list(self.allowed_values))
        return pl.lit(True)

    def describe(self) -> str:
        """Formats a human-readable summary of this QC variable's acceptance threshold.

        Returns:
            Concise string representation of the threshold (e.g. `"contig_count <= 150"`).
        """
        if self.min_value is not None and self.max_value is not None:
            return f"{self.column} in [{self.min_value:g}, {self.max_value:g}]"
        if self.min_value is not None:
            return f"{self.column} >= {self.min_value:g}"
        if self.max_value is not None:
            return f"{self.column} <= {self.max_value:g}"
        if self.allowed_values is not None:
            if len(self.allowed_values) == 1:
                return f"{self.column} == '{self.allowed_values[0]}'"
            vals = ", ".join(self.allowed_values)
            return f"{self.column} in {{{vals}}}"
        return f"{self.column} (custom QC)"


@dataclass(frozen=True, slots=True)
class Taxonomy(Variable):
    """Taxonomic or lineage classification variable (e.g. species, MLST sequence type, clonal group).

    Attributes:
        column: Physical column name.
        title: Human-readable display label.
        dtype: Polars data type (defaults to `pl.Categorical`).
        resolution: Taxonomic hierarchy resolution level (`Taxonomy.Resolution`).
    """

    class Resolution(OrderedStrEnum):
        """Hierarchical resolution levels for taxonomic classifications."""

        UNKNOWN = auto()
        SPECIES = auto()
        SUB_SPECIES = auto()
        CLONAL_GROUP = auto()
        SEQUENCE_TYPE = auto()
        GENOMIC_CLONE = auto()
        TRANSMISSION_CLONE = auto()

    dtype: pl.DataType = field(default_factory=pl.Categorical)
    resolution: Resolution = Resolution.UNKNOWN

    @property
    def is_compositional(self) -> bool:
        """Taxonomic classifications are mutually exclusive compositional categories."""
        return True


@dataclass(frozen=True, slots=True)
class Spatial(Variable):
    """Spatial variable representing geographic locality and administrative resolution.

    Supports automatic projection between country names, ISO3 codes, UN M49 subregions,
    and macro-continents via `CountryData`.

    Attributes:
        column: Physical column name.
        title: Human-readable display label.
        resolution: Geographic resolution level (`Spatial.Resolution`).
    """

    class Resolution(OrderedStrEnum):
        """Geographic resolution levels for spatial epidemiology."""

        UNKNOWN = auto()
        GLOBAL = auto()
        CONTINENT = auto()
        REGION = auto()
        COUNTRY = auto()
        CITY = auto()
        SITE = auto()
        POINT = auto()

    resolution: Resolution = Resolution.UNKNOWN

    def clean_expr(self) -> pl.Expr:
        """Returns a UTF-8 cast and whitespace-trimmed Polars expression for spatial lookups.

        Returns:
            Trimmed string `pl.Expr` aliased to `self.column`.
        """
        return pl.col(self.column).cast(pl.Utf8).str.strip_chars().alias(self.column)

    def iso3_expr(self, strict: bool = False, country_data: Any = None) -> pl.Expr:
        """Projects country strings to 3-letter ISO3 codes via the `CountryData` alias map.

        Args:
            strict: If `True`, unmatched country names become `None`. If `False`, retains trimmed input strings.
            country_data: Optional `CountryData` instance. Defaults to `CountryData.load()`.

        Returns:
            Polars expression resolving country strings to ISO3 codes.
        """
        from seroepi.country import CountryData

        try:
            cd = country_data or CountryData.load()
            alias_df = cd.aliases_df
        except FileNotFoundError:
            return self.clean_expr()

        clean = pl.col(self.column).cast(pl.Utf8).str.strip_chars()
        upper = clean.str.to_uppercase()
        default = None if strict else clean

        direct_iso = upper.replace_strict(old=alias_df["alias"], new=alias_df["iso3"], default=None)
        prefix_iso = (
            upper.str.split(" - ")
            .list.get(0)
            .str.strip_chars()
            .replace_strict(old=alias_df["alias"], new=alias_df["iso3"], default=None)
        )
        suffix_iso = (
            upper.str.split(" - ")
            .list.get(-1)
            .str.strip_chars()
            .replace_strict(old=alias_df["alias"], new=alias_df["iso3"], default=None)
        )

        return pl.coalesce(direct_iso, prefix_iso, suffix_iso, default).alias(self.column)

    def region_expr(self, strict: bool = False, country_data: Any = None) -> pl.Expr:
        """Projects country strings to UN M49 subregions (e.g. `'Western Europe'`).

        Args:
            strict: If `True`, unmatched values become `None`.
            country_data: Optional `CountryData` instance.

        Returns:
            Polars expression resolving country strings to UN M49 subregions.
        """
        from seroepi.country import CountryData

        try:
            cd = country_data or CountryData.load()
        except FileNotFoundError:
            return self.clean_expr()

        iso3 = self.iso3_expr(strict=True, country_data=cd)
        default = None if strict else pl.col(self.column).cast(pl.Utf8)
        return iso3.replace_strict(old=cd.data["iso3"], new=cd.data["subregion"], default=default).alias(self.column)

    def continent_expr(self, strict: bool = False, country_data: Any = None) -> pl.Expr:
        """Projects country strings to macro-continents (e.g. `'Europe'`, `'Africa'`, `'Americas'`).

        Args:
            strict: If `True`, unmatched values become `None`.
            country_data: Optional `CountryData` instance.

        Returns:
            Polars expression resolving country strings to macro-continents.
        """
        from seroepi.country import CountryData

        try:
            cd = country_data or CountryData.load()
        except FileNotFoundError:
            return self.clean_expr()

        iso3 = self.iso3_expr(strict=True, country_data=cd)
        default = None if strict else pl.col(self.column).cast(pl.Utf8)
        return iso3.replace_strict(old=cd.data["iso3"], new=cd.data["region"], default=default).alias(self.column)

    def to_resolution_expr(self, target_resolution: Resolution) -> pl.Expr:
        """Routes to the appropriate projection expression for `target_resolution`.

        Args:
            target_resolution: Target `Spatial.Resolution` level.

        Returns:
            Polars expression projecting `self.column` to `target_resolution`.
        """
        if target_resolution == self.Resolution.COUNTRY:
            return self.iso3_expr()
        if target_resolution == self.Resolution.REGION:
            return self.region_expr()
        if target_resolution == self.Resolution.CONTINENT:
            return self.continent_expr()
        return self.clean_expr()

    def to_expr(self) -> pl.Expr:
        """Projects the spatial column according to its configured `resolution`.

        Returns:
            Polars expression projecting `self.column`.
        """
        if self.resolution != self.Resolution.UNKNOWN:
            return self.to_resolution_expr(self.resolution)
        return pl.col(self.column)


@dataclass(frozen=True, slots=True)
class Temporal(Variable):
    """Temporal epidemiological variable supporting date parsing, ISO epi-weeks, and truncation.

    Attributes:
        column: Physical column name.
        title: Human-readable display label.
        resolution: Temporal resolution level (`Temporal.Resolution`).
        format: Optional explicit `strptime` format string.
        time_unit: Sub-second timestamp precision (`"ms"`, `"us"`, or `"ns"`).
    """

    class Resolution(OrderedStrEnum):
        """Epidemiological temporal resolutions supported by SeroEpi."""

        UNKNOWN = auto()
        YEAR = auto()
        QUARTER = auto()
        MONTH = auto()
        EPI_WEEK = auto()
        WEEK = auto()
        DAY = auto()
        HOUR = auto()

        @property
        def interval(self) -> str | None:
            """Polars interval duration string (e.g. `'1y'`, `'1mo'`) for `truncate` or `group_by_dynamic`."""
            return {
                self.YEAR: "1y",
                self.QUARTER: "1q",
                self.MONTH: "1mo",
                self.EPI_WEEK: "1w",
                self.WEEK: "1w",
                self.DAY: "1d",
                self.HOUR: "1h",
            }.get(self)

    resolution: Resolution = Resolution.UNKNOWN
    format: str | None = None
    time_unit: Literal["ms", "us", "ns"] = "us"

    def parse_expr(self) -> pl.Expr:
        """Parses strings, 4-digit years, `YYYY-MM`, or timestamps into a Polars `Datetime` expression.

        Returns:
            `pl.Datetime` expression aliased to `self.column`.
        """
        col = pl.col(self.column)
        if self.format:
            return col.cast(pl.Utf8).str.to_datetime(format=self.format, strict=False).alias(self.column)
        s_col = col.cast(pl.Utf8).str.strip_chars()
        s_len = s_col.str.len_bytes()
        norm_str = (
            pl.when(s_len == 4)
            .then(s_col + "-01-01")
            .when((s_len >= 6) & (s_len <= 7) & s_col.str.contains(r"-|/"))
            .then(s_col + "-01")
            .otherwise(s_col)
        )
        return norm_str.str.to_datetime(strict=False).alias(self.column)

    def epi_week_expr(self) -> pl.Expr:
        """Returns a Polars expression formatting timestamps as ISO/CDC Epidemiological Weeks (`'YYYY-Www'`).

        Returns:
            String `pl.Expr` aliased to `self.column`.
        """
        dt_col = self.parse_expr()
        return pl.concat_str(
            [
                dt_col.dt.iso_year().cast(pl.Utf8),
                pl.lit("-W"),
                dt_col.dt.week().cast(pl.Utf8).str.zfill(2),
            ]
        ).alias(self.column)

    def quarter_expr(self) -> pl.Expr:
        """Returns a Polars expression formatting timestamps as calendar quarters (`'YYYY-Q#'`).

        Returns:
            String `pl.Expr` aliased to `self.column`.
        """
        dt_col = self.parse_expr()
        return pl.concat_str(
            [
                dt_col.dt.year().cast(pl.Utf8),
                pl.lit("-Q"),
                dt_col.dt.quarter().cast(pl.Utf8),
            ]
        ).alias(self.column)

    def truncate_expr(self) -> pl.Expr:
        """Truncates the temporal column to the configured `self.resolution` boundary.

        Returns:
            Polars expression truncated to `self.resolution`.
        """
        if self.resolution == self.Resolution.EPI_WEEK:
            return self.epi_week_expr()
        if self.resolution == self.Resolution.QUARTER:
            return self.quarter_expr()

        dt_col = self.parse_expr()
        return (
            dt_col.dt.truncate(self.resolution.interval).alias(self.column)
            if self.resolution.interval
            else dt_col.alias(self.column)
        )

    def to_expr(self) -> pl.Expr:
        """Projects the temporal column according to its configured `resolution`.

        Returns:
            Polars expression projecting `self.column`.
        """
        if self.resolution != self.Resolution.UNKNOWN:
            return self.truncate_expr()
        return pl.col(self.column)


@dataclass(frozen=True, slots=True)
class Trait(Variable):
    """Biological, antigenic, or clinical trait variable (e.g. capsule K-locus, O-antigen serotype).

    Attributes:
        column: Physical column name.
        title: Human-readable display label (e.g. `'Capsule'`).
        dtype: Polars data type (defaults to `pl.Categorical`).
        compositional: Whether categories of this trait are mutually exclusive across isolates.
        phenotype: Whether this trait represents an observed lab phenotype rather than an in-silico genotype.
    """

    dtype: pl.DataType = field(default_factory=pl.Categorical)
    compositional: bool = True
    phenotype: bool = False

    @property
    def is_compositional(self) -> bool:
        """Whether this trait operates as a mutually exclusive compositional breakdown."""
        return self.compositional

    @staticmethod
    def _compile_pattern_str(
        pat: re.Pattern[str] | str | Sequence[str],
        *,
        field_name: str = "pattern",
        cls_name: str = "Trait",
    ) -> str:
        """Normalizes a compiled regex, pattern string, or sequence of literal tokens into a regex string."""
        if isinstance(pat, re.Pattern):
            return pat.pattern
        if isinstance(pat, (list, tuple, set)):
            tokens = [str(s) for s in pat if str(s)]
            if not tokens:
                raise ValueError(f"{cls_name}.{field_name} sequence cannot be empty.")
            return "|".join(f"(?:{re.escape(s)})" for s in tokens)
        return str(pat)

    def _validate_regex_fields(self, field_names: tuple[str, ...]) -> None:
        """Normalizes sequence regex fields to tuples and validates regex compilation."""
        cls_name = type(self).__name__
        for field_name in field_names:
            val = getattr(self, field_name)
            if val is None:
                continue
            if isinstance(val, (list, set)):
                object.__setattr__(self, field_name, tuple(val))
            pat_str = self._compile_pattern_str(
                getattr(self, field_name),
                field_name=field_name,
                cls_name=cls_name,
            )
            try:
                re.compile(pat_str)
            except Exception as exc:
                raise ValueError(f"Invalid regex for {cls_name}.{field_name} ({val!r}): {exc}") from exc


@dataclass(frozen=True, slots=True)
class Resistance(Trait):
    """Antimicrobial resistance trait associated with specific antibiotic compounds or classes.

    Provides regex-driven S/I/R (Sensitive / Intermediate / Resistant) tier classification
    expressions with precedence `R > I > S`.

    Attributes:
        column: Physical column name.
        title: Human-readable display label (e.g. `'Carbapenems'`).
        resolution: Antibiotic compound hierarchy resolution (`Resistance.Resolution`).
        sensitive_pattern: Optional regex, string, or sequence identifying Sensitive (`S`) calls.
        intermediate_pattern: Optional regex, string, or sequence identifying Intermediate (`I`) calls.
        resistant_pattern: Optional regex, string, or sequence identifying Resistant (`R`) calls.
    """

    class Resolution(OrderedStrEnum):
        """Compound resolution levels."""

        UNKNOWN = auto()
        SUPERFAMILY = auto()  # e.g. Beta-lactams
        CLASS = auto()  # e.g. Cephalosporins (3rd Generation)
        DRUG_NAME = auto()  # e.g. Ceftriaxone
        BRAND_NAME = auto()  # e.g. Rocephin

    resolution: Resolution = Resolution.UNKNOWN
    sensitive_pattern: re.Pattern[str] | str | Sequence[str] | None = None
    intermediate_pattern: re.Pattern[str] | str | Sequence[str] | None = None
    resistant_pattern: re.Pattern[str] | str | Sequence[str] | None = None

    def __post_init__(self) -> None:
        Trait.__post_init__(self)
        self._validate_regex_fields(("sensitive_pattern", "intermediate_pattern", "resistant_pattern"))

    def _raw_tier_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Returns mutually exclusive (sensitive, intermediate, resistant) boolean Polars expressions (R > I > S)."""
        explicit_i = (
            self.match_expr(self._compile_pattern_str(self.intermediate_pattern))
            if self.intermediate_pattern is not None
            else pl.lit(False)
        )
        explicit_s = (
            self.match_expr(self._compile_pattern_str(self.sensitive_pattern))
            if self.sensitive_pattern is not None
            else None
        )

        if self.resistant_pattern is not None:
            r_expr = self.match_expr(self._compile_pattern_str(self.resistant_pattern))
        else:
            r_expr = self.is_positive_expr() & ~explicit_i
            if explicit_s is not None:
                r_expr = r_expr & ~explicit_s

        i_expr = (explicit_i & ~r_expr).fill_null(False)

        if explicit_s is not None:
            s_expr = (explicit_s & ~r_expr & ~i_expr).fill_null(False)
        else:
            s_expr = (pl.col(self.column).is_not_null() & self.negative_expr() & ~r_expr & ~i_expr).fill_null(False)

        return s_expr, i_expr, r_expr.fill_null(False)

    def sensitive_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying sensitive (`S`) observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for susceptible isolates.
        """
        s_expr, _, _ = self._raw_tier_exprs()
        return s_expr

    def intermediate_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying intermediate (`I`) observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for intermediate isolates.
        """
        _, i_expr, _ = self._raw_tier_exprs()
        return i_expr

    def resistant_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying resistant (`R`) observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for resistant isolates.
        """
        _, _, r_expr = self._raw_tier_exprs()
        return r_expr

    def non_susceptible_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying non-susceptible (`I` or `R`) observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for intermediate or resistant isolates.
        """
        _, i_expr, r_expr = self._raw_tier_exprs()
        return (r_expr | i_expr).fill_null(False)

    def sir_expr(self) -> pl.Expr:
        """Returns a categorical Polars expression evaluating each observation to `'R'`, `'I'`, `'S'`, or `None`.

        Returns:
            String `pl.Expr` aliased to `self.column`.
        """
        s_expr, i_expr, r_expr = self._raw_tier_exprs()
        return (
            pl.when(r_expr)
            .then(pl.lit("R"))
            .when(i_expr)
            .then(pl.lit("I"))
            .when(s_expr)
            .then(pl.lit("S"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.column)
        )


@dataclass(frozen=True, slots=True)
class Virulence(Trait):
    """Virulence factor or pathogenic determinant trait with gene-cluster integrity classification.

    Classifies observations into mutually exclusive `intact`, `incomplete`, or `truncated`
    states with precedence `truncated > incomplete > intact`.

    Attributes:
        column: Physical column name.
        title: Human-readable display label (e.g. `'Yersiniabactin'`).
        factor: Functional virulence factor class (`Virulence.Factor`).
        requires_bgc: Whether a complete biosynthetic gene cluster is required for functional expression.
        compositional: Whether categories are mutually exclusive (defaults to `False`).
        intact_pattern: Optional regex, string, or sequence matching intact calls.
        incomplete_pattern: Regex, string, or sequence matching incomplete locus calls.
        truncated_pattern: Regex, string, or sequence matching truncated locus calls.
    """

    class Factor(StrEnum):
        """Virulence factor categories."""

        UNKNOWN = auto()
        ADHESIN = auto()
        INVASIN = auto()
        ENDOTOXIN = auto()
        EXOTOXIN = auto()
        CAPSULE = auto()
        SECRETION_SYSTEM = auto()
        SIDEROPHORE = auto()  # e.g. Yersiniabactin, Aerobactin, Salmochelin

    DEFAULT_TRUNCATED_REGEX: ClassVar[str] = r"\(truncated\)|-\d+%$"
    DEFAULT_INCOMPLETE_REGEX: ClassVar[str] = r"\(incomplete\)|\bunknown\b|\b0(?:;|$)"

    factor: Factor = Factor.UNKNOWN
    requires_bgc: bool = True  # Whether a complete biosynthetic gene cluster is required for expression
    compositional: bool = False
    intact_pattern: re.Pattern[str] | str | Sequence[str] | None = None
    incomplete_pattern: re.Pattern[str] | str | Sequence[str] | None = DEFAULT_INCOMPLETE_REGEX
    truncated_pattern: re.Pattern[str] | str | Sequence[str] | None = DEFAULT_TRUNCATED_REGEX

    def __post_init__(self) -> None:
        Trait.__post_init__(self)
        self._validate_regex_fields(("intact_pattern", "incomplete_pattern", "truncated_pattern"))

    def _raw_integrity_exprs(self) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Returns mutually exclusive (intact, incomplete, truncated) boolean Polars expressions (truncated > incomplete > intact)."""
        trunc_expr = (
            self.match_expr(self._compile_pattern_str(self.truncated_pattern))
            if self.truncated_pattern is not None
            else pl.lit(False)
        ).fill_null(False)
        incomp_raw = (
            self.match_expr(self._compile_pattern_str(self.incomplete_pattern))
            if self.incomplete_pattern is not None
            else pl.lit(False)
        )
        incomp_expr = (incomp_raw & ~trunc_expr).fill_null(False)

        if self.intact_pattern is not None:
            intact_raw = self.match_expr(self._compile_pattern_str(self.intact_pattern))
        else:
            intact_raw = self.is_positive_expr()
        intact_expr = (intact_raw & ~trunc_expr & ~incomp_expr).fill_null(False)

        return intact_expr, incomp_expr, trunc_expr

    def intact_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying intact/functional virulence observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for intact virulence loci.
        """
        intact_expr, _, _ = self._raw_integrity_exprs()
        return intact_expr

    def incomplete_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying incomplete/partial virulence observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for incomplete virulence loci.
        """
        _, incomp_expr, _ = self._raw_integrity_exprs()
        return incomp_expr

    def truncated_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying truncated virulence observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for truncated virulence loci.
        """
        _, _, trunc_expr = self._raw_integrity_exprs()
        return trunc_expr

    def defective_expr(self) -> pl.Expr:
        """Returns a boolean Polars expression identifying defective (truncated or incomplete) virulence observations.

        Returns:
            Boolean `pl.Expr` evaluating to `True` for truncated or incomplete virulence loci.
        """
        _, incomp_expr, trunc_expr = self._raw_integrity_exprs()
        return (trunc_expr | incomp_expr).fill_null(False)

    def integrity_expr(self) -> pl.Expr:
        """Returns a categorical Polars expression evaluating each observation to `'intact'`, `'incomplete'`, `'truncated'`, or `None`.

        Returns:
            String `pl.Expr` aliased to `self.column`.
        """
        intact_expr, incomp_expr, trunc_expr = self._raw_integrity_exprs()
        return (
            pl.when(trunc_expr)
            .then(pl.lit("truncated"))
            .when(incomp_expr)
            .then(pl.lit("incomplete"))
            .when(intact_expr)
            .then(pl.lit("intact"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias(self.column)
        )


@dataclass(frozen=True, slots=True)
class Metric(Variable):
    """Base class for quantitative statistical summaries and continuous model outputs.

    Attributes:
        column: Physical column name.
        title: Human-readable display label.
        dtype: Polars data type (defaults to `pl.Float64`).
        unit: Optional measurement unit string (e.g. `'bp'`, `'%'`).
    """

    dtype: pl.DataType = field(default_factory=pl.Float64)
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class Count(Metric):
    """Discrete integer frequency or sample size variable (e.g. `event`, denominator `n`, cluster count).

    Attributes:
        column: Physical column name.
        title: Human-readable display label.
        dtype: Polars data type (defaults to `pl.Int64`).
        is_denominator: Whether this count column represents a stratum sample size denominator (`n`).
    """

    dtype: pl.DataType = field(default_factory=pl.Int64)
    is_denominator: bool = False


@dataclass(frozen=True, slots=True)
class Estimate(Metric):
    """Statistical point estimate metadata with linked uncertainty bounds and formatting rules.

    Attributes:
        column: Physical column name of the point estimate (e.g. `'estimate'`).
        title: Human-readable display label (defaults to `'Prevalence'`).
        lower: Optional column name of the lower uncertainty bound (e.g. `'lower'`).
        upper: Optional column name of the upper uncertainty bound (e.g. `'upper'`).
        level: Confidence or credible interval coverage probability in `(0, 1)` (e.g. `0.95`).
        bounded: Whether values are bounded on `[0, 1]`.
        method: Identifier of the statistical estimator or interval method (e.g. `'wilson'`, `'svi'`).
        ci_prefix: Short label prefix for uncertainty intervals (`'CI'` or `'CrI'`).
        format: Numeric format string for axis and tooltip display (defaults to `'.1%'`).
        interval_description: Full human-readable description of the interval (e.g. `'95% Wilson Score CI'`).
    """

    title: str = "Prevalence"

    # Uncertainty bounds (column names in the dataframe)
    lower: str | None = None
    upper: str | None = None
    level: float = 0.95  # e.g. 0.95 for 95% CI

    # Statistical semantics
    bounded: bool = True  # True if bounded [0, 1] (prevalence, coverage)
    method: str | None = None  # e.g. "wilson", "clopper_pearson", "svi_hdi"

    # Presentation traits
    ci_prefix: str = "CI"
    format: str = ".1%"
    interval_description: str | None = None

    @property
    def has_interval(self) -> bool:
        """Whether this estimate has both `lower` and `upper` interval bound column names configured."""
        return self.lower is not None and self.upper is not None


@dataclass(frozen=True, slots=True)
class Criterion:
    """Single-variable matching rule pairing a `Variable` with a regex pattern and equivalence map.

    Attributes:
        variable: Target `Variable` to evaluate.
        pattern: Regular expression string or compiled pattern matching positive categories (`'.*'` for all positive).
        equivalence: Optional `{raw_category: canonical_category}` dictionary for cross-reactive alias collapse
            (e.g. `{'KL102': 'KL2'}`).
        negative_indicator: Optional override for `variable.negative_indicator`.

    Examples:
        >>> from seroepi.variables import Criterion, Trait
        >>> crit = Criterion(
        ...     variable=Trait("K_locus", title="Capsule"),
        ...     pattern="^(KL1|KL2|KL102)$",
        ...     equivalence={"KL102": "KL2"},
        ... )
        >>> crit.describe()
        'K_locus ~ /^(KL1|KL2|KL102)$/ [equiv: KL102=KL2]'
    """

    variable: Variable
    pattern: str | re.Pattern[str] = ".*"
    equivalence: dict[str, str] = field(default_factory=dict)
    negative_indicator: re.Pattern[str] | str | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.variable, Variable):
            raise TypeError(
                f"Criterion variable must be a Variable instance, got {type(self.variable).__name__}. "
                "Use dataset.var(...) to retrieve it."
            )
        try:
            re.compile(self.pattern)
        except Exception as e:
            raise ValueError(f"Criterion pattern must be a valid regex, got {self.pattern!r}: {e}") from e

    def __hash__(self) -> int:
        pat_key = self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern
        equiv_key = tuple(sorted(self.equivalence.items()))
        neg_key = (
            self.negative_indicator.pattern
            if isinstance(self.negative_indicator, re.Pattern)
            else tuple(self.negative_indicator)
            if isinstance(self.negative_indicator, (list, tuple))
            else self.negative_indicator
        )
        return hash((self.variable, pat_key, equiv_key, neg_key))

    @property
    def column(self) -> str:
        """Physical column name of the underlying `variable`."""
        return self.variable.column

    @property
    def title(self) -> str:
        """Human-readable display title of the underlying `variable`."""
        return self.variable.title

    @property
    def is_compositional(self) -> bool:
        """Whether this criterion's underlying `variable` is compositional."""
        return self.variable.is_compositional

    @property
    def pattern_str(self) -> str:
        """String representation of `self.pattern`."""
        return self.pattern.pattern if isinstance(self.pattern, re.Pattern) else str(self.pattern)

    @property
    def capture_group_index(self) -> int:
        """Regex group index to extract (`1` if `pattern` defines capture groups, else `0`)."""
        compiled = re.compile(self.pattern)
        return 1 if compiled.groups > 0 else 0

    def to_expr(self) -> pl.Expr:
        """Constructs a boolean Polars expression evaluating whether each row satisfies this criterion.

        Returns:
            Boolean `pl.Expr`.
        """
        return self.variable.match_expr(
            pattern=self.pattern,
            negative_indicator=self.negative_indicator,
            equivalence=self.equivalence or None,
        )

    def to_extract_expr(self) -> pl.Expr:
        """Constructs a Polars expression extracting the matched category string (after equivalence replacement).

        Returns:
            String `pl.Expr` evaluating to the matched category token or `None` for negative/unmatched rows.
        """
        col = pl.col(self.column)
        if self.equivalence:
            col = col.replace(self.equivalence)
        is_neg = self.variable.negative_expr(indicator=self.negative_indicator)
        pat_str = self.pattern_str

        if pat_str in (".*", "^.*$", ""):
            extracted = col.cast(pl.Utf8)
        else:
            extracted = col.cast(pl.Utf8).str.extract(pat_str, self.capture_group_index)
        return pl.when(~is_neg).then(extracted).otherwise(None)

    @staticmethod
    def parse_equivalence(raw: str) -> dict[str, str]:
        """Parses comma- or newline-delimited `key=value` or `key:value` pairs into a dictionary.

        Args:
            raw: Raw user input string containing `alias=canonical` pairs separated by commas or newlines.

        Returns:
            Dictionary mapping stripped source tokens to canonical target tokens.
        """
        mapping: dict[str, str] = {}
        if not raw or not raw.strip():
            return mapping
        for token in raw.replace("\n", ",").split(","):
            item = token.strip()
            if not item:
                continue
            for sep in ("=", ":"):
                if sep in item:
                    k, v = item.split(sep, 1)
                    if k.strip() and v.strip():
                        mapping[k.strip()] = v.strip()
                    break
        return mapping

    @classmethod
    def from_inputs(
        cls,
        variable: Variable,
        pattern: str = ".*",
        equivalence_text: str = "",
        negative_indicator: str = "",
    ) -> "Criterion":
        """Constructs a `Criterion` from raw UI text inputs.

        Args:
            variable: Bound semantic `Variable`.
            pattern: Regex pattern string (defaults to `'.*'`).
            equivalence_text: Comma- or newline-delimited `key=value` equivalence string.
            negative_indicator: Optional custom negative indicator regex string.

        Returns:
            Validated `Criterion` instance.
        """
        return cls(
            variable=variable,
            pattern=(pattern or ".*").strip() or ".*",
            equivalence=cls.parse_equivalence(equivalence_text),
            negative_indicator=(negative_indicator or "").strip() or None,
        )

    def with_variable(self, variable: Variable) -> "Criterion":
        """Returns a copy of this criterion bound to `variable`.

        Args:
            variable: Replacement `Variable` instance.

        Returns:
            Cloned `Criterion` referencing `variable`.
        """
        return replace(self, variable=variable)

    def describe(self) -> str:
        """Formats a compact human-readable summary of this criterion.

        Returns:
            Summary string showing column name, regex pattern, and equivalence mappings.
        """
        pat = self.pattern_str.strip() or ".*"
        base = f"{self.column}" if pat == ".*" else f"{self.column} ~ /{pat}/"
        if self.equivalence:
            equiv_str = ", ".join(f"{k}={v}" for k, v in self.equivalence.items())
            base += f" [equiv: {equiv_str}]"
        return base


@dataclass(slots=True)
class VariableFilter:
    """Toggleable filter specification bound to a semantic `Variable` for interactive subsetting.

    Attributes:
        column: Physical column name to filter.
        variable: Semantic `Variable` associated with `column`.
        enabled: Whether this filter is currently active.
        kind: Filter modality (`'categorical'`, `'numeric'`, `'temporal'`, or `'presence'`).
        selected_values: Tuple of allowed string values for `'categorical'` or `'values'` filters.
        numeric_range: Optional `(min_val, max_val)` inclusive range for `'numeric'` filters.
        year_range: Optional `(start_year, end_year)` inclusive range for `'temporal'` filters.
        presence_mode: Semantic state selector for `'presence'` filters on `Trait`, `Resistance`, or `Virulence`.
    """

    column: str
    variable: Variable
    enabled: bool = True
    kind: Literal["categorical", "numeric", "temporal", "presence"] = "categorical"
    selected_values: tuple[str, ...] = ()
    numeric_range: tuple[float, float] | None = None
    year_range: tuple[int, int] | None = None
    presence_mode: Literal[
        "positive",
        "negative",
        "resistant",
        "intermediate",
        "sensitive",
        "non_susceptible",
        "intact",
        "defective",
        "truncated",
        "incomplete",
        "values",
    ] = "positive"

    def to_expr(self) -> pl.Expr | None:
        """Compiles this filter specification into a Polars boolean predicate expression.

        Returns:
            Boolean `pl.Expr` if `self.enabled` is `True` and a valid rule is configured, otherwise `None`.
        """
        if not self.enabled:
            return None

        col = pl.col(self.column)
        if self.kind == "presence":
            if isinstance(self.variable, Resistance):
                if self.presence_mode == "resistant":
                    return self.variable.resistant_expr()
                if self.presence_mode == "intermediate":
                    return self.variable.intermediate_expr()
                if self.presence_mode == "sensitive":
                    return self.variable.sensitive_expr()
                if self.presence_mode == "non_susceptible":
                    return self.variable.non_susceptible_expr()
            if isinstance(self.variable, Virulence):
                if self.presence_mode == "intact":
                    return self.variable.intact_expr()
                if self.presence_mode == "defective":
                    return self.variable.defective_expr()
                if self.presence_mode == "truncated":
                    return self.variable.truncated_expr()
                if self.presence_mode == "incomplete":
                    return self.variable.incomplete_expr()
            if self.presence_mode == "positive":
                return self.variable.is_positive_expr()
            if self.presence_mode == "negative":
                return self.variable.negative_expr()
            if self.selected_values:
                return col.cast(pl.Utf8).is_in(list(self.selected_values))
            return None

        if self.kind == "numeric" and self.numeric_range is not None:
            lo, hi = self.numeric_range
            return col.cast(pl.Float64, strict=False).is_between(lo, hi)

        if self.kind == "temporal" and self.year_range is not None:
            y_lo, y_hi = self.year_range
            if self.variable.dtype in (pl.Date, pl.Datetime):
                return col.dt.year().is_between(y_lo, y_hi)
            extracted_yr = col.cast(pl.Utf8).str.extract(r"(\d{4})", 1).cast(pl.Int64, strict=False)
            return extracted_yr.is_between(y_lo, y_hi)

        if self.kind == "categorical" and self.selected_values:
            return col.cast(pl.Utf8).is_in(list(self.selected_values))

        return None

    def describe(self) -> str:
        """Returns a concise human-readable description of the filter rule.

        Returns:
            Formatted summary string suitable for UI filter pills.
        """
        if self.kind == "presence":
            if self.presence_mode == "resistant":
                return f"{self.column} is resistant (R)"
            if self.presence_mode == "intermediate":
                return f"{self.column} is intermediate (I)"
            if self.presence_mode == "sensitive":
                return f"{self.column} is sensitive (S)"
            if self.presence_mode == "non_susceptible":
                return f"{self.column} is non-susceptible (I/R)"
            if self.presence_mode == "intact":
                return f"{self.column} is intact"
            if self.presence_mode == "defective":
                return f"{self.column} is defective (truncated/incomplete)"
            if self.presence_mode == "truncated":
                return f"{self.column} is truncated"
            if self.presence_mode == "incomplete":
                return f"{self.column} is incomplete"
            if self.presence_mode == "positive":
                return f"{self.column} is positive (present)"
            if self.presence_mode == "negative":
                return f"{self.column} is negative (absent)"
            vals = ", ".join(self.selected_values[:4])
            suffix = "..." if len(self.selected_values) > 4 else ""
            return f"{self.column} in [{vals}{suffix}]"
        if self.kind == "numeric" and self.numeric_range is not None:
            return f"{self.column} in [{self.numeric_range[0]:g}, {self.numeric_range[1]:g}]"
        if self.kind == "temporal" and self.year_range is not None:
            return f"{self.column} year in [{self.year_range[0]}, {self.year_range[1]}]"
        vals = ", ".join(self.selected_values[:4])
        suffix = f" (+{len(self.selected_values) - 4})" if len(self.selected_values) > 4 else ""
        return f"{self.column} in [{vals}{suffix}]"

    def cache_key(self) -> tuple[Any, ...]:
        """Returns a hashable tuple representing the active filter specification.

        Returns:
            Tuple of scalar filter attributes for memoization keys.
        """
        return (
            self.column,
            self.enabled,
            self.kind,
            self.selected_values,
            self.numeric_range,
            self.year_range,
            self.presence_mode,
        )


@dataclass(frozen=True, slots=True)
class Target:
    """Composite epidemiological target defining a serotype breakdown, resistance phenotype, or vaccine formulation.

    Combines one or more `Criterion` rules with boolean conjunction (`'or'` / `'and'`),
    supporting both mutually exclusive compositional breakdowns (`is_compositional=True`)
    and binary coverage endpoints (`is_compositional=False`).

    Attributes:
        name: Programmatic identifier and output column alias for the target.
        criteria: Sequence of constituent `Criterion` matching rules.
        conjunction: Logical operator combining multiple criteria (`'or'` for union, `'and'` for intersection).
        compositional: Optional explicit override for compositional vs. binary mode.
        sep: Delimiter string joining extracted category tokens across multi-criterion compositional targets.
        title: Human-readable display label (defaults to `name`).

    Examples:
        >>> from seroepi.dataset import Dataset
        >>> from seroepi.variables import Criterion, Target
        >>> ds = Dataset.example()
        >>> tgt = Target(
        ...     name="K_and_O",
        ...     title="Capsule + O-Antigen",
        ...     criteria=[
        ...         Criterion(ds.var("K_locus"), equivalence={"KL102": "KL2"}),
        ...         Criterion(ds.var("O_locus")),
        ...     ],
        ...     conjunction="and",
        ... )
        >>> breakdown_df, kpis = tgt.evaluate(ds, top_n=5)
    """

    name: str
    criteria: Sequence[Criterion]
    conjunction: Literal["or", "and"] = "or"
    compositional: bool | None = None
    sep: str = " "
    title: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            object.__setattr__(self, "title", self.name)
        if self.compositional is None or self.compositional is True:
            has_comp = any(c.is_compositional for c in self.criteria)
            has_non_comp = any(not c.is_compositional for c in self.criteria)
            if has_comp and has_non_comp:
                raise ValueError(
                    f"Target '{self.name}' mixes compositional traits with non-compositional variables. "
                    "Filter non-compositional conditions beforehand or set Target(compositional=False) "
                    "to define a pure binary endpoint."
                )

    @classmethod
    def coerce(
        cls,
        spec: "Target | Criterion | Variable",
        *,
        compositional: bool | None = None,
        negative_indicator: re.Pattern[str] | str | Sequence[str] | None = None,
    ) -> "Target":
        """Normalizes a `Variable`, `Criterion`, or `Target` into a canonical `Target` instance.

        Args:
            spec: Input `Target`, `Criterion`, or `Variable`.
            compositional: Optional explicit compositional mode override.
            negative_indicator: Optional negative indicator override applied to `Variable` or `Criterion` inputs.

        Returns:
            Canonical `Target` instance.

        Raises:
            TypeError: If `spec` is not a `Target`, `Criterion`, or `Variable`.
        """
        if isinstance(spec, Target):
            if compositional is not None and spec.compositional != compositional:
                return replace(spec, compositional=compositional)
            return spec
        if isinstance(spec, Criterion):
            crit = replace(spec, negative_indicator=negative_indicator) if negative_indicator is not None else spec
            return cls(
                name=crit.column,
                criteria=[crit],
                compositional=compositional,
                title=crit.title,
            )
        if isinstance(spec, Variable):
            var = spec.copy(negative_indicator=negative_indicator) if negative_indicator is not None else spec
            crit = Criterion(variable=var, pattern=".*")
            return cls(
                name=var.column,
                criteria=[crit],
                compositional=compositional if compositional is not None else var.is_compositional,
                title=var.title,
            )
        raise TypeError(
            f"Target specification must be a Variable, Criterion, or Target instance, got {type(spec).__name__}."
        )

    @property
    def column(self) -> str:
        """Output column alias (`self.name`)."""
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def is_compositional(self) -> bool:
        """Whether this target operates in mutually exclusive compositional categorical mode."""
        if self.compositional is not None:
            return self.compositional
        return all(c.is_compositional for c in self.criteria) if self.criteria else False

    def to_expr(self) -> pl.Expr:
        """Evaluates `self.conjunction` (`'or'` or `'and'`) across all criteria into a boolean Polars expression.

        Returns:
            Boolean `pl.Expr` aliased to `self.name`.
        """
        if not self.criteria:
            return pl.lit(False).alias(self.name)
        exprs = [c.to_expr() for c in self.criteria]
        if self.conjunction == "and":
            return pl.all_horizontal(exprs).alias(self.name)
        return pl.any_horizontal(exprs).alias(self.name)

    def to_category_expr(self) -> pl.Expr:
        """Constructs a Polars expression returning the combined category string or `None` if unmatched.

        Returns:
            String `pl.Expr` aliased to `self.name`.
        """
        if not self.criteria:
            return pl.lit(None, dtype=pl.Utf8).alias(self.name)
        if len(self.criteria) == 1:
            return self.criteria[0].to_extract_expr().alias(self.name)

        extract_exprs = [c.to_extract_expr() for c in self.criteria]

        if self.conjunction == "and":
            all_matched = pl.all_horizontal([e.is_not_null() for e in extract_exprs])
            combined = pl.concat_list(extract_exprs).list.join(self.sep)
            return pl.when(all_matched).then(combined).otherwise(None).alias(self.name)
        else:
            joined = pl.concat_list(extract_exprs).list.drop_nulls().list.join(self.sep)
            return pl.when(joined != "").then(joined).otherwise(None).alias(self.name)

    def evaluate(
        self,
        data: pl.DataFrame | Any,
        *,
        top_n: int = 25,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Evaluates this `Target` on a `Dataset` or `pl.DataFrame`, returning a category breakdown and KPI stats.

        Args:
            data: Input `Dataset` or `pl.DataFrame` to evaluate against.
            top_n: Maximum number of top categories to return in the breakdown DataFrame.

        Returns:
            Tuple `(breakdown_df, kpi_dict)` where `breakdown_df` has columns
            `["Category", "Isolates", "Prevalence (%)"]` and `kpi_dict` contains
            `total_n`, `matched_n`, `coverage_pct`, `n_categories`, and `is_compositional`.
        """
        df: pl.DataFrame = data if isinstance(data, pl.DataFrame) else data.data
        total_n = df.height
        if total_n == 0:
            empty_df = pl.DataFrame(
                {"Category": [], "Isolates": [], "Prevalence (%)": []},
                schema={"Category": pl.String, "Isolates": pl.Int64, "Prevalence (%)": pl.Float64},
            )
            return empty_df, {
                "total_n": 0,
                "matched_n": 0,
                "coverage_pct": 0.0,
                "n_categories": 0,
                "is_compositional": self.is_compositional,
            }

        hit_series = df.select(self.to_expr().alias("_hit"))["_hit"]
        matched_n = int(hit_series.sum() or 0)
        coverage_pct = round(100.0 * matched_n / total_n, 2)

        if self.is_compositional:
            cat_df = df.select(self.to_category_expr().alias("_cat")).filter(
                pl.col("_cat").is_not_null() & (pl.col("_cat") != "")
            )
            if len(self.criteria) == 1 and self.criteria[0].variable.is_compositional:
                tokens_df = cat_df.select(pl.col("_cat").alias("Category"))
            else:
                tokens_df = (
                    cat_df.select(pl.col("_cat").str.split(self.sep).alias("_tokens"))
                    .explode("_tokens")
                    .select(pl.col("_tokens").str.strip_chars().alias("Category"))
                    .filter(pl.col("Category").is_not_null() & (pl.col("Category") != ""))
                )

            counts_df = (
                tokens_df.group_by("Category")
                .agg(pl.len().cast(pl.Int64).alias("Isolates"))
                .sort(["Isolates", "Category"], descending=[True, False])
                .with_columns((pl.col("Isolates") * 100.0 / total_n).round(2).alias("Prevalence (%)"))
            )
            return counts_df.head(top_n), {
                "total_n": total_n,
                "matched_n": matched_n,
                "coverage_pct": coverage_pct,
                "n_categories": counts_df.height,
                "is_compositional": True,
            }

        unmatched_n = total_n - matched_n
        binary_df = pl.DataFrame(
            {
                "Category": [f"{self.title} (+)", f"{self.title} (-)"],
                "Isolates": [matched_n, unmatched_n],
                "Prevalence (%)": [
                    round(100.0 * matched_n / total_n, 2),
                    round(100.0 * unmatched_n / total_n, 2),
                ],
            }
        )
        return binary_df, {
            "total_n": total_n,
            "matched_n": matched_n,
            "coverage_pct": coverage_pct,
            "n_categories": 2,
            "is_compositional": False,
        }

    @staticmethod
    def summarize_repertoire(
        targets: Mapping[str, "Target"],
        data: pl.DataFrame | Any | None = None,
    ) -> pl.DataFrame:
        """Builds a summary Polars DataFrame across a mapping of registered `Target` objects.

        Args:
            targets: Mapping of registry keys to `Target` instances.
            data: Optional `Dataset` or `pl.DataFrame` used to compute live category counts and coverage.

        Returns:
            Polars `DataFrame` with columns `["Name", "Title", "Mode", "Conjunction", "Criteria", "Categories", "Coverage (%)"]`.
        """
        if not targets:
            return pl.DataFrame(
                {
                    "Name": [],
                    "Title": [],
                    "Mode": [],
                    "Conjunction": [],
                    "Criteria": [],
                    "Categories": [],
                    "Coverage (%)": [],
                },
                schema={
                    "Name": pl.String,
                    "Title": pl.String,
                    "Mode": pl.String,
                    "Conjunction": pl.String,
                    "Criteria": pl.String,
                    "Categories": pl.Int64,
                    "Coverage (%)": pl.Float64,
                },
            )

        df: pl.DataFrame | None = None if data is None else (data if isinstance(data, pl.DataFrame) else data.data)
        rows: list[dict[str, Any]] = []
        for key, tgt in targets.items():
            crit_summary = f" {tgt.conjunction.upper()} ".join(
                f"{c.column}" if c.pattern_str == ".*" else f"{c.column}~/{c.pattern_str}/" for c in tgt.criteria
            )
            n_cats = 0
            cov_pct = 0.0
            if df is not None and all(c.column in df.columns for c in tgt.criteria):
                _, stats = tgt.evaluate(df, top_n=5)
                n_cats = int(stats["n_categories"])
                cov_pct = float(stats["coverage_pct"])

            rows.append(
                {
                    "Name": key,
                    "Title": tgt.title,
                    "Mode": "Compositional" if tgt.is_compositional else "Binary",
                    "Conjunction": tgt.conjunction.upper(),
                    "Criteria": crit_summary,
                    "Categories": n_cats,
                    "Coverage (%)": cov_pct,
                }
            )
        return pl.DataFrame(rows)


# Functions ------------------------------------------------------------------------------------------------------------
_KEYWORD_MAP: dict[str, tuple[type[Variable], dict[str, Any]]] = {
    # Identifiers
    "id": (Identifier, {}),
    "sample": (Identifier, {}),
    "isolate": (Identifier, {}),
    "strain": (Identifier, {}),
    "accession": (Identifier, {}),
    "barcode": (Identifier, {}),
    "genome": (Identifier, {}),
    # Spatial
    "latitude": (Spatial, {"resolution": Spatial.Resolution.POINT}),
    "lat": (Spatial, {"resolution": Spatial.Resolution.POINT}),
    "longitude": (Spatial, {"resolution": Spatial.Resolution.POINT}),
    "long": (Spatial, {"resolution": Spatial.Resolution.POINT}),
    "lon": (Spatial, {"resolution": Spatial.Resolution.POINT}),
    "country": (Spatial, {"resolution": Spatial.Resolution.COUNTRY}),
    "nation": (Spatial, {"resolution": Spatial.Resolution.COUNTRY}),
    "region": (Spatial, {"resolution": Spatial.Resolution.REGION}),
    "continent": (Spatial, {"resolution": Spatial.Resolution.CONTINENT}),
    "city": (Spatial, {"resolution": Spatial.Resolution.CITY}),
    "site": (Spatial, {"resolution": Spatial.Resolution.SITE}),
    # Temporal
    "year": (Temporal, {"resolution": Temporal.Resolution.YEAR}),
    "date": (Temporal, {"resolution": Temporal.Resolution.UNKNOWN}),
    "time": (Temporal, {"resolution": Temporal.Resolution.UNKNOWN}),
    # Taxonomy
    "species": (Taxonomy, {"resolution": Taxonomy.Resolution.SPECIES}),
    "st": (Taxonomy, {"resolution": Taxonomy.Resolution.SEQUENCE_TYPE}),
    # Metrics & Model Outputs
    "event": (Count, {"is_denominator": False}),
    "events": (Count, {"is_denominator": False}),
    "n": (Count, {"is_denominator": True}),
    "denominator": (Count, {"is_denominator": True}),
    "denominators": (Count, {"is_denominator": True}),
    "estimate": (Estimate, {"lower": "lower", "upper": "upper", "bounded": True}),
    "prevalence": (Estimate, {"lower": "lower", "upper": "upper", "bounded": True}),
    "proportion": (Estimate, {"lower": "lower", "upper": "upper", "bounded": True}),
}

_FUZZY_CANDIDATES = [k for k in _KEYWORD_MAP if len(k) >= 5]
