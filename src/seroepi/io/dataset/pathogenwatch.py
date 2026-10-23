"""Pathogenwatch core, MLST, LIN code, and PlasmidFinder analysis parsers."""

from typing import ClassVar

import polars as pl

from seroepi.io.dataset.base import DatasetParser, PathogenWatchFile
from seroepi.variables import (
    QC,
    Count,
    Identifier,
    Taxonomy,
    Trait,
    Variable,
)


class PathogenWatchCoreParser(PathogenWatchFile, DatasetParser):
    """Parser for Pathogenwatch Core Summary (`core`) exports.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"core"``).
        variables: Canonical tuple of ``Identifier``, ``Taxonomy``, ``Count``, ``QC``, and ``Variable`` definitions.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.pathogenwatch import PathogenWatchCoreParser
        >>> df = pl.DataFrame(
        ...     {
        ...         "Genome ID": ["g1"],
        ...         "Genome Name": ["SAMEA6155867"],
        ...         "Species Identifier": ["573"],
        ...         "% Kernel Matched": [99.8],
        ...     }
        ... )
        >>> ds = PathogenWatchCoreParser.parse(df, dataset_name="pw_core")
        >>> ds.var("Species Identifier").title
        'Species TaxID'
    """

    download_type: ClassVar[str | None] = "core"
    variables: ClassVar[tuple[Variable, ...]] = (
        Variable("Version", dtype=pl.String(), title="Core Version"),
        Taxonomy("Species Identifier", resolution=Taxonomy.Resolution.SPECIES, title="Species TaxID"),
        Count("No. Matched Families", title="Matched Core Families"),
        Count("No. Complete Alleles", title="Complete Core Alleles"),
        Count("Kernel Size (nts)", title="Kernel Size (bp)"),
        QC.min("% Kernel Matched", 90.0, title="Kernel Matched (%)"),
        Taxonomy("Pathogenwatch Reference", resolution=Taxonomy.Resolution.GENOMIC_CLONE, title="Reference Genome"),
    )


class MLSTParser(DatasetParser):
    """Parser for standalone MLST CLI (`mlst`) tabular outputs.

    Attributes:
        variables: Canonical tuple of ``Identifier``, ``Variable``, and ``Taxonomy`` definitions for MLST tables.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.pathogenwatch import MLSTParser
        >>> df = pl.DataFrame({"FILE": ["iso1.fasta"], "SCHEME": ["klebsiella"], "ST": ["15"]})
        >>> ds = MLSTParser.parse(df, dataset_name="mlst_cli")
        >>> ds.var("ST").title
        'MLST'
    """

    variables: ClassVar[tuple[Variable, ...]] = (
        Identifier("FILE"),
        Variable("SCHEME", dtype=pl.String(), title="MLST Scheme"),
        Taxonomy("ST", resolution=Taxonomy.Resolution.SEQUENCE_TYPE, title="MLST"),
    )


class PathogenWatchMLSTParser(PathogenWatchFile, MLSTParser):
    """Parser for Pathogenwatch MLST (`mlst`) exports.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"mlst"``).
        variables: Canonical tuple of ``Identifier``, ``Variable``, and ``Taxonomy`` definitions for Pathogenwatch MLST.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.pathogenwatch import PathogenWatchMLSTParser
        >>> df = pl.DataFrame({"Genome ID": ["g1"], "Genome Name": ["SAMEA6155867"], "Version": ["mlst-1"], "ST": ["15"]})
        >>> ds = PathogenWatchMLSTParser.parse(df, dataset_name="pw_mlst")
        >>> ds.var("ST").title
        'MLST'
    """

    download_type: ClassVar[str | None] = "mlst"
    variables: ClassVar[tuple[Variable, ...]] = (
        Variable("Version", dtype=pl.String(), title="MLST Scheme Version"),
        Taxonomy("ST", resolution=Taxonomy.Resolution.SEQUENCE_TYPE, title="MLST"),
    )


class PathogenWatchLINCodeParser(PathogenWatchFile, DatasetParser):
    """Parser for Pathogenwatch Klebsiella LIN code and cgMLST (`klebsiella-lincodes`) exports.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"klebsiella-lincodes"``).
        variables: Canonical tuple of ``Identifier``, ``Taxonomy``, ``QC``, ``Count``, and ``Variable`` definitions.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.pathogenwatch import PathogenWatchLINCodeParser
        >>> df = pl.DataFrame(
        ...     {
        ...         "Genome ID": ["g1"],
        ...         "Genome Name": ["SAMEA6155867"],
        ...         "cgST": ["18558"],
        ...         "Sublineage": ["15"],
        ...         "Clonal Group": ["15"],
        ...     }
        ... )
        >>> ds = PathogenWatchLINCodeParser.parse(df, dataset_name="pw_lincodes")
        >>> ds.var("Clonal Group").title
        'Clonal Group'
    """

    download_type: ClassVar[str | None] = "klebsiella-lincodes"
    variables: ClassVar[tuple[Variable, ...]] = (
        Taxonomy("cgST", resolution=Taxonomy.Resolution.GENOMIC_CLONE, title="cgST"),
        Variable("Closest cgST", dtype=pl.String(), title="Closest cgST"),
        Taxonomy("LIN code", resolution=Taxonomy.Resolution.GENOMIC_CLONE, title="LIN Code"),
        Taxonomy("Sublineage", resolution=Taxonomy.Resolution.SUB_SPECIES, title="Sublineage"),
        Taxonomy("Clonal Group", resolution=Taxonomy.Resolution.CLONAL_GROUP, title="Clonal Group"),
        QC.min("Identity", 95.0, title="cgMLST Identity (%)"),
        Count("Identical", title="Identical Loci"),
        Count("Compared Loci Count", title="Compared Loci"),
        Variable("Closest profile(s)", dtype=pl.String(), title="Closest Profiles"),
    )


class PathogenWatchPlasmidFinderParser(PathogenWatchFile, DatasetParser):
    """Parser for Pathogenwatch PlasmidFinder (`plasmidfinder`) exports.

    Automatically collapses multi-row plasmid replicon hits per isolate into one row per
    ``Genome Name`` with semicolon-delimited ``Inc Match`` and ``Group`` traits plus ``Plasmid Count``.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"plasmidfinder"``).
        variables: Canonical tuple of ``Identifier``, ``Trait``, ``Count``, and ``Variable`` definitions.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.pathogenwatch import PathogenWatchPlasmidFinderParser
        >>> df = pl.DataFrame(
        ...     {
        ...         "Genome ID": ["g1", "g1"],
        ...         "Genome Name": ["SAMEA6155867", "SAMEA6155867"],
        ...         "Group": ["IncF", "Col"],
        ...         "Inc Match": ["IncFIB(pKPHS1)", "ColRNAI"],
        ...     }
        ... )
        >>> ds = PathogenWatchPlasmidFinderParser.parse(df, dataset_name="pw_plasmids")
        >>> len(ds), ds.data["Plasmid Count"].to_list()
        (1, [2])
    """

    download_type: ClassVar[str | None] = "plasmidfinder"
    variables: ClassVar[tuple[Variable, ...]] = (
        Trait("Inc Match", title="Plasmid Replicon", compositional=False),
        Trait("Group", title="Plasmid Group", compositional=False),
        Count("Plasmid Count", title="Plasmid Replicon Count"),
    )

    @classmethod
    def _preprocess_dataframe(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Collapses per-hit PlasmidFinder rows into a single isolate-level record per ``Genome Name``."""
        if "Genome Name" not in df.columns or "Inc Match" not in df.columns:
            return df
        group_keys = [c for c in ("Genome ID", "Genome Name") if c in df.columns]
        aggs: list[pl.Expr] = [
            pl.col("Inc Match")
            .cast(pl.Utf8)
            .str.strip_chars()
            .filter(pl.col("Inc Match").is_not_null() & (pl.col("Inc Match").cast(pl.Utf8).str.strip_chars() != ""))
            .unique(maintain_order=True)
            .str.join(";")
            .alias("Inc Match"),
        ]
        if "Group" in df.columns:
            aggs.append(
                pl.col("Group")
                .cast(pl.Utf8)
                .str.strip_chars()
                .filter(pl.col("Group").is_not_null() & (pl.col("Group").cast(pl.Utf8).str.strip_chars() != ""))
                .unique(maintain_order=True)
                .str.join(";")
                .alias("Group")
            )
        aggs.append(
            pl.col("Inc Match")
            .filter(pl.col("Inc Match").is_not_null() & (pl.col("Inc Match").cast(pl.Utf8).str.strip_chars() != ""))
            .len()
            .cast(pl.Int64)
            .alias("Plasmid Count")
        )
        return df.group_by(group_keys, maintain_order=True).agg(aggs)
