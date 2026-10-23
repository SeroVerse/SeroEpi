"""Kaptive and Pathogenwatch Kaptive genotype parsing."""

from typing import ClassVar

import polars as pl

from seroepi.io.dataset.base import DatasetParser, PathogenWatchFile
from seroepi.variables import (
    QC,
    Identifier,
    Trait,
    Variable,
)


class KaptiveParser(DatasetParser):
    """Parser for standalone Kaptive CLI genotype tables mapping columns to SeroEpi ``Variable`` descriptors.

    Attributes:
        variables: Canonical tuple of ``Identifier``, ``Trait``, ``QC``, and ``Variable`` definitions
            for standalone Kaptive v3 tabular outputs.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.kaptive import KaptiveParser
        >>> df = pl.DataFrame(
        ...     {
        ...         "Assembly": ["iso1"],
        ...         "Best match locus": ["KL62"],
        ...         "Best match type": ["K62"],
        ...         "Match confidence": ["Typeable"],
        ...     }
        ... )
        >>> ds = KaptiveParser.parse(df, dataset_name="kaptive_cli")
        >>> ds.var("Best match locus").title
        'Locus'
    """

    variables: ClassVar[tuple[Variable, ...]] = (
        Identifier("Assembly"),
        Trait("Best match locus", title="Locus"),
        Trait("Best match type", title="Locus", phenotype=True),
        QC.equals("Match confidence", "Typeable", title="Match Confidence"),
        Variable("Problems", dtype=pl.String(), title="Problems"),
        Variable("Coverage", dtype=pl.String(), title="Coverage"),
        Variable("Identity", dtype=pl.String(), title="Identity"),
        Variable("Length discrepancy", dtype=pl.String(), title="Length Discrepancy"),
        Variable("Database name", dtype=pl.String(), title="Database Name"),
        Variable("Database version", dtype=pl.String(), title="Database Version"),
        Variable("Expected genes in locus", dtype=pl.String(), title="Expected Genes in Locus"),
        Variable("Expected genes in locus, details", dtype=pl.String(), title="Expected Genes in Locus (Details)"),
        Variable("Expected genes outside locus", dtype=pl.String(), title="Expected Genes Outside Locus"),
        Variable(
            "Expected genes outside locus, details",
            dtype=pl.String(),
            title="Expected Genes Outside Locus (Details)",
        ),
        Variable("Missing expected genes", dtype=pl.String(), title="Missing Expected Genes"),
        Variable("Other genes in locus", dtype=pl.String(), title="Other Genes in Locus"),
        Variable("Other genes in locus, details", dtype=pl.String(), title="Other Genes in Locus (Details)"),
        Variable("Other genes outside locus", dtype=pl.String(), title="Other Genes Outside Locus"),
        Variable("Other genes outside locus, details", dtype=pl.String(), title="Other Genes Outside Locus (Details)"),
        Variable("Truncated genes, details", dtype=pl.String(), title="Truncated Genes (Details)"),
        Variable("Extra genes, details", dtype=pl.String(), title="Extra Genes (Details)"),
    )


class PathogenWatchKaptiveParser(PathogenWatchFile, KaptiveParser):
    """Parser for Pathogenwatch Kaptive exports mapping prefixed K, O, and OC locus columns to ``Variable`` descriptors.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"kaptive"``).
        variables: Canonical tuple of ``Identifier``, ``Trait``, ``QC``, and ``Variable`` definitions
            for Pathogenwatch Kaptive CSV exports across ``K locus`` (Capsule), ``O locus`` (LPS),
            and ``OC locus`` (Outer Core).

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.kaptive import PathogenWatchKaptiveParser
        >>> df = pl.DataFrame(
        ...     {
        ...         "Genome ID": ["g1"],
        ...         "Genome Name": ["SAMN15355684"],
        ...         "Kaptive Version": ["3.2.1"],
        ...         "K locus: Best match locus": ["KL62"],
        ...         "K locus: Best match type": ["K62"],
        ...         "K locus: Match confidence": ["Typeable"],
        ...         "O locus: Best match locus": ["O1/O2v1"],
        ...         "O locus: Best match type": ["O1"],
        ...         "O locus: Match confidence": ["Typeable"],
        ...     }
        ... )
        >>> ds = PathogenWatchKaptiveParser.parse(df, dataset_name="pw_kaptive")
        >>> ds.var("K locus: Best match locus").title, ds.var("O locus: Best match locus").title
        ('Capsule', 'LPS')
    """

    download_type: ClassVar[str | None] = "kaptive"
    variables: ClassVar[tuple[Variable, ...]] = (
        Variable("Kaptive Version", dtype=pl.String(), title="Kaptive Version"),
        *PathogenWatchFile.prefix_variables(KaptiveParser.variables, "K locus", trait_title="Capsule"),
        *PathogenWatchFile.prefix_variables(KaptiveParser.variables, "O locus", trait_title="LPS"),
        *PathogenWatchFile.prefix_variables(KaptiveParser.variables, "OC locus", trait_title="Outer Core"),
    )
