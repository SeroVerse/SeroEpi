"""Kleborate parsing."""

from typing import ClassVar

import polars as pl

from seroepi.io.dataset.base import DatasetParser, PathogenWatchFile
from seroepi.variables import (
    QC,
    Count,
    Resistance,
    Taxonomy,
    Trait,
    Variable,
    Virulence,
)


class KleborateParser(PathogenWatchFile, DatasetParser):
    """Parser for Kleborate genotype outputs mapping raw columns to canonical SeroEpi ``Variable`` descriptors.

    Attributes:
        download_type: Pathogenwatch analysis download identifier (``"kleborate"``).
        variables: Canonical tuple of ``Identifier``, ``Taxonomy``, ``QC``, ``Count``, ``Trait``,
            ``Virulence``, and ``Resistance`` variables for Kleborate v2/v3 tables.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dataset.kleborate import KleborateParser
        >>> df = pl.DataFrame({"Genome Name": ["iso1"], "species": ["Klebsiella pneumoniae"], "K_locus": ["KL2"]})
        >>> ds = KleborateParser.parse(df, dataset_name="kleb")
        >>> ds.var("K_locus").title
        'Capsule'
    """

    download_type: ClassVar[str | None] = "kleborate"
    variables: ClassVar[tuple[Variable, ...]] = (
        # Tool metadata columns (Genome Name + Genome ID provided by PathogenWatchFile)
        Variable("Kleborate version"),
        Variable("Wrapper version"),
        # Taxonomy & QC
        Taxonomy("species", resolution=Taxonomy.Resolution.SPECIES, title="Species"),
        QC.equals("species_match", "strong", title="Species Match"),
        QC.max("contig_count", 500, title="Contig Count"),
        QC.min("N50", 10000),
        Count("largest_contig", title="Largest Contig"),
        Count("total_size", title="Total Size"),
        Variable("ambiguous_bases", dtype=pl.String(), title="Ambiguous Bases"),
        Variable("QC_warnings", dtype=pl.String(), title="QC Warnings"),
        Taxonomy("ST", title="MLST", resolution=Taxonomy.Resolution.SEQUENCE_TYPE),
        # Scores
        Count("virulence_score", title="Virulence Score"),
        Count("resistance_score", title="Resistance Score"),
        Count("num_resistance_classes", title="Number of Resistance Classes"),
        Count("num_resistance_genes", title="Number of Resistance Genes"),
        Count("resistance_class_count", title="Resistance Class Count"),
        Count("resistance_gene_count", title="Resistance Gene Count"),
        # Loci & Serotype
        Trait("K_locus", title="Capsule"),
        Trait("K_type", title="Capsule", phenotype=True),
        QC.equals("K_locus_confidence", "Typeable"),
        Trait("O_locus", title="LPS"),
        Trait("O_type", title="LPS", phenotype=True),
        QC.equals("O_locus_confidence", "Typeable"),
        # Virulence loci
        Virulence(
            "YbST",
            title="Yersiniabactin",
            factor=Virulence.Factor.SIDEROPHORE,
        ),
        Virulence(
            "Yersiniabactin",
            title="Yersiniabactin",
            factor=Virulence.Factor.SIDEROPHORE,
            phenotype=True,
        ),
        Virulence("CbST", title="Colibactin", factor=Virulence.Factor.EXOTOXIN, compositional=True),
        Virulence(
            "Colibactin",
            factor=Virulence.Factor.EXOTOXIN,
            phenotype=True,
        ),
        Virulence("AbST", title="Aerobactin", factor=Virulence.Factor.SIDEROPHORE, compositional=True),
        Virulence(
            "Aerobactin",
            factor=Virulence.Factor.SIDEROPHORE,
            phenotype=True,
        ),
        Virulence("SmST", title="Salmochelin", factor=Virulence.Factor.SIDEROPHORE, compositional=True),
        Virulence(
            "Salmochelin",
            title="Salmochelin",
            factor=Virulence.Factor.SIDEROPHORE,
            phenotype=True,
        ),
        Virulence("RmST", title="RmpADC", factor=Virulence.Factor.CAPSULE, compositional=True),
        Virulence(
            "RmpADC",
            title="RmpADC",
            factor=Virulence.Factor.CAPSULE,
            phenotype=True,
        ),
        Virulence(
            "rmpA2",
            title="RmpA2",
            factor=Virulence.Factor.CAPSULE,
            requires_bgc=False,
        ),
        # Acquired Resistance & Mutations
        Resistance(
            "AGly_acquired",
            title="Aminoglycoside",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Col_acquired",
            title="Colistin",
            resolution=Resistance.Resolution.DRUG_NAME,
        ),
        Resistance(
            "Fcyn_acquired",
            title="Fosfomycin",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Flq_acquired",
            title="Fluoroquinolone",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Gly_acquired",
            title="Glycopeptide",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "MLS_acquired",
            title="MLS",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Phe_acquired",
            title="Phenicol",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Rif_acquired",
            title="Rifampicin",
            resolution=Resistance.Resolution.DRUG_NAME,
        ),
        Resistance(
            "Sul_acquired",
            title="Sulfonamide",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Tet_acquired",
            title="Tetracycline",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Tgc_acquired",
            title="Tigecycline",
            resolution=Resistance.Resolution.DRUG_NAME,
        ),
        Resistance(
            "Tmt_acquired",
            title="Trimethoprim",
            resolution=Resistance.Resolution.DRUG_NAME,
        ),
        Resistance(
            "Bla_acquired",
            title="Beta-lactam",
            resolution=Resistance.Resolution.SUPERFAMILY,
        ),
        Resistance(
            "Bla_inhR_acquired",
            title="Beta-lactam + Inhibitor",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Bla_ESBL_acquired",
            title="Cephalosporin (3rd Gen)",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Bla_ESBL_inhR_acquired",
            title="Cephalosporin (3rd Gen) + Inhibitor",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Bla_Carb_acquired",
            title="Carbapenem",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Bla_chr",
            title="Beta-lactam",
            resolution=Resistance.Resolution.SUPERFAMILY,
        ),
        Resistance(
            "SHV_mutations",
            title="Beta-lactam",
            resolution=Resistance.Resolution.SUPERFAMILY,
        ),
        Resistance(
            "Omp_mutations",
            title="Carbapenem",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Col_mutations",
            title="Colistin",
            resolution=Resistance.Resolution.DRUG_NAME,
        ),
        Resistance(
            "Flq_mutations",
            title="Fluoroquinolone",
            resolution=Resistance.Resolution.CLASS,
        ),
        Resistance(
            "Ciprofloxacin_prediction",
            title="Ciprofloxacin",
            resolution=Resistance.Resolution.DRUG_NAME,
            phenotype=True,
            sensitive_pattern=r"\bS\b",
            intermediate_pattern=r"\bI\b",
            resistant_pattern=r"\bR\b",
        ),
        Resistance(
            "Ciprofloxacin_profile_support",
            title="Ciprofloxacin",
            resolution=Resistance.Resolution.DRUG_NAME,
            sensitive_pattern=r"\bS\b",
            intermediate_pattern=r"\bI(?:/R)?\b",
            resistant_pattern=r"(?:^|\s)R\b",
        ),
        Resistance(
            "Ciprofloxacin_profile",
            title="Ciprofloxacin",
            resolution=Resistance.Resolution.DRUG_NAME,
            sensitive_pattern=r"^0\^? QRDR, 0\^? PMQR\b",
            intermediate_pattern=r"^0\^? QRDR, qnrB\d+, 0\^? aac",
        ),
        Resistance(
            "Ciprofloxacin_MIC_prediction",
            title="Ciprofloxacin",
            resolution=Resistance.Resolution.DRUG_NAME,
            phenotype=True,
            sensitive_pattern=r"^(?:<=?\s*)?0\.(?:0\d+|1\d*|25)\b",
            intermediate_pattern=r"^0\.5\b",
            resistant_pattern=r"^(?:>=?\s*)?[1-9]\d*(?:\.\d+)?\b",
        ),
    )
