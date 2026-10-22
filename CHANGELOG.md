# Release Notes

All notable changes to **`SeroEpi`** are documented on this page. This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) specification.

---

## [Unreleased]

### Added

- **Surveillance Cohort & Semantic Variable Engine (`seroepi.dataset`, `seroepi.variables`, `seroepi.dist`)**:
  - Immutable Polars-backed `Dataset` container with $O(1)$ identity-mapped `Variable` lookup (`Trait`, `Spatial`, `Temporal`, `Target`, and composable multi-variable `Criterion` expressions).
  - Sparse CSR pairwise SNP distance matrices (`GenomicDistances`) and outbreak-aware clonal cluster deduplication (`TransmissionEstimator`).
  - Asynchronous `PathogenwatchClient` and UN/World Bank demographic `WorldBankClient` with bundled `KlebnetNeonatalSepsis` and `CountryData` assets.
- **Stratified Aggregation & Bayesian Prevalence Estimators (`seroepi.aggregate`, `seroepi.estimators`)**:
  - `Aggregator` sufficiency-reduction transformer compatible with `scikit-learn` pipelines.
  - Analytical/frequentist `UnpooledEstimator` (Wilson, Clopper-Pearson, Jeffreys, Dirichlet) and `BootstrapEstimator`.
  - `JAX`/`NumPyro` hierarchical Bayesian partial-pooling models (`HierarchicalMCMCEstimator` NUTS and `HierarchicalSVIEstimator` SVI) for Binomial and Beta-Binomial / Dirichlet-Multinomial likelihoods.
- **Multi-Valent Formulation Optimization (`seroepi.estimators.formulation`)**:
  - `Formulation` and `FormulationComparison` meta-estimators supporting pooled or equal-stratum weights, Leave-One-Stratum-Out (`LOSO`) cross-validation, and permutation/Bayesian False Discovery Rate (`BH`/`BY`) differential enrichment testing.
- **Publication-Grade Altair Plotting & Live PyShiny App (`seroepi.plotting`, `seroepi.app`)**:
  - Theme-aware Altair plotters (`ChoroplethPlotter`, `BarPlotter`, `HeatmapPlotter`, `PyramidPlotter`, `UpSetPlotter`, `FormulationCoveragePlotter`, `FormulationStabilityPlotter`, `DifferentialPrevalencePlotter`, `RankShiftPlotter`).
  - Modular OOP PyShiny web application (`HomeTab`, `DatasetTab`, `EstimationTab`, `FormulationTab`) deployed on Posit Connect Cloud and embedded in the documentation site.
