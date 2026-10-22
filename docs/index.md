---
title: SeroEpi
subtitle: Genomic Sero-Epidemiology & Vaccine Design
description: Python library for bacterial sero-epidemiology, stratified JAX/NumPyro Bayesian prevalence estimation, and LOSO-validated multi-valent vaccine formulation.
author: Tom Stanton
icon: lucide/chart-network
comments: true
tags:
  - epidemiology
  - bayesian
  - numpyro
  - vaccines
  - shiny
hide:
  - navigation
---

<div class="sv-hero" markdown>

# SeroEpi

**Precision target selection, hierarchical Bayesian prevalence modelling, and multi-valent formulation design from genomic surveillance.**

Built on `scikit-learn`, `Polars`, and `JAX`/`NumPyro`, **SeroEpi** transforms pathogen isolate collections and [`SeroGen`](https://seroverse.github.io/SeroGen/) / [`KBank`](https://seroverse.github.io/KBank/) trait calls into stratified prevalence estimates, spatial choropleths, and cross-validated multi-valent formulations.

<div class="sv-badges">
  <img class="off-glb" src="https://img.shields.io/badge/python-3.12%2B-8b5cf6?logo=python&logoColor=white" alt="Python 3.12+" />
  <img class="off-glb" src="https://img.shields.io/badge/inference-JAX%20%2B%20NumPyro-7c3aed" alt="JAX + NumPyro" />
  <img class="off-glb" src="https://img.shields.io/badge/api-scikit--learn%20%2B%20Polars-6366f1" alt="scikit-learn + Polars" />
  <img class="off-glb" src="https://img.shields.io/badge/app-PyShiny%20%2F%20Posit%20Cloud-db2777" alt="PyShiny Web App" />
  <img class="off-glb" src="https://img.shields.io/badge/license-GPLv3-6366f1" alt="GPLv3 License" />
</div>

<div class="sv-hero-actions" markdown>
  <a href="app/" class="sv-btn sv-btn--primary" markdown="span">:lucide-app-window: Open Embedded Web App</a>
  <a href="https://tomdstanton-seroepi.share.connect.posit.cloud" target="_blank" rel="noopener" class="sv-btn sv-btn--secondary" markdown="span">:lucide-external-link: Launch on Posit Connect Cloud ↗</a>
</div>

</div>

---

## Key Capabilities

<div class="grid cards" markdown>

-   :lucide-database: **Surveillance Cohort & Target Engineering (`seroepi.dataset`, `seroepi.variables`)**

    ---

    Ingests Pathogenwatch isolate collections (`PathogenwatchClient`) and UN/World Bank demographic indicators (`WorldBankClient`), enforces typed `Variable` contracts (`Spatial`, `Temporal`, `Trait`, `Target`), and composes multi-variable `Criterion` targets.

-   :lucide-layers: **Stratified Aggregation & Partial Pooling (`seroepi.aggregate`, `seroepi.estimators`)**

    ---

    Aggregates isolate counts across geographic, temporal, and clinical strata (`Aggregator`) and fits Unpooled (Wilson/Clopper-Pearson/Jeffreys), Bootstrap, or `JAX`/`NumPyro` Hierarchical Bayesian (`MCMC` NUTS and `SVI`) Binomial and Beta-Binomial models.

-   :lucide-shield-check: **LOSO-Validated Formulation Design (`seroepi.estimators.formulation`)**

    ---

    Ranks candidate targets under pooled or equal-stratum weights, controls Bayesian or Benjamini-Hochberg False Discovery Rates against burden thresholds, and evaluates held-out geographic coverage via Leave-One-Stratum-Out (`loso`) cross-validation.

-   :lucide-map: **Publication-Ready Visuals & Live PyShiny App (`seroepi.plotting`, `seroepi.app`)**

    ---

    Generates interactive Altair spatial choropleths (`ChoroplethPlotter`), cumulative coverage curves (`FormulationPlotter`), UpSet co-occurrence matrices (`UpSetPlotter`), and demographic pyramids (`PyramidPlotter`) inside a reactive PyShiny dashboard.

</div>

---

## End-to-End Epidemiological Workflow

```mermaid
flowchart LR
    PW["<b>Surveillance Cohort</b><br/><i>Pathogenwatch + World Bank</i>"]
    AGG["<b>Stratified Aggregator</b><br/><i>Target × Spatial / Temporal Strata</i>"]
    EST["<b>Prevalence Estimator</b><br/><i>Unpooled / NumPyro NUTS &amp; SVI</i>"]
    FRM["<b>Formulation Meta-Estimator</b><br/><i>Top-K Valency + LOSO Stability CV</i>"]
    PLT["<b>Interactive Altair &amp; Shiny</b><br/><i>Choropleth, UpSet &amp; Coverage</i>"]

    PW --> AGG --> EST --> FRM --> PLT
```

---

## Quickstart

```python
from seroepi.aggregate import Aggregator
from seroepi.estimators.formulation import FormulationEstimator
from seroepi.estimators.hierarchical import HierarchicalSVIEstimator
from seroepi.examples import KlebnetNeonatalSepsis
from seroepi.plotting.choropleth import ChoroplethPlotter

# 1. Load the KlebNET Neonatal Sepsis benchmark collection (981 K. pneumoniae isolates)
ds = KlebnetNeonatalSepsis.load_built()

# 2. Aggregate capsule K-locus counts across geographic strata
agg = Aggregator(target=ds.var("K_locus"), strata=[ds.var("country")]).fit(ds)

# 3. Fit a Hierarchical Bayesian SVI prevalence model and 10-valent vaccine formulation
est = HierarchicalSVIEstimator().fit(agg)
vaccine = FormulationEstimator(max_valency=10).fit(est)
chart = ChoroplethPlotter().plot(est)
```
