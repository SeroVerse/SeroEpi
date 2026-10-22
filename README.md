<div align="center">

# 🌐 SeroEpi

**Precision target selection, stratified JAX/NumPyro hierarchical Bayesian prevalence modelling, LOSO-validated multi-valent formulation design, and interactive genomic surveillance analytics.**

[![Docs](https://img.shields.io/github/actions/workflow/status/SeroVerse/SeroEpi/docs.yaml?branch=main&label=Docs&logo=readthedocs)](https://seroverse.github.io/SeroEpi/)
[![Web App](https://img.shields.io/badge/app-Posit%20Connect%20Cloud-db2777?logo=posit&logoColor=white)](https://seroverse-seroepi.share.connect.posit.cloud)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-8b5cf6?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-6366f1.svg)](https://www.gnu.org/licenses/gpl-3.0)

[**Documentation**](https://seroverse.github.io/SeroEpi/) · [**Web App**](https://seroverse-seroepi.share.connect.posit.cloud) · [**Architecture & ADRs**](https://seroverse.github.io/SeroEpi/glossary/) · [**API Reference**](https://seroverse.github.io/SeroEpi/reference/) · [**Issues**](https://github.com/SeroVerse/SeroEpi/issues)

</div>

---

## Overview

**SeroEpi** is the epidemiological modelling and vaccine formulation engine of the **SeroVerse** suite. Built on `scikit-learn` estimator contracts, `Polars` columnar dataframes, and `JAX`/`NumPyro` probabilistic inference, SeroEpi transforms pathogen isolate collections and genotype calls into stratified prevalence estimates, spatial choropleths, and cross-validated multi-valent formulations.

### Core Capabilities

- 🗂️ **Surveillance Cohort Ingestion & Compositional Target Engineering (`Dataset`, `Variable`, `Target`, `Criterion`)**: Ingests Pathogenwatch isolate collections (`PathogenwatchClient`) and World Bank / UN demographic indicators (`WorldBankClient`), enforces strict typed variable contracts (`Identifier`, `Spatial`, `Temporal`, `Trait`, `Continuous`, `Categorical`), and composes multi-variable boolean or intersectional `Criterion` targets (`&`, `|`, `~`, `@`).
- 🧬 **Stratified Aggregation & Outbreak De-Clonalization (`Aggregator`, `TransmissionEstimator`)**: Collapses isolate-level cohorts across arbitrary geographic, temporal, and clinical strata (`Aggregator`), clusters single-linkage SNP transmission complexes (`Dataset.cluster`), and fits exponential-decay genomic distance models (`TransmissionEstimator`) to mitigate local outbreak over-sampling bias.
- 🧠 **Hierarchical Bayesian & Bootstrap Prevalence Modelling (`HierarchicalMCMCEstimator`, `HierarchicalSVIEstimator`, `BootstrapEstimator`, `UnpooledEstimator`)**: Fits closed-form unpooled intervals (`wilson`, `clopper-pearson`, `jeffreys`, `wald`), Dirichlet-weighted Bayesian Bootstraps (`BootstrapEstimator`), and partially pooled `JAX`/`NumPyro` Hierarchical Binomial and Beta-Binomial models via No-U-Turn Sampler MCMC (`HierarchicalMCMCEstimator`) or Stochastic Variational Inference (`HierarchicalSVIEstimator`).
- 🛡️ **LOSO-Validated Multi-Valent Formulation & Differential Enrichment (`Formulation`, `FormulationComparison`)**: Ranks candidate targets under pooled or macro-averaged equal-stratum weights, evaluates minimum burden exceedance ($H_{0,j}: p_j \le \tau_{\min}$) with Bayesian posterior error probabilities or Benjamini-Hochberg FDR control, quantifies geographic generalizability via Leave-One-Stratum-Out (`cv="loso"`) cross-validation, and contrasts formulations across cohorts (`FormulationComparison.differential_enrichment()`).
- 🗺️ **Publication-Ready Altair Visuals & PyShiny Dashboard (`seroepi.plotting`, `seroepi.app`)**: Generates interactive spatial choropleths (`ChoroplethPlotter`), cumulative coverage curves, LOSO rank-stability heatmaps, and differential enrichment volcano plots (`FormulationPlotter`), target co-occurrence UpSet matrices (`UpSetPlotter`), and demographic pyramids (`PyramidPlotter`) inside a reactive PyShiny web dashboard (`seroepi-app`).

---

## Installation

```bash
uv add seroepi
# or with pip:
pip install seroepi
```

To include the interactive PyShiny web application (`seroepi-app`):

```bash
uv add "seroepi[app]"
# or with pip:
pip install "seroepi[app]"
```

---

## Quickstart

```python
from seroepi.aggregate import Aggregator
from seroepi.estimators.formulation import Formulation
from seroepi.estimators.hierarchical import HierarchicalSVIEstimator
from seroepi.examples import KlebnetNeonatalSepsis
from seroepi.plotting.choropleth import ChoroplethPlotter
from seroepi.plotting.formulation import FormulationPlotter

# 1. Load (or auto-build) the KlebNET Neonatal Sepsis benchmark cohort (981 K. pneumoniae isolates)
ds = KlebnetNeonatalSepsis.ensure()

# 2. Aggregate capsule K-locus counts across country strata
agg = Aggregator(target=ds.var("K_locus"), strata=[ds.var("country")]).fit(ds)

# 3. Fit a partially pooled Hierarchical Bayesian SVI Beta-Binomial prevalence model
est = HierarchicalSVIEstimator(likelihood="beta_binomial", num_steps=1500, random_state=42).fit(agg)

# 4. Optimize a 10-valent formulation with Leave-One-Stratum-Out (LOSO) cross-validation
form = Formulation(estimator=est, valency=10, weighting="equal", cv="loso", min_prevalence=0.02).fit(agg)
print(f"Selected 10-valent targets: {form.selected_targets_}")

# 5. Render interactive Altair spatial choropleth and cumulative coverage charts
choropleth = ChoroplethPlotter().plot(est)
coverage_curve = FormulationPlotter(plot_type="coverage").plot(form)
```

Launch the interactive PyShiny surveillance dashboard locally:

```bash
seroepi-app --host 127.0.0.1 --port 8000
```

---

## Development

```bash
# Install all development, test, lint, typing, docs, and app extras
uv sync --all-groups --all-extras

# Run linter and static type checker
uv run ruff check .
uv run ty check src/

# Run test suite
uv run pytest

# Launch the interactive PyShiny app in development mode
uv run seroepi-app --reload

# Build and serve documentation locally with Zensical
uv run zensical serve
```

---

## License

Released under the [GNU General Public License v3.0 (GPLv3)](LICENSE).
