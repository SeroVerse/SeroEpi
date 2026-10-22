---
title: API Reference
subtitle: Public Modules & Classes
description: Complete API reference for SeroEpi's surveillance dataset & variable system (seroepi.dataset, seroepi.variables), stratified aggregation & Bayesian estimators (seroepi.aggregate, seroepi.estimators), and interactive Altair plotting & PyShiny web app (seroepi.plotting, seroepi.app).
author: Tom Stanton
icon: lucide/book-open
comments: true
tags:
  - api
  - reference
---

# API Reference

The **SeroEpi** API is organized into three primary sections:

- **[`seroepi.dataset & variables`](dataset.md)** — Immutable isolate container ([`Dataset`][seroepi.dataset.Dataset]), persistent reference assets ([`LoadedData`][seroepi.models.LoadedData], [`CountryData`][seroepi.country.CountryData], [`KlebnetNeonatalSepsis`][seroepi.examples.KlebnetNeonatalSepsis]), semantic variable hierarchy ([`Variable`][seroepi.variables.Variable], [`Trait`][seroepi.variables.Trait], [`Spatial`][seroepi.variables.Spatial], [`Temporal`][seroepi.variables.Temporal], [`Target`][seroepi.variables.Target], [`Criterion`][seroepi.variables.Criterion]), sparse pairwise genomic distances ([`GenomicDistances`][seroepi.dist.GenomicDistances]), asynchronous API clients ([`PathogenwatchClient`][seroepi.client.pathogenwatch.PathogenwatchClient], [`WorldBankClient`][seroepi.client.world_bank.WorldBankClient]), and tabular/distance parsers (`seroepi.io`).
- **[`seroepi.aggregate & estimators`](estimators.md)** — Stratified count aggregation ([`Aggregator`][seroepi.aggregate.Aggregator]), estimator base contracts ([`BaseEstimator`][seroepi.estimators.base.BaseEstimator], [`StratifiedResultMixin`][seroepi.models.StratifiedResultMixin]), analytical and bootstrap estimators ([`UnpooledEstimator`][seroepi.estimators.unpooled.UnpooledEstimator], [`BootstrapEstimator`][seroepi.estimators.unpooled.BootstrapEstimator]), JAX/NumPyro hierarchical Bayesian partial-pooling models ([`HierarchicalMCMCEstimator`][seroepi.estimators.hierarchical.HierarchicalMCMCEstimator], [`HierarchicalSVIEstimator`][seroepi.estimators.hierarchical.HierarchicalSVIEstimator]), clonal outbreak deduplication ([`TransmissionEstimator`][seroepi.estimators.transmission.TransmissionEstimator]), and LOSO-validated multi-valent formulation optimization & differential prevalence testing ([`Formulation`][seroepi.estimators.formulation.Formulation], [`FormulationComparison`][seroepi.estimators.formulation.FormulationComparison]).
- **[`seroepi.plotting & app`](plotting.md)** — Theme-aware Altair visualizers ([`BasePlotter`][seroepi.plotting.base.BasePlotter], [`BarPlotter`][seroepi.plotting.bar.BarPlotter], [`ChoroplethPlotter`][seroepi.plotting.choropleth.ChoroplethPlotter], [`HeatmapPlotter`][seroepi.plotting.heatmap.HeatmapPlotter], [`PyramidPlotter`][seroepi.plotting.pyramid.PyramidPlotter], [`UpSetPlotter`][seroepi.plotting.upset.UpSetPlotter], [`FormulationCoveragePlotter`][seroepi.plotting.formulation.FormulationCoveragePlotter], [`FormulationStabilityPlotter`][seroepi.plotting.formulation.FormulationStabilityPlotter], [`DifferentialPrevalencePlotter`][seroepi.plotting.formulation.DifferentialPrevalencePlotter], [`RankShiftPlotter`][seroepi.plotting.formulation.RankShiftPlotter]) and the modular OOP PyShiny web application (`seroepi.app`).
