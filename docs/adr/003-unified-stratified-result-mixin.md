---
title: "ADR 003: Stratified Result Mixin"
subtitle: Unified Result Contract & Static Typing
description: "Architecture Decision Record 003: Unified StratifiedResultMixin and Elimination of Dynamic getattr."
author: Tom Stanton
status: new
comments: true
tags: [adr, architecture, DRY, estimators]
icon: lucide/layers
---

# ADR 003: Unified `StratifiedResultMixin` and Elimination of Dynamic `getattr`

* **Status:** Accepted
* **Date:** 2026-10-14

---

## Context

Previously, `Aggregator` and `BaseEstimator` duplicated nearly 150 lines of metadata properties and data-slicing methods:
- Both defined `target_col`, `target_name`, `target_title`, `strata_cols`, `strata_names`, `strata_titles`, `n_strata`, `is_stratified`, `is_single_stratum`, `single_stratum_col`, `single_stratum_title`, `infer_single_stratum_value`, `filter()`, and `split()`.
- Despite `self.strata_` and `self.target_` being known domain objects, `Aggregator` used defensive `getattr(s, "title", s.name.replace("_", " ").title())` calls, while `BaseEstimator` downgraded `strata_` and `target_` to raw strings (`list[str]` and `str | None`) and re-implemented string formatting from scratch.

---

## Decision

1. **Extract `StratifiedResultMixin` in `seroepi.models`:**
   - Inherited by both `Aggregator` and `BaseEstimator`.
   - Standardizes fitted metadata attributes across both classes:
     - `is_fitted_: bool = False`
     - `variables_: dict[str, Variable]`
     - `target_: Variable`
     - `strata_: tuple[Variable, ...]`
     - `adjust_for_: Variable | None`
     - `is_compositional_: bool`
     - `is_padded_: bool`
   - Defines an abstract/overridable `_frame` property (`self.data_` on `Aggregator`, `self.estimates_` on `BaseEstimator`) so `.filter()`, `.split()`, and `.infer_single_stratum_value()` are implemented **once**.

2. **Direct Typed Property Access (Zero `getattr`):**
   - `target_col -> str`: `return self.target_.column`
   - `target_title -> str`: `return self.target_.title`
   - `strata_cols -> tuple[str, ...]`: `return tuple(s.column for s in self.strata_)`
   - `strata_titles -> tuple[str, ...]`: `return tuple(s.title for s in self.strata_)`
   - `adjust_for_col -> str | None`: `return self.adjust_for_.column if self.adjust_for_ is not None else None`
   - `adjust_for_title -> str | None`: `return self.adjust_for_.title if self.adjust_for_ is not None else None`
   - Delete `target_name`, `strata_names`, and `adjust_for_name`.

---

## Consequences

* **Positive:** Shrinks `src/seroepi/aggregate.py` and `src/seroepi/estimators/base.py` significantly by consolidating all shared metadata, `.var()`, `.filter()`, and `.split()` logic in one place.
* **Positive:** Eliminates all `getattr()` duck-typing and `.replace("_", " ").title()` string manipulation outside `variables.py`.
* **Positive:** Strict static type checking (`ty check`) is enforced across `target_`, `strata_`, and `adjust_for_` without `typing.cast()`.
