---
author: Tom Stanton
comments: true
tags: [adr, architecture, variables]
icon: lucide/file-check-2
---

# ADR 001: Strict Variable Identity, $O(1)$ Column Lookup, and Polymorphic Expressions

* **Status:** Accepted
* **Date:** 2026-10-14

---

## Context

Previously, `Variable` carried both `column: str` and `alias: str | None = None`, along with a computed `@property def name(self) -> str` returning `self.alias or self.column` and a `@property def title(self) -> str` performing ad-hoc `.replace("_", " ").title()` string manipulation.

This dual identity caused systemic complexity across `Dataset`, `Aggregator`, `BaseEstimator`, and `Plotter`:
1. **Schema Divergence:** `Variable.to_expr()` projected `pl.col(self.column).alias(self.name)`. Whenever `alias` was set for human-readable display (e.g., `Trait("K_locus", alias="Capsule")` and `Trait("K_type", alias="Capsule")` in `KleborateParser`, or `Trait("target", alias=target_var.name)` in `Aggregator`), `var.name` diverged from `var.column` or caused naming collisions.
2. **Defensive Lookup Clutter:** Because callers could not trust whether a variable was indexed by `column`, `alias`, or passed as a `Variable` vs `str`, both `DataModel` and `Aggregator` implemented `resolve_variable()` with linear fallback scans over `variables.values()`, and downstream classes maintained redundant `target_col` vs `target_name` and `strata_cols` vs `strata_names` properties.
3. **Leaked Domain Logic:** `Aggregator` contained `isinstance` cascades checking `(isinstance(v, Trait) and v.compositional) or isinstance(v, Taxonomy)`, checking `Spatial`/`Temporal` resolutions to choose between `to_expr()`, `truncate_expr()`, and `to_resolution_expr()`, and branching on `Target | Criterion | Variable`.

---

## Decision

1. **Single Physical Identity (`column`) + Explicit Display Label (`title`):**
   - Remove `alias` and `name` from `Variable` completely.
   - `Variable.column: str` is strictly the physical column name in the Polars `DataFrame`.
   - `Variable.title: str` is the human-readable display title used in plots and tables. Users may pass `title: str | None = None` at initialization; if omitted (`None` or `""`), `__post_init__` sets `title` to `self.column` verbatim (except `Estimate`, which defaults `title` to `"Prevalence"`). Zero automatic `.replace("_", " ").title()` string manipulation is performed on column names.

2. **Strict $O(1)$ Variable Lookup & Removal of `resolve_variable()`:**
   - `variables` mappings on `DataModel` (`Dataset`) and `StratifiedResultMixin` (`Aggregator`, `BaseEstimator`) are strictly `dict[str, Variable]` keyed by `var.column`.
   - `.var(*cols: str)` accepts only string column names and performs direct $O(1)$ dictionary lookup (`self.variables[col]`).
   - Delete `resolve_variable()` across the entire codebase. Downstream constructors (`Aggregator`, `Criterion`, `Target`) require `Variable` instances at initialization and use them directly.

3. **Polymorphism in `seroepi.variables`:**
   - Define `@property def is_compositional(self) -> bool` on base `Variable` (`False` by default, `True` on `Taxonomy`, `self.compositional` on `Trait`).
   - Make `Spatial.to_expr()` and `Temporal.to_expr()` polymorphically apply their configured `resolution` (`to_resolution_expr()` and `truncate_expr()` respectively when `resolution != UNKNOWN`), always aliasing to `self.column`.
   - Add `Target.coerce(spec: Target | Criterion | Variable, *, compositional: bool | None = None, negative_indicator: ... = None) -> Target` to normalize any target specification into a canonical `Target` instance.

---

## Consequences

* **Positive:** Eliminates `resolve_variable()`, `target_name`, `strata_names`, `adjust_for_name`, and all `isinstance` type-switching cascades in `Aggregator`.
* **Positive:** Guarantees the invariant `var.column in df.columns` at every stage of the pipeline.
* **Positive:** Prevents silent column collisions in `KleborateParser` where multiple columns share the same biological display title (e.g., `K_locus` and `K_type` both having `title="Capsule"` while keeping distinct `column="K_locus"` and `column="K_type"`).
