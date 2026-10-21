---
author: Tom Stanton
comments: true
tags: [architecture, domain-model, glossary]
icon: lucide/book-a
---

# Domain Glossary

This glossary defines the ubiquitous language, core domain types, and invariants of the `seroepi` architecture.

---

## 1. Core Variable Hierarchy (`seroepi.variables`)

| Term | Class | Invariants & Responsibilities |
| :--- | :--- | :--- |
| **Variable** | `Variable` | Immutable descriptor binding a physical Polars column (`column: str`) to semantic metadata (`title: str`, `dtype: pl.DataType`, `negative_indicator`). Always satisfies `var.column in df.columns`. When `title` is not explicitly provided, `var.title` defaults verbatim to `var.column` (zero implicit string manipulation). |
| **Identifier** | `Identifier(Variable)` | Primary sample key. Every `Dataset` must contain **strictly one** `Identifier` variable. |
| **Trait** | `Trait(Variable)` | Biological or clinical attribute (e.g., capsule `K_locus`, lipopolysaccharide `O_type`). Carries `compositional: bool = True` and `phenotype: bool = False`. |
| **Taxonomy** | `Taxonomy(Variable)` | Hierarchical taxonomic classification (`SPECIES` $\to$ `TRANSMISSION_CLONE`). Always compositional (`is_compositional == True`). |
| **Spatial** | `Spatial(Variable)` | Geographic variable with `Resolution` (`GLOBAL` $\to$ `POINT`). Its polymorphic `to_expr()` automatically projects raw strings to the configured `resolution` (e.g., ISO3 country codes, UN M49 subregions, or macro-continents) aliased to `self.column`. |
| **Temporal** | `Temporal(Variable)` | Epidemiological timestamp or date with `Resolution` (`YEAR` $\to$ `HOUR`). Its polymorphic `to_expr()` truncates or formats timestamps to the configured `resolution` aliased to `self.column`. |
| **Quality Control (QC)** | `QC(Variable)` | Declarative sample-level acceptance filter (`min`, `max`, `between`, `equals`, `one_of`). Evaluates to a boolean mask via `filter_expr()`. Non-compositional (`is_compositional == False`). |
| **Metric** | `Metric(Variable)` | Quantitative statistical output (`Count`, `Estimate`). |
| **Count** | `Count(Metric)` | Discrete integer frequency (`event` numerator or `n` denominator). |
| **Estimate** | `Estimate(Metric)` | Point estimate column (`column="estimate"`) paired with uncertainty interval columns (`lower`, `upper`), confidence `level`, `ci_prefix`, and `format`. Defaults `title` to `"Prevalence"`. |

---

## 2. Composite Targets (`seroepi.variables`)

| Term | Class | Invariants & Responsibilities |
| :--- | :--- | :--- |
| **Criterion** | `Criterion` | Pairs a single `Variable` with a validated regular expression `pattern` (`re.Pattern[str] \| str = ".*"`) and optional `equivalence` mapping. Extracts regex matches via `to_extract_expr()` (compositional) or boolean flags via `to_expr()` (binary). |
| **Target** | `Target` | Named collection of `Criterion` objects combined via `conjunction` (`"or"` or `"and"`) and joined by `sep` (`" "`). Strictly forbids mixing compositional and non-compositional criteria unless `compositional=False` is explicitly set. `Target.coerce()` normalizes any `Variable`, `Criterion`, or `Target` into a canonical `Target`. |

---

## 3. Data & Pipeline Containers (`seroepi.models`, `seroepi.dataset`, `seroepi.aggregate`, `seroepi.estimators`)

| Term | Class | Observational Unit | Fitted Table Attribute | Key Invariants |
| :--- | :--- | :--- | :--- | :--- |
| **DataModel** | `DataModel` | Generic | `data: pl.DataFrame` | Immutable container pairing `data` with `variables: dict[str, Variable]` keyed **strictly** by `v.column`. Lookup via `.var(*cols: str)` is a strict $O(1)$ dictionary lookup. |
| **Dataset** | `Dataset(DataModel)` | Isolate / Sample ($N$ rows) | `data: pl.DataFrame` | Microdata container with strictly one `Identifier` and optional `GenomicDistances`. |
| **StratifiedResultMixin** | `StratifiedResultMixin` | Stratum $\times$ Target Cell ($G \times K$ rows) | `_frame` hook (`data_` or `estimates_`) | Shared mixin for fitted `Aggregator` and `BaseEstimator` instances. Provides typed `target_: Variable`, `strata_: tuple[Variable, ...]`, `adjust_for_: Variable \| None`, `variables_: dict[str, Variable]`, `.var()`, `.filter()`, `.split()`, and direct zero-`getattr` properties (`target_col`, `target_title`, `strata_cols`, `strata_titles`, `single_stratum_col`, `single_stratum_title`). |
| **Aggregator** | `Aggregator` | Stratum $\times$ Target Cell ($G \times K$ rows) | `data_: pl.DataFrame` | Sufficiency-reduction `sklearn` transformer mapping $N$ isolate rows in `Dataset` into $G \times K$ cell counts (`strata_cols + ["target", "event", "n"]`). `transform(X)` is stateless and returns a newly fitted clone without mutating `self`. |
| **BaseEstimator** | `BaseEstimator` | Stratum $\times$ Target Cell ($G \times K$ rows) | `estimates_: pl.DataFrame` | Statistical or hierarchical Bayesian model mapping aggregated sufficient statistics (`event`, `n`) into prevalence/coverage estimates (`estimate`, `lower`, `upper`). |
