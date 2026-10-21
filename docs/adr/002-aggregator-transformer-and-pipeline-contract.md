---
author: Tom Stanton
comments: true
tags: [adr, architecture, sklearn, aggregator]
icon: lucide/workflow
---

# ADR 002: `Aggregator` as a Sufficiency-Reduction Transformer and `sklearn.pipeline.Pipeline` Contract

* **Status:** Accepted
* **Date:** 2026-10-14

---

## Context

We want `Aggregator` and `BaseEstimator` to compose seamlessly inside a `scikit-learn` `Pipeline`:

```python
pipe = Pipeline(
    [
        ("agg", Aggregator(target=ds.var("K_locus"), strata=[ds.var("region")])),
        ("est", HierarchicalSVIEstimator()),
    ]
)
pipe.fit(ds)
```

Auditing the previous `Aggregator` implementation revealed two critical issues:

1. **Broken `transform()` / `fit_transform()` Mechanics:**
   - `Aggregator.transform(X)` was implemented as `return self.fit(X)`.
   - Because `sklearn.base.TransformerMixin.fit_transform(X)` calls `self.fit(X).transform(X)`, calling `pipe.fit(ds)` executed the full Polars aggregation **twice**.
   - Worse, calling `pipe.predict(ds_test)` invoked `Xt = agg.transform(ds_test)`, which **mutated the fitted `Aggregator` in-place** (overwriting `self.data_` with `ds_test`) and returned `self`. Downstream in `BaseEstimator.predict(Xt)`, the identity check `if X is self.aggregator_:` evaluated to `True` (since `Xt` was the mutated `self`), causing `pipe.predict(ds_test)` to silently return the cached training estimates instead of predicting on `ds_test`!

2. **Mathematical & Statistical ML Ethos (Change of Observational Unit):**
   - In standard row-wise `scikit-learn` transformers ($T: \mathcal{X}_{N \times D} \to \mathcal{X}'_{N \times D'}$), sample cardinality $N$ is preserved ($|T(X)| = |X| = N$).
   - Mathematically, `Aggregator` is a **sufficiency reduction** operator: it maps $N$ isolate-level records ($i \in \{1, \dots, N\}$) to $G \times K$ stratum-by-target binomial count cells $(k_{g,t}, n_g)$.
   - While `Pipeline.fit(ds)` and `Pipeline.predict(ds_test)` do not enforce $|T(X)| = |X|$, standard row-wise `sklearn` cross-validators (`cross_val_score(pipe, ds, y)`) expect prediction arrays of length $N_{\text{test}}$ rather than $G \times K$.

---

## Decision

1. **Retain `Aggregator` as an `sklearn` `Transformer` (`BaseEstimator, TransformerMixin`) with Non-Mutating `transform()`:**
   - `fit(X: Dataset, y=None) -> Self`: Aggregates `X`, stores `self.data_`, `self.variables_`, `self.target_`, `self.strata_`, and marks `self.is_fitted_ = True`.
   - `fit_transform(X: Dataset, y=None) -> Self`: Overridden explicitly to `return self.fit(X, y)` so `Pipeline.fit(ds)` executes aggregation **strictly once**.
   - `transform(X: Dataset) -> Self`: Validates `self.check_is_fitted()`, creates a shallow clone (`cloned = copy.copy(self)`), runs `cloned.fit(X)` on the clone, and returns `cloned`.
     - This guarantees `transform(X_test)` is **stateless/non-mutating** on the training `Aggregator` (`self.data_` remains untouched) and ensures `Xt is not self.aggregator_` in `BaseEstimator.predict(Xt)`.

2. **Strict `sklearn` Trailing-Underscore Convention:**
   - Remove the non-underscored `@property def data` and `@property def variables` from `Aggregator`.
   - Fitted state is accessed strictly via `agg.data_` and `agg.variables_` (matching `est.estimates_` and `est.variables_` on `BaseEstimator`).

3. **Cross-Validation Boundary:**
   - Chaining `Pipeline([("agg", agg), ("est", est)])` is endorsed for end-to-end fitting (`.fit(ds)`) and out-of-sample prediction (`.predict(ds_test)`).
   - Cross-validation stability analyses (e.g., `Formulation`) operate on the aggregated cell matrix `agg.data_` (splitting across strata/groups) rather than row-wise isolate `cross_val_score`.

---

## Consequences

* **Positive:** `Pipeline([("agg", agg), ("est", est)]).fit(ds)` runs in $1\times$ time instead of $2\times$.
* **Positive:** `pipe.predict(ds_test)` accurately computes predictions on `ds_test` without corrupting the training `Aggregator` or hitting the cached training short-circuit.
* **Positive:** Full compliance with `scikit-learn` trailing-underscore conventions (`data_`, `estimates_`, `variables_`).
