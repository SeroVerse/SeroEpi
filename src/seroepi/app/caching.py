"""In-memory content-addressable `ModelCache` subclassing `shiny_kit.caching.BaseCache`."""

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
from shiny_kit.caching import BaseCache
from shiny_kit.warnings import WarningCatcher
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import clone


class ModelCache(BaseCache):
    """Thread-safe in-memory LRU content-addressable cache for SeroEpi models.

    Subclasses [`shiny_kit.caching.BaseCache`](file:///Users/tsta0015/Programming/shiny-kit/src/shiny_kit/caching.py)
    to fingerprint Polars `DataFrame`s, `Dataset`s, and scikit-learn `BaseEstimator`s,
    and provide single-flight `fit_or_load()` memoization in memory.

    Args:
        cache_dir: Optional unused directory path forwarded to `BaseCache`.
        max_memory_items: Maximum number of fitted artifacts retained in the in-memory LRU tier.
        persist_to_disk: Whether to persist fitted models to disk. Defaults to `False`.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        max_memory_items: int = 64,
        persist_to_disk: bool = False,
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            max_memory_items=max_memory_items,
            persist_to_disk=persist_to_disk,
        )

    @classmethod
    def _update_custom_hasher(cls, hasher: "hashlib._Hash", obj: Any) -> bool:
        if hasattr(obj, "data") and isinstance(obj.data, pl.DataFrame):
            id_col = getattr(getattr(obj, "identifier", None), "column", None)
            hasher.update(f"{type(obj).__qualname__}:{getattr(obj, 'name', '')}:{obj.data.shape}:".encode())
            cls._update_dataframe_hasher(hasher, obj.data, id_col=id_col)
            if hasattr(obj, "variables") and isinstance(obj.variables, Mapping):
                for k in sorted(obj.variables.keys(), key=str):
                    hasher.update(f"var:{k}:".encode())
                    cls._update_hasher(hasher, obj.variables[k])
            if hasattr(obj, "distances"):
                hasher.update(f"dist:{obj.distances is not None};".encode())
            return True

        if isinstance(obj, pl.DataFrame):
            cls._update_dataframe_hasher(hasher, obj)
            return True

        if isinstance(obj, SklearnBaseEstimator):
            hasher.update(f"Estimator:{type(obj).__module__}.{type(obj).__qualname__}:".encode())
            params = obj.get_params(deep=False)
            for k in sorted(params.keys()):
                hasher.update(f"param:{k}=".encode())
                cls._update_hasher(hasher, params[k])
            if hasattr(obj, "data_") and isinstance(obj.data_, pl.DataFrame):
                hasher.update(b"fitted_data:")
                cls._update_dataframe_hasher(hasher, obj.data_)
            elif hasattr(obj, "estimates_") and isinstance(obj.estimates_, pl.DataFrame):
                hasher.update(b"fitted_estimates:")
                cls._update_dataframe_hasher(hasher, obj.estimates_)
            elif hasattr(obj, "rankings_") and isinstance(obj.rankings_, pl.DataFrame):
                hasher.update(b"fitted_rankings:")
                cls._update_dataframe_hasher(hasher, obj.rankings_)
            return True

        return False

    @staticmethod
    def _update_dataframe_hasher(hasher: "hashlib._Hash", df: pl.DataFrame, id_col: str | None = None) -> None:
        schema_repr = [(col, str(dtype)) for col, dtype in df.schema.items()]
        hasher.update(f"DF:{df.shape}:{schema_repr}:".encode())
        if len(df) == 0:
            return
        sample_col = id_col if (id_col and id_col in df.columns) else df.columns[0]
        head_vals = df[sample_col].head(5).to_list()
        tail_vals = df[sample_col].tail(5).to_list()
        hasher.update(f"head:{head_vals}:tail:{tail_vals}:".encode())
        for col, dtype in df.schema.items():
            if dtype.is_numeric():
                s_sum = df[col].sum()
                hasher.update(f"sum({col})={s_sum};".encode())

    def compute_fit_key(self, estimator: Any, X: Any, y: Any = None) -> str:
        """Computes a composite cache key for fitting `estimator` on `(X, y)`.

        Args:
            estimator: Unfitted or configured estimator instance.
            X: Input data object.
            y: Optional target argument.

        Returns:
            32-character SHA-256 fingerprint string.
        """
        return self.fingerprint(("fit", estimator, X, y))

    def fit_or_load[E: SklearnBaseEstimator](self, estimator: E, X: Any = None, y: Any = None) -> E:
        """Returns a cached fitted clone of `estimator` on `(X, y)`, or fits and caches it on miss.

        Uses per-key single-flight locking so concurrent sessions fitting the same model and
        dataset wait on the first fit and share its cached output without redundant execution.

        Args:
            estimator: Unfitted or configured scikit-learn estimator.
            X: Input data (or `None` when wrapping pre-fitted models).
            y: Optional target argument.

        Returns:
            The fitted estimator instance.
        """
        key = self.compute_fit_key(estimator, X, y)
        with self._lock:
            hit, cached = self._lru_get(key)
            if hit:
                self.hits += 1
                WarningCatcher.emit_messages(getattr(cached, "warnings_", ()), stacklevel=2)
                return cached

        key_lock = self._get_key_lock(key)
        with key_lock:
            hit, cached = self.get(key)
            if hit:
                WarningCatcher.emit_messages(getattr(cached, "warnings_", ()), stacklevel=2)
                return cached

            if (hasattr(estimator, "estimator") and getattr(estimator.estimator, "is_fitted_", False)) or hasattr(
                estimator, "formulations"
            ):
                fitted: E = copy.copy(estimator)
            else:
                fitted = clone(estimator)
            with WarningCatcher() as catcher:
                if y is not None:
                    fitted.fit(X, y)  # ty: ignore[unresolved-attribute]
                else:
                    fitted.fit(X)  # ty: ignore[unresolved-attribute]
            inner_warnings = getattr(getattr(fitted, "estimator_", None), "warnings_", ())
            own_warnings = getattr(fitted, "warnings_", ())
            merged_warnings = WarningCatcher.merge(inner_warnings, own_warnings, catcher.messages)
            fitted.warnings_ = merged_warnings  # ty: ignore[invalid-assignment]
            stored = self.put(key, fitted)
            WarningCatcher.emit_messages(merged_warnings, stacklevel=2)
            return stored


default_cache = ModelCache()
