"""Module to handle and represent distance measures between samples."""

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from scipy.sparse import csr_array, issparse
from scipy.sparse.csgraph import connected_components as sp_connected_components
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


# Classes --------------------------------------------------------------------------------------------------------------
class SparseCompatibleClusterer(ClusterMixin, BaseEstimator):
    """Base class for clustering estimators compatible with sparse distance matrices.

    Attributes:
        labels_: 1D integer NumPy array of cluster labels assigned during ``fit()``.
    """

    def __sklearn_tags__(self) -> Any:
        """Returns scikit-learn estimator tags declaring sparse matrix input support.

        Returns:
            Scikit-learn ``Tags`` instance with ``input_tags.sparse = True``.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fits the clusterer to a distance or feature matrix ``X``.

        Args:
            X: Square distance matrix or feature matrix.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted clusterer instance (``self``).

        Raises:
            NotImplementedError: If not overridden by a concrete subclass.
        """
        raise NotImplementedError("Subclasses must implement fit.")


class ConnectedComponentsClusterer(SparseCompatibleClusterer):
    """Graph connected components clusterer for sparse distance or adjacency matrices.

    Args:
        threshold: Maximum distance threshold below which two samples are connected in the graph.

    Attributes:
        labels_: 1D int32 NumPy array of connected component labels assigned during ``fit()``.
        n_features_in_: Number of columns in the fitted matrix ``X``.

    Examples:
        >>> from scipy.sparse import csr_array
        >>> from seroepi.dist import ConnectedComponentsClusterer
        >>> mat = csr_array([[0.0, 2.0, 25.0], [2.0, 0.0, 25.0], [25.0, 25.0, 0.0]])
        >>> ConnectedComponentsClusterer(threshold=5.0).fit_predict(mat).tolist()
        [0, 0, 1]
    """

    def __init__(self, threshold: float = 0.0) -> None:
        """Initializes the connected components clusterer.

        Args:
            threshold: Maximum distance threshold below which two samples are connected in the graph.
        """
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> Self:
        """Fits the connected components model from a square distance matrix ``X``.

        Args:
            X: Square sparse or dense distance matrix.
            y: Ignored; present for API consistency by convention.

        Returns:
            The fitted ``ConnectedComponentsClusterer`` instance (``self``).

        Raises:
            ValueError: If ``X`` is not a square 2D matrix.
        """
        X_arr = check_array(X, accept_sparse=["csr", "csc"], ensure_min_samples=0, ensure_min_features=0)
        if X_arr.shape[0] != X_arr.shape[1]:
            raise ValueError(f"Distance matrix must be square, got shape {X_arr.shape}.")

        if X_arr.shape[0] == 0:
            self.labels_ = np.array([], dtype=np.int32)
            self.n_features_in_ = 0
            return self

        adj = X_arr.copy()
        if issparse(adj):
            adj.data = (np.asarray(adj.data) <= self.threshold).astype(np.int8)
            adj.eliminate_zeros()
            adj.setdiag(1)
        else:
            adj = (adj <= self.threshold).astype(np.int8)
            np.fill_diagonal(adj, 1)

        _, labels = sp_connected_components(csgraph=adj, directed=False, return_labels=True)
        self.labels_ = labels
        self.n_features_in_ = X_arr.shape[1]
        return self


@dataclass(frozen=True, slots=True)
class GenomicDistances:
    """Container for pairwise genomic distance matrices and sample identifiers.

    Attributes:
        array: Square SciPy ``csr_array`` storing measured pairwise distances.
        samples: 1D string NumPy array of sample identifiers aligned with ``array`` axes.

    Examples:
        >>> from scipy.sparse import csr_array
        >>> from seroepi.dist import ConnectedComponentsClusterer, GenomicDistances
        >>> gd = GenomicDistances(csr_array([[0.0, 3.0], [3.0, 0.0]]), ("s1", "s2"))
        >>> len(gd)
        2
        >>> gd.cluster(ConnectedComponentsClusterer(threshold=5.0))
        ['0', '0']
    """

    array: csr_array
    samples: np.ndarray

    def __post_init__(self) -> None:
        """Validates the consistency of the distance matrix and sample labels.

        Raises:
            ValueError: If ``array`` is not square or ``len(samples)`` does not match ``array.shape[0]``.
        """
        if not isinstance(self.samples, np.ndarray):
            object.__setattr__(self, "samples", np.asarray(self.samples, dtype=str))
        if self.array.shape[0] != self.array.shape[1]:
            raise ValueError("Distance matrix must be square.")
        if len(self.samples) != self.array.shape[0]:
            raise ValueError("Number of labels must match matrix dimensions.")

    @property
    def n_samples(self) -> int:
        """Returns the total number of samples in the distance matrix.

        Returns:
            Integer sample count.
        """
        return len(self.samples)

    def __len__(self) -> int:
        """Returns the total number of samples.

        Returns:
            Integer sample count.
        """
        return self.n_samples

    @property
    def memory(self) -> float:
        """Returns the approximate memory footprint in gigabytes if converted to dense float64.

        Returns:
            Estimated dense array size in GB.
        """
        return (self.__len__() ** 2 * 8) / (1024**3)

    @property
    def data(self) -> np.ndarray:
        """Returns the underlying CSR matrix non-zero data array.

        Returns:
            1D NumPy array of stored distance values.
        """
        return self.array.data

    def subset(self, sample_ids: Any) -> "GenomicDistances":
        """Slices the distance matrix to retain only the specified unique sample IDs in order.

        Args:
            sample_ids: Iterable of sample identifier strings to retain.

        Returns:
            A new ``GenomicDistances`` instance restricted and ordered to the matching ``sample_ids``.
        """
        target_ids = list(dict.fromkeys(str(s) for s in sample_ids))
        sample_to_idx = {str(s): idx for idx, s in enumerate(self.samples)}
        valid_indices = [sample_to_idx[s] for s in target_ids if s in sample_to_idx]
        if not valid_indices:
            empty_csr = csr_array((0, 0), dtype=self.array.dtype)
            return GenomicDistances(array=empty_csr, samples=np.array([], dtype=str))
        idx_arr = np.asarray(valid_indices, dtype=np.intp)
        sub_csr = self.array[idx_arr, :][:, idx_arr]
        return GenomicDistances(array=csr_array(sub_csr), samples=self.samples[idx_arr])

    def concat(self, *others: "GenomicDistances") -> "GenomicDistances":
        """Combines multiple ``GenomicDistances`` into a unified sparse distance graph.

        Preserves sample ordering by first-seen priority across datasets. If an isolate
        pair ``(i, j)`` is present in multiple distance matrices, retains the distance from
        the earliest matrix in concatenation order. Unmeasured cross-dataset pairs remain
        absent from the sparse structure.

        Args:
            *others: Additional ``GenomicDistances`` instances to merge.

        Returns:
            A new unified ``GenomicDistances`` instance.
        """
        from scipy.sparse import coo_array

        valid_others = [o for o in others if isinstance(o, GenomicDistances) and o.n_samples > 0]
        if not valid_others:
            return self
        if self.n_samples == 0:
            return valid_others[0].concat(*valid_others[1:])

        all_dists = (self, *valid_others)
        seen_samples: dict[str, int] = {}
        for d in all_dists:
            for s in d.samples:
                s_str = str(s)
                if s_str not in seen_samples:
                    seen_samples[s_str] = len(seen_samples)

        total_n = len(seen_samples)
        merged_samples = np.asarray(list(seen_samples.keys()), dtype=str)

        row_chunks: list[np.ndarray] = []
        col_chunks: list[np.ndarray] = []
        val_chunks: list[np.ndarray] = []

        for d in all_dists:
            if d.n_samples == 0 or d.array.nnz == 0:
                continue
            global_idx = np.asarray([seen_samples[str(s)] for s in d.samples], dtype=np.int64)
            coo = d.array.tocoo()
            r_local = coo.row
            c_local = coo.col
            v_local = np.asarray(coo.data, dtype=np.float64)
            off_diag = r_local != c_local
            if not np.any(off_diag):
                continue
            r_glob = global_idx[r_local[off_diag]]
            c_glob = global_idx[c_local[off_diag]]
            v_glob = v_local[off_diag]
            row_chunks.extend((r_glob, c_glob))
            col_chunks.extend((c_glob, r_glob))
            val_chunks.extend((v_glob, v_glob))

        if not row_chunks:
            empty_csr = csr_array((total_n, total_n), dtype=np.float64)
            return GenomicDistances(array=empty_csr, samples=merged_samples)

        all_rows = np.concatenate(row_chunks)
        all_cols = np.concatenate(col_chunks)
        all_vals = np.concatenate(val_chunks)

        # Deduplicate (row, col) pairs keeping the first-seen distance in dataset priority order
        linear_keys = all_rows * total_n + all_cols
        _, first_indices = np.unique(linear_keys, return_index=True)
        first_indices.sort()

        u_rows = all_rows[first_indices]
        u_cols = all_cols[first_indices]
        u_vals = all_vals[first_indices]

        merged_csr = coo_array((u_vals, (u_rows, u_cols)), shape=(total_n, total_n)).tocsr()
        return GenomicDistances(array=merged_csr, samples=merged_samples)

    def to_dense(self, unmeasured_fill: float = 1e9) -> np.ndarray:
        """Converts the sparse distance graph to a dense 2D array, filling unmeasured off-diagonal pairs.

        Args:
            unmeasured_fill: Sentinel distance assigned to unmeasured off-diagonal sample pairs.

        Returns:
            A 2D float64 NumPy array of shape ``(n_samples, n_samples)`` with zeros on the diagonal.
        """
        n = self.n_samples
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        coo = self.array.tocoo()
        if len(coo.data) >= n * (n - 1):
            dense = self.array.toarray().astype(np.float64, copy=False)
            np.fill_diagonal(dense, 0.0)
            return dense
        dense = np.full((n, n), unmeasured_fill, dtype=np.float64)
        np.fill_diagonal(dense, 0.0)
        dense[coo.row, coo.col] = coo.data
        return dense

    def cluster(
        self,
        clusterer: ClusterMixin | BaseEstimator,
        *args: Any,
        allow_dense: bool = False,
        **kwargs: Any,
    ) -> list[str | None]:
        """Clusters samples using a scikit-learn compatible clustering estimator.

        Attempts to fit the estimator on the internal sparse CSR matrix. If the estimator
        requires dense input, it falls back to a dense NumPy array for datasets up to 5,000
        samples. For larger matrices, explicit confirmation (`allow_dense=True`) is required
        to prevent out-of-memory exhaustion.

        Noise points (labeled as -1, e.g. from DBSCAN) are mapped to null.

        Args:
            clusterer: A scikit-learn compatible clustering estimator.
            *args: Positional arguments passed to fit / fit_predict.
            allow_dense: Set to True to allow automatic conversion to dense array for matrices
                larger than 5,000 samples.
            **kwargs: Keyword arguments passed to fit / fit_predict.

        Returns:
            A Polars Categorical Series containing cluster labels aligned with `self.index`.
        """
        fit_predict_fn = getattr(clusterer, "fit_predict", None)
        fit_fn = getattr(clusterer, "fit", None)

        raw_labels = None
        try:
            if callable(fit_predict_fn):
                raw_labels = fit_predict_fn(self.array, *args, **kwargs)
            elif callable(fit_fn):
                fit_fn(self.array, *args, **kwargs)
                raw_labels = getattr(clusterer, "labels_", None)
        except (TypeError, ValueError) as err:
            err_msg = str(err).lower()
            if "sparse" in err_msg or "dense" in err_msg:
                if self.__len__() > 5000 and not allow_dense:
                    raise MemoryError(
                        f"Distance matrix dimension ({self.__len__()}x{self.__len__()}) exceeds 5,000 threshold for "
                        f"automatic dense conversion. Converting to dense array requires ~{self.memory:.1f} GB of "
                        f"RAM. Pass allow_dense=True to override, or use a sparse-compatible clusterer."
                    ) from err
                dense_mat = self.to_dense()
                if callable(fit_predict_fn):
                    raw_labels = fit_predict_fn(dense_mat, *args, **kwargs)
                elif callable(fit_fn):
                    fit_fn(dense_mat, *args, **kwargs)
                    raw_labels = getattr(clusterer, "labels_", None)
            else:
                raise

        if raw_labels is None:
            raw_labels = getattr(clusterer, "labels_", None)
        if raw_labels is None:
            raise ValueError(
                f"Clusterer {clusterer.__class__.__name__} did not produce cluster labels or a 'labels_' attribute."
            )
        return [None if (lbl is None or str(lbl) == "-1") else str(lbl) for lbl in raw_labels]
