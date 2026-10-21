"""Module to handle and represent distance measures between samples."""

from abc import ABCMeta
from collections.abc import Callable, Sequence
from io import IOBase
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import coo_array

from seroepi.dist import GenomicDistances
from seroepi.io.base import BaseParser


# Classes --------------------------------------------------------------------------------------------------------------
class DistanceParser(BaseParser, metaclass=ABCMeta):
    """Abstract base parser for pairwise distance and similarity matrices.

    Attributes:
        _READERS: Mapping of file extensions to Polars reader functions.
    """

    @staticmethod
    def from_pairwise(
        df: pl.DataFrame | pl.LazyFrame,
        query_col: str | None = None,
        target_col: str | None = None,
        dist_col: str | None = None,
    ) -> GenomicDistances:
        """Creates a GenomicDistances instance from long-format pairwise data.

        If column names are not specified, defaults to the first 3 columns of the DataFrame:
        column 0 = query, column 1 = target, column 2 = distance.

        Args:
            df: Long-format DataFrame or LazyFrame containing pairwise comparisons.
            query_col: Column name for the query/first sample ID.
            target_col: Column name for the target/second sample ID.
            dist_col: Column name for the scalar distance measure.

        Returns:
            GenomicDistances containing a symmetric CSR matrix and unique sample labels.
        """
        lf = df.lazy() if isinstance(df, pl.DataFrame) else df
        cols = lf.collect_schema().names()
        if len(cols) < 3 and not (query_col and target_col and dist_col):
            raise ValueError(f"from_pairwise requires at least 3 columns, found {len(cols)}")

        q_col = query_col or cols[0]
        t_col = target_col or cols[1]
        d_col = dist_col or cols[2]

        # 1. Deterministically collect sorted unique sample IDs
        samples_df = (
            pl.concat(
                [
                    lf.select(pl.col(q_col).alias("id")),
                    lf.select(pl.col(t_col).alias("id")),
                ]
            )
            .unique()
            .sort("id")
            .collect()
        )
        sample_array = samples_df["id"].to_numpy().astype(str)
        if len(samples_df) == 0:
            empty_mat = coo_array((0, 0), dtype=float)
            return GenomicDistances(empty_mat.tocsr(), np.array([], dtype=str))

        sample_map = samples_df.with_row_index("idx").lazy()

        # 2. Map string labels to integer coordinates and symmetrize in Polars
        # (Avoiding scipy `.maximum(mat.T)` preserves explicit 0.0 SNP distances in csr_array.data)
        raw_indexed = (
            lf.join(sample_map, left_on=q_col, right_on="id")
            .rename({"idx": "row"})
            .join(sample_map, left_on=t_col, right_on="id")
            .rename({"idx": "col"})
            .select(["row", "col", pl.col(d_col).cast(pl.Float64).alias("dist")])
        )
        indexed_df = (
            pl.concat(
                [
                    raw_indexed,
                    raw_indexed.select([pl.col("col").alias("row"), pl.col("row").alias("col"), pl.col("dist")]),
                ]
            )
            .filter(pl.col("row") != pl.col("col"))
            .group_by(["row", "col"])
            .agg(pl.col("dist").max())
            .collect()
        )
        rows = indexed_df["row"].to_numpy()
        cols_arr = indexed_df["col"].to_numpy()
        weights = indexed_df["dist"].to_numpy().astype(float)

        n = len(sample_array)
        mat = coo_array((weights, (rows, cols_arr)), shape=(n, n))
        return GenomicDistances(mat.tocsr(), sample_array)


class PairwiseDistanceParser(DistanceParser, metaclass=ABCMeta):
    """Abstract parser for long/tabular pairwise distance files with fixed schemas.

    Attributes:
        _READERS: Mapping of file extensions to Polars reader functions.

    Examples:
        >>> import polars as pl
        >>> from seroepi.io.dist import PairwiseDistanceParser
        >>> df = pl.DataFrame({"s1": ["A", "B"], "s2": ["B", "C"], "dist": [1.0, 4.0]})
        >>> gd = PairwiseDistanceParser.from_pairwise(df)
        >>> len(gd)
        3
    """

    @classmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> GenomicDistances:
        """Parses a long-format pairwise file using the parser schema.

        Args:
            source: File path, bytes, or stream containing long-format pairwise distances.
            separator: Optional delimiter override.
            reader: Optional reader override.
            **kwargs: Additional keyword arguments passed to ``polars.read_csv``.

        Returns:
            Parsed ``GenomicDistances`` instance.
        """
        src = cls._normalize_source(source)
        df = pl.read_csv(src, schema=cls.schema(), **kwargs)
        return cls.from_pairwise(df)


class SquareDistanceParser(DistanceParser):
    """Parser for wide NxN dense square distance matrices.

    Attributes:
        id_col: Name of the leading sample identifier column.
        value_dtype: Polars numeric dtype used to cast pairwise distance columns.

    Examples:
        >>> from seroepi.io.dist import SquareDistanceParser
        >>> gd = SquareDistanceParser.parse("Name,s1,s2\\ns1,0,3\\ns2,3,0\\n")
        >>> len(gd)
        2
    """

    id_col: str = "Name"
    value_dtype: pl.DataType = pl.Float64()

    @classmethod
    def schema(cls, columns: Sequence[str] | None = None) -> pl.Schema:
        """Constructs an exact Polars schema if ``columns`` are provided, or an empty ``Schema`` otherwise.

        Args:
            columns: Optional sequence of header column names where ``columns[0]`` is the sample ID column.

        Returns:
            Polars ``Schema`` mapping ``columns[0]`` to ``String`` and remaining columns to ``cls.value_dtype``.
        """
        if not columns:
            return pl.Schema()
        return pl.Schema(
            {
                columns[0]: pl.String(),
                **{col: cls.value_dtype for col in columns[1:]},
            }
        )

    @classmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> GenomicDistances:
        """Parses a square distance matrix safely from files, bytes, or in-memory streams.

        Args:
            source: File path, CSV string/bytes, or stream containing a wide NxN distance matrix.
            separator: Optional delimiter override.
            reader: Optional reader override.
            **kwargs: Additional keyword arguments forwarded to ``polars.read_csv``.

        Returns:
            Symmetric ``GenomicDistances`` instance.

        Raises:
            ValueError: If the parsed distance matrix is not square.
        """
        src = cls._normalize_source(source)
        df = pl.read_csv(src, **kwargs)
        if df.width == 0:
            empty_mat = coo_array((0, 0), dtype=float)
            return GenomicDistances(empty_mat.tocsr(), np.array([], dtype=str))

        sample_cols = df.columns[1:]
        matrix_data = df.select(pl.col(sample_cols).cast(cls.value_dtype)).to_numpy()

        if matrix_data.shape[0] != matrix_data.shape[1]:
            raise ValueError(
                f"Expected square distance matrix, got shape ({matrix_data.shape[0]}, {matrix_data.shape[1]})."
            )

        n = matrix_data.shape[0]
        sym = np.maximum(matrix_data, matrix_data.T)
        rows, cols = np.where(~np.eye(n, dtype=bool))
        vals = sym[rows, cols]
        mat = coo_array((vals, (rows, cols)), shape=(n, n))
        return GenomicDistances(mat.tocsr(), np.asarray(sample_cols, dtype=str))


class Ska2DistanceParser(PairwiseDistanceParser):
    """Parses pairwise distance tables generated by SKA2.

    Attributes:
        _READERS: Mapping of file extensions to Polars reader functions.
    """

    @classmethod
    def schema(cls) -> pl.Schema:
        """Returns the 6-column tabular schema for SKA2 pairwise output.

        Returns:
            Polars ``Schema`` for ``Sample1``, ``Sample2``, ``Distance``, and match/mismatch columns.
        """
        return pl.Schema(
            {
                "Sample1": pl.String(),
                "Sample2": pl.String(),
                "Distance": pl.Float64(),
                "Mismatches (proportion)": pl.Float64(),
                "Match count": pl.UInt64(),
                "Mismatch count": pl.UInt64(),
            }
        )

    @classmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> GenomicDistances:
        """Parses a pairwise distance matrix from SKA2 tab-separated output.

        Args:
            source: File path, string/bytes, or stream containing SKA2 pairwise output.
            separator: Optional delimiter override (defaults to ``"\\t"``).
            reader: Optional reader override.
            **kwargs: Additional keyword arguments forwarded to ``polars.read_csv``.

        Returns:
            Parsed ``GenomicDistances`` instance.
        """
        src = cls._normalize_source(source)
        csv_kwargs: dict[str, Any] = {"separator": "\t", "has_header": False, "schema": cls.schema()}
        csv_kwargs.update(kwargs)
        df = pl.read_csv(src, **csv_kwargs)
        return cls.from_pairwise(df, query_col="Sample1", target_col="Sample2", dist_col="Distance")


class NewickDistanceParser(DistanceParser):
    """Parses Newick phylogenetic trees into pairwise patristic distance matrices.

    Attributes:
        _READERS: Mapping of file extensions to Polars reader functions.

    Examples:
        >>> from seroepi.io.dist import NewickDistanceParser
        >>> gd = NewickDistanceParser.from_newick("(A:0.1,B:0.2):0.0;")
        >>> len(gd)
        2
    """

    @classmethod
    def schema(cls) -> pl.Schema:
        """Returns an empty schema as Newick trees are non-tabular.

        Returns:
            Empty Polars ``Schema``.
        """
        return pl.Schema()

    @staticmethod
    def from_newick(
        newick: str,
        label_map: dict[str, str] | None = None,
    ) -> GenomicDistances:
        """Computes a symmetric pairwise patristic distance matrix from a Newick string in ``O(N^2)`` time.

        Args:
            newick: Newick-formatted phylogenetic tree string.
            label_map: Optional dictionary mapping raw leaf node labels to canonical sample IDs.

        Returns:
            Symmetric ``GenomicDistances`` matrix of pairwise leaf-to-leaf patristic distances.
        """
        s = newick.strip().rstrip(";").strip()
        if not s:
            empty_mat = coo_array((0, 0), dtype=float)
            return GenomicDistances(empty_mat.tocsr(), np.array([], dtype=str))

        parent: list[int] = [-1]
        length: list[float] = [0.0]
        labels: dict[int, str] = {}
        children: list[list[int]] = [[]]

        cur = 0
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch in " \t\r\n":
                i += 1
            elif ch == "(":
                nid = len(parent)
                parent.append(cur)
                length.append(0.0)
                children.append([])
                children[cur].append(nid)
                cur = nid
                i += 1
            elif ch == ",":
                cur = parent[cur]
                nid = len(parent)
                parent.append(cur)
                length.append(0.0)
                children.append([])
                children[cur].append(nid)
                cur = nid
                i += 1
            elif ch == ")":
                cur = parent[cur]
                i += 1
            elif ch == ":":
                i += 1
                j = i
                while j < n and s[j] not in ",() \t\r\n;":
                    j += 1
                length[cur] = float(s[i:j])
                i = j
            else:
                j = i
                while j < n and s[j] not in ":,() \t\r\n;":
                    j += 1
                lbl = s[i:j].strip("'\"")
                labels[cur] = label_map.get(lbl, lbl) if label_map else lbl
                i = j

        leaf_nodes = [nid for nid, ch_list in enumerate(children) if not ch_list and nid in labels]
        n_leaves = len(leaf_nodes)
        if n_leaves == 0:
            empty_mat = coo_array((0, 0), dtype=float)
            return GenomicDistances(empty_mat.tocsr(), np.array([], dtype=str))

        leaf_names = np.asarray([labels[nid] for nid in leaf_nodes], dtype=str)
        dist_mat = np.zeros((n_leaves, n_leaves), dtype=np.float64)
        node_to_leaf_idx = {nid: idx for idx, nid in enumerate(leaf_nodes)}

        order: list[int] = []
        stack = [0]
        while stack:
            u = stack.pop()
            order.append(u)
            stack.extend(children[u])
        order.reverse()

        subtree_leaves: list[list[int]] = [[] for _ in range(len(parent))]
        dist_to_root = np.zeros(n_leaves, dtype=np.float64)

        for u in order:
            if u in node_to_leaf_idx:
                subtree_leaves[u] = [node_to_leaf_idx[u]]
            else:
                ch_subs = [subtree_leaves[v] for v in children[u] if subtree_leaves[v]]
                for a_idx in range(len(ch_subs)):
                    leaves_a = ch_subs[a_idx]
                    for b_idx in range(a_idx + 1, len(ch_subs)):
                        leaves_b = ch_subs[b_idx]
                        d_ab = dist_to_root[leaves_a][:, None] + dist_to_root[leaves_b][None, :]
                        dist_mat[np.ix_(leaves_a, leaves_b)] = d_ab
                        dist_mat[np.ix_(leaves_b, leaves_a)] = d_ab.T
                merged: list[int] = []
                for sub in ch_subs:
                    merged.extend(sub)
                subtree_leaves[u] = merged

            if parent[u] != -1 and subtree_leaves[u]:
                dist_to_root[subtree_leaves[u]] += length[u]

        clamped = np.maximum(0.0, dist_mat)
        clamped = np.maximum(clamped, clamped.T)
        np.fill_diagonal(clamped, 0.0)
        rows, cols = np.where(~np.eye(n_leaves, dtype=bool))
        vals = clamped[rows, cols]
        mat = coo_array((vals, (rows, cols)), shape=(n_leaves, n_leaves))
        return GenomicDistances(mat.tocsr(), leaf_names)

    @classmethod
    def parse(
        cls,
        source: str | Path | IOBase | bytes | pl.DataFrame | pl.LazyFrame,
        separator: str | None = None,
        reader: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> GenomicDistances:
        """Parses a Newick tree file, string, or byte stream into a ``GenomicDistances`` matrix.

        Args:
            source: File path, Newick string, ``bytes``, or ``IOBase`` stream.
            separator: Unused; present for ``BaseParser`` interface compatibility.
            reader: Unused; present for ``BaseParser`` interface compatibility.
            **kwargs: Optional keyword arguments (supports ``label_map: dict[str, str]``).

        Returns:
            Parsed ``GenomicDistances`` instance.

        Raises:
            TypeError: If ``source`` is not a path, string, bytes, or stream.
        """
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, bytes):
            text = source.decode("utf-8")
        elif isinstance(source, IOBase):
            raw = source.read()
            text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        elif isinstance(source, str):
            p = Path(source)
            text = (
                p.read_text(encoding="utf-8")
                if (len(source) < 1024 and "\n" not in source and "(" not in source and p.exists())
                else source
            )
        else:
            raise TypeError(f"Unsupported Newick source type: {type(source)}")
        label_map = kwargs.get("label_map")
        return cls.from_newick(text, label_map=label_map)


class PathogenwatchDistanceParser(SquareDistanceParser):
    """Parses square distance matrices and Newick trees produced by Pathogenwatch.

    Attributes:
        id_col: Leading sample identifier column name (``"Name"``).
        value_dtype: Polars numeric dtype for distance entries (``Float64``).
    """

    id_col: str = "Name"
    value_dtype: pl.DataType = pl.Float64()
    from_newick = staticmethod(NewickDistanceParser.from_newick)
