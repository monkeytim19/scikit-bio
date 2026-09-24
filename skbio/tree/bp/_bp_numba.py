# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Derived from improved-octo-waddle (https://github.com/biocore/improved-octo-waddle)
# originally authored by Daniel McDonald, distributed under the Modified BSD License.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

"""Numba compute engine for :class:`skbio.tree.BPTree`.

The navigation primitives of the Cython engine (``_bp_cy._BPKernel``), ported
to Numba functions over the tree's flat arrays, and the batch kernels built on
them. The primitives follow the Cython implementation line by line, with one
exception: the range query over the range min-max (rmM) tree is iterative
(bottom-up) instead of recursive. It visits a different but equivalent set of
nodes, covering the same blocks, so it returns the same value.

Numba is an optional dependency; everything here is defined only if it is
installed (``NUMBA_AVAILABLE``).
"""

from collections import namedtuple

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# The arrays and rmM-tree geometry of a tree, as passed to the kernels.
BPArrays = namedtuple(
    "BPArrays",
    [
        "B",  # parentheses (uint8)
        "e_index",  # excess at each position
        "k_index_0",  # position of the k-th closing parenthesis
        "k_index_1",  # position of the k-th opening parenthesis
        "m",  # rmM tree: minimum excess per node, heap order
        "M",  # rmM tree: maximum excess per node, heap order
        "r",  # rmM tree: rank (opening parentheses) before each node
        "b",  # block size
        "height",  # rmM tree height
        "n_internal",  # rmM tree internal node count
        "size",  # number of parentheses
    ],
)


def bp_arrays(tree):
    """Collect the arrays of a :class:`~skbio.tree.BPTree` for the kernels."""
    return BPArrays(
        tree._data,
        tree._e_index,
        tree._k_index_0,
        tree._k_index_1,
        tree._m,
        tree._M,
        tree._r,
        tree._b,
        tree._height,
        (1 << tree._height) - 1,
        tree._size,
    )


if NUMBA_AVAILABLE:
    _SIZE_MAX = np.iinfo(np.intp).max

    # -- complete binary tree in breadth-first (heap) order -------------------

    @njit(inline="always")
    def _bt_is_root(v):
        return v == 0

    @njit(inline="always")
    def _bt_is_left_child(v):
        return 0 if v == 0 else v % 2

    @njit(inline="always")
    def _bt_is_right_child(v):
        return 0 if v == 0 else 1 - (v % 2)

    @njit(inline="always")
    def _bt_parent(v):
        return 0 if v == 0 else (v - 1) // 2

    @njit(inline="always")
    def _bt_is_leaf(v, n_internal):
        return v >= n_internal

    # -- primitives -----------------------------------------------------------

    @njit
    def _scan_block_forward(T, i, k, d):
        lower = max(max(k, 0) * T.b, i + 1)
        upper = min((k + 1) * T.b, T.size)
        for j in range(lower, upper):
            if T.e_index[j] == d:
                return j
        return -1

    @njit
    def _scan_block_backward(T, i, k, d):
        lower = max(k, 0) * T.b - 1
        if lower >= 0:
            lower -= 1
        upper = min((k + 1) * T.b, T.size) - 1
        upper = min(i - 1, upper)
        if upper <= 0:
            return -1
        for j in range(upper, lower, -1):
            if T.e_index[j] == d:
                return j
        return -1

    @njit
    def _fwdsearch(T, i, d):
        """Position after ``i`` with excess ``excess(i) + d``, or -1."""
        k = i // T.b
        d += T.e_index[i]
        node = T.n_internal + k
        result = -1
        if T.m[node] <= d <= T.M[node]:
            result = _scan_block_forward(T, i, k, d)
        if result == -1:
            while not _bt_is_root(node):
                if _bt_is_left_child(node):
                    node += 1
                    if T.m[node] <= d <= T.M[node]:
                        break
                node = _bt_parent(node)
            if _bt_is_root(node):
                return -1
            while not _bt_is_leaf(node, T.n_internal):
                node = 2 * node + 1
                if not (T.m[node] <= d <= T.M[node]):
                    node += 1
            k = node - T.n_internal
            result = _scan_block_forward(T, i, k, d)
        return result

    @njit
    def _bwdsearch(T, i, d):
        """Position before ``i`` with excess ``excess(i) + d``, or -1."""
        k = i // T.b
        d += T.e_index[i]
        result = _scan_block_backward(T, i, k, d)
        node = T.n_internal + k
        if result == -1 and _bt_is_right_child(node):
            node -= 1
            k = node - T.n_internal
            result = _scan_block_backward(T, i, k, d)
            k = i // T.b
            node += 1
        if result == -1:
            while not _bt_is_root(node):
                if _bt_is_right_child(node):
                    node -= 1
                    if T.m[node] <= d <= T.M[node]:
                        break
                node = _bt_parent(node)
            if _bt_is_root(node):
                return -1
            while not _bt_is_leaf(node, T.n_internal):
                node = 2 * node + 2
                if not (T.m[node] <= d <= T.M[node]):
                    node -= 1
            k = node - T.n_internal
            result = _scan_block_backward(T, i, k, d)
        return result

    @njit
    def _close(T, i):
        if not T.B[i]:
            return i
        return _fwdsearch(T, i, -1)

    @njit
    def _open(T, i):
        if T.B[i] or i <= 0:
            return i
        return _bwdsearch(T, i, 0) + 1

    @njit
    def _enclose(T, i):
        if T.B[i]:
            return _bwdsearch(T, i, -2) + 1
        return _bwdsearch(T, i - 1, -2) + 1

    @njit
    def _parent(T, i):
        if i == 0 or i == T.size - 1:
            return -1
        return _enclose(T, i)

    @njit
    def _is_ancestor(T, i, j):
        if i == j:
            return False
        if not T.B[i]:
            i = _open(T, i)
        return i <= j < _close(T, i)

    @njit
    def _tree_min(T, lo, hi):
        """Minimum excess over the rmM leaf blocks ``[lo, hi]`` (bottom-up)."""
        res = _SIZE_MAX
        lo += T.n_internal
        hi += T.n_internal
        while lo <= hi:
            if lo % 2 == 0:  # a right child: take it, continue to its right
                res = min(res, T.m[lo])
                lo += 1
            if hi % 2 == 1:  # a left child: take it, continue to its left
                res = min(res, T.m[hi])
                hi -= 1
            lo = (lo - 1) // 2
            hi = (hi - 1) // 2
        return res

    @njit
    def _rmq(T, i, j):
        """Leftmost position of the minimum excess in ``[i, j]``."""
        if i >= j:
            return i
        b = T.b
        bi = i // b
        bj = j // b
        e_i = T.e_index[i]
        d_star = e_i
        if bi == bj:
            for p in range(i + 1, j + 1):
                if T.e_index[p] < d_star:
                    d_star = T.e_index[p]
        else:
            for p in range(i + 1, min((bi + 1) * b, T.size)):
                if T.e_index[p] < d_star:
                    d_star = T.e_index[p]
            if bi + 1 <= bj - 1:
                tree_v = _tree_min(T, bi + 1, bj - 1)
                if tree_v < d_star:
                    d_star = tree_v
            for p in range(bj * b, j + 1):
                if T.e_index[p] < d_star:
                    d_star = T.e_index[p]
        if e_i == d_star:
            return i
        return _fwdsearch(T, i, d_star - e_i)

    @njit
    def _lca(T, i, j):
        if _is_ancestor(T, i, j):
            return i
        elif _is_ancestor(T, j, i):
            return j
        return _parent(T, _rmq(T, i, j) + 1)

    @njit
    def _level_ancestor(T, i, d):
        if d <= 0:
            return -1
        if not T.B[i]:
            i = _open(T, i)
        return _bwdsearch(T, i, -d - 1) + 1

    # -- batch kernels (engine="numba") ---------------------------------------

    @njit(parallel=True)
    def close_batch(T, idx):
        """Kernel of :meth:`skbio.tree.BPTree.close_batch`."""
        out = np.empty(idx.shape[0], dtype=np.intp)
        for t in prange(idx.shape[0]):
            out[t] = _close(T, idx[t])
        return out

    @njit(parallel=True)
    def parent_batch(T, idx):
        """Kernel of :meth:`skbio.tree.BPTree.parent_batch`."""
        out = np.empty(idx.shape[0], dtype=np.intp)
        for t in prange(idx.shape[0]):
            out[t] = _parent(T, idx[t])
        return out

    @njit(parallel=True)
    def lca_batch(T, i, j):
        """Kernel of :meth:`skbio.tree.BPTree.lca_batch`."""
        out = np.empty(i.shape[0], dtype=np.intp)
        for t in prange(i.shape[0]):
            out[t] = _lca(T, min(i[t], j[t]), max(i[t], j[t]))
        return out

    @njit(parallel=True)
    def level_ancestor_batch(T, idx, d):
        """Kernel of :meth:`skbio.tree.BPTree.level_ancestor_batch`."""
        out = np.empty(idx.shape[0], dtype=np.intp)
        for t in prange(idx.shape[0]):
            out[t] = _level_ancestor(T, idx[t], d[t])
        return out

    @njit
    def root_distances(T, lengths):
        """Sum of branch lengths from the root to each node, per position."""
        out = np.zeros(T.size, dtype=np.float64)
        stack = np.zeros(T.size // 2 + 1, dtype=np.float64)
        top = 0
        for i in range(1, T.size):
            if T.B[i]:
                top += 1
                stack[top] = stack[top - 1] + lengths[i]
                out[i] = stack[top]
            else:
                top -= 1
        return out

    @njit
    def _tip_distance_row(tips, slot, parent, end, dist, out, row, n):
        r = slot[row]
        out[r, r] = 0.0
        start = row + 1
        da = dist[tips[row]]
        v = parent[tips[row]]
        while start < n:
            stop = end[v]
            dv = dist[v]
            for col in range(start, stop):
                d = (da - dv) + (dist[tips[col]] - dv)
                c = slot[col]
                out[r, c] = d
                out[c, r] = d
            start = stop
            v = parent[v]

    @njit(parallel=True)
    def tip_distances(tips, slot, parent, end, dist):
        """Pairwise path distances between tips, as a square matrix.

        See ``_bp_cy.tip_distances``; rows ``h`` and ``n - 2 - h`` share an
        iteration to balance the triangular workload.
        """
        n = tips.shape[0]
        n_rows = n - 1 if n > 0 else 0
        out = np.empty((n, n), dtype=np.float64)
        for h in prange((n_rows + 1) // 2):
            _tip_distance_row(tips, slot, parent, end, dist, out, h, n)
            if n_rows - 1 - h != h:
                _tip_distance_row(tips, slot, parent, end, dist, out, n_rows - 1 - h, n)
        if n:
            out[slot[n - 1], slot[n - 1]] = 0.0
        return out
