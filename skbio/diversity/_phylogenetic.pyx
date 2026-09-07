# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from libc.math cimport pow

# platform-specific
INDEX_DTYPE = np.intp

# use 64-bit ints for counts (otherwise, overflows on 32-bit systems ie WASM)
COUNT_DTYPE = np.int64

ctypedef np.npy_intp intp_t
ctypedef np.int64_t count_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _tip_distances(np.ndarray[np.double_t, ndim=1] a, object t,
                   np.ndarray[intp_t, ndim=1] tip_indices):
    """Sets each tip to its distance from the root

    Parameters
    ----------
    a : np.ndarray of double
        A matrix in which each row corresponds to a node in ``t``.
    t : skbio.tree.TreeNode
        The tree that corresponds to the rows in ``a``.
    tip_indices : np.ndarray of int
        The index positions in ``a`` of the tips in ``t``.

    Returns
    -------
    np.ndarray of double
        A matrix in which each row corresponds to a node in ``t``, Only the
        rows that correspond to tips are nonzero, and the values in these rows
        are the distance from that tip to the root of the tree.
    """
    cdef:
        object n
        Py_ssize_t i, p_i, n_rows
        np.ndarray[np.double_t, ndim=1] mask
        np.ndarray[np.double_t, ndim=1] tip_ds = a.copy()

    # preorder reduction over the tree to gather distances at the tips
    n_rows = tip_ds.shape[0]
    for n in t.preorder(include_self=False):
        i = n.id
        p_i = n.parent.id

        tip_ds[i] += tip_ds[p_i]

    # construct a mask that represents the locations of the tips
    mask = np.zeros(n_rows, dtype=np.double)
    for i in range(tip_indices.shape[0]):
        mask[tip_indices[i]] = 1.0

    # apply the mask such that tip_ds only includes values which correspond to
    # the tips of the tree.
    for i in range(n_rows):
        tip_ds[i] *= mask[i]

    return tip_ds


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _traverse_reduce(np.ndarray[intp_t, ndim=2] child_index,
                      np.ndarray[count_t, ndim=2] a):
    """Apply a[k] = sum[i:j]

    Parameters
    ----------
    child_index: np.array of int
        A matrix in which the first column corresponds to an index position in
        ``a``, which represents a node in a tree. The second column is the
        starting index in ``a`` for the node's children, and the third column
        is the ending index in ``a`` for the node's children.
    a : np.ndarray of int
        A matrix of the environment data. Each row corresponds to a node in a
        tree, and each column corresponds to an environment. On input, it is
        assumed that only tips have counts.

    Notes
    -----
    This is effectively a postorder reduction over the tree. For example,
    given the following tree:

                            /-A
                  /E-------|
                 |          \-B
        -root----|
                 |          /-C
                  \F-------|
                            \-D

    And assuming counts for [A, B, C, D] in environment FOO of [1, 1, 1, 0] and
    counts for environment BAR of [0, 1, 1, 1], the input counts matrix ``a``
    would be:

        [1 0  -> A
         1 1  -> B
         1 1  -> C
         0 1  -> D
         0 0  -> E
         0 0  -> F
         0 0] -> root

    The method will perform the following reduction:

        [1 0     [1 0     [1 0     [1 0
         1 1      1 1      1 1      1 1
         1 1      1 1      1 1      1 1
         0 1  ->  0 1  ->  0 1  ->  0 1
         0 0      2 1      2 1      2 1
         0 0      0 0      1 2      1 2
         0 0]     0 0]     0 0]     3 3]

    The index positions of the above are encoded in ``child_index`` which
    describes the node to aggregate into, and the start and stop index
    positions of the nodes immediate descendents.

    This method operates inplace on ``a``
    """
    cdef:
        Py_ssize_t i, j, k
        intp_t node, start, end
        count_t n_envs = a.shape[1]

    # possible GPGPU target
    for i in range(child_index.shape[0]):
        node = child_index[i, 0]
        start = child_index[i, 1]
        end = child_index[i, 2]

        for j in range(start, end + 1):
            for k in range(n_envs):
                a[node, k] += a[j, k]


@cython.boundscheck(False)
@cython.wraparound(False)
def _nodes_by_counts(np.ndarray counts,
                     np.ndarray tip_ids,
                     dict indexed):
    """Construct the count array, and the counts up the tree

    Parameters
    ----------
    counts : np.array of int
        A 1D or 2D vector in which each row corresponds to the observed counts
        in an environment. The rows are expected to be in order with respect to
        `tip_ids`.
    tip_ids : np.array of str
        A vector of tip names that correspond to the columns in the `counts`
        matrix.
    indexed : dict
        The result of `index_tree`.

    Returns
    -------
    np.array of int
        The observed counts of every node and the counts if its descendents.

    """
    cdef:
        np.ndarray nodes, observed_ids
        np.ndarray[count_t, ndim=2] count_array, counts_t
        np.ndarray[intp_t, ndim=1] observed_indices, taxa_in_nodes
        Py_ssize_t i, j
        set observed_ids_set
        object n
        dict node_lookup
        count_t n_count_vectors, n_count_taxa

    nodes = indexed['name']

    # allow counts to be a vector
    counts = np.atleast_2d(counts)

    counts = counts.astype(COUNT_DTYPE, copy=False)

    # determine observed IDs. It may be possible to unroll these calls to
    # squeeze a little more performance
    observed_indices = counts.sum(0).nonzero()[0]
    observed_ids = tip_ids[observed_indices]
    observed_ids_set = set(observed_ids)

    # construct mappings of the observed to their positions in the node array
    node_lookup = {}
    for i in range(nodes.shape[0]):
        n = nodes[i]
        if n in observed_ids_set:
            node_lookup[n] = i

    # determine the positions of the observed IDs in nodes
    taxa_in_nodes = np.zeros(observed_ids.shape[0], dtype=INDEX_DTYPE)

    for i in range(observed_ids.shape[0]):
        n = observed_ids[i]
        taxa_in_nodes[i] = node_lookup[n]

    # count_array has a row per node (not tip) and a column per env.
    n_count_vectors = counts.shape[0]
    count_array = np.zeros((nodes.shape[0], n_count_vectors), dtype=COUNT_DTYPE)

    # populate the counts array with the counts of each observation in each
    # env
    counts_t = counts.transpose()
    n_count_taxa = taxa_in_nodes.shape[0]
    for i in range(n_count_taxa):
        for j in range(n_count_vectors):
            count_array[taxa_in_nodes[i], j] = counts_t[observed_indices[i], j]

    child_index = indexed['child_index'].astype(INDEX_DTYPE, copy=False)
    _traverse_reduce(child_index, count_array)

    return count_array


@cython.boundscheck(False)
@cython.wraparound(False)
def _faith_pd_bp(np.uint8_t[:, ::1] presence,
                 np.int32_t[::1] lo,
                 np.int32_t[::1] hi,
                 np.double_t[::1] lengths,
                 np.int32_t[:, ::1] pref,
                 np.double_t[::1] out):
    """Batched Faith's PD over a BPTree's compacted tip-range arrays.

    An OpenMP-parallel (``prange``) reduction that supersedes the sequential
    post-order accumulation in :func:`_traverse_reduce`. For each sample it
    builds an exclusive prefix of the present-taxon indicator, then sums each
    node's branch length iff that node's descendant-taxa range ``[lo, hi)``
    contains a present taxon.

    Parameters
    ----------
    presence : memoryview of uint8, shape (n_samples, n_taxa)
        Present-taxon indicator, already reordered into ascending tip order.
    lo, hi : memoryview of int32, shape (n_nodes,)
        Half-open descendant-taxa bounds of each node in that column space.
    lengths : memoryview of double, shape (n_nodes,)
        Branch length of each node.
    pref : memoryview of int32, shape (n_samples, n_taxa + 1)
        Caller-allocated scratch for the per-sample prefix, indexed by the
        loop variable (never by thread id, so no OpenMP runtime is required).
    out : memoryview of double, shape (n_samples,)
        Faith's PD per sample; written in place.
    """
    cdef:
        Py_ssize_t s, j, k
        Py_ssize_t n_samples = presence.shape[0]
        Py_ssize_t n_taxa = presence.shape[1]
        Py_ssize_t n_nodes = lengths.shape[0]
        double acc

    for s in prange(n_samples, nogil=True):
        pref[s, 0] = 0
        for j in range(n_taxa):
            pref[s, j + 1] = pref[s, j] + presence[s, j]
        acc = 0.0
        for k in range(n_nodes):
            if pref[s, hi[k]] - pref[s, lo[k]] > 0:
                acc = acc + lengths[k]
        out[s] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
def _phydiv_bp(double[:, ::1] counts,
               np.int32_t[::1] lo,
               np.int32_t[::1] hi,
               np.double_t[::1] lengths,
               double[:, ::1] pref,
               bint rooted,
               bint weighted,
               double theta,
               np.double_t[::1] out):
    """Batched generalized phylogenetic diversity over a BPTree's tip-ranges.

    The ``phydiv`` counterpart of :func:`_faith_pd_bp`: an OpenMP-parallel
    (``prange``) reduction that supersedes the sequential post-order
    accumulation in :func:`_traverse_reduce`. For each sample it builds an
    exclusive prefix of the sample's **abundance** in tip order, from which each
    node's descendant-taxa count is the range sum
    ``cbn = pref[hi[k]] - pref[lo[k]]`` and the sample total is ``pref[n_taxa]``
    (the root spans ``[0, n_taxa)``). The four ``phydiv`` modes then reduce to a
    per-node test/weight:

    * unweighted, rooted:   add ``lengths[k]`` iff ``cbn > 0`` (== Faith's PD).
    * unweighted, unrooted: add ``lengths[k]`` iff ``0 < cbn < total`` (drop the
      branches that subtend every taxon).
    * weighted, rooted:     add ``lengths[k] * (cbn/total) ** theta``.
    * weighted, unrooted:   add ``lengths[k] * (2*min(f, 1-f)) ** theta`` where
      ``f = cbn/total`` (the abundance "balance").

    Parameters
    ----------
    counts : memoryview of double, shape (n_samples, n_taxa)
        Abundances, already reordered into ascending tip order.
    lo, hi : memoryview of int32, shape (n_nodes,)
        Half-open descendant-taxa bounds of each node in that column space.
    lengths : memoryview of double, shape (n_nodes,)
        Branch length of each node.
    pref : memoryview of double, shape (n_samples, n_taxa + 1)
        Caller-allocated scratch for the per-sample prefix, indexed by the loop
        variable (never by thread id, so no OpenMP runtime is required).
    rooted : bool
        Whether the root branches are retained.
    weighted : bool
        Whether branch lengths are weighted by relative abundance.
    theta : double
        Weighting exponent; only applied when ``theta < 1.0`` (fully-weighted
        modes pass ``1.0`` and skip the ``pow``, matching the Python reference).
    out : memoryview of double, shape (n_samples,)
        Phylogenetic diversity per sample; written in place.
    """
    cdef:
        Py_ssize_t s, j, k
        Py_ssize_t n_samples = counts.shape[0]
        Py_ssize_t n_taxa = counts.shape[1]
        Py_ssize_t n_nodes = lengths.shape[0]
        double acc, total, cbn, f, g

    for s in prange(n_samples, nogil=True):
        pref[s, 0] = 0.0
        for j in range(n_taxa):
            pref[s, j + 1] = pref[s, j] + counts[s, j]
        total = pref[s, n_taxa]
        if total == 0.0:
            out[s] = 0.0
        else:
            acc = 0.0
            for k in range(n_nodes):
                cbn = pref[s, hi[k]] - pref[s, lo[k]]
                if not weighted:
                    if rooted:
                        if cbn > 0.0:
                            acc = acc + lengths[k]
                    else:
                        if cbn > 0.0 and cbn < total:
                            acc = acc + lengths[k]
                else:
                    f = cbn / total
                    if not rooted:
                        g = 1.0 - f
                        if g < f:
                            f = g
                        f = 2.0 * f
                    if theta < 1.0:
                        f = pow(f, theta)
                    acc = acc + lengths[k] * f
            out[s] = acc
