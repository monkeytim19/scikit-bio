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

### NOTE: some doctext strings are copied and pasted from manuscript
### http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf

import math
from operator import attrgetter

import numpy as np
import array_api_compat as aac

from skbio._base import SkbioObject
from skbio.io.descriptors import Read, Write

from . import _bp_cy
from ._bp_cy import _BPKernel


# Navigation methods implemented by the compiled kernel. ``BPTree`` binds each
# of these onto the instance at construction, so ``bp.close(i)`` resolves in the
# instance ``__dict__`` straight to the kernel's compiled method (~5 ns over a
# direct call) instead of running the class-level forwarding method (~35 ns).
# The class-level methods remain the documented API and serve unbound calls.
_KERNEL_METHODS = (
    "name",
    "length",
    "edge",
    "edge_from_number",
    "close",
    "rmq",
    "rMq",
    "depth",
    "root",
    "parent",
    "is_tip",
    "first_child",
    "last_child",
    "next_sibling",
    "previous_sibling",
    "preorder_rank",
    "preorder_select",
    "postorder_rank",
    "postorder_select",
    "is_ancestor",
    "count",
    "level_ancestor",
    "level_next",
    "lca",
    "deepest_node",
    "height",
)
_get_kernel_methods = attrgetter(*_KERNEL_METHODS)


def _rmm_geometry(n):
    """Block size and height of the range min-max tree for ``n`` parentheses.

    The block size is ``ceil(ln(n) * ln(ln(n)))``, at least 1 (the product is
    not positive for ``n = 2``, a single-node tree). The ``ceil(n / b)`` blocks
    are the leaves of a complete binary tree of height ``ceil(log2(n / b))``.
    """
    b = max(1, math.ceil(math.log(n) * math.log(math.log(n))))
    n_tip = -(-n // b)
    height = (n_tip - 1).bit_length()  # exact ceil(log2(n_tip))
    return b, height


def _build_index(B):
    """Build the navigation index of a balanced-parentheses array.

    NumPy input is handled by the compiled builder (``_bp_cy.build_index``);
    other array backends by the array API implementation
    (:func:`_build_index_xp`). Both produce identical results.
    """
    b, height = _rmm_geometry(B.shape[0])
    if aac.is_numpy_array(B):
        return _bp_cy.build_index(B, b, height)
    return _build_index_xp(B)


def _build_index_xp(B, xp=None):
    """Build the navigation index of a balanced-parentheses array (array API).

    The index is the excess array, the select indexes of opening and closing
    parentheses, and the range min-max (rmM) tree of Navarro and Sadakane
    (http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf) over blocks of the
    excess array.

    The construction only uses array API standard operations (and no in-place
    assignment), so it runs on any array backend, e.g. on a GPU.

    Parameters
    ----------
    B : array of uint8, shape (n,)
        The parentheses, 1 for an opening and 0 for a closing parenthesis.
    xp : namespace, optional
        The array namespace of ``B``. Inferred if not given.

    Returns
    -------
    dict
        ``e_index`` (excess at each position), ``k_index_1`` and ``k_index_0``
        (select indexes: position of the k-th opening / closing parenthesis,
        with position 0 at k = 0 for the bit that is absent there), ``m``,
        ``M`` and ``r`` (minimum excess, maximum excess and rank of the rmM tree
        nodes in heap order), ``b`` (block size) and ``height`` (rmM tree
        height). Arrays are int64 and belong to ``xp``.

    """
    if xp is None:
        xp = aac.array_namespace(B)
    idx = xp.int64
    n = B.shape[0]

    # excess: running count of opening minus closing parentheses. The +1/-1
    # steps are formed in int8 so the only full-size int64 array allocated is
    # the result.
    step = xp.astype(B, xp.int8) * 2 - 1
    e_index = xp.cumulative_sum(step, dtype=idx)
    del step

    # select indexes: _k_index_t[k] is the position of the k-th occurrence of
    # bit t. Position 0 is always included, so the index of the bit that is not
    # at position 0 gets a leading 0.
    opens = B != 0
    zero = xp.zeros(1, dtype=idx)
    k_index_1 = xp.astype(xp.nonzero(opens)[0], idx, copy=False)
    k_index_0 = xp.astype(xp.nonzero(~opens)[0], idx, copy=False)
    if bool(opens[0]):
        k_index_0 = xp.concat([zero, k_index_0])
    else:
        k_index_1 = xp.concat([zero, k_index_1])
    del opens

    # rmM tree geometry: n_tip blocks of b parentheses form the leaves of a
    # complete binary tree of the given height
    b, height = _rmm_geometry(n)
    n_tip = -(-n // b)

    # leaves: per-block minimum / maximum excess (the maximum is floored at 0,
    # its initial value in the scan-based definition) and the rank (number of
    # opening parentheses) before the block, recovered from the excess as
    # rank(i) = (excess(i) + i + 1) / 2
    n_full = n // b
    mins, maxs = [], []
    if n_full:
        blocks = xp.reshape(e_index[: n_full * b], (n_full, b))
        mins.append(xp.min(blocks, axis=1))
        maxs.append(xp.max(blocks, axis=1))
        del blocks
    if n_full < n_tip:
        tail = e_index[n_full * b :]
        mins.append(xp.reshape(xp.min(tail), (1,)))
        maxs.append(xp.reshape(xp.max(tail), (1,)))
    m = xp.concat(mins)
    M = xp.concat(maxs)
    M = xp.maximum(M, xp.zeros_like(M))
    starts = xp.arange(1, n_tip, dtype=idx) * b
    r = xp.concat([zero, (e_index[b - 1 : (n_tip - 1) * b : b] + starts) // 2])

    # internal nodes, bottom-up one level at a time. A node's children are the
    # pairs of the level below; a node without a left child stays 0, and one
    # without a right child copies its left child.
    levels_m, levels_M, levels_r = [m], [M], [r]
    n_child = n_tip
    for lvl in range(height - 1, -1, -1):
        width = 1 << lvl
        pad = 2 * width - n_child
        if pad:
            fill = xp.zeros(pad, dtype=idx)
            m, M, r = (xp.concat([x, fill]) for x in (m, M, r))
        m, M, r = (xp.reshape(x, (width, 2)) for x in (m, M, r))
        pos = xp.arange(width, dtype=idx) * 2
        has_left = pos < n_child
        has_right = (pos + 1) < n_child
        nil = xp.zeros(width, dtype=idx)
        m = xp.where(
            has_right,
            xp.minimum(m[:, 0], m[:, 1]),
            xp.where(has_left, m[:, 0], nil),
        )
        M = xp.where(
            has_right,
            xp.maximum(M[:, 0], M[:, 1]),
            xp.where(has_left, M[:, 0], nil),
        )
        r = xp.where(has_left, r[:, 0], nil)
        levels_m.append(m)
        levels_M.append(M)
        levels_r.append(r)
        n_child = width

    # heap order is root level first
    return {
        "e_index": e_index,
        "k_index_0": k_index_0,
        "k_index_1": k_index_1,
        "m": xp.concat(levels_m[::-1]),
        "M": xp.concat(levels_M[::-1]),
        "r": xp.concat(levels_r[::-1]),
        "b": b,
        "height": height,
    }


def _check_array(arr, dtype, name, size=None):
    """Validate a 1-D NumPy array of an exact dtype, made C-contiguous."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, not {type(arr).__name__}.")
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, not {arr.ndim}-D.")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}, not {arr.dtype}.")
    if size is not None and arr.shape[0] != size:
        raise ValueError(
            f"{name} must have one entry per parenthesis ({size}), not {arr.shape[0]}."
        )
    return np.ascontiguousarray(arr)


def _readonly(arr):
    arr.flags.writeable = False
    return arr


class BPTree(SkbioObject):
    """A balanced parentheses succinct data structure tree representation.

    The basis for this implementation is the data structure described by
    Cordova and Navarro [1]. In some instances, some docstring text was copied
    verbatim from the manuscript. This does not implement the bucket-based
    trees.

    A node in this data structure is represented by 2 bits, an open parenthesis
    and a close parenthesis. The implementation uses a NumPy uint8 type where
    an open parenthesis is a 1 and a close is a 0. In general, operations on
    this tree are best suited for passing in the opening parenthesis index, so
    for instance, if you'd like to use BPTree.is_tip to determine if a node is a
    leaf, the operation is defined only for using the opening parenthesis. At
    this time, there is some ambiguity over what methods can handle a closing
    parenthesis.

    Node attributes, such as names, are stored external to this data structure.

    The motivator for this data structure is pure performance both in space and
    time. As such, there is minimal sanity checking. It is advised to use this
    structure with care, and ideally within a framework which can assure
    sanity.

    Parameters
    ----------
    B : numpy.ndarray of uint8
        The parentheses bit array encoding the tree topology, where an open
        parenthesis is 1 and a close parenthesis is 0. A bool array is also
        accepted, and is viewed as uint8.
    lengths : numpy.ndarray of float64, optional
        Branch length per parenthesis (read at opening parentheses). Defaults
        to 0.
    names : numpy.ndarray of object, optional
        Node name per parenthesis (read at opening parentheses). Defaults to
        None.
    edges : numpy.ndarray of int32, optional
        Edge number per parenthesis (read at opening parentheses). Defaults to
        0, without an edge-number lookup.

    Attributes
    ----------
    data : numpy.ndarray of uint8
        The parentheses bit array encoding the tree topology, where an open
        parenthesis is 1 and a close parenthesis is 0.

    Notes
    -----
    The tree's arrays, including the navigation index built from ``data``, are
    held by this Python object. Operations on them are computed by a compiled
    (Cython) engine: per-node navigation methods (e.g., :meth:`parent`,
    :meth:`lca`) are bound directly to the engine when the tree is created, so
    they run at compiled speed.

    References
    ----------
    [1] http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    """

    default_write_format = "newick"

    # ``read`` and ``write`` are provided by the ``skbio.io`` registry via the
    # descriptor protocol, as for ``TreeNode`` and other SkbioObjects.
    read = Read()
    write = Write()

    def __init__(self, B, lengths=None, names=None, edges=None):
        # a bool array has the same one-byte 0/1 layout, which the compiled
        # engine has always accepted: view it as uint8, without a copy
        if isinstance(B, np.ndarray) and B.dtype == np.bool_:
            B = B.view(np.uint8)
        B = _check_array(B, np.uint8, "B")
        size = B.shape[0]
        if size == 0:
            raise ValueError("The topology array is empty.")

        # the tree is only valid if it is balanced (equal opens and closes)
        if int(B.sum()) * 2 != size:
            raise ValueError(
                "The topology array is unbalanced; it must contain an equal "
                "number of opening (1) and closing (0) parentheses."
            )

        if names is None:
            names = np.full(size, None, dtype=object)
        else:
            names = _check_array(names, object, "names", size)

        if lengths is None:
            lengths = np.zeros(size, dtype=np.float64)
        else:
            lengths = _check_array(lengths, np.float64, "lengths", size)

        if edges is None:
            edges = np.full(size, 0, dtype=np.int32)
            edge_lookup = None
        else:
            edges = _check_array(edges, np.int32, "edges", size)
            edge_lookup = self._edge_lookup_for(B, edges)

        index = _build_index(B)
        self._data = B
        self._size = size
        self._names = names
        self._lengths = lengths
        self._edges = edges
        self._edge_lookup = edge_lookup
        self._e_index = _readonly(index["e_index"])
        self._k_index_0 = _readonly(index["k_index_0"])
        self._k_index_1 = _readonly(index["k_index_1"])
        self._m = _readonly(index["m"])
        self._M = _readonly(index["M"])
        self._r = _readonly(index["r"])
        self._b = index["b"]
        self._height = index["height"]

        self._kernel = _BPKernel(
            B,
            self._e_index,
            self._k_index_0,
            self._k_index_1,
            self._m,
            self._M,
            self._r,
            self._b,
            self._height,
            names,
            lengths,
            edges,
            edge_lookup,
        )
        self.__dict__.update(zip(_KERNEL_METHODS, _get_kernel_methods(self._kernel)))

    @property
    def data(self):
        """The parentheses bit array (1 = open, 0 = close)."""
        return self._data

    @staticmethod
    def _edge_lookup_for(B, edges):
        opens = edges[B == 1]
        if opens.size and (opens.min() < 0 or opens.max() >= B.shape[0]):
            raise ValueError("Edge numbers must be in [0, %d)." % B.shape[0])
        return _bp_cy.edge_lookup(B, edges)

    # ------------------------------------------------------------------
    # Construction, conversion and serialization
    # ------------------------------------------------------------------

    def to_npz(self, file):
        """Save the tree to a compressed NumPy ``.npz`` archive.

        The parentheses bit array, node names, and branch lengths are stored;
        edge numbers are not. This is a lightweight binary dump, distinct from
        the registry-based :meth:`write` (which defaults to the ``newick``
        format).

        Parameters
        ----------
        file : str or file-like object
            Path or open file handle to write to.

        See Also
        --------
        from_npz
        write

        """
        np.savez_compressed(
            file, names=self._names, lengths=self._lengths, B=self._data
        )

    @classmethod
    def from_npz(cls, file):
        """Load a tree from a NumPy ``.npz`` archive written by ``to_npz``.

        Parameters
        ----------
        file : str or file-like object
            Path or open file handle to read from.

        Returns
        -------
        BPTree
            The reconstructed tree, with node names and branch lengths.

        Warnings
        --------
        This method calls :func:`numpy.load` with ``allow_pickle=True`` in
        order to restore the object-dtype ``names`` array. Loading a pickled
        array can execute arbitrary code, so only read ``.npz`` archives from
        trusted sources.

        See Also
        --------
        to_npz
        read

        """
        # names is an object array pickled by ``to_npz``, so unpickling must be
        # allowed to restore it
        data = np.load(file, allow_pickle=True)
        return cls(data["B"], names=data["names"], lengths=data["lengths"])

    @classmethod
    def from_treenode(cls, tree):
        """Construct a BPTree from a :class:`~skbio.tree.TreeNode`.

        Parameters
        ----------
        tree : skbio.tree.TreeNode
            The tree to convert.

        Returns
        -------
        BPTree
            The tree represented in balanced-parentheses form.

        See Also
        --------
        skbio.tree.TreeNode.from_bptree

        """
        topo, names, lengths, edges = _bp_cy.from_treenode_arrays(tree)
        return cls(topo, names=names, lengths=lengths, edges=edges)

    def to_array(self):
        """Return an array representation of the tree.

        This mirrors :meth:`skbio.tree.TreeNode.to_array`.

        Returns
        -------
        dict
            Dictionary with keys ``'child_index'``, ``'length'``,
            ``'id_index'`` and ``'name'``.

        See Also
        --------
        skbio.tree.TreeNode.to_array

        """
        return _bp_cy.to_array(self._kernel)

    def _to_node_arrays(self):
        """Return preorder per-node arrays for :meth:`TreeNode.from_bptree`.

        The balanced-parentheses traversal is performed in a single compiled
        pass so that :meth:`skbio.tree.TreeNode.from_bptree` pays no per-node
        Python/C call overhead; the pure-Python side is then left with only the
        ``TreeNode`` object assembly.

        Returns
        -------
        tuple of numpy.ndarray
            ``(name, length, edge, parent)``, each indexed by preorder position.
            ``name`` is dtype ``object``, ``length`` is ``float64`` and ``edge``
            is ``int32`` (mirroring :meth:`name`, :meth:`length`, :meth:`edge`).
            ``parent`` (dtype ``intp``) holds the preorder index of each node's
            parent, or ``-1`` for the root.

        See Also
        --------
        skbio.tree.TreeNode.from_bptree

        """
        return _bp_cy.to_node_arrays(self._kernel)

    def __reduce__(self):
        return (BPTree, (self._data, self._lengths, self._names))

    # ------------------------------------------------------------------
    # Node attributes
    # ------------------------------------------------------------------

    def set_names(self, names):
        """Replace the node names.

        Parameters
        ----------
        names : numpy.ndarray of object
            Node name per parenthesis (read at opening parentheses).

        """
        names = _check_array(names, object, "names", self._size)
        self._names = self._kernel._names = names

    def set_lengths(self, lengths):
        """Replace the branch lengths.

        Parameters
        ----------
        lengths : numpy.ndarray of float64
            Branch length per parenthesis (read at opening parentheses).

        """
        lengths = _check_array(lengths, np.float64, "lengths", self._size)
        self._lengths = self._kernel._lengths = lengths

    def set_edges(self, edges):
        """Replace the edge numbers and rebuild the edge-number lookup.

        Parameters
        ----------
        edges : numpy.ndarray of int32
            Edge number per parenthesis (read at opening parentheses).

        """
        edges = _check_array(edges, np.int32, "edges", self._size)
        edge_lookup = self._edge_lookup_for(self._data, edges)
        self._edges = self._kernel._edges = edges
        self._edge_lookup = self._kernel._edge_lookup = edge_lookup

    def name(self, i):
        """Name of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        str or None
            The name of node ``i``.
        """
        return self._kernel.name(i)

    def length(self, i):
        """Branch length of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        float
            The length of the branch leading to node ``i``.
        """
        return self._kernel.length(i)

    def edge(self, i):
        """Edge number of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            The edge number of the branch leading to node ``i``.
        """
        return self._kernel.edge(i)

    def edge_from_number(self, n):
        """Index of the node carrying an edge number.

        Parameters
        ----------
        n : int
            The edge number to look up.

        Returns
        -------
        int
            Index of the node whose edge number is ``n``.
        """
        return self._kernel.edge_from_number(n)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def __len__(self):
        """The number of nodes in the tree."""
        return self._size // 2

    def __str__(self):
        """Return a concise summary of the tree.

        Implements the abstract ``__str__`` required of scikit-bio objects.
        """
        return self.__repr__()

    def __repr__(self):
        """Returns summary of the tree.

        Returns
        -------
        str
            A summary of this tree

        Notes
        -----
        This method returns the name of the node and a count of tips and the
        number of internal nodes in the tree.
        """
        total_nodes = len(self)
        tip_count = self.count(tips=True)

        return "<BPTree, name: %s, internal node count: %d, tips count: %d>" % (
            self.name(0),
            total_nodes - tip_count,
            tip_count,
        )

    # ------------------------------------------------------------------
    # Navigation (computed by the compiled kernel)
    # ------------------------------------------------------------------

    def rmq(self, i, j):
        """The leftmost minimum excess in i -> j.

        Parameters
        ----------
        i : int
            Start position (inclusive).
        j : int
            End position (inclusive).

        Returns
        -------
        int
            The leftmost position in ``[i, j]`` with the minimum excess.
        """
        return self._kernel.rmq(i, j)

    def rMq(self, i, j):
        """The leftmost maximum excess in i -> j.

        Parameters
        ----------
        i : int
            Start position (inclusive).
        j : int
            End position (inclusive).

        Returns
        -------
        int
            The leftmost position in ``[i, j]`` with the maximum excess.
        """
        return self._kernel.rMq(i, j)

    def close(self, i):
        """The position of the closing parenthesis that matches B[i].

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Position of the matching closing parenthesis.
        """
        return self._kernel.close(i)

    def depth(self, i):
        """The depth of given node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Depth of node relative to the root of tree.
        """
        return self._kernel.depth(i)

    def root(self):
        """The index of the root node of the tree."""
        return self._kernel.root()

    def parent(self, i):
        """The parent of node.

        Parameters
        ----------
        i : int
            Index of node to evaluate.

        Returns
        -------
        int
            Index of parent node. Returns -1 if node does not have a parent.
        """
        return self._kernel.parent(i)

    def is_tip(self, i):
        """Whether the node is a tip of a tree.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        bool
            Whether the node is a tip of a tree or not.
        """
        return self._kernel.is_tip(i)

    def first_child(self, i):
        """Index of the first (leftmost) child of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the first child of node ``i``, or 0 if ``i`` is a tip
            (0 is the root, which can never be a child).

        See Also
        --------
        last_child
        skbio.tree.TreeNode

        Notes
        -----
        Returns an integer index into the parentheses bit array, not a node
        object. Corresponds to accessing ``children[0]`` on a
        :class:`~skbio.tree.TreeNode`.
        """
        return self._kernel.first_child(i)

    def last_child(self, i):
        """Index of the last (rightmost) child of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the last child of node ``i``, or 0 if ``i`` is a tip
            (0 is the root, which can never be a child).

        See Also
        --------
        first_child
        skbio.tree.TreeNode

        Notes
        -----
        Returns an integer index into the parentheses bit array, not a node
        object. Corresponds to accessing ``children[-1]`` on a
        :class:`~skbio.tree.TreeNode`.
        """
        return self._kernel.last_child(i)

    def mincount(self, i, j):
        """Number of occurrences of the minimum in excess(i), ..., excess(j)."""
        excess, counts = np.unique(self._e_index[i : j + 1], return_counts=True)
        return counts[excess.argmin()]

    def minselect(self, i, j, q):
        """Position of the qth minimum in excess(i), ..., excess(j)."""
        counts = self._e_index[i : j + 1]
        index = counts == counts.min()

        if index.sum() < q:
            return None
        else:
            return i + index.nonzero()[0][q - 1]

    def next_sibling(self, i):
        """Index of the next (right) sibling of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the next sibling of node ``i``, or 0 if ``i`` has no
            next sibling (0 is the root, which can never be a sibling).

        See Also
        --------
        previous_sibling
        skbio.tree.TreeNode.siblings

        Notes
        -----
        Returns the integer index of a single sibling, unlike
        :meth:`~skbio.tree.TreeNode.siblings`, which returns a list of all
        sibling nodes.
        """
        return self._kernel.next_sibling(i)

    def previous_sibling(self, i):
        """Index of the previous (left) sibling of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the previous sibling of node ``i``, or 0 if ``i`` has no
            previous sibling (0 is the root, which can never be a sibling).

        See Also
        --------
        next_sibling
        skbio.tree.TreeNode.siblings

        Notes
        -----
        Returns the integer index of a single sibling, unlike
        :meth:`~skbio.tree.TreeNode.siblings`, which returns a list of all
        sibling nodes.
        """
        return self._kernel.previous_sibling(i)

    def preorder_rank(self, i):
        """Preorder rank of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            The position of node ``i`` in a preorder traversal of the tree.

        See Also
        --------
        preorder_select
        skbio.tree.TreeNode.preorder

        Notes
        -----
        Returns the node's integer position in preorder, not a node object.
        This differs from :meth:`~skbio.tree.TreeNode.preorder`, which yields
        the nodes of the tree in preorder. The inverse of
        :meth:`preorder_select`.
        """
        return self._kernel.preorder_rank(i)

    def preorder_select(self, k):
        """Index of the node with a given preorder rank.

        Parameters
        ----------
        k : int
            Preorder rank to look up.

        Returns
        -------
        int
            Index of the node whose preorder rank is ``k``.

        See Also
        --------
        preorder_rank
        skbio.tree.TreeNode.preorder

        Notes
        -----
        The inverse of :meth:`preorder_rank`. Returns an integer index into
        the parentheses bit array, not a node object.
        :meth:`~skbio.tree.TreeNode.preorder` yields the nodes in this order.
        """
        return self._kernel.preorder_select(k)

    def postorder_rank(self, i):
        """Postorder rank of a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            The position of node ``i`` in a postorder traversal of the tree.

        See Also
        --------
        postorder_select
        skbio.tree.TreeNode.postorder

        Notes
        -----
        Returns the node's integer position in postorder, not a node object.
        This differs from :meth:`~skbio.tree.TreeNode.postorder`, which yields
        the nodes of the tree in postorder. The inverse of
        :meth:`postorder_select`.
        """
        return self._kernel.postorder_rank(i)

    def postorder_select(self, k):
        """Index of the node with a given postorder rank.

        Parameters
        ----------
        k : int
            Postorder rank to look up.

        Returns
        -------
        int
            Index of the node whose postorder rank is ``k``.

        See Also
        --------
        postorder_rank
        skbio.tree.TreeNode.postorder

        Notes
        -----
        The inverse of :meth:`postorder_rank`. Returns an integer index into
        the parentheses bit array, not a node object.
        :meth:`~skbio.tree.TreeNode.postorder` yields the nodes in this order.
        """
        return self._kernel.postorder_select(k)

    def is_ancestor(self, i, j):
        """Whether a node is an ancestor of another node.

        Parameters
        ----------
        i : int
            A node index
        j : int
            A node index

        Note
        ----
        False is returned if i == j. A node cannot be an ancestor of itself.

        Returns
        -------
        bool
            True if i is an ancestor of j, False otherwise.
        """
        return self._kernel.is_ancestor(i, j)

    def count(self, i=0, tips=False):
        """Get the count of nodes in the subtree rooted at a node.

        Parameters
        ----------
        i : int, optional
            Index of the node whose subtree is evaluated. Defaults to the root
            (``0``), i.e., the whole tree.
        tips : bool, optional
            If True, only count the tips (leaves) in the subtree (default:
            False).

        Returns
        -------
        int
            The number of nodes (or tips, if ``tips`` is True) in the subtree
            rooted at node ``i``, including ``i`` itself.

        See Also
        --------
        skbio.tree.TreeNode.count

        Notes
        -----
        Returns a node count, not a subtree.

        """
        return self._kernel.count(i, tips)

    def level_ancestor(self, i, d):
        """Index of the ancestor a given number of levels above a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.
        d : int
            Number of levels to ascend toward the root.

        Returns
        -------
        int
            Index of the ancestor ``d`` levels above node ``i``, or -1 if
            ``d`` is not positive.

        See Also
        --------
        level_next
        skbio.tree.TreeNode.ancestors

        Notes
        -----
        Returns an integer index into the parentheses bit array, not a node
        object. :meth:`~skbio.tree.TreeNode.ancestors` returns the full list
        of ancestor nodes from a node toward the root.
        """
        return self._kernel.level_ancestor(i, d)

    def level_next(self, i):
        """Index of the next node at the same depth.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the next node at the same depth as node ``i``, or -1 if
            there is no such node.

        See Also
        --------
        level_ancestor
        skbio.tree.TreeNode.levelorder

        Notes
        -----
        Returns an integer index into the parentheses bit array, not a node
        object. :meth:`~skbio.tree.TreeNode.levelorder` traverses all nodes
        depth by depth.
        """
        return self._kernel.level_next(i)

    def lca(self, i, j):
        """The lowest common ancestor of two nodes.

        Parameters
        ----------
        i : int
            A node index to evaluate
        j : int
            A node index to evaluate

        Returns
        -------
        int
           The index of the lowest common ancestor
        """
        return self._kernel.lca(i, j)

    def deepest_node(self, i):
        """Index of the deepest node descending from a node.

        Parameters
        ----------
        i : int
            Index of the node to evaluate.

        Returns
        -------
        int
            Index of the deepest (most distant) tip descending from node ``i``.

        See Also
        --------
        height
        skbio.tree.TreeNode.height

        Notes
        -----
        Returns an integer index into the parentheses bit array, not a node
        object. This is the tip that :meth:`~skbio.tree.TreeNode.height`
        returns as the second element of its ``(height, tip)`` result.
        """
        return self._kernel.deepest_node(i)

    def height(self, i):
        """The height of node i with respect to its deepest descendent

        Parameters
        ----------
        i : int
            The node to evaluate

        Notes
        -----
        Height is in terms of number of edges, not in terms of branch length

        Returns
        -------
        int
            The number of edges between node i and its deepest node
        """
        return self._kernel.height(i)

    # ------------------------------------------------------------------
    # Whole-tree operations
    # ------------------------------------------------------------------

    def shear(self, tips):
        """Remove all nodes from the tree except tips and ancestors of tips.

        Parameters
        ----------
        tips : set of str
            The set of tip names to retain

        Returns
        -------
        BPTree
            A new BPTree corresponding to only the described tips and their
            ancestors.
        """
        if not isinstance(tips, set):
            raise TypeError("tips must be a set, not %s." % type(tips).__name__)
        mask, count = _bp_cy.shear_mask(self._kernel, tips)
        if count == 0:
            raise ValueError("No requested tips found")
        return self._from_mask(mask, self._lengths)

    def collapse(self):
        """Collapse single-child internal nodes.

        Every internal node with exactly one child is removed from the tree,
        and the removed node's branch length is added to that of its single
        child so that root-to-tip path lengths are preserved. The root and all
        tips are always retained, as are internal nodes with two or more
        children.

        Returns
        -------
        BPTree
            A new tree with all single-child internal nodes removed. Node names
            and the merged branch lengths are carried over; edge numbers are
            not retained.

        Notes
        -----
        A new ``BPTree`` is returned; the original tree is not modified.

        """
        mask, lengths = _bp_cy.collapse_mask(self._kernel)
        return self._from_mask(mask, lengths)

    def _from_mask(self, mask, lengths):
        """A new tree of the positions set in ``mask``."""
        keep = mask.view(bool)
        return BPTree(self._data[keep], names=self._names[keep], lengths=lengths[keep])
