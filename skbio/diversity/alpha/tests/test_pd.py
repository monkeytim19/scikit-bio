# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.txt, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import TestCase, main
import warnings
from io import StringIO
import os

import numpy as np
import pandas as pd

from skbio import TreeNode
from skbio.util import get_data_path
from skbio.tree import BPTree, DuplicateNodeError, MissingNodeError
from skbio.diversity import alpha_diversity
from skbio.diversity.alpha import faith_pd, phydiv
from skbio.diversity.alpha._pd import NUMBA_AVAILABLE

# engines to exercise; numba only if the optional dependency is importable
_BP_ENGINES = ["cython"] + (["numba"] if NUMBA_AVAILABLE else [])


class FaithPDTests(TestCase):

    def setUp(self):
        self.counts = np.array([0, 1, 1, 4, 2, 5, 2, 4, 1, 2])
        self.b1 = np.array([[1, 3, 0, 1, 0],
                            [0, 2, 0, 4, 4],
                            [0, 0, 6, 2, 1],
                            [0, 0, 1, 1, 1],
                            [2, 0, 3, 0, 0]])
        self.sids1 = list('ABCDE')
        self.oids1 = ['OTU%d' % i for i in range(1, 6)]
        self.t1 = TreeNode.read(StringIO(
            '(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):'
            '0.0,(OTU4:0.75,OTU5:0.75):1.25):0.0)root;'))
        self.t1_w_extra_tips = TreeNode.read(
           StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                    '0.75,(OTU5:0.25,(OTU6:0.5,OTU7:0.5):0.5):0.5):1.25):0.0'
                    ')root;'))

    def test_faith_pd(self):
        # expected results derived from QIIME 1.9.1, which
        # is a completely different implementation skbio's initial
        # phylogenetic diversity implementation
        actual = faith_pd(self.b1[0], self.oids1, self.t1)
        expected = 4.5
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[0], self.oids1, self.t1, validate=False)
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[1], self.oids1, self.t1)
        expected = 4.75
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[2], self.oids1, self.t1)
        expected = 4.75
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[3], self.oids1, self.t1)
        expected = 4.75
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[4], self.oids1, self.t1)
        expected = 3.0
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_extra_tips(self):
        # results are the same despite presences of unobserved tips in tree
        actual = faith_pd(self.b1[0], self.oids1, self.t1_w_extra_tips)
        expected = faith_pd(self.b1[0], self.oids1, self.t1)
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[1], self.oids1, self.t1_w_extra_tips)
        expected = faith_pd(self.b1[1], self.oids1, self.t1)
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[2], self.oids1, self.t1_w_extra_tips)
        expected = faith_pd(self.b1[2], self.oids1, self.t1)
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[3], self.oids1, self.t1_w_extra_tips)
        expected = faith_pd(self.b1[3], self.oids1, self.t1)
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd(self.b1[4], self.oids1, self.t1_w_extra_tips)
        expected = 3.0
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_none_observed(self):
        actual = faith_pd(np.array([], dtype=int),
                          np.array([], dtype=int),
                          self.t1)
        expected = 0.0
        self.assertAlmostEqual(actual, expected)
        actual = faith_pd([0, 0, 0, 0, 0], self.oids1, self.t1)
        expected = 0.0
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_all_observed(self):
        actual = faith_pd([1, 1, 1, 1, 1], self.oids1, self.t1)
        expected = sum(n.length for n in self.t1.traverse()
                       if n.length is not None)
        self.assertAlmostEqual(actual, expected)

        actual = faith_pd([1, 2, 3, 4, 5], self.oids1, self.t1)
        expected = sum(n.length for n in self.t1.traverse()
                       if n.length is not None)
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_one_observed(self):
        actual = faith_pd([1, 0, 0, 0, 0], self.oids1, self.t1)
        expected = 2.0
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_minimal(self):
        # two tips
        tree = TreeNode.read(StringIO('(OTU1:0.25, OTU2:0.25)root;'))
        actual = faith_pd([1, 0], ['OTU1', 'OTU2'], tree)
        expected = 0.25
        self.assertEqual(actual, expected)

    def test_faith_pd_qiime_tiny_test(self):
        # the following table and tree are derived from the QIIME 1.9.1
        # "tiny-test" data
        tt_table_fp = get_data_path(
            os.path.join('qiime-191-tt', 'otu-table.tsv'), 'data')
        tt_tree_fp = get_data_path(
            os.path.join('qiime-191-tt', 'tree.nwk'), 'data')

        self.q_table = pd.read_csv(tt_table_fp, sep='\t', skiprows=1,
                                   index_col=0)
        self.q_tree = TreeNode.read(tt_tree_fp)

        expected_fp = get_data_path(
            os.path.join('qiime-191-tt', 'faith-pd.txt'), 'data')
        expected = pd.read_csv(expected_fp, sep='\t', index_col=0)
        for sid in self.q_table.columns:
            actual = faith_pd(self.q_table[sid],
                              taxa=self.q_table.index,
                              tree=self.q_tree)
            self.assertAlmostEqual(actual, expected['PD_whole_tree'][sid])

    def test_faith_pd_root_not_observed(self):
        # expected values computed by hand
        tree = TreeNode.read(
            StringIO('((OTU1:0.1, OTU2:0.2):0.3, (OTU3:0.5, OTU4:0.7):1.1)'
                     'root;'))
        taxa = ['OTU%d' % i for i in range(1, 5)]
        # root node not observed, but branch between (OTU1, OTU2) and root
        # is considered observed
        actual = faith_pd([1, 1, 0, 0], taxa, tree)
        expected = 0.6
        self.assertAlmostEqual(actual, expected)

        # root node not observed, but branch between (OTU3, OTU4) and root
        # is considered observed
        actual = faith_pd([0, 0, 1, 1], taxa, tree)
        expected = 2.3
        self.assertAlmostEqual(actual, expected)

    def test_faith_pd_invalid_input(self):
        # tree has duplicated tip ids
        t = TreeNode.read(
            StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU2:0.75):1.25):0.0)root;'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(DuplicateNodeError, faith_pd, counts, taxa,
                          t)

        # unrooted tree as input
        t = TreeNode.read(StringIO('((OTU1:0.1, OTU2:0.2):0.3, OTU3:0.5,'
                                   'OTU4:0.7);'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # taxa has duplicated ids
        t = TreeNode.read(
            StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU5:0.75):1.25):0.0)root;'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU2']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # len of vectors not equal
        t = TreeNode.read(
            StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU5:0.75):1.25):0.0)root;'))
        counts = [1, 2]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # negative counts
        t = TreeNode.read(
            StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU5:0.75):1.25):0.0)root;'))
        counts = [1, 2, -3]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # tree with no branch lengths
        t = TreeNode.read(
            StringIO('((((OTU1,OTU2),OTU3)),(OTU4,OTU5));'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # tree missing some branch lengths
        t = TreeNode.read(
            StringIO('(((((OTU1,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU5:0.75):1.25):0.0)root;'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU3']
        self.assertRaises(ValueError, faith_pd, counts, taxa, t)

        # taxa not present in tree
        t = TreeNode.read(
            StringIO('(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:'
                     '0.75,OTU5:0.75):1.25):0.0)root;'))
        counts = [1, 2, 3]
        taxa = ['OTU1', 'OTU2', 'OTU42']
        self.assertRaises(MissingNodeError, faith_pd, counts, taxa, t)


class FaithPDBPTreeEngineTests(TestCase):
    """faith_pd over a BPTree, across the cython and numba engines, must agree
    with the TreeNode reference and preserve its error behavior."""

    def setUp(self):
        self.oids1 = ["OTU%d" % i for i in range(1, 6)]
        self.t1 = TreeNode.read(
            StringIO(
                "(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):"
                "0.0,(OTU4:0.75,OTU5:0.75):1.25):0.0)root;"
            )
        )
        self.b1 = np.array(
            [
                [1, 3, 0, 1, 0],
                [0, 2, 0, 4, 4],
                [0, 0, 6, 2, 1],
                [0, 0, 1, 1, 1],
                [2, 0, 3, 0, 0],
            ]
        )

    def _assert_match(self, counts, taxa, tree, **kw):
        ref = faith_pd(counts, taxa, tree, **kw)
        bp = BPTree.from_treenode(tree)
        for eng in _BP_ENGINES:
            got = faith_pd(counts, taxa, bp, engine=eng, **kw)
            self.assertAlmostEqual(got, ref, places=10, msg="engine=%s" % eng)
        return ref

    # --- topology builders (rooted, bifurcating, all branches have length) ---
    @staticmethod
    def _balanced(n):
        nodes = [TreeNode(name="t%d" % i, length=1.0) for i in range(n)]
        while len(nodes) > 1:
            nxt = [
                TreeNode(children=[nodes[i], nodes[i + 1]], length=1.0)
                for i in range(0, len(nodes) - 1, 2)
            ]
            if len(nodes) % 2:
                nxt.append(nodes[-1])
            nodes = nxt
        nodes[0].length = 0.0
        return nodes[0]

    @staticmethod
    def _caterpillar(n):
        node = TreeNode(
            children=[TreeNode(name="t0", length=1.0), TreeNode(name="t1", length=1.0)],
            length=1.0,
        )
        for i in range(2, n):
            node = TreeNode(
                children=[node, TreeNode(name="t%d" % i, length=1.0)], length=1.0
            )
        node.length = 0.0
        return node

    @staticmethod
    def _random(n, rng):
        nodes = [
            TreeNode(name="t%d" % i, length=round(float(rng.uniform(0.1, 1.0)), 3))
            for i in range(n)
        ]
        while len(nodes) > 1:
            a = nodes.pop(int(rng.integers(0, len(nodes))))
            b = nodes.pop(int(rng.integers(0, len(nodes))))
            nodes.append(
                TreeNode(
                    children=[a, b], length=round(float(rng.uniform(0.1, 1.0)), 3)
                )
            )
        nodes[0].length = 0.0
        return nodes[0]

    def test_engines_match_reference(self):
        for row in self.b1:
            self._assert_match(row, self.oids1, self.t1)
            self._assert_match(row, self.oids1, self.t1, validate=False)
        # none / all / one observed
        self._assert_match([0, 0, 0, 0, 0], self.oids1, self.t1)
        self._assert_match([1, 1, 1, 1, 1], self.oids1, self.t1)
        self._assert_match([1, 0, 0, 0, 0], self.oids1, self.t1)

    def test_extra_tips(self):
        # tree tips are a superset of taxa
        t = TreeNode.read(
            StringIO(
                "(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:"
                "0.75,(OTU5:0.25,(OTU6:0.5,OTU7:0.5):0.5):0.5):1.25):0.0)root;"
            )
        )
        for row in self.b1:
            self._assert_match(row, self.oids1, t)

    def test_root_not_observed(self):
        tree = TreeNode.read(
            StringIO("((OTU1:0.1, OTU2:0.2):0.3, (OTU3:0.5, OTU4:0.7):1.1)root;")
        )
        taxa = ["OTU%d" % i for i in range(1, 5)]
        self.assertAlmostEqual(self._assert_match([1, 1, 0, 0], taxa, tree), 0.6)
        self.assertAlmostEqual(self._assert_match([0, 0, 1, 1], taxa, tree), 2.3)

    def test_topologies_and_subset(self):
        rng = np.random.default_rng(42)
        trees = [self._balanced(64), self._caterpillar(50), self._random(300, rng)]
        for tree in trees:
            tips = [t.name for t in tree.tips()]
            # all tips
            self._assert_match(rng.integers(0, 4, len(tips)), tips, tree)
            # strict subset of tips (exercises the searchsorted compaction)
            k = max(2, len(tips) // 2)
            sub = list(rng.choice(tips, k, replace=False))
            self._assert_match(rng.integers(0, 4, k), sub, tree)
            # sparse
            self._assert_match(
                (rng.random(len(tips)) < 0.1).astype(int), tips, tree
            )

    def test_alpha_diversity_engines(self):
        ref = alpha_diversity(
            "faith_pd", self.b1, ids=list("ABCDE"), taxa=self.oids1, tree=self.t1
        )
        bp = BPTree.from_treenode(self.t1)
        for eng in _BP_ENGINES:
            got = alpha_diversity(
                "faith_pd", self.b1, ids=list("ABCDE"), taxa=self.oids1,
                tree=bp, engine=eng,
            )
            pd.testing.assert_series_equal(ref, got)

    def test_treenode_noncython_engine_warns_and_falls_back(self):
        # an explicit non-cython engine on a TreeNode warns once and returns
        # the cython answer (faith_pd and alpha_diversity)
        ref = faith_pd(self.b1[0], self.oids1, self.t1)
        with self.assertWarnsRegex(UserWarning, "BPTree"):
            got = faith_pd(self.b1[0], self.oids1, self.t1, engine="numba")
        self.assertAlmostEqual(got, ref, places=10)
        with self.assertWarnsRegex(UserWarning, "BPTree"):
            got = alpha_diversity(
                "faith_pd", self.b1, taxa=self.oids1, tree=self.t1, engine="numba"
            )
        pd.testing.assert_series_equal(
            alpha_diversity("faith_pd", self.b1, taxa=self.oids1, tree=self.t1), got
        )

    def test_global_engine_default_does_not_warn_on_treenode(self):
        # a global set_config("engine", ...) must degrade silently: only an
        # explicit per-call engine= that cannot be served is worth warning
        # about, else every pre-existing TreeNode call in a script warns.
        from skbio import get_config, set_config

        prev = get_config("engine")
        set_config("engine", "numba")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                got = faith_pd(self.b1[0], self.oids1, self.t1)
                alpha_diversity("faith_pd", self.b1, taxa=self.oids1,
                                tree=self.t1)
            self.assertEqual([str(w.message) for w in caught], [])
            # ... while an explicit request still does warn
            with self.assertWarnsRegex(UserWarning, "BPTree"):
                faith_pd(self.b1[0], self.oids1, self.t1, engine="numba")
        finally:
            set_config("engine", prev)
        self.assertAlmostEqual(got, faith_pd(self.b1[0], self.oids1, self.t1),
                               places=10)

    def test_unsupported_engine_rejected(self):
        bp = BPTree.from_treenode(self.t1)
        with self.assertRaises(ValueError):
            faith_pd(self.b1[0], self.oids1, bp, engine="nonesuch")

    def test_error_parity(self):
        for eng in _BP_ENGINES:
            # duplicated tip ids -> DuplicateNodeError
            t = TreeNode.read(
                StringIO(
                    "(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:"
                    "0.75,OTU2:0.75):1.25):0.0)root;"
                )
            )
            self.assertRaises(
                DuplicateNodeError, faith_pd, [1, 2, 3], ["OTU1", "OTU2", "OTU3"],
                BPTree.from_treenode(t), engine=eng,
            )
            # unrooted (trifurcating root) -> ValueError
            t = TreeNode.read(
                StringIO("((OTU1:0.1, OTU2:0.2):0.3, OTU3:0.5, OTU4:0.7)root;")
            )
            self.assertRaises(
                ValueError, faith_pd, [1, 2, 3], ["OTU1", "OTU2", "OTU3"],
                BPTree.from_treenode(t), engine=eng,
            )
            good = TreeNode.read(
                StringIO(
                    "(((((OTU1:0.5,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:"
                    "0.75,OTU5:0.75):1.25):0.0)root;"
                )
            )
            bp = BPTree.from_treenode(good)
            # duplicated taxa ids -> ValueError
            self.assertRaises(
                ValueError, faith_pd, [1, 2, 3], ["OTU1", "OTU2", "OTU2"], bp,
                engine=eng,
            )
            # length mismatch -> ValueError
            self.assertRaises(
                ValueError, faith_pd, [1, 2], ["OTU1", "OTU2", "OTU3"], bp, engine=eng
            )
            # negative counts -> ValueError
            self.assertRaises(
                ValueError, faith_pd, [1, 2, -3], ["OTU1", "OTU2", "OTU3"], bp,
                engine=eng,
            )
            # taxa not present in tree -> MissingNodeError
            self.assertRaises(
                MissingNodeError, faith_pd, [1, 2, 3], ["OTU1", "OTU2", "OTU42"], bp,
                engine=eng,
            )

    def test_missing_length_is_a_known_divergence(self):
        # BPTree collapses a missing branch length to 0.0 at construction, so
        # unlike the TreeNode path it CANNOT reject a missing-length tree; it
        # computes with the 0.0 lengths instead. This documents that gap.
        t = TreeNode.read(
            StringIO(
                "(((((OTU1,OTU2:0.5):0.5,OTU3:1.0):1.0):0.0,(OTU4:"
                "0.75,OTU5:0.75):1.25):0.0)root;"
            )
        )
        # TreeNode path rejects it ...
        self.assertRaises(
            ValueError, faith_pd, [1, 2, 3], ["OTU1", "OTU2", "OTU3"], t
        )
        # ... the BPTree path does not (missing length is already 0.0)
        bp = BPTree.from_treenode(t)
        val = faith_pd([1, 2, 3], ["OTU1", "OTU2", "OTU3"], bp)
        self.assertTrue(np.isfinite(val))


class PhyDivTests(TestCase):

    def setUp(self):
        self.tree = TreeNode.read(StringIO(
            '(((a:0.4,b:0.5):0.7,((c:0.1,d:0.2):0.6,(e:0.2,f:0.3):0.4):0.2)'
            ':0.1,g:1.2):0.2;'))

        # each dash (-) represents branch length = 0.05
        #
        #                          /--------a
        #          /--------------|
        #         |                \----------b
        #         |
        #      /--|                    /--c
        #     |   |      /------------|
        #     |   |     |              \----d
        #     |    \----|
        # ----|         |          /----e
        #     |          \--------|
        #     |                    \------f
        #     |
        #      \------------------------g

        self.taxa = list('abcdef')
        self.data = np.array([
            [1, 2, 0, 0, 0, 0],   # clade (a, b)
            [0, 0, 3, 4, 0, 0],   # clade (c, d)
            [0, 0, 0, 0, 5, 6],   # clade (e, f)
            [0, 0, 3, 4, 5, 6],   # non-basal, monophyletic group (c, d, e, f)
            [1, 2, 3, 4, 0, 0]])  # basal, non-monophyletic group (a, b, c, d)

    def test_phydiv_rooted_unweighted(self):
        # equivalent to faith_pd
        for datum in self.data:
            obs = phydiv(datum, self.taxa, self.tree)
            exp = faith_pd(datum, self.taxa, self.tree)
            self.assertAlmostEqual(obs, exp)

    def test_phydiv_unrooted_unweighted(self):
        # equivalent to faith_pd without path to root
        exps = [0.9, 0.3, 0.5, 1.8, 2.7]
        for datum, exp in zip(self.data, exps):
            obs = phydiv(datum, self.taxa, self.tree, rooted=False)
            self.assertAlmostEqual(obs, exp)

        # edge case: one taxon
        obs = phydiv([1], ['a'], self.tree, rooted=False)
        self.assertEqual(obs, 0)

        # edge case: zero taxon
        obs = phydiv([0], ['a'], self.tree, rooted=False)
        self.assertEqual(obs, 0)

        # edge case: no taxon
        obs = phydiv([], [], self.tree, rooted=False)
        self.assertEqual(obs, 0)

    def test_phydiv_rooted_weighted(self):
        # group (a, b)
        # = (0.4 * 1 + 0.5 * 2 + (0.7 + 0.1 + 0.2) * (1 + 2)) / (1 + 2)
        obs = phydiv(self.data[0], self.taxa, self.tree, weight=True)
        exp = 1.46666667
        self.assertAlmostEqual(obs, exp)

        # group (c, d)
        # = (0.1 * 3 + 0.2 * 4 + (0.6 + 0.2 + 0.1 + 0.2) * (3 + 4)) / (3 + 4)
        obs = phydiv(self.data[1], self.taxa, self.tree, weight=True)
        exp = 1.25714286
        self.assertAlmostEqual(obs, exp)

        # group (c, d, e, f)
        # = (0.1 * 3 + 0.2 * 4 + 0.2 * 5 + 0.3 * 6 + 0.6 * (3 + 4) + 0.4 *
        #   (5 + 6) + (0.2 + 0.1 + 0.2) * (3 + 4 + 5 + 6)) / (3 + 4 + 5 + 6)
        obs = phydiv(self.data[3], self.taxa, self.tree, weight=True)
        exp = 1.19444444
        self.assertAlmostEqual(obs, exp)

        # group (a, b, c, d)
        # = (0.4 * 1 + 0.5 * 2 + 0.1 * 3 + 0.2 * 4 + 0.7 * (1 + 2) + (0.6 +
        #   0.2) * (3 + 4) + (0.1 + 0.2) * (1 + 2 + 3 + 4)) / (1 + 2 + 3 + 4)
        obs = phydiv(self.data[4], self.taxa, self.tree, weight=True)
        exp = 1.32
        self.assertAlmostEqual(obs, exp)

    def test_phydiv_unrooted_weighted(self):
        # a.k.a., balance-weighted PD
        # group (a, b)
        # = (0.4 + 0.5) * 2 * min(1, 2) / (1 + 2)
        obs = phydiv(self.data[0], self.taxa, self.tree,
                     rooted=False, weight=True)
        exp = 0.6
        self.assertAlmostEqual(obs, exp)

        # group (c, d)
        # = (0.1 + 0.2) * 2 * min(3, 4) / (3 + 4)
        obs = phydiv(self.data[1], self.taxa, self.tree,
                     rooted=False, weight=True)
        exp = 0.25714286
        self.assertAlmostEqual(obs, exp)

        # group (c, d, e, f)
        # = 2 * (0.1 * min(3, 4 + 5 + 6) + 0.2 * min(4, 3 + 5 + 6) + 0.2 *
        #   min(5, 3 + 4 + 6) + 0.3 * min(6, 3 + 4 + 5) + (0.6 + 0.4) *
        #   min(3 + 4, 5 + 6)) / (3 + 4 + 5 + 6)
        obs = phydiv(self.data[3], self.taxa, self.tree,
                     rooted=False, weight=True)
        exp = 1.21111111
        self.assertAlmostEqual(obs, exp)

        # group (a, b, c, d)
        # = 2 * (0.4 * min(1, 2 + 3 + 4) + 0.5 * min(2, 1 + 3 + 4) + (0.7 +
        #   0.6 + 0.2) * min(1 + 2, 3 + 4) + 0.1 * min(3, 1 + 2 + 4) + 0.2 *
        #   min(4, 1 + 2 + 3)) / (1 + 2 + 3 + 4)
        obs = phydiv(self.data[4], self.taxa, self.tree,
                     rooted=False, weight=True)
        exp = 1.4
        self.assertAlmostEqual(obs, exp)

        # edge cases
        self.assertEqual(phydiv([1], ['a'], self.tree, False, True), 0)
        self.assertEqual(phydiv([0], ['a'], self.tree, False, True), 0)
        self.assertEqual(phydiv([], [], self.tree, False, True), 0)

    def test_phydiv_weight_param(self):
        # group (a, b), unrooted
        # = (0.4 + 0.5) * (2 * min(1, 2) / (1 + 2)) ** theta
        obs = phydiv(self.data[0], self.taxa, self.tree, False, 0.5)
        exp = 0.73484692
        self.assertAlmostEqual(obs, exp)
        obs = phydiv(self.data[0], self.taxa, self.tree, False, 0.25)
        exp = 0.81324180
        self.assertAlmostEqual(obs, exp)
        # fall back to unweighted
        obs = phydiv(self.data[0], self.taxa, self.tree, False, 0)
        exp = 0.9
        # fall back to fully-weighted
        self.assertAlmostEqual(obs, exp)
        obs = phydiv(self.data[0], self.taxa, self.tree, False, 1)
        exp = 0.6
        self.assertAlmostEqual(obs, exp)

        # rooted
        # = (0.4 * 1 ** theta + 0.5 * 2 ** theta + (0.7 + 0.1 + 0.2) *
        #   (1 + 2) ** theta) / (1 + 2) ** theta
        obs = phydiv(self.data[0], self.taxa, self.tree, True, 0.5)
        exp = 1.63918840
        self.assertAlmostEqual(obs, exp)
        obs = phydiv(self.data[0], self.taxa, self.tree, True, 0.25)
        exp = 1.75573528
        self.assertAlmostEqual(obs, exp)

        # edge cases
        self.assertEqual(phydiv([1], ['a'], self.tree, False, 0.5), 0)
        self.assertEqual(phydiv([0], ['a'], self.tree, False, 0.5), 0)
        self.assertEqual(phydiv([], [], self.tree, False, 0.5), 0)

    def test_phydiv_tree_unrooted(self):
        # convert tree to unrooted
        outgroup = self.tree.find('g')
        ingroup = outgroup.siblings()[0]
        unrooted = ingroup.copy()
        unrooted.extend([outgroup.copy()])
        unrooted.length += self.tree.length or 0.0

        # auto-enter unrooted mode
        obs = phydiv(self.data[0], self.taxa, unrooted)
        exp = phydiv(self.data[0], self.taxa, self.tree, rooted=False)
        self.assertEqual(obs, exp)

        # force rooted mode
        obs = phydiv(self.data[0], self.taxa, unrooted, rooted=True)
        exp = phydiv(self.data[0], self.taxa, self.tree)
        self.assertEqual(obs, exp)

    def test_phydiv_invalid_weight(self):
        params = (self.data[0], self.taxa, self.tree)
        self.assertRaises(ValueError, phydiv, *params, weight='hello')
        self.assertRaises(ValueError, phydiv, *params, weight=-0.5)
        self.assertRaises(ValueError, phydiv, *params, weight=2.0)


if __name__ == "__main__":
    main()
