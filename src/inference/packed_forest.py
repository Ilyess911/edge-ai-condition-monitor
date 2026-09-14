"""Isolation Forest inference from exported flat arrays.

scikit-learn scores an Isolation Forest with a Python loop over its trees and
validates the input on every call. For one window at a time that overhead is
the whole latency. This module exports the fitted trees into five (n_trees,
max_nodes) arrays and walks all trees at once, one depth level per NumPy
operation. Same trees, same score (tested against score_samples), a fraction of
the cost. The same flat layout is what one would write to a C header for a
microcontroller.
"""

from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329


def average_path_length(n: np.ndarray) -> np.ndarray:
    """Expected path length of an unsuccessful BST search among n points."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + EULER_GAMMA) - 2.0 * (n[big] - 1.0) / n[big]
    return out


class PackedIsolationForest:
    def __init__(self, sk_forest):
        trees = sk_forest.estimators_
        feats = sk_forest.estimators_features_
        T = len(trees)
        M = max(t.tree_.node_count for t in trees)
        self.feature = np.zeros((T, M), dtype=np.int16)
        self.threshold = np.zeros((T, M), dtype=np.float64)
        self.left = np.zeros((T, M), dtype=np.int32)
        self.right = np.zeros((T, M), dtype=np.int32)
        # Path-length contribution of stopping at a node: its depth is added
        # during traversal, c(n_samples) is added once a leaf is reached.
        self.leaf_bonus = np.zeros((T, M), dtype=np.float64)
        max_depth = 0
        for i, (est, fmap) in enumerate(zip(trees, feats)):
            tr = est.tree_
            n = tr.node_count
            is_leaf = tr.children_left[:n] == -1
            idx = np.arange(n)
            # Leaves point to themselves so extra iterations are harmless.
            self.left[i, :n] = np.where(is_leaf, idx, tr.children_left[:n])
            self.right[i, :n] = np.where(is_leaf, idx, tr.children_right[:n])
            # Trees see a permuted feature subset: map back to global columns.
            self.feature[i, :n] = np.where(is_leaf, 0, np.asarray(fmap)[np.maximum(tr.feature[:n], 0)])
            self.threshold[i, :n] = np.where(is_leaf, np.inf, tr.threshold[:n])
            self.leaf_bonus[i, :n] = average_path_length(tr.n_node_samples[:n])
            max_depth = max(max_depth, est.get_depth())
        self.is_leaf = self.left == np.arange(M)[None, :]
        self.max_depth = max_depth
        self._rows = np.arange(T)
        self._denominator = T * float(average_path_length(np.array([sk_forest.max_samples_]))[0])

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score, identical to -IsolationForest.score_samples(X)."""
        # scikit-learn trees compare float32 inputs against float64 thresholds.
        X = np.asarray(X, dtype=np.float32).astype(np.float64)
        out = np.empty(len(X))
        rows = self._rows
        for j, x in enumerate(X):
            node = np.zeros(len(rows), dtype=np.int32)
            depth = np.zeros(len(rows))
            for _ in range(self.max_depth):
                f = self.feature[rows, node]
                go_left = x[f] <= self.threshold[rows, node]
                depth += ~self.is_leaf[rows, node]
                node = np.where(go_left, self.left[rows, node], self.right[rows, node])
            total = np.sum(depth + self.leaf_bonus[rows, node])
            out[j] = 2.0 ** (-total / self._denominator)
        return out

    def n_parameters(self) -> int:
        return int((~self.is_leaf).sum() * 4 + self.is_leaf.sum())


class PackedIsolationForestDetector:
    """Streaming-engine wrapper around a fitted IsolationForestDetector."""

    def __init__(self, detector):
        self.name = f"{detector.name}-packed"
        self.packed = PackedIsolationForest(detector.model_)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.packed.score(X)

    def n_parameters(self) -> int:
        return self.packed.n_parameters()

    def serialized_bytes(self) -> int:
        p = self.packed
        return int(sum(a.nbytes for a in (p.feature, p.threshold, p.left, p.right, p.leaf_bonus)))
