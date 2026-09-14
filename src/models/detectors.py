"""Four lightweight one-class anomaly detectors.

All are trained on healthy windows only (failures are rare and unlabelled in
practice) and expose the same contract:

    fit(X)      X: (n_windows, n_features) healthy data
    score(X)    anomaly score per row, higher = more abnormal
    to_onnx()   the same scoring function as an ONNX graph (see src/inference)

Inference is written the way it would be deployed: z-score, PCA and the
autoencoder are a handful of matrix operations evaluated with NumPy on the
exported parameters, Isolation Forest goes through scikit-learn because tree
traversal has no cheap NumPy form. The ONNX Runtime path runs all four under
one runtime, which removes that framework asymmetry from the comparison.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor


class _Standardizer:
    def fit(self, X: np.ndarray) -> "_Standardizer":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        # A constant feature (e.g. an empty FFT band) must not divide by zero.
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


class Detector:
    name = "base"

    def fit(self, X: np.ndarray) -> "Detector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_parameters(self) -> int:
        raise NotImplementedError

    def serialized_bytes(self) -> int:
        """Size of the pickled trained object: what a gateway would store on disk."""
        return len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))


class ZScoreDetector(Detector):
    """Per-feature alarm limits: score = max_j |x_j - mu_j| / sigma_j.

    This is what a SCADA alarm list does. It ignores correlations, so a normal
    speed increase that raises vibration RMS looks abnormal to it.
    """

    name = "zscore"

    def fit(self, X):
        self.std_ = _Standardizer().fit(X)
        return self

    def score(self, X):
        return np.max(np.abs(self.std_(X)), axis=1)

    def n_parameters(self):
        return 2 * len(self.std_.mean_)


class PCADetector(Detector):
    """Linear subspace model, classical multivariate statistical process control.

    Score is the combined index of Yue & Qin (2001):
        phi = SPE / delta2 + T2 / tau2
    where SPE is the squared residual outside the retained subspace, T2 the
    Hotelling statistic inside it, and delta2, tau2 their 99th percentiles on
    the training set. SPE catches broken correlations, T2 catches abnormal
    magnitude along normal directions; neither alone catches both.
    """

    name = "pca"

    def __init__(self, variance: float = 0.95):
        self.variance = variance

    def fit(self, X):
        self.std_ = _Standardizer().fit(X)
        Z = self.std_(X)
        _, s, vt = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
        eig = s**2 / (len(Z) - 1)
        k = int(np.searchsorted(np.cumsum(eig) / eig.sum(), self.variance) + 1)
        self.components_ = vt[:k]  # (k, d)
        self.eigenvalues_ = eig[:k]
        spe, t2 = self._stats(Z)
        self.delta2_ = float(np.percentile(spe, 99))
        self.tau2_ = float(np.percentile(t2, 99))
        return self

    def _stats(self, Z):
        proj = Z @ self.components_.T
        resid = Z - proj @ self.components_
        spe = np.sum(resid * resid, axis=1)
        t2 = np.sum(proj * proj / self.eigenvalues_, axis=1)
        return spe, t2

    def score(self, X):
        spe, t2 = self._stats(self.std_(X))
        return spe / self.delta2_ + t2 / self.tau2_

    def n_parameters(self):
        k, d = self.components_.shape
        return 2 * d + k * d + k + 2


class IsolationForestDetector(Detector):
    """Random isolation trees; anomalies need fewer splits to be isolated.

    No distance, no density, no scaling needed, and the cost per window is
    n_estimators tree walks of depth <= log2(max_samples): bounded and known
    in advance, which is what a real-time budget needs.
    """

    name = "iforest"

    def __init__(self, n_estimators: int = 100, max_samples: int = 256, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed

    def fit(self, X):
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(X)),
            random_state=self.seed,
            n_jobs=1,
        ).fit(X)
        return self

    def score(self, X):
        return -self.model_.score_samples(X)

    def n_parameters(self):
        # Every node stores feature, threshold, two children and a sample count.
        return int(sum(est.tree_.node_count for est in self.model_.estimators_)) * 5


class AutoencoderDetector(Detector):
    """Small dense autoencoder d -> 8 -> 3 -> 8 -> d, reconstruction error as score.

    Included for one reason: it is the non-linear counterpart of PCA. If it
    does not beat PCA on detection, its extra cost is not justified at the edge.
    scikit-learn's MLPRegressor is used for training to avoid a deep learning
    dependency; inference is a NumPy forward pass over the exported weights.
    """

    name = "autoencoder"

    def __init__(self, hidden: tuple[int, ...] = (8, 3, 8), seed: int = 0):
        self.hidden = hidden
        self.seed = seed

    def fit(self, X):
        self.std_ = _Standardizer().fit(X)
        Z = self.std_(X)
        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="tanh",
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=self.seed,
        ).fit(Z, Z)
        self.weights_ = [w.copy() for w in mlp.coefs_]
        self.biases_ = [b.copy() for b in mlp.intercepts_]
        self.n_iter_ = mlp.n_iter_
        return self

    def _forward(self, Z):
        h = Z
        last = len(self.weights_) - 1
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            h = h @ w + b
            if i < last:
                h = np.tanh(h)
        return h

    def score(self, X):
        Z = self.std_(X)
        r = Z - self._forward(Z)
        return np.mean(r * r, axis=1)

    def n_parameters(self):
        return sum(w.size for w in self.weights_) + sum(b.size for b in self.biases_) + 2 * len(
            self.std_.mean_
        )


def build(name: str, **kwargs) -> Detector:
    table = {
        "zscore": ZScoreDetector,
        "pca": PCADetector,
        "iforest": IsolationForestDetector,
        "autoencoder": AutoencoderDetector,
    }
    if name not in table:
        raise KeyError(f"unknown detector {name!r}, choose from {sorted(table)}")
    return table[name](**kwargs)
