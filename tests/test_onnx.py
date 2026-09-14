import gc

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from src.inference.onnx_export import OnnxDetector, to_onnx  # noqa: E402
from src.models.detectors import build  # noqa: E402


@pytest.mark.parametrize(
    "name,kwargs",
    [("zscore", {}), ("pca", {}), ("iforest", {"n_estimators": 25}), ("autoencoder", {})],
)
def test_onnx_scores_match_native(rng, name, kwargs):
    X = rng.normal(size=(1500, 12)) * rng.uniform(0.1, 50, size=12) + rng.uniform(-5, 60, 12)
    det = build(name, **kwargs).fit(X[:1000])
    onx = OnnxDetector(to_onnx(det, 12), name)
    test = X[1000:] * 1.3
    native = det.score(test)
    ort = onx.score(test)
    np.testing.assert_allclose(ort, native, rtol=1e-4, atol=1e-4)
    single = onx.score(test[:1])
    assert single.shape == (1,)
    # Release the ONNX Runtime session now: sessions still alive at interpreter
    # exit can abort the process on macOS after the tests have passed.
    del onx
    gc.collect()


def test_packed_forest_matches_sklearn(rng):
    from src.inference.packed_forest import PackedIsolationForestDetector

    X = rng.normal(size=(3000, 12)) * rng.uniform(0.1, 50, size=12)
    det = build("iforest", n_estimators=60).fit(X[:2000])
    packed = PackedIsolationForestDetector(det)
    test = np.vstack([X[2000:2300], X[2300:2400] * 3.0])
    np.testing.assert_allclose(packed.score(test), det.score(test), rtol=1e-12, atol=1e-12)
