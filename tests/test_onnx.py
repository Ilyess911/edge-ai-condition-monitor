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
