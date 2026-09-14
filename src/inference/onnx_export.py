"""Export trained detectors to ONNX and score them with ONNX Runtime.

Why: the native path mixes two things, the algorithm's cost and the cost of
the Python framework that hosts it (scikit-learn's input validation dominates
a single Isolation Forest call). ONNX Runtime executes all four detectors as
compiled graphs, which is also the realistic deployment format for a Raspberry
Pi, a Jetson or an industrial gateway.

Each graph reproduces `Detector.score` exactly; tests check parity to 1e-4
(float32 inside the graph).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.models.detectors import (
    AutoencoderDetector,
    Detector,
    IsolationForestDetector,
    PCADetector,
    ZScoreDetector,
)

OPSET = 17


def _const(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name)


def _graph(nodes, inits, n_features: int, name: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])
    g = helper.make_graph(nodes, name, [x], [y], initializer=inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def _standardize(det, inits, nodes, out="Z"):
    inits += [_const("mu", det.std_.mean_), _const("sigma", det.std_.scale_)]
    nodes += [helper.make_node("Sub", ["X", "mu"], ["Xc"]),
              helper.make_node("Div", ["Xc", "sigma"], [out])]


def _zscore(det: ZScoreDetector) -> onnx.ModelProto:
    inits, nodes = [], []
    _standardize(det, inits, nodes)
    nodes += [helper.make_node("Abs", ["Z"], ["A"]),
              helper.make_node("ReduceMax", ["A"], ["score"], axes=[1], keepdims=0)]
    return _graph(nodes, inits, len(det.std_.mean_), "zscore")


def _pca(det: PCADetector) -> onnx.ModelProto:
    inits, nodes = [], []
    _standardize(det, inits, nodes)
    P = det.components_
    inits += [
        _const("Pt", P.T),
        _const("P", P),
        _const("inv_eig", 1.0 / det.eigenvalues_),
        _const("inv_delta2", 1.0 / det.delta2_),
        _const("inv_tau2", 1.0 / det.tau2_),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
    ]
    nodes += [
        helper.make_node("MatMul", ["Z", "Pt"], ["proj"]),
        helper.make_node("MatMul", ["proj", "P"], ["recon"]),
        helper.make_node("Sub", ["Z", "recon"], ["resid"]),
        helper.make_node("Mul", ["resid", "resid"], ["resid2"]),
        helper.make_node("ReduceSum", ["resid2", "axis1"], ["spe"], keepdims=0),
        helper.make_node("Mul", ["proj", "proj"], ["proj2"]),
        helper.make_node("Mul", ["proj2", "inv_eig"], ["t2_terms"]),
        helper.make_node("ReduceSum", ["t2_terms", "axis1"], ["t2"], keepdims=0),
        helper.make_node("Mul", ["spe", "inv_delta2"], ["a"]),
        helper.make_node("Mul", ["t2", "inv_tau2"], ["b"]),
        helper.make_node("Add", ["a", "b"], ["score"]),
    ]
    return _graph(nodes, inits, P.shape[1], "pca")


def _autoencoder(det: AutoencoderDetector) -> onnx.ModelProto:
    inits, nodes = [], []
    _standardize(det, inits, nodes)
    h = "Z"
    last = len(det.weights_) - 1
    for i, (w, b) in enumerate(zip(det.weights_, det.biases_)):
        inits += [_const(f"W{i}", w), _const(f"b{i}", b)]
        nodes += [helper.make_node("MatMul", [h, f"W{i}"], [f"m{i}"]),
                  helper.make_node("Add", [f"m{i}", f"b{i}"], [f"a{i}"])]
        h = f"a{i}"
        if i < last:
            nodes.append(helper.make_node("Tanh", [h], [f"h{i}"]))
            h = f"h{i}"
    nodes += [
        helper.make_node("Sub", ["Z", h], ["r"]),
        helper.make_node("Mul", ["r", "r"], ["r2"]),
        helper.make_node("ReduceMean", ["r2"], ["score"], axes=[1], keepdims=0),
    ]
    return _graph(nodes, inits, len(det.std_.mean_), "autoencoder")


def _iforest(det: IsolationForestDetector, n_features: int) -> onnx.ModelProto:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    model = convert_sklearn(
        det.model_,
        initial_types=[("X", FloatTensorType([None, n_features]))],
        target_opset={"": OPSET, "ai.onnx.ml": 3},
    )
    # skl2onnx outputs decision_function = score_samples - offset_.
    # Our score is -score_samples = -(decision + offset_).
    g = model.graph
    scores_out = [o.name for o in g.output if o.name != "label"][0]
    g.initializer.append(_const("neg_one", -1.0))
    g.initializer.append(_const("offset", det.model_.offset_))
    g.node.extend([
        helper.make_node("Add", [scores_out, "offset"], ["raw"]),
        helper.make_node("Flatten", ["raw"], ["raw2"], axis=0),
        helper.make_node("Mul", ["raw2", "neg_one"], ["score_2d"]),
        helper.make_node("Reshape", ["score_2d", "flat_shape"], ["score"]),
    ])
    g.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), "flat_shape"))
    while len(g.output):
        g.output.pop()
    g.output.append(helper.make_tensor_value_info("score", TensorProto.FLOAT, [None]))
    onnx.checker.check_model(model)
    return model


def to_onnx(det: Detector, n_features: int) -> onnx.ModelProto:
    if isinstance(det, ZScoreDetector):
        return _zscore(det)
    if isinstance(det, PCADetector):
        return _pca(det)
    if isinstance(det, AutoencoderDetector):
        return _autoencoder(det)
    if isinstance(det, IsolationForestDetector):
        return _iforest(det, n_features)
    raise TypeError(f"no ONNX exporter for {type(det).__name__}")


class OnnxDetector:
    """Drop-in replacement for a Detector inside the streaming engine."""

    def __init__(self, model: onnx.ModelProto, name: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # edge-like: one core, no thread pool jitter
        opts.inter_op_num_threads = 1
        self.name = f"{name}-onnx"
        self.model_bytes = model.SerializeToString()
        self.session = ort.InferenceSession(
            self.model_bytes, opts, providers=["CPUExecutionProvider"]
        )
        self._input = self.session.get_inputs()[0].name

    def score(self, X: np.ndarray) -> np.ndarray:
        out = self.session.run(["score"], {self._input: X.astype(np.float32, copy=False)})[0]
        return out.reshape(-1).astype(np.float64)

    def serialized_bytes(self) -> int:
        return len(self.model_bytes)
