from __future__ import annotations

import numpy as np


def binary_accuracy(predictions: np.ndarray | object, inputs: np.ndarray | object = None, targets: np.ndarray | None = None) -> float:
    if hasattr(predictions, "forward"):
        network = predictions
        assert inputs is not None and targets is not None
        outputs = network.forward(inputs)
    else:
        outputs = predictions
    outputs = np.asarray(outputs)
    predicted = outputs.reshape(-1) >= 0.0
    return float(np.mean(predicted == np.asarray(targets).reshape(-1)))
