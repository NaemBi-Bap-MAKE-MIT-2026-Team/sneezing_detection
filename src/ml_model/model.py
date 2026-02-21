from pathlib import Path

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    try:
        from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
    except Exception:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter


class LiteModel:
    """Thin wrapper around a TFLite interpreter for single-output classification."""

    def __init__(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TFLite model not found: {path}")
        self.interp = TFLiteInterpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

    def expected_input_shape(self):
        return self.in_det.get("shape", None)

    def predict_proba(self, x: np.ndarray) -> float:
        """Run inference and return the scalar output probability."""
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)
