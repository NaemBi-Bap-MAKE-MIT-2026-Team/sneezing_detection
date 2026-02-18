"""
Mock TFLite Interpreter
=======================
Drop-in replacement for ``tflite_runtime.interpreter.Interpreter`` that
returns deterministic fake outputs without requiring a real .tflite model
or the TFLite runtime library.

Designed for RPi4 simulation — ``invoke()`` injects a configurable sleep
to mimic on-device inference latency.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Union

import numpy as np


class MockInterpreter:
    """Mimics the ``tflite_runtime.interpreter.Interpreter`` API.

    Parameters
    ----------
    model_path : str or None
        Ignored.  Accepted only for API compatibility.
    input_shape : tuple[int, ...]
        Shape that ``get_input_details`` will report.
        Default matches v4 model: ``(1, 198, 64, 1)``.
    output_shape : tuple[int, ...]
        Shape that ``get_output_details`` will report.
        Default: ``(1, 1)`` — single probability value.
    fake_output : float or list[float]
        Value(s) returned by ``get_tensor`` for the output index.
        If a list is given, values are consumed sequentially on each
        ``invoke()`` call and the list wraps around.
    inference_delay_ms : float
        Artificial delay (milliseconds) injected during ``invoke()``
        to simulate RPi4 TFLite inference latency.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        input_shape: tuple = (1, 198, 64, 1),
        output_shape: tuple = (1, 1),
        fake_output: Union[float, List[float]] = 0.95,
        inference_delay_ms: float = 10.0,
    ) -> None:
        self._model_path = model_path
        self._input_shape = tuple(input_shape)
        self._output_shape = tuple(output_shape)
        self._inference_delay_ms = inference_delay_ms

        # Normalise fake_output to a list for sequential consumption
        if isinstance(fake_output, (int, float)):
            self._fake_outputs: List[float] = [float(fake_output)]
        else:
            self._fake_outputs = [float(v) for v in fake_output]
        self._invoke_count: int = 0

        # Internal tensor storage
        self._input_tensor: Optional[np.ndarray] = None
        self._allocated: bool = False

        # Detail dicts (immutable once created)
        self._input_details = [
            {
                "index": 0,
                "name": "serving_default_input:0",
                "shape": np.array(self._input_shape, dtype=np.int32),
                "shape_signature": np.array(self._input_shape, dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
                "quantization_parameters": {
                    "scales": np.array([], dtype=np.float32),
                    "zero_points": np.array([], dtype=np.int32),
                    "quantized_dimension": 0,
                },
                "sparsity_parameters": {},
            }
        ]
        self._output_details = [
            {
                "index": 1,
                "name": "StatefulPartitionedCall:0",
                "shape": np.array(self._output_shape, dtype=np.int32),
                "shape_signature": np.array(self._output_shape, dtype=np.int32),
                "dtype": np.float32,
                "quantization": (0.0, 0),
                "quantization_parameters": {
                    "scales": np.array([], dtype=np.float32),
                    "zero_points": np.array([], dtype=np.int32),
                    "quantized_dimension": 0,
                },
                "sparsity_parameters": {},
            }
        ]

    # ------------------------------------------------------------------ #
    # Public API (mirrors tflite_runtime.interpreter.Interpreter)
    # ------------------------------------------------------------------ #
    def allocate_tensors(self) -> None:
        """No-op.  Marks the interpreter as allocated."""
        self._allocated = True

    def get_input_details(self) -> list:
        return self._input_details

    def get_output_details(self) -> list:
        return self._output_details

    def set_tensor(self, index: int, data: np.ndarray) -> None:
        """Store *data* as the input tensor after basic validation."""
        if not self._allocated:
            raise RuntimeError(
                "Interpreter has not been allocated. "
                "Call allocate_tensors() first."
            )

        expected_shape = self._input_shape
        if tuple(data.shape) != expected_shape:
            raise ValueError(
                f"Input shape mismatch: expected {expected_shape}, "
                f"got {tuple(data.shape)}"
            )
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self._input_tensor = data

    def invoke(self) -> None:
        """Simulate inference with an artificial delay."""
        if self._input_tensor is None:
            raise RuntimeError(
                "No input tensor has been set. Call set_tensor() first."
            )
        # Simulate RPi4 inference latency
        if self._inference_delay_ms > 0:
            time.sleep(self._inference_delay_ms / 1000.0)
        self._invoke_count += 1

    def get_tensor(self, index: int) -> np.ndarray:
        """Return the fake output tensor.

        If *index* matches the output detail index, returns the next
        value from the ``fake_output`` sequence.  Otherwise returns
        the stored input tensor (for debugging).
        """
        output_index = self._output_details[0]["index"]
        if index == output_index:
            # Cycle through the fake output list
            idx = (self._invoke_count - 1) % len(self._fake_outputs)
            value = self._fake_outputs[idx]
            return np.full(self._output_shape, value, dtype=np.float32)

        # Fallback: return the stored input tensor if available
        if self._input_tensor is not None:
            return self._input_tensor.copy()

        raise ValueError(f"No tensor stored for index {index}")

    # ------------------------------------------------------------------ #
    # Convenience helpers (not part of TFLite API)
    # ------------------------------------------------------------------ #
    @property
    def invoke_count(self) -> int:
        """Number of times ``invoke()`` has been called."""
        return self._invoke_count

    @property
    def inference_delay_ms(self) -> float:
        return self._inference_delay_ms

    @inference_delay_ms.setter
    def inference_delay_ms(self, value: float) -> None:
        self._inference_delay_ms = max(0.0, float(value))

    def reset(self) -> None:
        """Reset internal state (invoke count, stored tensors)."""
        self._invoke_count = 0
        self._input_tensor = None

    def __repr__(self) -> str:
        return (
            f"MockInterpreter("
            f"input={self._input_shape}, "
            f"output={self._output_shape}, "
            f"fake_output={self._fake_outputs}, "
            f"delay_ms={self._inference_delay_ms})"
        )
