"""
tests/test_ml_model.py
----------------------
Standalone tests for the ml_model module.

Run from src/:
    python tests/test_ml_model.py

Results:
    PASS  — assertion satisfied
    SKIP  — required file not found (model / stats missing)
    FAIL  — unexpected error
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def _pass(name: str) -> None:
    print(f"  PASS  {name}")


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP  {name}  ({reason})")


def _fail(name: str, err: Exception) -> None:
    print(f"  FAIL  {name}  — {err}")


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #

def test_load_stats():
    name = "load_stats — shape (64,)"
    from ml_model import config as cfg
    if not cfg.STATS_PATH.exists():
        _skip(name, f"{cfg.STATS_PATH} not found")
        return
    try:
        from ml_model import load_stats
        mu, sdv = load_stats(cfg.STATS_PATH)
        assert mu.shape == (64,),  f"mu shape {mu.shape}"
        assert sdv.shape == (64,), f"sdv shape {sdv.shape}"
        assert mu.dtype == np.float32
        _pass(name)
    except Exception as e:
        _fail(name, e)


def test_preproc_shape():
    name = "preproc — output shape (1, frames, 64, 1)"
    from ml_model import config as cfg
    if not cfg.STATS_PATH.exists():
        _skip(name, f"{cfg.STATS_PATH} not found")
        return
    try:
        from ml_model import load_stats, preproc
        mu, sdv = load_stats(cfg.STATS_PATH)
        dummy = np.zeros(int(cfg.MODEL_SR * cfg.CLIP_SECONDS), dtype=np.float32)
        x_in = preproc(dummy, mu, sdv)
        assert x_in.ndim == 4,        f"expected 4-D, got {x_in.ndim}"
        assert x_in.shape[0] == 1,    f"batch dim {x_in.shape[0]}"
        assert x_in.shape[2] == 64,   f"mel dim {x_in.shape[2]}"
        assert x_in.shape[3] == 1,    f"channel dim {x_in.shape[3]}"
        assert x_in.dtype == np.float32
        _pass(name)
    except Exception as e:
        _fail(name, e)


def test_model_inference():
    name = "LiteModel.predict_proba — returns float in [0, 1]"
    from ml_model import config as cfg
    if not cfg.TFLITE_PATH.exists() or not cfg.STATS_PATH.exists():
        _skip(name, "model or stats file not found")
        return
    try:
        from ml_model import LiteModel, load_stats, preproc
        mu, sdv = load_stats(cfg.STATS_PATH)
        model   = LiteModel(cfg.TFLITE_PATH)
        dummy   = np.zeros(int(cfg.MODEL_SR * cfg.CLIP_SECONDS), dtype=np.float32)
        x_in    = preproc(dummy, mu, sdv)
        p = model.predict_proba(x_in)
        assert isinstance(p, float),  f"expected float, got {type(p)}"
        assert 0.0 <= p <= 1.0,       f"prob out of range: {p}"
        _pass(f"{name}  (p={p:.4f})")
    except Exception as e:
        _fail(name, e)


def test_model_input_shape():
    name = "LiteModel.expected_input_shape — matches preproc output"
    from ml_model import config as cfg
    if not cfg.TFLITE_PATH.exists() or not cfg.STATS_PATH.exists():
        _skip(name, "model or stats file not found")
        return
    try:
        from ml_model import LiteModel, load_stats, preproc
        mu, sdv = load_stats(cfg.STATS_PATH)
        model   = LiteModel(cfg.TFLITE_PATH)
        dummy   = np.zeros(int(cfg.MODEL_SR * cfg.CLIP_SECONDS), dtype=np.float32)
        x_in    = preproc(dummy, mu, sdv)
        expected = model.expected_input_shape()
        # expected may be None on some TFLite builds — skip shape check if so
        if expected is not None:
            for dim_model, dim_input in zip(expected[1:], x_in.shape[1:]):
                if dim_model > 0:  # -1 means dynamic
                    assert dim_model == dim_input, \
                        f"model expects {expected}, got {x_in.shape}"
        _pass(f"{name}  (model={expected}, input={x_in.shape})")
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# Runner                                                                  #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== test_ml_model ===")
    test_load_stats()
    test_preproc_shape()
    test_model_inference()
    test_model_input_shape()
    print("=====================")
