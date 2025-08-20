import pytest

from modules.configs import PreprocessorConfig, InferenceEngineConfig

def test_preprocessor_config_from_dict_allows_unknown_keys():
    cfg = PreprocessorConfig.from_dict({
        "normalization": "standard",
        "handle_nan": True,
        "enforce_float32": True,  # unknown to dataclass, should be attached dynamically
    })
    # Known field still parsed
    assert cfg.handle_nan is True
    # Unknown flag should be accessible via getattr gate used in code paths
    assert getattr(cfg, "enforce_float32", False) is True


def test_inference_engine_config_from_dict_allows_micro_batch_flags():
    cfg = InferenceEngineConfig.from_dict({
        "enable_batching": False,
        "batch_timeout": 0.05,
        "enable_micro_batch_fallback": True,
        "micro_batch_min_size": 3,
        "micro_batch_max_size": 17,
        "micro_batch_window_ms": 2.5,
    })
    # Known fields parsed
    assert cfg.enable_batching is False
    assert cfg.batch_timeout == 0.05
    # Unknown fields attached and readable by getattr in engine
    assert getattr(cfg, "enable_micro_batch_fallback", False) is True
    assert getattr(cfg, "micro_batch_min_size", None) == 3
    assert getattr(cfg, "micro_batch_max_size", None) == 17
    assert getattr(cfg, "micro_batch_window_ms", None) == 2.5
