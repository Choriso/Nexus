from threading import Lock

from config import config

from .core import AIProfiler

_profiler = None
_lock = Lock()


def get_profiler() -> AIProfiler:
    global _profiler
    if _profiler is None:
        with _lock:
            if _profiler is None:
                _profiler = AIProfiler(
                    use_local_models=config.USE_LOCAL_AI_MODELS,
                    config_obj=config,
                )
    return _profiler
