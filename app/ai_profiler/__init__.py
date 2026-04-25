from threading import Lock

from .core import AIProfiler

_profiler = None
_lock = Lock()


def get_profiler() -> AIProfiler:
    global _profiler
    if _profiler is None:
        with _lock:
            if _profiler is None:
                _profiler = AIProfiler()
    return _profiler
