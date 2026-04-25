# profiler_singleton.py
from app.ai_profiler.core import AIProfiler
import threading

_profiler = None
_lock = threading.Lock()

def get_profiler() -> AIProfiler:
    global _profiler
    if _profiler is None:
        with _lock:
            if _profiler is None:
                _profiler = AIProfiler()
    return _profiler