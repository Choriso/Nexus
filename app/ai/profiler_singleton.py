"""Re-export do profiler singleton canónico em :mod:`app.ai_profiler`."""

from app.ai_profiler import get_profiler

__all__ = ["get_profiler"]
