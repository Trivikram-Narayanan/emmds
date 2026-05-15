"""
EMMDS Fault Tolerance
Ensures the system never crashes completely.
Every pipeline stage is wrapped in safe_execute() so one bad model
or one bad column never kills the entire run.
"""

import time
import traceback
import functools
from typing import Any, Callable, Optional, TypeVar
from src.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def safe_execute(
    func: Callable,
    *args,
    fallback: Any = None,
    stage: str = "",
    reraise: bool = False,
    retries: int = 0,
    retry_delay: float = 0.5,
    **kwargs,
) -> Any:
    """
    Execute func(*args, **kwargs) safely.

    On success  → return result.
    On failure  → log error, return fallback (default None).

    Args:
        func:        The callable to execute
        *args:       Positional arguments for func
        fallback:    Value returned on failure
        stage:       Human-readable name for logging
        reraise:     If True, re-raise after logging
        retries:     Number of retry attempts on failure
        retry_delay: Seconds to wait between retries
        **kwargs:    Keyword arguments for func

    Returns:
        func result on success, fallback on failure.
    """
    label = stage or getattr(func, "__name__", "unknown")
    attempt = 0

    while attempt <= retries:
        try:
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"[{label}] succeeded on retry {attempt}")
            return result

        except KeyboardInterrupt:
            raise  # Never swallow Ctrl+C

        except Exception as exc:
            tb = traceback.format_exc()
            if attempt < retries:
                logger.warning(
                    f"[{label}] attempt {attempt+1}/{retries+1} failed: {exc}. "
                    f"Retrying in {retry_delay}s…"
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"[{label}] FAILED after {attempt+1} attempt(s): {exc}\n"
                    f"{tb}"
                )
                if reraise:
                    raise
                return fallback

        attempt += 1

    return fallback


def safe_stage(
    stage_name: str,
    fallback: Any = None,
    retries: int = 0,
    critical: bool = False,
):
    """
    Decorator that wraps a function with safe_execute.

    Usage:
        @safe_stage("Training", fallback={}, critical=True)
        def train_all(X, y): ...

    Args:
        stage_name: Human-readable stage name for logs
        fallback:   Value returned on failure
        retries:    Retry count
        critical:   If True, re-raise the exception after logging
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                func, *args,
                fallback=fallback,
                stage=stage_name,
                reraise=critical,
                retries=retries,
                **kwargs,
            )
        return wrapper  # type: ignore
    return decorator


class FaultTolerantPipeline:
    """
    Wraps an ordered list of pipeline stages so that:
    - Each stage runs inside safe_execute
    - Failures are recorded but don't stop subsequent stages
    - A summary of all stage outcomes is returned
    """

    def __init__(self):
        self.stages: list = []        # [(name, fn, fallback, critical)]
        self.results: dict = {}       # {name: result}
        self.errors: list  = []       # [(name, error_msg)]

    def add(
        self,
        name: str,
        fn: Callable,
        fallback: Any = None,
        critical: bool = False,
    ) -> "FaultTolerantPipeline":
        """Register a pipeline stage. Returns self for chaining."""
        self.stages.append((name, fn, fallback, critical))
        return self

    def run(self, **initial_context) -> dict:
        """
        Execute all stages in order.
        Each stage receives the accumulated context dict as kwargs.

        Returns:
            {stage_name: result, "_errors": [...], "_success": bool}
        """
        context = dict(initial_context)
        aborted = False

        for name, fn, fallback, critical in self.stages:
            if aborted:
                self.results[name] = None
                self.errors.append((name, "Skipped (pipeline aborted)"))
                continue

            result = safe_execute(fn, fallback=fallback, stage=name, reraise=False, **context)

            if result is None and fallback is None:
                # Stage produced no output
                self.errors.append((name, "Returned None"))
                if critical:
                    logger.error(f"Critical stage '{name}' failed — aborting pipeline.")
                    aborted = True
            else:
                if isinstance(result, dict):
                    context.update(result)  # Propagate outputs to next stages
                self.results[name] = result

        self.results["_errors"] = self.errors
        self.results["_success"] = not aborted and len(self.errors) == 0
        return self.results

    def summary(self) -> str:
        total    = len(self.stages)
        failed   = len(self.errors)
        passed   = total - failed
        lines = [f"Pipeline summary: {passed}/{total} stages succeeded"]
        for name, err in self.errors:
            lines.append(f"  ✗ {name}: {err}")
        return "\n".join(lines)
