"""
EMMDS Experiment Tracker
Tracks every pipeline run with full metrics, parameters, and metadata.
Default backend: JSON files in outputs/logs/
Optional backend: MLflow (if installed)
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

LOG_DIR = Path("outputs/logs")


def _serialise(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, pd.Series):      return obj.to_list()
    raise TypeError(f"Non-serialisable: {type(obj)}")


class ExperimentTracker:
    """
    Records every EMMDS pipeline run as a structured experiment.

    Each experiment contains:
      - run metadata (id, timestamp, dataset)
      - parameters (task, scaler, cv_folds, ...)
      - model results (metrics per model)
      - best model summary
      - trust scores
    """

    def __init__(self, log_dir: str | Path = LOG_DIR, use_mlflow: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self._mlflow = None
        self._active_run_id: Optional[str] = None

        if use_mlflow:
            self._init_mlflow()

    # ── Start / End run ──────────────────────────────────────────

    def start_run(
        self,
        dataset_name: str,
        task: str,
        target_col: str,
        n_samples: int,
        n_features: int,
        params: Optional[dict] = None,
    ) -> str:
        """
        Begin a tracked experiment run.

        Returns:
            run_id string
        """
        run_id = str(uuid.uuid4())[:8]
        ts     = datetime.now().isoformat()

        self._active = {
            "run_id":      run_id,
            "timestamp":   ts,
            "dataset":     dataset_name,
            "task":        task,
            "target_col":  target_col,
            "n_samples":   n_samples,
            "n_features":  n_features,
            "params":      params or {},
            "model_results": {},
            "best_model":  None,
            "trust_scores": {},
            "tags":        {},
        }
        self._active_run_id = run_id

        if self.use_mlflow and self._mlflow:
            self._mlflow.start_run(run_name=f"{dataset_name}_{run_id}")
            self._mlflow.log_params({"dataset": dataset_name, "task": task, **(params or {})})

        logger.info(f"Experiment started | run_id={run_id} | dataset={dataset_name}")
        return run_id

    def log_model_result(self, model_name: str, metrics: dict):
        """Log metrics for a single model."""
        if not hasattr(self, "_active"):
            logger.warning("No active run. Call start_run() first.")
            return
        clean = {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))}
        self._active["model_results"][model_name] = clean

        if self.use_mlflow and self._mlflow:
            for k, v in clean.items():
                if v is not None:
                    self._mlflow.log_metric(f"{model_name}_{k}", float(v))

    def log_trust_scores(self, trust_scores: dict):
        """Log all model trust scores."""
        if hasattr(self, "_active"):
            self._active["trust_scores"] = trust_scores

    def log_best_model(self, model_name: str, metrics: dict, trust_score: float):
        """Record the final selected model."""
        if hasattr(self, "_active"):
            self._active["best_model"] = {
                "name":        model_name,
                "metrics":     {k: v for k, v in metrics.items() if not isinstance(v, (list, dict))},
                "trust_score": trust_score,
            }
        if self.use_mlflow and self._mlflow:
            self._mlflow.log_param("best_model", model_name)
            self._mlflow.log_metric("best_trust_score", trust_score)

    def set_tag(self, key: str, value: str):
        if hasattr(self, "_active"):
            self._active["tags"][key] = value

    def end_run(self) -> str:
        """
        Finalise and persist the active run.

        Returns:
            Path to saved JSON log
        """
        if not hasattr(self, "_active"):
            logger.warning("No active run to end.")
            return ""

        out_path = self.log_dir / f"run_{self._active['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(self._active, f, indent=2, default=_serialise)

        if self.use_mlflow and self._mlflow:
            self._mlflow.end_run()

        logger.info(f"Experiment ended | saved → {out_path}")
        run_id = self._active_run_id
        del self._active
        self._active_run_id = None
        return str(out_path)

    # ── History ──────────────────────────────────────────────────

    def list_runs(self) -> List[dict]:
        """Return summary of all logged runs, newest first."""
        runs = []
        for p in sorted(self.log_dir.glob("run_*.json"), reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                best = data.get("best_model", {})
                runs.append({
                    "run_id":      data.get("run_id"),
                    "timestamp":   data.get("timestamp"),
                    "dataset":     data.get("dataset"),
                    "task":        data.get("task"),
                    "best_model":  best.get("name") if best else None,
                    "trust_score": best.get("trust_score") if best else None,
                    "file":        str(p),
                })
            except Exception:
                pass
        return runs

    def load_run(self, run_id: str) -> dict:
        path = self.log_dir / f"run_{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run log not found: {path}")
        with open(path) as f:
            return json.load(f)

    def compare_runs(self, metric: str = "f1") -> List[dict]:
        """
        Compare all runs on a given metric for the best model.
        Returns list sorted by metric descending.
        """
        rows = []
        for run in self.list_runs():
            run_data = self.load_run(run["run_id"])
            best = run_data.get("best_model", {})
            if best:
                val = best.get("metrics", {}).get(metric)
                rows.append({
                    "run_id":    run["run_id"],
                    "dataset":   run["dataset"],
                    "model":     best.get("name"),
                    metric:      val,
                    "trust":     best.get("trust_score"),
                    "timestamp": run["timestamp"],
                })
        return sorted(rows, key=lambda x: (x.get(metric) or 0), reverse=True)

    # ── MLflow ───────────────────────────────────────────────────

    def _init_mlflow(self):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_experiment("EMMDS")
            logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not installed — falling back to JSON logging")
            self.use_mlflow = False
            self._mlflow = None
