"""
EMMDS Result Store
Unified storage layer for all pipeline results:
  - Model metrics
  - Trust scores
  - SHAP explanations
  - Decision outputs

Persists to JSON in outputs/reports/.
Supports loading previous runs.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

STORE_DIR = Path("outputs/reports")


def _default_serializer(obj):
    """JSON serializer for numpy/pandas types."""
    if isinstance(obj, (np.integer,)):        return int(obj)
    if isinstance(obj, (np.floating,)):       return float(obj)
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, pd.DataFrame):         return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):            return obj.to_list()
    raise TypeError(f"Non-serialisable type: {type(obj)}")


class ResultStore:
    """
    Stores and retrieves EMMDS pipeline results.
    Each run is saved as a timestamped JSON file.
    """

    def __init__(self, store_dir: str | Path = STORE_DIR):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._current: dict = {}

    # ── Save ─────────────────────────────────────────────────────

    def save_run(
        self,
        pipeline_result: dict,
        dataset_name: str = "dataset",
        run_id: Optional[str] = None,
    ) -> str:
        """
        Persist a full pipeline result to disk.

        Args:
            pipeline_result: Output of EMPipeline.run()
            dataset_name:    Label for this run (e.g. filename stem)
            run_id:          Override run ID; auto-generated if None

        Returns:
            Path to saved JSON file
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or f"{dataset_name}_{ts}"

        record = self._build_record(pipeline_result, run_id, dataset_name, ts)
        self._current = record

        out_path = self.store_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=_default_serializer)

        logger.info(f"Run saved → {out_path}")
        return str(out_path)

    def save_metrics(self, metrics: dict, model_name: str, dataset_name: str = "dataset") -> str:
        """Quick save of just the metrics for a single model."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "run_id":       f"{dataset_name}_{model_name}_{ts}",
            "dataset":      dataset_name,
            "model":        model_name,
            "metrics":      metrics,
            "timestamp":    ts,
        }
        out_path = self.store_dir / f"metrics_{dataset_name}_{model_name}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=_default_serializer)
        logger.info(f"Metrics saved → {out_path}")
        return str(out_path)

    # ── Load ─────────────────────────────────────────────────────

    def load_run(self, run_id: str) -> dict:
        """Load a specific run by ID."""
        path = self.store_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {path}")
        with open(path) as f:
            return json.load(f)

    def list_runs(self) -> List[dict]:
        """
        List all stored runs as summary dicts, newest first.
        """
        runs = []
        for p in sorted(self.store_dir.glob("*.json"), reverse=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                runs.append({
                    "run_id":     data.get("run_id", p.stem),
                    "dataset":    data.get("dataset", "—"),
                    "best_model": data.get("best_model", "—"),
                    "trust_score": data.get("trust_score"),
                    "primary_score": data.get("primary_score"),
                    "timestamp":  data.get("timestamp", "—"),
                    "file":       str(p),
                })
            except Exception:
                pass
        return runs

    def get_current(self) -> dict:
        """Return the last saved record (in-memory)."""
        return self._current

    # ── Internal ─────────────────────────────────────────────────

    def _build_record(
        self, result: dict, run_id: str, dataset_name: str, ts: str
    ) -> dict:
        """Extract key fields from a full pipeline result into a flat record."""
        decision = result.get("decision", {})
        steps    = result.get("steps", {})

        return {
            "run_id":          run_id,
            "timestamp":       ts,
            "dataset":         dataset_name,
            "task":            result.get("task"),
            "target_col":      result.get("target_col"),

            # Decision summary
            "best_model":      decision.get("best_model"),
            "primary_metric":  decision.get("primary_metric"),
            "primary_score":   decision.get("primary_score"),
            "accuracy":        decision.get("accuracy"),
            "trust_score":     decision.get("trust_score"),
            "trust_label":     decision.get("trust_label"),
            "top_features":    decision.get("top_features", [])[:10],

            # Dataset info
            "dataset_info":    decision.get("dataset_info", {}),

            # Full leaderboard
            "leaderboard":     steps.get("leaderboard", []),

            # Evaluation metrics (strip confusion matrices)
            "eval_results": {
                name: {k: v for k, v in m.items() if k != "confusion_matrix"}
                for name, m in steps.get("evaluation", {}).items()
            },

            # Trust breakdown
            "trust_breakdown": decision.get("trust_breakdown", {}),
            "all_trust_scores": decision.get("all_trust_scores", {}),

            # Calibration
            "calibration_scores": {
                k: v for k, v in steps.get("calibration_scores", {}).items()
            },

            # SHAP top features
            "shap_top_features": steps.get("shap_global", {}).get("ranking", [])[:10],

            # Preprocessing info
            "preprocessing": {
                k: v for k, v in steps.get("preprocessing", {}).items()
                if k != "X_train_ref"
            },
        }
