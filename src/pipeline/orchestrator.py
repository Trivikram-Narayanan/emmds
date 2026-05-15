"""
EMMDS Orchestrator
Higher-level wrapper around EMPipeline.
Handles file I/O, result persistence, and batch runs.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from src.pipeline.pipeline import EMPipeline
from src.utils.helpers import save_json, timestamp
from src.utils.logger import get_logger
from src.utils.config import get

logger = get_logger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates full pipeline runs from file paths.
    Saves results to outputs/ automatically.
    """

    def __init__(self):
        self.pipeline = EMPipeline()
        self.last_result: Optional[dict] = None

    def run_from_file(
        self,
        csv_path: str,
        target_col: str,
        task: Optional[str] = None,
        scaler: str = "standard",
        save_results: bool = True,
    ) -> dict:
        """
        Load a CSV and run the full pipeline.

        Args:
            csv_path:     Path to input CSV file
            target_col:   Target column name
            task:         "classification" or "regression" (auto-detect if None)
            scaler:       Feature scaling method
            save_results: Whether to persist results to outputs/

        Returns:
            Full pipeline result dict
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

        result = self.pipeline.run(
            df=df,
            target_col=target_col,
            task=task,
            scaler=scaler,
        )

        self.last_result = result

        if save_results and result.get("status") == "success":
            self._save_results(result, stem=csv_path.stem)

        return result

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: Optional[str] = None,
        scaler: str = "standard",
    ) -> dict:
        """Run pipeline directly from a DataFrame."""
        result = self.pipeline.run(
            df=df,
            target_col=target_col,
            task=task,
            scaler=scaler,
        )
        self.last_result = result
        return result

    def _save_results(self, result: dict, stem: str = "run") -> None:
        """Persist key results to outputs/reports/."""
        ts = timestamp()
        out_dir = Path(get("paths.outputs_reports", "outputs/reports"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Serializable subset of results
        decision = result.get("decision", {})
        leaderboard = result["steps"].get("leaderboard", [])
        shap = result["steps"].get("shap_global", {})

        report = {
            "timestamp": ts,
            "dataset": stem,
            "task": result.get("task"),
            "target_col": result.get("target_col"),
            "decision": {k: v for k, v in decision.items() if not k.startswith("_")},
            "leaderboard": leaderboard,
            "shap_top_features": shap.get("ranking", [])[:10],
        }

        out_path = out_dir / f"emmds_{stem}_{ts}.json"
        save_json(report, out_path)
        logger.info(f"Results saved → {out_path}")

    def print_summary(self) -> None:
        """Print a clean console summary of the last run."""
        if not self.last_result:
            print("No results available.")
            return

        d = self.last_result.get("decision", {})
        top = d.get("top_features", [])
        lb  = self.last_result["steps"].get("leaderboard", [])

        print("\n" + "═" * 55)
        print("  EMMDS RESULT SUMMARY")
        print("═" * 55)
        print(f"  Best Model     : {d.get('best_model', 'N/A')}")
        print(f"  Task           : {d.get('task', 'N/A')}")
        print(f"  {d.get('primary_metric','Score').upper():14s} : {d.get('primary_score', 'N/A')}")
        print(f"  Accuracy       : {d.get('accuracy', 'N/A')}")
        print(f"  Trust Score    : {d.get('trust_score', 'N/A')} — {d.get('trust_label', '')}")
        print("─" * 55)
        print("  Top Features:")
        for f in top[:5]:
            print(f"    {f}")
        print("─" * 55)
        print("  Leaderboard:")
        for row in lb[:5]:
            print(f"    #{row['rank']}  {row['model']:25s}  {d.get('primary_metric','f1')}={row.get(d.get('primary_metric','f1'), 0):.4f}")
        print("═" * 55 + "\n")
