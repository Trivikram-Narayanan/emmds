"""
EMMDS Benchmark Engine
Evaluates the full EMMDS pipeline across multiple datasets.
Produces a comparative report showing which models win on which data profiles.
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

BENCH_DIR = Path("outputs/benchmarks")


class BenchmarkEngine:
    """
    Runs the EMMDS pipeline on multiple datasets and aggregates results
    into a comparison matrix for research reporting.

    Usage:
        bench = BenchmarkEngine()
        bench.add_dataset("breast_cancer", df_bc, "target")
        bench.add_dataset("iris",          df_iris, "target")
        bench.add_dataset("wine",          df_wine, "target")
        report = bench.run()
        bench.save_report(report)
    """

    def __init__(self, output_dir: str | Path = BENCH_DIR):
        self.datasets: list  = []   # [(name, df, target_col)]
        self.results:  list  = []   # Per-dataset result rows
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_dataset(
        self,
        name:       str,
        df:         pd.DataFrame,
        target_col: str,
        task:       Optional[str] = None,
    ) -> "BenchmarkEngine":
        self.datasets.append((name, df, target_col, task))
        return self

    def add_sklearn_dataset(self, name: str) -> "BenchmarkEngine":
        """Load a standard sklearn toy dataset by name."""
        from sklearn import datasets as skds
        loaders = {
            "breast_cancer": skds.load_breast_cancer,
            "iris":          skds.load_iris,
            "wine":          skds.load_wine,
            "digits":        skds.load_digits,
        }
        if name not in loaders:
            logger.warning(f"Unknown sklearn dataset: {name}")
            return self
        data = loaders[name](as_frame=True)
        df   = data.frame.copy()
        df["target"] = data.target
        return self.add_dataset(name, df, "target")

    def run(self, scaler: str = "standard") -> dict:
        """
        Run the full pipeline on every registered dataset.

        Returns:
            {
                "summary":   DataFrame as records,
                "per_dataset": {name: pipeline_result},
                "timestamp": str,
            }
        """
        from src.pipeline.pipeline import EMPipeline

        logger.info(f"Benchmark starting: {len(self.datasets)} datasets")
        per_dataset = {}
        rows        = []

        for name, df, target_col, task in self.datasets:
            logger.info(f"\n  ── Benchmarking: {name} ──")
            t0 = time.time()
            try:
                pipeline = EMPipeline()
                result   = pipeline.run(
                    df=df, target_col=target_col,
                    task=task, scaler=scaler,
                    dataset_name=name, track=False,
                )
                elapsed  = round(time.time() - t0, 2)
                d        = result.get("decision", {})
                dq       = d.get("data_quality", {})

                rows.append({
                    "dataset":       name,
                    "task":          d.get("task"),
                    "n_rows":        d.get("dataset_info", {}).get("rows"),
                    "n_features":    d.get("dataset_info", {}).get("features"),
                    "best_model":    d.get("best_model"),
                    "primary_metric": d.get("primary_metric"),
                    "primary_score": d.get("primary_score"),
                    "accuracy":      d.get("accuracy"),
                    "trust_score":   d.get("trust_score"),
                    "trust_label":   d.get("trust_label"),
                    "data_quality":  dq.get("quality_score"),
                    "agreement":     d.get("agreement", {}).get("agreement_score"),
                    "elapsed_s":     elapsed,
                    "status":        "success",
                })
                per_dataset[name] = result
                logger.info(f"  ✅ {name} | best={d.get('best_model')} "
                            f"| trust={d.get('trust_score')} | {elapsed}s")

            except Exception as e:
                elapsed = round(time.time() - t0, 2)
                logger.error(f"  ❌ {name} failed: {e}")
                rows.append({
                    "dataset": name, "status": "failed", "error": str(e),
                    "elapsed_s": elapsed,
                })

        summary_df  = pd.DataFrame(rows)
        self.results = rows

        report = {
            "timestamp":   datetime.now().isoformat(),
            "n_datasets":  len(self.datasets),
            "n_success":   sum(1 for r in rows if r.get("status") == "success"),
            "summary":     rows,
            "per_dataset": {k: {kk: vv for kk, vv in v.items()
                                if not kk.startswith("_")}
                            for k, v in per_dataset.items()},
        }
        logger.info(
            f"\nBenchmark complete: {report['n_success']}/{report['n_datasets']} succeeded"
        )
        return report

    def save_report(self, report: dict, name: str = "benchmark") -> str:
        """Save benchmark report as JSON."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}_{ts}.json"

        def _serial(obj):
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, np.ndarray):      return obj.tolist()
            return str(obj)

        path.write_text(json.dumps(report, default=_serial, indent=2))
        logger.info(f"Benchmark report saved → {path}")
        return str(path)

    def print_summary(self) -> None:
        """Print a readable comparison table."""
        if not self.results:
            print("No benchmark results yet.")
            return
        df = pd.DataFrame(self.results)
        cols = [c for c in ["dataset","best_model","primary_score",
                             "trust_score","data_quality","elapsed_s"] if c in df.columns]
        print("\n" + "="*70)
        print("  EMMDS BENCHMARK SUMMARY")
        print("="*70)
        print(df[cols].to_string(index=False))
        print("="*70 + "\n")
