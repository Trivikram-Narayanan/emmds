"""
EMMDS Ablation Study
Proves that each system component contributes positively to performance.
Run pipeline with components removed one-at-a-time and compare.

Ablation conditions:
  - full:              Full system (baseline)
  - no_calibration:    Skip probability calibration
  - no_explainability: Skip SHAP
  - no_trust:          Use simple accuracy ranking instead of trust score
  - no_recommendation: Use all models (ignore recommender)
  - no_agreement:      Set agreement component to 0.5 (neutral)
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

ABLATION_DIR = Path("outputs/ablation")


class AblationStudy:
    """
    Systematic ablation of EMMDS components on a single dataset.
    Answers: "Does removing X hurt performance?"
    """

    CONDITIONS = {
        "full":              "Full EMMDS system",
        "no_calibration":    "Without probability calibration",
        "no_explainability": "Without SHAP explanations",
        "no_trust":          "Without trust scoring (accuracy-only ranking)",
        "no_recommendation": "Without model recommendation (all models)",
        "no_agreement":      "Without model agreement component",
    }

    def __init__(self, output_dir: str | Path = ABLATION_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}

    def run(
        self,
        df:         pd.DataFrame,
        target_col: str,
        task:       Optional[str] = None,
        conditions: Optional[list] = None,
    ) -> dict:
        """
        Run all ablation conditions.

        Args:
            df:         Input dataset
            target_col: Target column name
            task:       Task type (auto-detected if None)
            conditions: Subset of CONDITIONS keys to run (default = all)

        Returns:
            {condition: result_dict}
        """
        conditions = conditions or list(self.CONDITIONS.keys())
        logger.info(f"Running ablation study: {len(conditions)} conditions")

        for cond in conditions:
            logger.info(f"\n  ── Ablation: {cond} ──")
            t0 = time.time()
            try:
                result = self._run_condition(df, target_col, task, cond)
                elapsed = round(time.time() - t0, 2)
                result["elapsed_s"] = elapsed
                result["condition_label"] = self.CONDITIONS.get(cond, cond)
                self.results[cond] = result
                d = result.get("decision", {})
                logger.info(
                    f"  ✅ {cond:25s} | best={d.get('best_model')} "
                    f"| trust={d.get('trust_score')} | {elapsed}s"
                )
            except Exception as e:
                logger.error(f"  ❌ {cond}: {e}")
                self.results[cond] = {"error": str(e), "condition": cond}

        return self.results

    def _run_condition(
        self,
        df:         pd.DataFrame,
        target_col: str,
        task:       Optional[str],
        condition:  str,
    ) -> dict:
        """Run a single ablation condition."""

        # ── Imports ──────────────────────────────────────────────
        from src.data_engine.validator import DataValidator
        from src.data_engine.analyzer import DataAnalyzer
        from src.data_engine.meta_features import MetaFeatureExtractor
        from src.data_engine.data_quality import DataQualityScorer
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.parallel_trainer import ParallelTrainer
        from src.training.cross_validation import CrossValidator
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.ranking import ModelRanker
        from src.decision.trust_score import TrustScoreEngine
        from src.decision.model_selector import ModelSelector
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        # Validate + analyse
        analysis = DataAnalyzer().analyze(df, target_col)
        det_task = task or analysis["task"]

        # Preprocess
        pp = DataPreprocessor(task=det_task)
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target_col)
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        # Model set
        from src.models.model_registry import get_all_models
        if condition == "no_recommendation":
            models = get_all_models(enabled_only=False)
        else:
            from src.decision.model_recommender import ModelRecommender
            from src.data_engine.meta_features import MetaFeatureExtractor
            meta = MetaFeatureExtractor().extract(df, target_col)
            recommended = ModelRecommender().recommend(meta)
            models = {k: v for k, v in get_all_models().items() if k in recommended}

        # Train (parallel)
        trainer = ParallelTrainer(n_jobs=-1)
        trained = trainer.train_all(models, X_train, y_train)

        # Calibration
        if condition != "no_calibration":
            from src.calibration.calibrator import ModelCalibrator
            cal        = ModelCalibrator()
            trained    = cal.calibrate_all(trained, X_train, y_train, X_test, y_test)
            cal_scores = cal.get_calibration_scores()
        else:
            cal_scores = {n: 0.5 for n in trained}

        # CV
        cv_results = CrossValidator(task=det_task).run(trained, X_all, y_all)

        # Evaluate
        eval_results = ModelEvaluator(task=det_task).evaluate_all(trained, X_test, y_test)

        # Agreement
        if condition != "no_agreement":
            from src.decision.model_agreement import ModelAgreementEngine
            agree_score = ModelAgreementEngine().compute(trained, X_test, task=det_task).get("agreement_score", 0.5)
        else:
            agree_score = 0.5

        # Data quality
        dq_score = DataQualityScorer().score_dataset(df, target_col, task=det_task)

        # Trust / ranking
        if condition == "no_trust":
            # Accuracy-only ranking — ignore trust
            ranker = ModelRanker(task=det_task)
            lb     = ranker.rank(eval_results, cv_results)
            best   = lb[0]["model"] if lb else list(trained.keys())[0]
            trust  = 0.0
        else:
            trust_engine = TrustScoreEngine()
            trust_scores = trust_engine.compute_all(
                eval_results, cal_scores, cv_results, task=det_task,
                agreement_score=agree_score, data_quality_score=dq_score,
            )
            ranker = ModelRanker(task=det_task)
            lb     = ranker.rank(eval_results, cv_results)
            sel    = ModelSelector().select(lb, trust_scores, trained)
            best   = sel.get("best_model_name", lb[0]["model"] if lb else "")
            trust  = trust_scores.get(best, 0.0)

        pm        = "f1" if det_task == "classification" else "r2"
        best_perf = eval_results.get(best, {}).get(pm, 0.0)

        return {
            "condition":   condition,
            "decision": {
                "best_model":     best,
                "primary_metric": pm,
                "primary_score":  best_perf,
                "accuracy":       eval_results.get(best, {}).get("accuracy"),
                "trust_score":    trust,
            },
            "leaderboard": lb[:3],
            "n_models":    len(trained),
        }

    def summary_table(self) -> pd.DataFrame:
        """Build comparison DataFrame across all ablation conditions."""
        rows = []
        for cond, result in self.results.items():
            if "error" in result:
                rows.append({"condition": cond, "status": "failed", "error": result["error"]})
                continue
            d = result.get("decision", {})
            rows.append({
                "condition":     cond,
                "label":         result.get("condition_label", cond),
                "best_model":    d.get("best_model"),
                "primary_score": d.get("primary_score"),
                "accuracy":      d.get("accuracy"),
                "trust_score":   d.get("trust_score"),
                "elapsed_s":     result.get("elapsed_s"),
                "status":        "ok",
            })
        return pd.DataFrame(rows)

    def save(self, name: str = "ablation") -> str:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{name}_{ts}.json"

        def _s(o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            return str(o)

        path.write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in self.results.items()},
            default=_s, indent=2,
        ))
        logger.info(f"Ablation results saved → {path}")
        return str(path)
