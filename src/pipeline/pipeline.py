"""
EMMDS Pipeline  v3.0  — Research-Grade
========================================
Clean single-pass flow:
  meta-features → model recommendation → sklearn pipelines →
  CV (train only, no leakage) → evaluation → calibration →
  agreement → explainability → trust → decision → report

Fixes over v2:
  - Removed duplicate raw ModelTrainer path; sklearn pipelines are the sole
    training mechanism.
  - CV runs on X_train_raw only (no test leakage).
  - /api/predict works directly against the stored sklearn pipeline objects.
  - Regression path is fully wired and tested.
  - Settings.yaml trust_score weights are ignored; empirical weights in
    TrustScoreEngine are the authority.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EMPipeline:
    """Single .run() call executes the entire EMMDS system."""

    def __init__(self):
        self.result: dict = {}

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: Optional[str] = None,
        scaler: str = "standard",
        dataset_name: str = "dataset",
        explain_instance: Optional[np.ndarray] = None,
        save_models: bool = False,
        track: bool = True,
    ) -> dict:

        self._log("STARTING EMMDS PIPELINE v3.0")
        steps = {}

        # ── 1. VALIDATION ─────────────────────────────────────────────
        self._log("Step 1/9: Validating data")
        from src.data_engine.validator import DataValidator
        validation = DataValidator().validate(df, target_col)
        steps["validation"] = validation
        if not validation["passed"]:
            logger.error("Pipeline halted: validation failed")
            return {"error": "Validation failed", "details": validation}

        # ── 2. ANALYSIS + META-FEATURES ───────────────────────────────
        self._log("Step 2/9: Analysing dataset + extracting meta-features")
        from src.data_engine.analyzer import DataAnalyzer
        from src.data_engine.profiler import DataProfiler
        from src.data_engine.meta_features import MetaFeatureExtractor
        from src.data_engine.data_quality import DataQualityScorer

        analysis = DataAnalyzer().analyze(df, target_col)
        task = task or analysis["task"]

        profiler = DataProfiler()
        profile  = profiler.profile_dataframe(df, target_col)

        meta_extractor = MetaFeatureExtractor()
        meta = meta_extractor.extract(df, target_col)

        dq_scorer = DataQualityScorer()
        dq_score  = dq_scorer.score_dataset(df, target_col, task=task)

        steps["analysis"]      = analysis
        steps["profile"]       = {k: v for k, v in profile.items()
                                   if k != "correlation_matrix"}
        steps["meta_features"] = meta
        steps["data_quality"]  = dq_scorer.get_breakdown()

        # ── 3. MODEL RECOMMENDATION ───────────────────────────────────
        self._log("Step 3/9: Model recommendation")
        from src.decision.model_recommender import ModelRecommender
        recommender  = ModelRecommender()
        recommended  = recommender.recommend(meta, task=task)
        steps["recommendation"] = recommender.get_report()

        # ── 4. EXPERIMENT TRACKING (start) ────────────────────────────
        from src.utils.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker()
        if track:
            tracker.start_run(
                dataset_name=dataset_name, task=task,
                target_col=target_col,
                n_samples=meta["n_samples"], n_features=meta["n_features"],
                params={"scaler": scaler, "recommended_models": recommended},
            )

        # ── 5. SPLIT + BUILD SKLEARN PIPELINES ────────────────────────
        self._log("Step 4/9: Splitting + building sklearn Pipelines")
        from src.training.data_split import DataSplitter
        from src.training.pipeline_builder import build_all_pipelines

        if task == "regression":
            from src.models.regression_registry import get_all_regression_models
            all_models = get_all_regression_models(enabled_only=True)
        else:
            from src.models.model_registry import get_all_models
            all_models = get_all_models(enabled_only=True)

        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col]

        splitter = DataSplitter(task=task)
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitter.split(X_raw, y_raw)

        from sklearn.preprocessing import LabelEncoder
        le = None
        if task == "classification":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_raw)
            y_test  = le.transform(y_test_raw)
        else:
            y_train = pd.to_numeric(y_train_raw, errors="coerce").fillna(0).to_numpy()
            y_test  = pd.to_numeric(y_test_raw,  errors="coerce").fillna(0).to_numpy()

        num_cols = list(X_train_raw.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X_train_raw.select_dtypes(
            include=["object", "category", "bool"]).columns)
        feature_names = num_cols + cat_cols

        model_subset = {k: v for k, v in all_models.items() if k in recommended}
        if not model_subset:
            logger.warning("No recommended models matched registry. Using all enabled.")
            model_subset = all_models

        model_pipelines = build_all_pipelines(
            model_subset, num_cols, cat_cols, scaler, X_sample=X_train_raw
        )

        steps["preprocessing"] = {
            "strategy":      splitter.get_split_info().get("strategy"),
            "train_shape":   (len(X_train_raw), len(feature_names)),
            "test_shape":    (len(X_test_raw),  len(feature_names)),
            "feature_names": feature_names,
            "num_cols":      num_cols,
            "cat_cols":      cat_cols,
            "scaler":        scaler,
        }

        # ── 6. TRAINING (sklearn pipelines only) ──────────────────────
        self._log("Step 5/9: Training sklearn Pipelines")
        from src.training.cross_validation import CrossValidator

        trained_models: dict = {}
        train_times:    dict = {}
        failed_models:  list = []

        import time
        for name, pipe in model_pipelines.items():
            try:
                t0 = time.time()
                pipe.fit(X_train_raw, y_train)
                train_times[name] = round(time.time() - t0, 3)
                trained_models[name] = pipe
                logger.info(f"  ✅ {name:30s} trained in {train_times[name]:.3f}s")
            except Exception as e:
                failed_models.append(name)
                logger.warning(f"  ❌ {name}: {e}")

        # CV on training data only — no leakage from test set
        cv = CrossValidator(task=task)
        cv_results = cv.run(trained_models, X_train_raw, y_train)

        steps["training"] = {
            "trained":       list(trained_models.keys()),
            "failed":        failed_models,
            "training_times": train_times,
            "total_time":    round(sum(train_times.values()), 3),
        }
        steps["cv_results"] = cv_results

        # ── 7. EVALUATION + CALIBRATION ───────────────────────────────
        self._log("Step 6/9: Evaluating + calibrating")
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.ranking import ModelRanker
        from src.calibration.calibrator import ModelCalibrator

        evaluator = ModelEvaluator(task=task)
        eval_results = evaluator.evaluate_all(trained_models, X_test_raw, y_test)

        if track:
            for name, metrics in eval_results.items():
                tracker.log_model_result(name, metrics)

        calibrator = ModelCalibrator(task=task)
        calibrated_models = calibrator.calibrate_all(
            trained_models,
            X_train_raw, y_train,
            X_test_raw,  y_test,
        )
        calibration_scores = calibrator.get_calibration_scores()

        ranker = ModelRanker(task=task)
        leaderboard = ranker.rank(eval_results, cv_results)

        steps["evaluation"]         = eval_results
        steps["calibration_scores"] = calibration_scores
        steps["leaderboard"]        = leaderboard

        # ── 8. MODEL AGREEMENT ────────────────────────────────────────
        self._log("Step 7/9: Computing model agreement")
        from src.decision.model_agreement import ModelAgreementEngine
        agreement_engine = ModelAgreementEngine()
        agreement_result = agreement_engine.compute(
            calibrated_models, X_test_raw, task=task)
        agreement_score  = agreement_result.get("agreement_score", 0.5)
        steps["agreement"] = agreement_result

        # ── 9. EXPLAINABILITY ─────────────────────────────────────────
        self._log("Step 8/9: Generating explanations")
        from src.explainability.shap_explainer import SHAPExplainer
        from src.explainability.lime_explainer import LIMEExplainer

        best_name_prelim  = ranker.get_best_model_name()
        best_model_prelim = calibrated_models.get(best_name_prelim)

        shap_global = {}
        lime_local  = {}

        try:
            shap_exp = SHAPExplainer(max_samples=100)
            if hasattr(best_model_prelim, "named_steps"):
                X_train_t = best_model_prelim.named_steps["preprocessor"].transform(
                    X_train_raw)
                X_test_t  = best_model_prelim.named_steps["preprocessor"].transform(
                    X_test_raw)
                inner_model = best_model_prelim.named_steps["model"]
                shap_exp.fit(inner_model, X_train_t, feature_names=feature_names)
                shap_global = shap_exp.explain_global(X_test_t)
            else:
                shap_exp.fit(best_model_prelim, X_train_raw.values,
                             feature_names=feature_names)
                shap_global = shap_exp.explain_global(X_test_raw.values)
        except Exception as e:
            logger.warning(f"SHAP skipped: {e}")

        if explain_instance is not None:
            try:
                lime_exp = LIMEExplainer()
                class_names = [str(c) for c in le.classes_] if le else None
                lime_exp.fit(
                    X_train_raw.values if hasattr(X_train_raw, "values")
                    else X_train_raw,
                    feature_names=feature_names,
                    class_names=class_names,
                    task=task,
                )
                lime_local = lime_exp.explain_instance(
                    explain_instance, best_model_prelim)
            except Exception as e:
                logger.warning(f"LIME skipped: {e}")

        steps["shap_global"] = shap_global
        steps["lime_local"]  = lime_local

        # ── 10. DECISION ENGINE ───────────────────────────────────────
        self._log("Step 9/9: Decision Engine")
        from src.decision.trust_score import TrustScoreEngine
        from src.decision.model_selector import ModelSelector
        from src.decision.decision_engine import DecisionEngine

        trust_engine = TrustScoreEngine()
        trust_scores = trust_engine.compute_all(
            eval_results=eval_results,
            calibration_scores=calibration_scores,
            cv_results=cv_results,
            task=task,
            agreement_score=agreement_score,
            data_quality_score=dq_score,
        )

        engine = DecisionEngine(task=task)
        final_decision = engine.decide(
            trained_models=calibrated_models,
            eval_results=eval_results,
            calibration_scores=calibration_scores,
            cv_results=cv_results,
            leaderboard=leaderboard,
            shap_global=shap_global,
            analysis_report=analysis,
        )
        final_decision["all_trust_scores"] = trust_scores
        final_decision["trust_score"]      = trust_scores.get(
            final_decision["best_model"], 0.0)
        final_decision["trust_label"]      = trust_engine.get_trust_label(
            final_decision["trust_score"])
        final_decision["trust_breakdown"]  = trust_engine.get_breakdown().get(
            final_decision["best_model"], {})
        final_decision["agreement"]        = agreement_result
        final_decision["data_quality"]     = dq_scorer.get_breakdown()
        final_decision["recommendation"]   = recommender.get_report()

        steps["decision"] = final_decision

        # ── SAVE MODELS (optional) ────────────────────────────────────
        if save_models:
            from src.utils.model_saver import ModelSaver
            saver = ModelSaver()
            saver.save_all(
                calibrated_models,
                eval_results=eval_results,
                trust_scores=trust_scores,
                feature_names=feature_names,
            )

        # ── HYPERPARAMETER TUNING on trust-selected model ─────────────
        try:
            from src.training.hyperparameter import TrustAwareTuner
            best_name  = final_decision.get("best_model")
            best_model = calibrated_models.get(best_name)
            if best_name and X_train_raw is not None:
                tuner = TrustAwareTuner(n_iter=15, cv_folds=3)
                dq_s    = steps.get("data_quality", {}).get("quality_score", 0.8)
                agr_s   = steps.get("agreement",    {}).get("agreement_score", 0.75)
                from src.models.model_registry import get_model
                try:
                    raw_model = get_model(best_name)
                except Exception:
                    raw_model = None
                Xtr_num = X_train_raw.select_dtypes(include=[float, int]).values
                Xte_num = X_test_raw.select_dtypes(include=[float, int]).values
                from sklearn.preprocessing import StandardScaler as _SS
                _sc  = _SS().fit(Xtr_num)
                tune_result = tuner.tune(
                    model_name=best_name,
                    model=raw_model or best_model,
                    X_train=_sc.transform(Xtr_num), y_train=y_train,
                    X_test=_sc.transform(Xte_num),  y_test=y_test,
                    agreement_score=agr_s, data_quality=dq_s,
                )
                if (tune_result.get("tuned")
                        and tune_result["trust_improvement"] > 0):
                    final_decision["hyperparameter_tuning"] = {
                        "tuned":             True,
                        "trust_before":      tune_result["before"]["trust"],
                        "trust_after":       tune_result["after"]["trust"],
                        "trust_improvement": tune_result["trust_improvement"],
                        "f1_change":         tune_result["f1_change"],
                        "best_params":       tune_result["best_params"],
                        "note": (f"Trust improved by "
                                 f"{tune_result['trust_improvement']:+.4f} after tuning"),
                    }
                    final_decision["trust_score"] = tune_result["after"]["trust"]
                    final_decision["trust_label"] = trust_engine.get_trust_label(
                        tune_result["after"]["trust"])
                else:
                    final_decision["hyperparameter_tuning"] = {
                        "tuned": True,
                        "trust_improvement": tune_result.get("trust_improvement", 0),
                        "note": "Default params already optimal",
                    }
        except Exception as e:
            final_decision["hyperparameter_tuning"] = {"tuned": False, "error": str(e)}

        # ── TRACKING: end run ─────────────────────────────────────────
        if track:
            best = final_decision.get("best_model")
            tracker.log_trust_scores(trust_scores)
            tracker.log_best_model(
                model_name=best,
                metrics=eval_results.get(best, {}),
                trust_score=final_decision.get("trust_score", 0.0),
            )
            tracker.end_run()

        # ── BUILD FINAL RESULT ────────────────────────────────────────
        self.result = {
            "status":          "success",
            "task":            task,
            "target_col":      target_col,
            "decision":        final_decision,
            "steps":           steps,
            # Sklearn pipeline objects — used directly by /api/predict
            "_trained_models": calibrated_models,
            "_feature_names":  feature_names,
            "_label_encoder":  le,
            "_num_cols":       num_cols,
            "_cat_cols":       cat_cols,
            # Train reference kept for LIME (not exposed to API clients)
            "_X_train_raw":    X_train_raw,
        }

        # ── GENERATE REPORT ───────────────────────────────────────────
        try:
            from src.utils.report_generator import ReportGenerator
            rg = ReportGenerator()
            rg.generate(self.result, dataset_name=dataset_name,
                        fmt="markdown", save=True)
            rg.generate(self.result, dataset_name=dataset_name,
                        fmt="text",     save=True)
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        self._log("PIPELINE COMPLETE ✅")
        return self.result

    def _log(self, msg: str):
        logger.info(f"\n{'─'*52}\n  {msg}\n{'─'*52}")
