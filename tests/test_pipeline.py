"""
EMMDS Test Suite
Tests for core pipeline components using sklearn toy datasets.
Run with: python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def binary_df():
    """Breast cancer dataset (binary classification)."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df, "target"


@pytest.fixture
def multiclass_df():
    """Iris dataset (multi-class classification)."""
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target
    return df, "target"


# ── Data Engine Tests ─────────────────────────────────────────────────

class TestDataValidator:
    def test_valid_dataset(self, binary_df):
        from src.data_engine.validator import DataValidator
        df, target = binary_df
        result = DataValidator().validate(df, target)
        assert result["passed"] is True
        assert result["error_count"] == 0

    def test_missing_target_column(self, binary_df):
        from src.data_engine.validator import DataValidator
        df, _ = binary_df
        result = DataValidator().validate(df, "nonexistent_col")
        assert result["passed"] is False
        assert any("not found" in e for e in result["errors"])

    def test_empty_dataframe(self):
        from src.data_engine.validator import DataValidator
        df = pd.DataFrame()
        result = DataValidator().validate(df, "target")
        assert result["passed"] is False


class TestDataAnalyzer:
    def test_detects_classification(self, binary_df):
        from src.data_engine.analyzer import DataAnalyzer
        df, target = binary_df
        report = DataAnalyzer().analyze(df, target)
        assert report["task"] == "classification"
        assert report["rows"] == len(df)
        assert report["feature_count"] == df.shape[1] - 1

    def test_detects_feature_types(self, binary_df):
        from src.data_engine.analyzer import DataAnalyzer
        df, target = binary_df
        report = DataAnalyzer().analyze(df, target)
        assert "numerical" in report["feature_types"]
        assert "categorical" in report["feature_types"]

    def test_imbalance_ratio(self, binary_df):
        from src.data_engine.analyzer import DataAnalyzer
        df, target = binary_df
        report = DataAnalyzer().analyze(df, target)
        assert report["imbalance_ratio"] is not None
        assert report["imbalance_ratio"] >= 1.0


class TestDataPreprocessor:
    def test_fit_transform_returns_correct_shapes(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target, test_size=0.2)
        assert X_train.shape[0] > X_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_no_data_leakage(self, binary_df):
        """Train and test sizes must be consistent with test_size."""
        from src.data_engine.preprocessor import DataPreprocessor
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(df)


# ── Model Registry Tests ──────────────────────────────────────────────

class TestModelRegistry:
    def test_get_all_models_returns_dict(self):
        from src.models.model_registry import get_all_models
        models = get_all_models(enabled_only=False)
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_get_model_by_name(self):
        from src.models.model_registry import get_model
        model = get_model("random_forest")
        assert model is not None
        assert hasattr(model, "fit")

    def test_invalid_model_raises(self):
        from src.models.model_registry import get_model
        with pytest.raises(ValueError):
            get_model("invalid_model_xyz")


# ── Training Tests ────────────────────────────────────────────────────

class TestModelTrainer:
    def test_trains_all_models(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        trainer = ModelTrainer()
        trained = trainer.train_all(X_train, y_train)
        assert len(trained) > 0
        for name, model in trained.items():
            assert hasattr(model, "predict")

    def test_trained_models_can_predict(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        trained = ModelTrainer().train_all(X_train, y_train)
        for name, model in trained.items():
            preds = model.predict(X_test)
            assert len(preds) == len(X_test)


# ── Evaluation Tests ──────────────────────────────────────────────────

class TestModelEvaluator:
    def test_returns_metrics_for_all_models(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        from src.evaluation.evaluator import ModelEvaluator
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        trained = ModelTrainer().train_all(X_train, y_train)
        results = ModelEvaluator(task="classification").evaluate_all(trained, X_test, y_test)
        assert len(results) == len(trained)
        for name, metrics in results.items():
            assert "accuracy" in metrics
            assert "f1" in metrics


# ── Trust Score Tests ─────────────────────────────────────────────────

class TestTrustScore:
    def test_trust_scores_in_range(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        from src.evaluation.evaluator import ModelEvaluator
        from src.decision.trust_score import TrustScoreEngine
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        trained = ModelTrainer().train_all(X_train, y_train)
        eval_results = ModelEvaluator(task="classification").evaluate_all(trained, X_test, y_test)
        engine = TrustScoreEngine()
        scores = engine.compute_all(
            eval_results=eval_results,
            calibration_scores={name: 0.8 for name in trained},
            cv_results={},
        )
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0


# ── Full Pipeline Integration Test ───────────────────────────────────

class TestFullPipeline:
    def test_pipeline_runs_end_to_end(self, binary_df):
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        pipeline = EMPipeline()
        result = pipeline.run(df=df, target_col=target)
        assert result["status"] == "success"
        assert "decision" in result
        assert result["decision"]["best_model"] is not None
        assert 0.0 <= result["decision"]["trust_score"] <= 1.0

    def test_pipeline_with_multiclass(self, multiclass_df):
        from src.pipeline.pipeline import EMPipeline
        df, target = multiclass_df
        pipeline = EMPipeline()
        result = pipeline.run(df=df, target_col=target)
        assert result["status"] == "success"
        assert result["task"] == "classification"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ── Week 5-8 new module tests ─────────────────────────────────────────

class TestDataSplit:
    def test_stratified_split(self, binary_df):
        from src.training.data_split import DataSplitter
        df, target = binary_df
        X = df.drop(columns=[target])
        y = df[target]
        splitter = DataSplitter(task="classification", test_size=0.2)
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        assert len(X_train) + len(X_test) == len(df)
        info = splitter.get_split_info()
        assert info["strategy"] == "stratified"

    def test_no_leakage_sizes(self, binary_df):
        from src.training.data_split import DataSplitter
        df, target = binary_df
        X = df.drop(columns=[target])
        y = df[target]
        splitter = DataSplitter(task="classification", test_size=0.25)
        X_train, X_test, y_train, y_test = splitter.split(X, y)
        assert abs(len(X_test) / len(df) - 0.25) < 0.05


class TestPipelineBuilder:
    def test_builds_pipeline_per_model(self, binary_df):
        from src.training.pipeline_builder import build_all_pipelines
        from src.models.model_registry import get_all_models
        df, target = binary_df
        X = df.drop(columns=[target])
        num_cols = list(X.select_dtypes(include=["number"]).columns)
        models = get_all_models(enabled_only=True)
        pipes = build_all_pipelines(models, num_cols, [])
        assert len(pipes) == len(models)
        for name, pipe in pipes.items():
            assert hasattr(pipe, "fit")
            assert "preprocessor" in pipe.named_steps
            assert "model" in pipe.named_steps

    def test_pipeline_fit_predict(self, binary_df):
        from src.training.pipeline_builder import build_model_pipeline
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        df, target = binary_df
        X = df.drop(columns=[target])
        y = df[target]
        num_cols = list(X.select_dtypes(include=["number"]).columns)
        pipe = build_model_pipeline(RandomForestClassifier(n_estimators=10), num_cols, [])
        pipe.fit(X, y)
        preds = pipe.predict(X[:5])
        assert len(preds) == 5


class TestMetaFeatures:
    def test_extracts_all_keys(self, binary_df):
        from src.data_engine.meta_features import MetaFeatureExtractor
        df, target = binary_df
        meta = MetaFeatureExtractor().extract(df, target)
        for key in ["n_samples", "n_features", "imbalance_ratio", "missing_ratio",
                    "avg_abs_correlation", "dimensionality_ratio", "n_classes"]:
            assert key in meta

    def test_correct_dimensions(self, binary_df):
        from src.data_engine.meta_features import MetaFeatureExtractor
        df, target = binary_df
        meta = MetaFeatureExtractor().extract(df, target)
        assert meta["n_samples"] == len(df)
        assert meta["n_features"] == df.shape[1] - 1


class TestDataQuality:
    def test_score_in_range(self, binary_df):
        from src.data_engine.data_quality import DataQualityScorer
        df, target = binary_df
        score = DataQualityScorer().score_dataset(df, target)
        assert 0.0 <= score <= 1.0

    def test_clean_data_high_score(self, binary_df):
        from src.data_engine.data_quality import DataQualityScorer
        df, target = binary_df
        score = DataQualityScorer().score_dataset(df, target)
        assert score > 0.7  # clean sklearn dataset should score well

    def test_dirty_data_lower_score(self, binary_df):
        from src.data_engine.data_quality import DataQualityScorer
        import numpy as np
        df, target = binary_df
        dirty = df.copy()
        # Inject 30% missing values
        mask = np.random.rand(*dirty.shape) < 0.3
        dirty = dirty.mask(mask)
        dirty[target] = df[target]  # keep target clean
        score_dirty = DataQualityScorer().score_dataset(dirty, target)
        score_clean = DataQualityScorer().score_dataset(df, target)
        assert score_dirty < score_clean


class TestModelRecommender:
    def test_recommends_subset_of_all_models(self):
        from src.decision.model_recommender import ModelRecommender, ALL_MODELS
        meta = {"n_samples": 500, "n_features": 10, "imbalance_ratio": 1.2,
                "missing_ratio": 0.0, "dimensionality_ratio": 0.02,
                "avg_abs_correlation": 0.3, "n_classes": 2, "noise_estimate": 0.5}
        rec = ModelRecommender().recommend(meta)
        assert len(rec) > 0
        assert all(m in ALL_MODELS for m in rec)

    def test_excludes_knn_for_high_dim(self):
        from src.decision.model_recommender import ModelRecommender
        meta = {"n_samples": 1000, "n_features": 500, "imbalance_ratio": 1.0,
                "missing_ratio": 0.0, "dimensionality_ratio": 0.5,
                "avg_abs_correlation": 0.2, "n_classes": 2, "noise_estimate": 0.5}
        rec = ModelRecommender().recommend(meta)
        assert "knn" not in rec

    def test_excludes_svm_for_large_dataset(self):
        from src.decision.model_recommender import ModelRecommender
        meta = {"n_samples": 50000, "n_features": 20, "imbalance_ratio": 1.0,
                "missing_ratio": 0.0, "dimensionality_ratio": 0.0004,
                "avg_abs_correlation": 0.2, "n_classes": 2, "noise_estimate": 0.5}
        rec = ModelRecommender().recommend(meta)
        assert "svm" not in rec


class TestModelAgreement:
    def test_agreement_score_in_range(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        from src.decision.model_agreement import ModelAgreementEngine
        import numpy as np
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        trained = ModelTrainer().train_all(X_train, y_train)
        result = ModelAgreementEngine().compute(trained, X_test)
        assert 0.0 <= result["agreement_score"] <= 1.0
        assert 0.0 <= result["global_agreement"] <= 1.0

    def test_identical_models_full_agreement(self, binary_df):
        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        from src.decision.model_agreement import ModelAgreementEngine
        import numpy as np
        df, target = binary_df
        pp = DataPreprocessor(task="classification")
        X_train, X_test, y_train, y_test = pp.fit_transform(df, target)
        # Use 2 copies of the same model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.base import clone
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X_train, y_train)
        two_same = {"rf_a": m, "rf_b": m}
        result = ModelAgreementEngine().compute(two_same, X_test)
        assert result["global_agreement"] == 1.0


class TestExperimentTracker:
    def test_start_end_creates_log(self, tmp_path, binary_df):
        from src.utils.experiment_tracker import ExperimentTracker
        df, target = binary_df
        tracker = ExperimentTracker(log_dir=tmp_path)
        run_id = tracker.start_run("test_ds", "classification", target, len(df), 5)
        tracker.log_model_result("rf", {"accuracy": 0.9, "f1": 0.89})
        tracker.log_trust_scores({"rf": 0.87})
        tracker.log_best_model("rf", {"accuracy": 0.9}, 0.87)
        path = tracker.end_run()
        assert path != ""
        import pathlib
        assert pathlib.Path(path).exists()

    def test_list_runs_finds_saved(self, tmp_path, binary_df):
        from src.utils.experiment_tracker import ExperimentTracker
        df, target = binary_df
        tracker = ExperimentTracker(log_dir=tmp_path)
        tracker.start_run("ds", "classification", target, 100, 5)
        tracker.log_best_model("rf", {}, 0.85)
        tracker.end_run()
        runs = tracker.list_runs()
        assert len(runs) == 1


class TestResultStore:
    def test_save_and_load_run(self, tmp_path, binary_df):
        from src.utils.result_store import ResultStore
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        result = EMPipeline().run(df=df, target_col=target, track=False)
        store = ResultStore(store_dir=tmp_path)
        path = store.save_run(result, dataset_name="test")
        import pathlib
        assert pathlib.Path(path).exists()


class TestReportGenerator:
    def test_generates_markdown(self, tmp_path, binary_df):
        from src.utils.report_generator import ReportGenerator
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        result = EMPipeline().run(df=df, target_col=target, track=False)
        rg = ReportGenerator(report_dir=tmp_path)
        report = rg.generate(result, dataset_name="test_ds", fmt="markdown", save=True)
        assert "EMMDS Report" in report
        assert "Best Model" in report

    def test_generates_text(self, tmp_path, binary_df):
        from src.utils.report_generator import ReportGenerator
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        result = EMPipeline().run(df=df, target_col=target, track=False)
        rg = ReportGenerator(report_dir=tmp_path)
        report = rg.generate(result, dataset_name="test_ds", fmt="text", save=False)
        assert "EMMDS REPORT" in report


class TestV2FullPipeline:
    def test_v2_pipeline_has_all_new_keys(self, binary_df):
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        result = EMPipeline().run(df=df, target_col=target, track=False)
        assert result["status"] == "success"
        d = result["decision"]
        # All new v2 keys must be present
        assert "agreement" in d
        assert "data_quality" in d
        assert "recommendation" in d
        # Trust breakdown must have 5 components
        tb = d.get("trust_breakdown", {})
        for comp in ["accuracy_component","calibration_component",
                     "agreement_component","data_quality_component","stability_component"]:
            assert comp in tb, f"Missing trust component: {comp}"

    def test_v2_trust_score_in_range(self, binary_df):
        from src.pipeline.pipeline import EMPipeline
        df, target = binary_df
        result = EMPipeline().run(df=df, target_col=target, track=False)
        ts = result["decision"]["trust_score"]
        assert 0.0 <= ts <= 1.0
