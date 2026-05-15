"""
EMMDS Model Recommender
Suggests the best subset of models to train based on dataset meta-features.
Handles both classification and regression tasks.

Rules are derived from empirical ML heuristics and meta-learning literature.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Backwards-compatible alias used by existing tests
ALL_MODELS = [
    "logistic_regression", "decision_tree", "random_forest", "extra_trees",
    "gradient_boosting", "hist_gradient_boosting", "knn", "naive_bayes", "mlp",
]

ALL_CLASSIFICATION_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "knn",
    "naive_bayes",
    "mlp",
]

ALL_REGRESSION_MODELS = [
    "ridge",
    "elasticnet",
    "decision_tree_reg",
    "random_forest_reg",
    "extra_trees_reg",
    "gradient_boosting_reg",
    "hist_gradient_boosting_reg",
    "knn_reg",
    "mlp_reg",
]


class ModelRecommender:
    """
    Analyses dataset meta-features and returns a prioritised list
    of recommended models, with explanations for each decision.
    Supports both classification and regression tasks.
    """

    def __init__(self):
        self.recommended: list = []
        self.excluded: dict = {}
        self.reasoning: list = []
        self._task: str = "classification"

    def recommend(self, meta: dict, task: str = "classification") -> list:
        """
        Return a recommended list of model names based on meta-features.

        Args:
            meta: Output of MetaFeatureExtractor.extract()
            task: "classification" or "regression"

        Returns:
            List of model names to train
        """
        self._task     = task
        all_models     = (ALL_CLASSIFICATION_MODELS if task == "classification"
                          else ALL_REGRESSION_MODELS)

        n          = meta.get("n_samples", 0)
        p          = meta.get("n_features", 0)
        imbalance  = meta.get("imbalance_ratio") or 1.0
        missing    = meta.get("missing_ratio", 0.0)
        dim_ratio  = meta.get("dimensionality_ratio", 0.0)
        avg_corr   = meta.get("avg_abs_correlation", 0.0)
        n_classes  = meta.get("n_classes", 2)
        noise      = meta.get("noise_estimate", 0.0)

        include = set(all_models)
        self.reasoning = []
        self.excluded  = {}

        # ── Shared rules ─────────────────────────────────────────────

        # Rule 1: Very small dataset → avoid complex ensembles
        if n < 200:
            for m in self._ensemble_names(task):
                self._exclude(m, f"Dataset too small ({n} samples) — ensembles overfit easily")
            self._note("Small dataset: preferring simple/regularised models")

        # Rule 2: High dimensionality → KNN unreliable
        if dim_ratio > 0.1 or p > 100:
            for m in self._knn_names(task):
                self._exclude(m, f"High dimensionality (p={p}, p/n={dim_ratio:.3f}) — KNN suffers from curse of dimensionality")
            self._note("High-dim data: excluding KNN")

        # Rule 3: Many missing values → prefer HistGradientBoosting (handles natively)
        if missing > 0.3:
            self._note(
                f"High missing ratio ({missing:.2f}) — "
                "hist_gradient_boosting handles missing values natively; preferred"
            )

        # ── Classification-specific rules ────────────────────────────
        if task == "classification":

            # Rule 4: Large dataset → SVM too slow (classification only)
            if n > 20_000:
                self._note("Large dataset: SVM not in registry (excluded by default)")

            # Rule 5: Strong class imbalance
            if imbalance and imbalance > 3.0:
                self._note(
                    f"Class imbalance ratio={imbalance:.1f} → "
                    "tree-based models prioritised (handle imbalance better)"
                )

            # Rule 6: Naive Bayes assumes feature independence
            if avg_corr > 0.5:
                self._exclude(
                    "naive_bayes",
                    f"High avg feature correlation ({avg_corr:.2f}) — "
                    "Naive Bayes assumes independence (violated)"
                )

            # Rule 7: Noisy data → avoid single tree
            if noise > 2.0:
                self._exclude(
                    "decision_tree",
                    f"High noise ({noise:.2f}) — single trees unstable on noisy data"
                )
                self._note("Noisy data: preferring ensembles over single tree")

            # Rule 8: Many classes → Naive Bayes may struggle
            if n_classes > 10:
                self._exclude(
                    "naive_bayes",
                    f"Many classes ({n_classes}) — NB often underperforms in high-cardinality multi-class"
                )

            # Fallbacks always included
            for fb in ["logistic_regression", "random_forest"]:
                include.add(fb)

        # ── Regression-specific rules ─────────────────────────────────
        else:
            # Rule R1: Noisy data → avoid single tree
            if noise > 2.0:
                self._exclude(
                    "decision_tree_reg",
                    f"High noise ({noise:.2f}) — single trees unstable; use ensemble"
                )

            # Rule R2: High correlation → regularised linear models preferred
            if avg_corr > 0.7:
                self._note(
                    f"High feature correlation ({avg_corr:.2f}) → "
                    "ridge/elasticnet preferred to handle multicollinearity"
                )

            # Fallbacks always included
            for fb in ["ridge", "random_forest_reg"]:
                include.add(fb)

        # Remove excluded, preserve registry order
        include -= set(self.excluded.keys())
        self.recommended = [m for m in all_models if m in include]

        logger.info(
            f"Recommender [{task}] → {len(self.recommended)} models: {self.recommended}\n"
            f"  Excluded: {list(self.excluded.keys())}"
        )
        return self.recommended

    # ── Helpers ───────────────────────────────────────────────────────

    def _ensemble_names(self, task: str) -> list:
        if task == "classification":
            return ["random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting"]
        return ["random_forest_reg", "extra_trees_reg", "gradient_boosting_reg", "hist_gradient_boosting_reg"]

    def _knn_names(self, task: str) -> list:
        return ["knn"] if task == "classification" else ["knn_reg"]

    def _exclude(self, model: str, reason: str):
        self.excluded[model] = reason
        logger.debug(f"Excluding {model}: {reason}")

    def _note(self, msg: str):
        self.reasoning.append(msg)

    def get_report(self) -> dict:
        return {
            "recommended": self.recommended,
            "excluded": self.excluded,
            "reasoning": self.reasoning,
            "total_recommended": len(self.recommended),
            "total_excluded": len(self.excluded),
            "task": self._task,
        }
