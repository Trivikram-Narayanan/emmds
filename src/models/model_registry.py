"""
EMMDS Model Registry  v2.0
===========================
10 models across 5 families:
  Classical:   Logistic Regression, Linear Discriminant Analysis
  Tree-based:  Decision Tree, Random Forest, Extra Trees
  Boosting:    Gradient Boosting, HistGradientBoosting (sklearn XGBoost-equivalent)
  Instance:    K-Nearest Neighbours
  Probabilistic: Naive Bayes
  Neural:      MLP Neural Network

HistGradientBoostingClassifier is sklearn's modern gradient boosting
implementation using the same histogram-based algorithm as LightGBM/XGBoost.
MLP provides neural network evaluation where overconfidence is documented
(Guo et al. 2017).

XGBoost and LightGBM are included as optional extras — they are used
if the packages are installed, otherwise HistGradientBoosting serves
as the equivalent.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = get("training.random_state", 42)
MAX_ITER     = get("training.max_iter", 1000)


# ── Core registry ─────────────────────────────────────────────────────
CLASSIFICATION_MODELS: dict = {

    # Classical linear
    "logistic_regression": LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    ),
    "lda": LinearDiscriminantAnalysis(),

    # Tree-based
    "decision_tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=10,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    # Boosting
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        learning_rate=0.1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=100,
        random_state=RANDOM_STATE,
        # Same histogram-based algorithm as LightGBM/XGBoost
        # Handles missing values natively
    ),

    # Instance-based
    "knn": KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1,
    ),

    # Probabilistic
    "naive_bayes": GaussianNB(),

    # Neural network
    "mlp": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    ),
}

# ── Optional: XGBoost / LightGBM if installed ─────────────────────────
try:
    from xgboost import XGBClassifier
    CLASSIFICATION_MODELS["xgboost"] = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
        use_label_encoder=False,
    )
    logger.info("XGBoost detected and added to registry")
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier
    CLASSIFICATION_MODELS["lightgbm"] = LGBMClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    logger.info("LightGBM detected and added to registry")
except ImportError:
    pass

# ── Default enabled set (config-driven) ───────────────────────────────
DEFAULT_ENABLED = [
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
# Add optional boosters if present
for _opt in ("xgboost", "lightgbm"):
    if _opt in CLASSIFICATION_MODELS:
        DEFAULT_ENABLED.append(_opt)


def get_model(name: str):
    """Return a fresh unfitted clone of a model by name."""
    from sklearn.base import clone
    if name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Model '{name}' not in registry. "
            f"Available: {list(CLASSIFICATION_MODELS.keys())}"
        )
    return clone(CLASSIFICATION_MODELS[name])


def get_all_models(enabled_only: bool = True) -> dict:
    """
    Return dict of {name: fresh_model_instance}.
    If enabled_only=True, returns config-driven subset.
    """
    from sklearn.base import clone
    enabled = get("models.enabled", DEFAULT_ENABLED) if enabled_only else list(CLASSIFICATION_MODELS.keys())
    result = {}
    for name in enabled:
        if name in CLASSIFICATION_MODELS:
            result[name] = clone(CLASSIFICATION_MODELS[name])
        else:
            logger.warning(f"Model '{name}' in enabled list but not in registry — skipping")
    logger.info(f"Loaded {len(result)} model(s): {list(result.keys())}")
    return result


def list_available_models() -> list:
    return list(CLASSIFICATION_MODELS.keys())


def get_model_family(name: str) -> str:
    """Return the model family for reporting purposes."""
    families = {
        "logistic_regression": "Linear",
        "lda":                 "Linear",
        "decision_tree":       "Tree",
        "random_forest":       "Ensemble",
        "extra_trees":         "Ensemble",
        "gradient_boosting":   "Boosting",
        "hist_gradient_boosting": "Boosting",
        "knn":                 "Instance",
        "naive_bayes":         "Probabilistic",
        "mlp":                 "Neural Network",
        "xgboost":             "Boosting",
        "lightgbm":            "Boosting",
    }
    return families.get(name, "Unknown")
