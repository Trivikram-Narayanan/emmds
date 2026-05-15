"""
EMMDS Regression Model Registry
================================
9 regressors across 5 families matching the classification registry structure.
HistGradientBoostingRegressor handles missing values natively (no imputation needed).
"""
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from src.utils.config import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = get("training.random_state", 42)
MAX_ITER     = get("training.max_iter", 1000)

REGRESSION_MODELS: dict = {

    # Linear
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=0.1, max_iter=MAX_ITER),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=MAX_ITER),
    "bayesian_ridge": BayesianRidge(),

    # Tree-based
    "decision_tree_reg": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=10),
    "random_forest_reg": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "extra_trees_reg": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),

    # Boosting
    "gradient_boosting_reg": GradientBoostingRegressor(
        n_estimators=100, random_state=RANDOM_STATE, learning_rate=0.1
    ),
    "hist_gradient_boosting_reg": HistGradientBoostingRegressor(
        max_iter=100, random_state=RANDOM_STATE
    ),

    # Instance
    "knn_reg": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),

    # Neural
    "mlp_reg": MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
    ),
}

# Optional boosters
try:
    from xgboost import XGBRegressor
    REGRESSION_MODELS["xgboost_reg"] = XGBRegressor(
        n_estimators=100, random_state=RANDOM_STATE, verbosity=0
    )
    logger.info("XGBoost regressor added")
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    REGRESSION_MODELS["lightgbm_reg"] = LGBMRegressor(
        n_estimators=100, random_state=RANDOM_STATE, verbose=-1
    )
    logger.info("LightGBM regressor added")
except ImportError:
    pass

DEFAULT_REGRESSION_ENABLED = [
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
for _opt in ("xgboost_reg", "lightgbm_reg"):
    if _opt in REGRESSION_MODELS:
        DEFAULT_REGRESSION_ENABLED.append(_opt)

REGRESSION_FAMILIES = {
    "ridge":                      "Linear",
    "lasso":                      "Linear",
    "elasticnet":                 "Linear",
    "bayesian_ridge":             "Linear",
    "decision_tree_reg":          "Tree",
    "random_forest_reg":          "Ensemble",
    "extra_trees_reg":            "Ensemble",
    "gradient_boosting_reg":      "Boosting",
    "hist_gradient_boosting_reg": "Boosting",
    "knn_reg":                    "Instance",
    "mlp_reg":                    "Neural Network",
    "xgboost_reg":                "Boosting",
    "lightgbm_reg":               "Boosting",
}


def get_regression_model(name: str):
    from sklearn.base import clone
    if name not in REGRESSION_MODELS:
        raise ValueError(
            f"Regressor '{name}' not found. Available: {list(REGRESSION_MODELS.keys())}"
        )
    return clone(REGRESSION_MODELS[name])


def get_all_regression_models(enabled_only: bool = True) -> dict:
    from sklearn.base import clone
    enabled = DEFAULT_REGRESSION_ENABLED if enabled_only else list(REGRESSION_MODELS.keys())
    result = {}
    for name in enabled:
        if name in REGRESSION_MODELS:
            result[name] = clone(REGRESSION_MODELS[name])
        else:
            logger.warning(f"Regressor '{name}' in list but not in registry — skipping")
    logger.info(f"Loaded {len(result)} regressor(s): {list(result.keys())}")
    return result


def get_regression_family(name: str) -> str:
    return REGRESSION_FAMILIES.get(name, "Unknown")
