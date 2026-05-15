"""
Model card generator following Mitchell et al. (2019) format.

Produces structured model cards for models selected by the EMMDS trust pipeline.
Covers: intended use, quantitative analysis, trust breakdown, fairness, and caveats.

Reference: Mitchell et al. (2019), "Model Cards for Model Reporting", FAccT.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional, Any


def generate_model_card(
    model_name: str,
    dataset_name: str,
    trust_report: Dict[str, Any],
    accuracy_metrics: Dict[str, float],
    intended_use: str = "Binary classification in a supervised ML workflow.",
    out_of_scope: str = "Not intended for use on datasets with severe distribution shift from training data.",
    sensitive_attribute: Optional[str] = None,
    fairness_report: Optional[Dict[str, Any]] = None,
    conformal_intervals: Optional[Dict[str, float]] = None,
    training_data_description: str = "Tabular dataset, details in pipeline run config.",
    caveats: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a model card dict from trust pipeline outputs.

    Args:
        model_name:       Sklearn estimator name (e.g., "RandomForestClassifier").
        dataset_name:     Human-readable dataset label.
        trust_report:     Output dict from TrustScoreEngine.compute_trust_score().
        accuracy_metrics: Dict with keys like 'f1', 'accuracy', 'roc_auc'.
        intended_use:     Natural-language description of deployment intent.
        out_of_scope:     What this model should NOT be used for.
        sensitive_attribute: Name of protected attribute if fairness was evaluated.
        fairness_report:  Output from fairness_metrics.fairness_summary().
        conformal_intervals: Output from conformal_trust.evaluate_conformal_coverage().
        training_data_description: Free-text dataset summary.
        caveats:          Additional limitations or warnings.

    Returns:
        Dict matching the model card schema (serialisable to JSON / Markdown).
    """
    trust_score = trust_report.get("trust_score", trust_report.get("overall", None))
    trust_label = trust_report.get("trust_label", _label_from_score(trust_score))

    card: Dict[str, Any] = {
        "model_details": {
            "name": model_name,
            "generated": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset_name,
            "framework": "scikit-learn via EMMDS pipeline",
            "version": "1.0",
        },
        "intended_use": {
            "primary_use": intended_use,
            "out_of_scope": out_of_scope,
        },
        "training_data": {
            "description": training_data_description,
        },
        "quantitative_analysis": {
            "accuracy_metrics": accuracy_metrics,
            "trust_score": trust_score,
            "trust_label": trust_label,
            "trust_components": _extract_components(trust_report),
        },
        "deployment_risk": _deployment_risk_block(trust_report, conformal_intervals),
        "fairness": _fairness_block(sensitive_attribute, fairness_report),
        "caveats_and_recommendations": _caveats_block(trust_report, caveats),
    }
    return card


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_from_score(score: Optional[float]) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= 0.85:
        return "HIGH"
    if score >= 0.65:
        return "MEDIUM"
    return "LOW"


def _extract_components(trust_report: Dict[str, Any]) -> Dict[str, float]:
    keys = ["accuracy", "calibration", "agreement", "data_quality", "stability", "fairness"]
    return {k: round(float(trust_report[k]), 4) for k in keys if k in trust_report}


def _deployment_risk_block(
    trust_report: Dict[str, Any],
    conformal_intervals: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    risk = trust_report.get("deployment_risk")
    if risk is not None:
        block["point_estimate"] = round(float(risk), 4)
    if conformal_intervals:
        block["conformal_interval"] = {
            "lower": conformal_intervals.get("lower"),
            "upper": conformal_intervals.get("upper"),
            "coverage_guarantee": f"{(1 - conformal_intervals.get('alpha', 0.1)) * 100:.0f}%",
        }
        block["mean_interval_width"] = conformal_intervals.get("mean_interval_width")
    return block


def _fairness_block(
    sensitive_attribute: Optional[str],
    fairness_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if fairness_report is None:
        return {"evaluated": False, "note": "No sensitive attribute provided."}
    return {
        "evaluated": True,
        "sensitive_attribute": sensitive_attribute or "unspecified",
        "demographic_parity_gap": fairness_report.get("demographic_parity_gap"),
        "equalized_odds_tpr_gap": fairness_report.get("equalized_odds_tpr_gap"),
        "equalized_odds_fpr_gap": fairness_report.get("equalized_odds_fpr_gap"),
        "dp_pass": fairness_report.get("dp_pass"),
        "eo_pass": fairness_report.get("eo_pass"),
        "overall_fairness_score": fairness_report.get("overall_fairness_score"),
        "per_group": fairness_report.get("per_group", {}),
    }


def _caveats_block(
    trust_report: Dict[str, Any],
    extra_caveats: Optional[str],
) -> Dict[str, Any]:
    caveats = []
    trust_score = trust_report.get("trust_score", trust_report.get("overall", 1.0))
    if trust_score is not None and trust_score < 0.65:
        caveats.append(
            "Low trust score — model should not be deployed without human review."
        )
    if trust_report.get("data_quality", 1.0) < 0.70:
        caveats.append(
            "Data quality score below 0.70 — check for missing values, outliers, or imbalance."
        )
    if trust_report.get("stability", 1.0) < 0.80:
        caveats.append(
            "CV stability below 0.80 — predictions may vary substantially across data splits."
        )
    if extra_caveats:
        caveats.append(extra_caveats)
    return {
        "caveats": caveats,
        "recommendations": _recommendations(trust_score),
    }


def _recommendations(trust_score: Optional[float]) -> list:
    if trust_score is None:
        return []
    if trust_score >= 0.85:
        return ["Model meets deployment criteria. Monitor for distribution shift post-deployment."]
    if trust_score >= 0.65:
        return [
            "Consider increasing training data before deployment.",
            "Run calibration check on holdout set.",
        ]
    return [
        "Do not deploy without further validation.",
        "Investigate data quality issues.",
        "Consider alternative model architectures or more training data.",
    ]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def card_to_json(card: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(card, indent=indent, default=str)


def card_to_markdown(card: Dict[str, Any]) -> str:
    """Convert a model card dict to a human-readable Markdown string."""
    md_lines = []
    md = card

    md_lines.append(f"# Model Card: {md['model_details']['name']}")
    md_lines.append(f"\n**Dataset:** {md['model_details']['dataset']}  ")
    md_lines.append(f"**Generated:** {md['model_details']['generated']}  ")
    md_lines.append(f"**Framework:** {md['model_details']['framework']}\n")

    md_lines.append("## Intended Use")
    md_lines.append(f"- **Primary use:** {md['intended_use']['primary_use']}")
    md_lines.append(f"- **Out of scope:** {md['intended_use']['out_of_scope']}\n")

    md_lines.append("## Quantitative Analysis")
    acc = md["quantitative_analysis"]["accuracy_metrics"]
    for k, v in acc.items():
        md_lines.append(f"- **{k}:** {v:.4f}")
    ts = md["quantitative_analysis"].get("trust_score")
    tl = md["quantitative_analysis"].get("trust_label")
    if ts is not None:
        md_lines.append(f"- **Trust Score:** {ts:.4f} ({tl})")
    comps = md["quantitative_analysis"].get("trust_components", {})
    if comps:
        md_lines.append("\n### Trust Components")
        for k, v in comps.items():
            md_lines.append(f"  - {k}: {v:.4f}")

    dr = md.get("deployment_risk", {})
    if dr:
        md_lines.append("\n## Deployment Risk")
        if "point_estimate" in dr:
            md_lines.append(f"- **Point estimate:** {dr['point_estimate']:.4f}")
        if "conformal_interval" in dr:
            ci = dr["conformal_interval"]
            md_lines.append(
                f"- **{ci['coverage_guarantee']} prediction interval:** [{ci['lower']:.4f}, {ci['upper']:.4f}]"
            )

    fair = md.get("fairness", {})
    md_lines.append("\n## Fairness")
    if fair.get("evaluated"):
        md_lines.append(f"- **Sensitive attribute:** {fair['sensitive_attribute']}")
        md_lines.append(f"- **Demographic parity gap:** {fair['demographic_parity_gap']:.4f} (pass={fair['dp_pass']})")
        md_lines.append(f"- **Equalized odds TPR gap:** {fair['equalized_odds_tpr_gap']:.4f}")
        md_lines.append(f"- **Overall fairness score:** {fair['overall_fairness_score']:.4f}")
    else:
        md_lines.append("- Not evaluated (no sensitive attribute provided).")

    caveats_block = md.get("caveats_and_recommendations", {})
    if caveats_block.get("caveats"):
        md_lines.append("\n## Caveats")
        for c in caveats_block["caveats"]:
            md_lines.append(f"- {c}")
    if caveats_block.get("recommendations"):
        md_lines.append("\n## Recommendations")
        for r in caveats_block["recommendations"]:
            md_lines.append(f"- {r}")

    return "\n".join(md_lines)
