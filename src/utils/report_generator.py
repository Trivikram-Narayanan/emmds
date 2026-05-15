"""
EMMDS Report Generator
Auto-generates structured text and Markdown reports from pipeline results.
Saved to outputs/reports/ as .md and .txt files.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORT_DIR = Path("outputs/reports")


class ReportGenerator:
    """
    Produces human-readable reports from EMMDS pipeline output.
    Formats: Markdown (.md) and plain text (.txt).
    """

    def __init__(self, report_dir: str | Path = REPORT_DIR):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        pipeline_result: dict,
        dataset_name: str = "dataset",
        fmt: str = "markdown",
        save: bool = True,
    ) -> str:
        """
        Generate and optionally save a report.

        Args:
            pipeline_result: Full pipeline result dict from EMPipeline.run()
            dataset_name:    Label for this dataset
            fmt:             "markdown" or "text"
            save:            Write to disk

        Returns:
            Report string
        """
        if fmt == "markdown":
            report = self._build_markdown(pipeline_result, dataset_name)
            ext = ".md"
        else:
            report = self._build_text(pipeline_result, dataset_name)
            ext = ".txt"

        if save:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.report_dir / f"report_{dataset_name}_{ts}{ext}"
            out.write_text(report, encoding="utf-8")
            logger.info(f"Report saved → {out}")

        return report

    # ── Markdown ─────────────────────────────────────────────────

    def _build_markdown(self, result: dict, dataset_name: str) -> str:
        d  = result.get("decision", {})
        st = result.get("steps",    {})
        lb = st.get("leaderboard", [])
        ev = st.get("evaluation",  {})
        cs = st.get("calibration_scores", {})
        cv = st.get("cv_results",  {})
        pp = st.get("preprocessing", {})
        an = st.get("analysis",    {})
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# 🧠 EMMDS Report — {dataset_name}",
            f"> Generated: {ts}",
            "",
            "---",
            "",
            "## 1. Dataset Summary",
            "",
            f"| Property | Value |",
            f"|---|---|",
            f"| Dataset | `{dataset_name}` |",
            f"| Task | {d.get('task', '—')} |",
            f"| Target Column | `{result.get('target_col', '—')}` |",
            f"| Rows | {d.get('dataset_info', {}).get('rows', '—')} |",
            f"| Features | {d.get('dataset_info', {}).get('features', '—')} |",
            f"| Imbalance Ratio | {d.get('dataset_info', {}).get('imbalance_ratio', '—')} |",
            "",
            "---",
            "",
            "## 2. 🏆 Best Model Decision",
            "",
            f"| Property | Value |",
            f"|---|---|",
            f"| Best Model | **{d.get('best_model', '—')}** |",
            f"| {d.get('primary_metric','Score').upper()} | **{d.get('primary_score', '—')}** |",
            f"| Accuracy | {d.get('accuracy', '—')} |",
            f"| Trust Score | **{d.get('trust_score', '—')}** |",
            f"| Trust Label | {d.get('trust_label', '—')} |",
            "",
        ]

        # Trust breakdown
        tb = d.get("trust_breakdown", {})
        if tb:
            lines += [
                "### Trust Score Breakdown",
                "",
                "| Component | Score | Weight |",
                "|---|---|---|",
                f"| Accuracy | {tb.get('accuracy_component', '—')} | 0.25 |",
                f"| Calibration | {tb.get('calibration_component', '—')} | 0.20 |",
                f"| Stability | {tb.get('stability_component', '—')} | 0.20 |",
                f"| Agreement | {tb.get('agreement_component', '—')} | 0.20 |",
                f"| Data Quality | {tb.get('data_quality_component', '—')} | 0.15 |",
                "",
            ]

        # Leaderboard
        lines += [
            "---",
            "",
            "## 3. 📊 Model Leaderboard",
            "",
            "| Rank | Model | F1 | Accuracy | CV Mean | Trust |",
            "|---|---|---|---|---|---|",
        ]
        for row in lb:
            lines.append(
                f"| #{row.get('rank','—')} | {row.get('model','—')} "
                f"| {row.get('f1', row.get('r2','—'))} "
                f"| {row.get('accuracy','—')} "
                f"| {row.get('cv_mean','—')} "
                f"| {d.get('all_trust_scores',{}).get(row.get('model'),'—')} |"
            )

        # Top features
        top = d.get("top_features", [])
        if top:
            lines += ["", "---", "", "## 4. 🔍 Top Predictive Features (SHAP)", ""]
            for f in top:
                lines.append(f"- {f}")

        # Full metrics table
        if ev:
            lines += ["", "---", "", "## 5. 📋 Detailed Metrics", ""]
            metric_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
            header = "| Model | " + " | ".join(c.upper() for c in metric_cols) + " |"
            sep    = "|---|" + "|---|" * len(metric_cols)
            lines += [header, sep]
            for mname, metrics in ev.items():
                vals = " | ".join(
                    str(round(metrics.get(c, 0) or 0, 4)) for c in metric_cols
                )
                lines.append(f"| {mname} | {vals} |")

        lines += [
            "",
            "---",
            "",
            "*Generated by EMMDS — Ensemble Multi-Model Decision System v1.0*",
        ]
        return "\n".join(lines)

    # ── Plain text ────────────────────────────────────────────────

    def _build_text(self, result: dict, dataset_name: str) -> str:
        d  = result.get("decision", {})
        st = result.get("steps",    {})
        lb = st.get("leaderboard",  [])
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sep = "=" * 60

        lines = [
            sep,
            "  EMMDS REPORT",
            f"  Dataset : {dataset_name}",
            f"  Generated: {ts}",
            sep,
            "",
            "  BEST MODEL",
            f"  Model        : {d.get('best_model', '—')}",
            f"  Task         : {d.get('task', '—')}",
            f"  {d.get('primary_metric','Score').upper():12s} : {d.get('primary_score', '—')}",
            f"  Accuracy     : {d.get('accuracy', '—')}",
            f"  Trust Score  : {d.get('trust_score', '—')}",
            f"  Trust Label  : {d.get('trust_label', '—')}",
            "",
            "-" * 60,
            "  LEADERBOARD",
        ]

        for row in lb[:7]:
            primary = d.get("primary_metric", "f1")
            lines.append(
                f"  #{row.get('rank'):<2} {row.get('model',''):<26} "
                f"{primary}={row.get(primary, row.get('r2', 0)):.4f}"
            )

        top = d.get("top_features", [])
        if top:
            lines += ["", "-" * 60, "  TOP FEATURES (SHAP)"]
            for f in top[:5]:
                lines.append(f"  {f}")

        lines += ["", sep, "  EMMDS v1.0", sep]
        return "\n".join(lines)
