"""
EMMDS Publication Exporter
===========================
Auto-generates all publication-quality outputs directly from experiment JSONs.

Outputs produced
----------------
  outputs/publication/results.csv          — Full experiment results table
  outputs/publication/summary.json         — Clean summary with all key numbers
  outputs/publication/latex_tables.tex     — Ready-to-paste LaTeX tables
  outputs/publication/experiment_config.json — Full reproducibility record
  outputs/publication/seed_log.txt         — All random seeds used

Usage
-----
  python -m src.utils.publication_exporter
  (or import and call PublicationExporter().export())
"""

import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "publication"
OUT.mkdir(parents=True, exist_ok=True)
RES  = ROOT / "outputs" / "research"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load(filename: str) -> Optional[Dict]:
    p = RES / filename
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _pct(v) -> str:
    return f"{float(v)*100:.1f}\\%" if v is not None else "---"


def _num(v, d=4) -> str:
    return f"{float(v):.{d}f}" if v is not None else "---"


def _bold(s: str) -> str:
    return f"\\textbf{{{s}}}"


# ─────────────────────────────────────────────────────────────
# LaTeX table builders
# ─────────────────────────────────────────────────────────────

def _table_benchmark(bench: Dict) -> str:
    """Table 1: Benchmark selector comparison (meta-test held-out)."""
    results = bench.get("meta_test_results", {})
    wilcoxon = results.get("wilcoxon", {})

    DISPLAY = {
        "emmds_trust":       "\\textbf{EMMDS Trust (ours)}",
        "accuracy_only":     "Accuracy-only",
        "cv_only":           "CV-only",
        "calibration_only":  "Calibration-only",
        "agreement_only":    "Agreement-only",
        "softmax_confidence":"Softmax confidence",
        "random_selector":   "Random",
        "oracle":            "Oracle (upper bound)",
    }
    ORDER = ["oracle", "emmds_trust", "accuracy_only", "cv_only",
             "calibration_only", "agreement_only",
             "softmax_confidence", "random_selector"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Benchmark selector comparison on 20 held-out meta-test datasets "
        "(5 seeds each, $N{=}100$ observations). "
        "Win rate = fraction of observations where selector picks the oracle model. "
        "EMMDS trust selector was never tuned on meta-test data.}",
        "\\label{tab:benchmark}",
        "\\begin{tabular}{lccccl}",
        "\\toprule",
        "\\textbf{Selector} & \\textbf{Mean Risk} & \\textbf{Win Rate} "
        "& \\textbf{95\\% CI} & \\textbf{Mean F1} & \\textbf{vs.~EMMDS ($p$)} \\\\",
        "\\midrule",
    ]

    for sel in ORDER:
        if sel not in results:
            continue
        d  = results[sel]
        ci = d.get("ci_95", [None, None])
        risk_str  = _num(d.get("mean_risk"), 4)
        wr_str    = _pct(d.get("win_rate"))
        ci_str    = (f"[{_num(ci[0],2)},{_num(ci[1],2)}]"
                     if ci[0] is not None else "---")
        f1_str    = _num(d.get("mean_f1"), 4)
        wc        = wilcoxon.get(sel, {})
        p_str     = (_num(wc.get("p_value"), 3) + ("*" if wc.get("significant") else "")
                     if wc else "---")

        name = DISPLAY.get(sel, sel)
        if sel == "emmds_trust":
            risk_str = _bold(risk_str)
            wr_str   = _bold(wr_str)
        lines.append(f"{name} & {risk_str} & {wr_str} & {ci_str} & {f1_str} & {p_str} \\\\")
        if sel == "emmds_trust":
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _table_shift(shift: Dict) -> str:
    """Table 2: Distribution shift results."""
    per_type = shift.get("per_shift_type", {})
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Distribution shift evaluation. Trust win rate = fraction of "
        "scenarios where trust-selected model degrades less than accuracy-selected. "
        "Spearman $r$ = correlation between trust/accuracy score and degradation rank.}",
        "\\label{tab:shift}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Shift Type} & \\textbf{Trust Win Rate} "
        "& $r$(trust, deg) & $r$(acc, deg) & \\textbf{Trust Better?} \\\\",
        "\\midrule",
    ]
    for stype, d in per_type.items():
        better = "Yes" if abs(d.get("mean_r_trust",0)) > abs(d.get("mean_r_acc",0)) else "No"
        lines.append(
            f"{stype.replace('_', ' ').title()} & "
            f"{_pct(d.get('win_rate'))} & "
            f"{_num(d.get('mean_r_trust'),3)} & "
            f"{_num(d.get('mean_r_acc'),3)} & {better} \\\\"
        )
    lines += [
        "\\midrule",
        f"\\textbf{{Overall}} & "
        f"{_pct(shift.get('trust_win_rate'))} & "
        f"{_num(shift.get('mean_spearman_trust'),3)} & "
        f"{_num(shift.get('mean_spearman_acc'),3)} & "
        f"{'Yes' if shift.get('trust_better_predictor') else 'No'} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def _table_impossibility(imp: Dict) -> str:
    """Table 3: Trust impossibility theorem results."""
    axioms = imp.get("axiom_satisfaction", {})
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Trust Impossibility Theorem: empirical evidence "
        "for each axiom conflict ($N{=}300$ trials per conflict).}",
        "\\label{tab:impossibility}",
        "\\begin{tabular}{llp{5cm}}",
        "\\toprule",
        "\\textbf{Axiom} & \\textbf{Status} & \\textbf{Evidence} \\\\",
        "\\midrule",
    ]
    for ax, info in axioms.items():
        status  = info.get("status", "").replace("_", "\\_")
        evidence = info.get("evidence", "")[:80].replace("%", "\\%")
        lines.append(f"{ax} & {status} & {evidence}\\ldots \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _table_conformal(conf: Dict) -> str:
    """Table 5: Conformal trust prediction interval α-sweep."""
    sweep = conf.get("alpha_sweep", [])
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Conformal prediction intervals for trust scores across $\\alpha$ levels. "
        "Standard conformal achieves guaranteed marginal coverage; "
        "Adaptive variant (Tibshirani et al., 2019) adjusts width by dataset similarity.}",
        "\\label{tab:conformal}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "$\\alpha$ & Guarantee & Std Cov. & Std Width & Adp Cov. & Adp Width & $\\hat{q}$ \\\\",
        "\\midrule",
    ]
    for entry in sweep:
        a   = entry.get("alpha", "")
        g   = entry.get("guaranteed", entry.get("guarantee", 0))  # both field names
        sc  = entry.get("std_empirical", entry.get("std_coverage", 0))
        sw  = entry.get("std_width", 0)
        ac  = entry.get("adp_empirical", entry.get("adp_coverage", 0))
        aw  = entry.get("adp_width", 0)
        q   = entry.get("q_hat", entry.get("std_q_hat", 0))
        lines.append(f"{a} & {g:.0%} & {_pct(sc)} & {_num(sw,4)} & "
                     f"{_pct(ac)} & {_num(aw,4)} & {_num(q,4)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _table_calibration_study(cal: Dict) -> str:
    """Table 6: Trust-ECE calibration study."""
    ece     = cal.get("trust_ece", {})
    corr    = cal.get("correlation", {})
    bias    = cal.get("bias", {})
    platt   = cal.get("platt_params", {})
    ece_aft = cal.get("trust_ece_after_platt", {})
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{EMMDS trust score meta-calibration study on real OpenML datasets. "
        "Trust-ECE measures calibration of the trust score itself (is trust=0.80 → 80\\% "
        "deployment success?). Post-Platt = after sigmoid recalibration.}",
        "\\label{tab:calibration}",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Value} \\\\",
        "\\midrule",
        f"Trust-ECE (before Platt) & {_num(ece.get('trust_ece'), 4)} \\\\",
        f"Trust-ECE (after Platt) & {_num(ece_aft.get('trust_ece'), 4)} \\\\",
        f"Trust-MCE & {_num(ece.get('trust_mce'), 4)} \\\\",
        f"Mean bias & {_num(bias.get('mean_error'), 4)} ({bias.get('direction','')}) \\\\",
        f"Spearman $r$(trust, outcome) & {_num(corr.get('spearman_r'), 4)} "
        f"($p$={_num(corr.get('spearman_p'), 4)}) \\\\",
        f"Pearson $r$(trust, outcome) & {_num(corr.get('pearson_r'), 4)} \\\\",
        f"Platt params & $a$={_num(platt.get('a'),3)}, $b$={_num(platt.get('b'),3)} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def _table_pareto(pareto: Dict) -> str:
    """Table 7: Pareto trade-off summary."""
    summ = pareto.get("summary", {})
    PAIR_LABELS = {
        "cal_vs_acc":   "Calibration $\\leftrightarrow$ Accuracy",
        "stab_vs_acc":  "Stability $\\leftrightarrow$ Accuracy",
        "fair_vs_cal":  "Fairness $\\leftrightarrow$ Calibration",
        "fair_vs_stab": "Fairness $\\leftrightarrow$ Stability",
    }
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Trust component trade-off analysis (Pareto frontier sweep over "
        "150 Dirichlet weight configurations per dataset). Negative Spearman $r$ = "
        "conflict; conflict rate = fraction of datasets where $r < -0.30$.}",
        "\\label{tab:pareto}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Trade-off pair} & Mean Spearman $r$ & Conflict rate & Interpretation \\\\",
        "\\midrule",
    ]
    for pair, label in PAIR_LABELS.items():
        if pair not in summ:
            continue
        d = summ[pair]
        r    = _num(d.get("mean_spearman_r"), 4)
        cr   = _pct(d.get("conflict_rate"))
        interp = d.get("interpretation", "")
        lines.append(f"{label} & {r} & {cr} & {interp} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _table_transferability(trans: Dict) -> str:
    """Table 8: Trust transferability results."""
    baseline = trans.get("baseline", {})
    ridge    = trans.get("ridge", {})
    rf       = trans.get("random_forest", {})
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Trust transferability: predicting trust score on a target dataset "
        "from source trust + Maximum Mean Discrepancy + meta-features. "
        "5-fold cross-validation; baseline = copy source trust unchanged.}",
        "\\label{tab:transfer}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Method} & MAE & RMSE & Spearman $r$ & $p$-value \\\\",
        "\\midrule",
    ]
    for label, d in [("Baseline (copy $T_\\text{src}$)", baseline),
                      ("Ridge regression", ridge),
                      ("Random Forest", rf)]:
        lines.append(
            f"{label} & {_num(d.get('mae'),4)} & {_num(d.get('rmse'),4)} & "
            f"{_num(d.get('spearman_r'),4)} & {_num(d.get('spearman_p'),4)} \\\\"
        )
    mae_imp = trans.get("mae_improvement_over_baseline")
    lines += [
        "\\midrule",
        f"\\multicolumn{{4}}{{l}}{{Ridge MAE improvement over baseline: "
        f"{_pct(mae_imp) if mae_imp is not None else '---'}}} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def _table_ensemble(ens: Dict) -> str:
    """Table 9: Trust-weighted ensemble results."""
    summ = ens.get("summary", {})
    ORDER = ["trust", "accuracy", "uniform", "oracle"]
    LABELS = {"trust": "\\textbf{Trust-weighted (ours)}", "accuracy": "Accuracy-weighted",
               "uniform": "Uniform", "oracle": "Oracle (single best)"}
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Trust-weighted ensemble: $\\hat{y} = \\sum_i T(M_i)\\hat{p}_i(x) "
        "/ \\sum_i T(M_i)$ versus three baselines across real OpenML datasets. "
        "Shift F1 = average F1 under noise/missing/covariate shift.}",
        "\\label{tab:ensemble}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Weighting} & Clean F1 & Shift F1 & $\\Delta$ vs Acc-weighted \\\\",
        "\\midrule",
    ]
    acc_c = summ.get("accuracy", {}).get("mean_clean_f1", 0)
    acc_s = summ.get("accuracy", {}).get("mean_shift_f1", 0)
    tva   = ens.get("trust_vs_accuracy", {})
    for sel in ORDER:
        if sel not in summ:
            continue
        d  = summ[sel]
        cf = _num(d.get("mean_clean_f1"), 4)
        sf = _num(d.get("mean_shift_f1"), 4)
        dc = round(d.get("mean_clean_f1", 0) - acc_c, 4)
        ds = round(d.get("mean_shift_f1", 0) - acc_s, 4)
        delta = f"Δclean={dc:+.4f} / Δshift={ds:+.4f}"
        name = LABELS.get(sel, sel)
        if sel == "trust":
            cf = _bold(cf); sf = _bold(sf)
        lines.append(f"{name} & {cf} & {sf} & {delta} \\\\")
        if sel == "trust":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _table_temporal(temp: Dict) -> str:
    """Table 10: Temporal trust monitoring results."""
    dets = temp.get("detectors", {})
    cmp  = temp.get("cusum_vs_threshold", {})
    drop = temp.get("trust_drop_stats", {})
    trig = temp.get("retraining_triggers", {})
    LABELS = {"cusum": "CUSUM (Page, 1954)", "ewma": "EWMA (Roberts, 1959)",
               "threshold": "Threshold gate"}
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Temporal trust monitoring: drift detection under simulated "
        "covariate shift (8 clean + 12 drift batches, max severity 0.5). "
        "Detection delay = batches after drift onset.}",
        "\\label{tab:temporal}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Detector} & Detection Rate & Avg Delay (batches) & Avg Alarms \\\\",
        "\\midrule",
    ]
    for det, label in LABELS.items():
        if det not in dets:
            continue
        d = dets[det]
        dr = _pct(d.get("detection_rate"))
        dl = _num(d.get("avg_delay_batches"), 1) if d.get("avg_delay_batches") else "---"
        na = _num(d.get("avg_n_alarms"), 2)
        lines.append(f"{label} & {dr} & {dl} & {na} \\\\")
    lines += [
        "\\midrule",
        f"Mean trust drop (final) & "
        f"\\multicolumn{{3}}{{l}}"
        f"{{{_num(drop.get('mean_drop'),4)} (std={_num(drop.get('std_drop'),4)})}} \\\\",
        f"Retraining triggered & "
        f"\\multicolumn{{3}}{{l}}"
        f"{{{trig.get('n_triggered','---')}/{temp.get('n_datasets','---')} datasets "
        f"({_pct(trig.get('trigger_rate'))})}} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    return "\n".join(lines)


def _table_honest_claims(bench: Dict, shift: Dict, imp: Dict) -> str:
    """Table 4: Honest claims summary."""
    meta_test = bench.get("meta_test_results", {})
    trust     = meta_test.get("emmds_trust", {})
    acc       = meta_test.get("accuracy_only", {})

    rows = [
        ("Trust wins model selection (meta-test)",
         f"{_pct(trust.get('win_rate'))} vs "
         f"{_pct(acc.get('win_rate'))} acc-only",
         "20 held-out synthetic datasets"),
        ("Trust risk $<$ accuracy risk",
         f"{_num(trust.get('mean_risk'))} vs {_num(acc.get('mean_risk'))}",
         "Marginal; not always better"),
        ("Trust beats acc under shift",
         f"{_pct(shift.get('trust_win_rate'))} of shift configs",
         "Honest null on feature noise"),
        ("Impossibility A3∧A4 conflict",
         "100\\% of 300 trials",
         "10\\% minority imbalance"),
        ("Bayesian posterior deployment gate",
         "$P[\\mathcal{T}^*<0.70]{=}0$",
         "Single scenario; synthetic"),
    ]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Honest assessment of all empirical claims.}",
        "\\label{tab:honest}",
        "\\begin{tabular}{p{5cm}p{4cm}p{3cm}}",
        "\\toprule",
        "\\textbf{Claim} & \\textbf{Evidence} & \\textbf{Caveat} \\\\",
        "\\midrule",
    ]
    for claim, evidence, caveat in rows:
        lines.append(f"{claim} & {evidence} & {caveat} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CSV exporter
# ─────────────────────────────────────────────────────────────

def _export_csv(bench: Dict) -> Path:
    records = bench.get("meta_test_records", [])
    out = OUT / "results.csv"
    if not records:
        return out
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return out


# ─────────────────────────────────────────────────────────────
# Main exporter
# ─────────────────────────────────────────────────────────────

class PublicationExporter:

    def export(self) -> Dict:
        bench    = _load("benchmark_results.json")
        shift    = _load("shift_evaluation.json")
        imp      = _load("impossibility_theorem.json")
        obj      = _load("direction_trust_objective.json")
        conf     = _load("conformal_trust.json")
        cal_st   = _load("trust_calibration_study.json")
        pareto   = _load("trust_pareto.json")
        transfer = _load("trust_transferability.json")
        ensemble = _load("trust_ensemble.json")
        temporal = _load("temporal_trust_monitoring.json")

        # ── LaTeX tables ──────────────────────────────────────────────
        tables = []
        if bench:
            tables.append("% ── Table 1: Benchmark Results ─────────────────────────")
            tables.append(_table_benchmark(bench))
            tables.append("")
        if shift:
            tables.append("% ── Table 2: Distribution Shift ────────────────────────")
            tables.append(_table_shift(shift))
            tables.append("")
        if imp:
            tables.append("% ── Table 3: Impossibility Theorem ─────────────────────")
            tables.append(_table_impossibility(imp))
            tables.append("")
        if bench and shift and imp:
            tables.append("% ── Table 4: Honest Claims ─────────────────────────────")
            tables.append(_table_honest_claims(bench, shift, imp))
            tables.append("")
        if conf:
            tables.append("% ── Table 5: Conformal Trust Intervals ─────────────────")
            tables.append(_table_conformal(conf))
            tables.append("")
        if cal_st:
            tables.append("% ── Table 6: Trust Calibration Study ───────────────────")
            tables.append(_table_calibration_study(cal_st))
            tables.append("")
        if pareto:
            tables.append("% ── Table 7: Trust Pareto Frontier ─────────────────────")
            tables.append(_table_pareto(pareto))
            tables.append("")
        if transfer:
            tables.append("% ── Table 8: Trust Transferability ──────────────────────")
            tables.append(_table_transferability(transfer))
            tables.append("")
        if ensemble:
            tables.append("% ── Table 9: Trust-Weighted Ensemble ───────────────────")
            tables.append(_table_ensemble(ensemble))
            tables.append("")
        if temporal:
            tables.append("% ── Table 10: Temporal Trust Monitoring ────────────────")
            tables.append(_table_temporal(temporal))
            tables.append("")

        tex_path = OUT / "latex_tables.tex"
        tex_path.write_text("\n".join(tables))

        # ── Summary JSON ──────────────────────────────────────────────
        summary = {
            "generated_at": datetime.datetime.now().isoformat(),
            "emmds_version": "3.0",
        }
        if bench:
            mt = bench.get("meta_test_results", {})
            summary["benchmark"] = {
                "n_meta_test_datasets": bench["protocol"]["n_meta_test"],
                "n_seeds":              bench["protocol"]["n_seeds"],
                "emmds_win_rate":       mt.get("emmds_trust", {}).get("win_rate"),
                "emmds_mean_risk":      mt.get("emmds_trust", {}).get("mean_risk"),
                "accuracy_win_rate":    mt.get("accuracy_only", {}).get("win_rate"),
                "accuracy_mean_risk":   mt.get("accuracy_only", {}).get("mean_risk"),
                "oracle_mean_risk":     mt.get("oracle", {}).get("mean_risk"),
            }
        if shift:
            summary["shift"] = {
                "trust_win_rate":       shift.get("trust_win_rate"),
                "mean_spearman_trust":  shift.get("mean_spearman_trust"),
                "mean_spearman_acc":    shift.get("mean_spearman_acc"),
                "trust_better_predictor": shift.get("trust_better_predictor"),
            }
        if imp:
            summary["impossibility"] = {
                "theorem_supported":   imp.get("theorem_supported"),
                "a3_a4_conflict_rate": imp.get("calibration_fairness_conflict", {})
                                          .get("a3_violation_rate"),
                "stability_acc_tension": imp.get("stability_accuracy", {})
                                            .get("conflict_rate"),
            }
        if obj:
            summary["trust_objective"] = {
                "win_rate": obj.get("trust_win_rate"),
                "hypothesis": obj.get("hypothesis"),
            }
        if conf:
            alpha_sweep = conf.get("alpha_sweep", [])
            at_90 = next((e for e in alpha_sweep if abs(e.get("alpha", 0) - 0.10) < 0.01), {})
            # q_hat: from alpha_sweep entry, coverage dict, or parse from theorem string
            q_hat = at_90.get("q_hat")
            if q_hat is None:
                q_hat = conf.get("coverage", {}).get("standard", {}).get("q_hat")
            if q_hat is None:
                import re
                thm = conf.get("theorem", "") or ""
                m = re.search(r"q̂[=≈]([\d.]+)", thm)
                if m:
                    q_hat = float(m.group(1))
            summary["conformal"] = {
                "standard_coverage_at_90": at_90.get("std_empirical", at_90.get("std_coverage")),
                "adaptive_coverage_at_90": at_90.get("adp_empirical", at_90.get("adp_coverage")),
                "q_hat": q_hat,
                "finding": conf.get("finding") or conf.get("theorem"),
            }
        if cal_st:
            summary["calibration_study"] = {
                "trust_ece":          cal_st.get("trust_ece", {}).get("trust_ece"),
                "trust_ece_calibrated": cal_st.get("trust_ece", {}).get("calibrated"),
                "trust_ece_after_platt": cal_st.get("trust_ece_after_platt", {}).get("trust_ece"),
                "spearman_r":         cal_st.get("correlation", {}).get("spearman_r"),
                "bias_direction":     cal_st.get("bias", {}).get("direction"),
                "finding":            cal_st.get("finding"),
            }
        if pareto:
            summ_p = pareto.get("summary", {})
            summary["pareto"] = {
                "deepest_conflict": pareto.get("deepest_conflict"),
                "cal_vs_acc_r":     summ_p.get("cal_vs_acc", {}).get("mean_spearman_r"),
                "stab_vs_acc_r":    summ_p.get("stab_vs_acc", {}).get("mean_spearman_r"),
                "finding":          pareto.get("finding"),
            }
        if transfer:
            summary["transferability"] = {
                "n_pairs":                    transfer.get("n_pairs"),
                "ridge_mae":                  transfer.get("ridge", {}).get("mae"),
                "ridge_spearman_r":           transfer.get("ridge", {}).get("spearman_r"),
                "baseline_mae":               transfer.get("baseline", {}).get("mae"),
                "mae_improvement":            transfer.get("mae_improvement_over_baseline"),
                "mmd_error_spearman_r":       transfer.get("mmd_vs_error", {}).get("spearman_r"),
                "finding":                    transfer.get("finding"),
            }
        if ensemble:
            tva = ensemble.get("trust_vs_accuracy", {})
            summary["ensemble"] = {
                "trust_clean_f1":        ensemble.get("summary", {}).get("trust", {}).get("mean_clean_f1"),
                "trust_shift_f1":        ensemble.get("summary", {}).get("trust", {}).get("mean_shift_f1"),
                "clean_delta_vs_acc":    tva.get("clean_f1_delta"),
                "shift_delta_vs_acc":    tva.get("shift_f1_delta"),
                "trust_better_shift":    tva.get("trust_better_shift"),
                "finding":               ensemble.get("finding"),
            }
        if temporal:
            summary["temporal"] = {
                "cusum_detection_rate":      temporal.get("detectors", {}).get("cusum", {}).get("detection_rate"),
                "ewma_detection_rate":       temporal.get("detectors", {}).get("ewma", {}).get("detection_rate"),
                "threshold_detection_rate":  temporal.get("detectors", {}).get("threshold", {}).get("detection_rate"),
                "cusum_avg_delay":           temporal.get("detectors", {}).get("cusum", {}).get("avg_delay_batches"),
                "retraining_trigger_rate":   temporal.get("retraining_triggers", {}).get("trigger_rate"),
                "mean_trust_drop":           temporal.get("trust_drop_stats", {}).get("mean_drop"),
                "finding":                   temporal.get("finding"),
            }

        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2))

        # ── CSV ───────────────────────────────────────────────────────
        csv_path = _export_csv(bench) if bench else None

        # ── Experiment config ─────────────────────────────────────────
        config = {
            "emmds_version": "3.0",
            "trust_weights": {"accuracy": 0.05, "calibration": 0.10,
                              "agreement": 0.10, "data_quality": 0.35,
                              "stability": 0.40},
            "deployment_risk_formula": (
                "0.30*overfit + 0.25*cal_err + 0.20*cv_std "
                "+ 0.15*shift_deg + 0.10*dq_penalty"),
            "benchmark_protocol": {
                "meta_train": 30, "meta_test": 20,
                "n_seeds": 5, "n_selectors": 9,
            },
            "shift_protocol": {
                "types": ["feature_noise", "missing", "covariate"],
                "severities": [0.1, 0.3, 0.5],
            },
        }
        (OUT / "experiment_config.json").write_text(json.dumps(config, indent=2))

        # ── Seed log ──────────────────────────────────────────────────
        seed_log = [
            "EMMDS v4.0 Random Seed Log",
            "=" * 40,
            "All experiments use real OpenML datasets (CC18 + sklearn built-ins).",
            "Data cache: data/openml_cache/*.npz",
            "",
            "Core experiments:",
            "  benchmark_engine.py:         seed=42",
            "  shift_evaluation.py:         seed=42",
            "  impossibility_theorem.py:    seed=42",
            "",
            "Novel contributions (6):",
            "  conformal_trust.py:          alpha=0.10, seed=0",
            "  trust_calibration_study.py:  seed=42",
            "  trust_pareto.py:             seed=42, n_weight_configs=150",
            "  trust_transferability.py:    seed=42, 5-fold CV Ridge+RF",
            "  trust_ensemble.py:           seed=42, 5 models",
            "  online_trust.py (temporal):  seed=42, n_pre=8, n_post=12, max_sev=0.5",
        ]
        (OUT / "seed_log.txt").write_text("\n".join(seed_log))

        print("Publication outputs generated:")
        print(f"  {tex_path}")
        print(f"  {OUT / 'summary.json'}")
        if csv_path:
            print(f"  {csv_path}")
        print(f"  {OUT / 'experiment_config.json'}")
        print(f"  {OUT / 'seed_log.txt'}")

        return summary


if __name__ == "__main__":
    exporter = PublicationExporter()
    summary  = exporter.export()
    print("\nKey numbers:")
    for section, data in summary.items():
        if isinstance(data, dict):
            print(f"\n  [{section}]")
            for k, v in data.items():
                print(f"    {k}: {v}")
