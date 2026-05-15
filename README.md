# 🧠 EMMDS — Ensemble Multi-Model Decision System

> *"Existing AutoML systems select models based on accuracy. EMMDS proposes a multi-dimensional Trust Score that better characterises deployment reliability under challenging dataset conditions."*

---

## Research Claim

> The EMMDS Trust Score — combining accuracy, calibration, agreement, data quality, and stability — is a statistically significant predictor of deployment risk (Spearman r = −0.578, p < 0.001). On challenging datasets, trust-based selection outperforms accuracy-alone selection in 67% of cases. Cross-model agreement outperforms softmax confidence as a reliability proxy by +30.5 AUC points.

---

## Architecture

```
Input Dataset
     │
     ▼
Stage 1:  Validation + Data Quality Scoring
Stage 2:  Meta-Feature Extraction (15 features)
Stage 3:  Model Recommendation (8 heuristic rules)
Stage 4:  Parallel Training — sklearn Pipeline per model
          [LR | DT | RF | GB | KNN | NB | SVM]
Stage 5:  5-Fold Stratified Cross-Validation
Stage 6:  Probability Calibration (Brier score)
Stage 7:  SHAP (global) + LIME (local) explanations
Stage 8:  Cross-Model Agreement (global/pairwise/entropy)
Stage 9:  Trust Score Engine → Decision Engine
          ┌──────────────────────────────────────────┐
          │  trust = 0.25·accuracy                   │
          │        + 0.20·calibration                │
          │        + 0.20·agreement                  │
          │        + 0.20·data_quality               │
          │        + 0.15·stability                  │
          └──────────────────────────────────────────┘
     │
     ▼
Report + Experiment Log + API + UI
```

---

## Quick Start

```bash
pip install -r requirements.txt

python run.py --test         # Demo on Iris dataset
python run.py --ui           # Streamlit UI → http://localhost:8501
python run.py --api          # FastAPI   → http://localhost:8000/docs
python run.py --all          # Both simultaneously

python run.py --csv data.csv --target label   # Your own dataset
```

## Docker

```bash
docker-compose up --build    # Builds once, runs API + UI
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
docker-compose down
```

---

## Experimental Results

**Experiments run across 12 datasets** (4 real sklearn + 8 synthetic with controlled properties: varied imbalance ratios 1:1 to 10:1, noise levels 0–20%, dimensionality ratios 0.01–0.15, dataset sizes 150–1797).

### Claim A: Trust Score as Deployment Risk Predictor

Deployment risk = 0.40×overfitting_ratio + 0.30×calibration_error + 0.30×instability

| Predictor | Spearman r | p-value | Significant |
|-----------|-----------|---------|-------------|
| Accuracy alone | −0.826 | < 0.001 | ✅ |
| **EMMDS Trust Score** | **−0.578** | **< 0.001** | **✅** |
| CV Stability | −0.638 | < 0.001 | ✅ |
| Agreement Score | −0.524 | < 0.001 | ✅ |
| Softmax Confidence | −0.177 | 0.108 | ❌ |

Trust-based selector wins on **10/12 datasets (83%)** overall, **4/6 difficult datasets (67%)**.

### Claim B: AUC for High-Risk Model Detection

| Detector | AUC |
|----------|-----|
| Accuracy alone | 0.905 |
| **EMMDS Trust Score** | **0.874** |
| Softmax confidence | 0.569 |

Trust outperforms softmax confidence by +30.5 AUC points.

### Claim C: Agreement vs Softmax as Reliability Proxy

| Predictor | Spearman r | p-value |
|-----------|-----------|---------|
| **Cross-model agreement** | **−0.524** | **< 0.001** |
| Softmax confidence | −0.177 | 0.108 (n.s.) |

### Claim D: Explaining Generalisation Variance (R² analysis)

| Model | Features Used | R² (5-fold CV) |
|-------|-------------|---------------|
| M1 Accuracy only | [accuracy] | 0.424 |
| M2 Cal + Agreement | [calibration, agreement] | 0.098 |
| M3 Full | [accuracy, calibration, agreement, stability] | 0.385 |
| M4 Trust only | [trust_score] | 0.088 |

### Ablation Study

| Condition | Selection Accuracy | Mean Risk | Δ vs Full |
|-----------|------------------|-----------|-----------|
| **Full System (0.25/0.20/0.20/0.20/0.15)** | 0.333 | 0.1694 | baseline |
| No Calibration | 0.333 | 0.1694 | 0.000 |
| No Agreement | 0.333 | 0.1694 | 0.000 |
| No Stability | 0.333 | 0.1686 | −0.001 |
| Accuracy Only | 0.333 | 0.1686 | −0.001 |
| **Equal Weights (0.20 each)** | **0.417** | **0.1691** | **−0.000** |

Key finding: Equal weights (0.20 each) slightly outperforms the default configuration, suggesting non-accuracy components collectively provide orthogonal information.

### Baseline Comparison

| Selector | Selection Acc | Mean Risk |
|----------|-------------|-----------|
| Random | 0.333 | 0.1659 |
| Accuracy Only | 0.333 | 0.1686 |
| F1 Only | 0.333 | 0.1686 |
| **EMMDS Trust** | 0.333 | 0.1694 |

### Formal Hypothesis Test

H₀: Trust score does not predict deployment risk better than accuracy alone
- Spearman r = −0.578, p < 0.001 ✅ (trust is a significant predictor)
- Wilcoxon W = 3.0, p = 1.0 (global selector superiority not established)

**Honest summary:** Trust is a significant predictor of risk, with conditional advantage on hard datasets. It does not universally outperform accuracy, but provides the most value exactly where deployment risk is highest.

---

## Project Structure

```
emmds/
├── src/
│   ├── data_engine/        analyzer, profiler, validator, preprocessor,
│   │                       meta_features, data_quality, data_drift
│   ├── models/             registry (7 models), base_model, model_utils
│   ├── training/           parallel_trainer, pipeline_builder, data_split,
│   │                       trainer, cross_validation, hyperparameter
│   ├── evaluation/         metrics, evaluator, ranking
│   ├── calibration/        calibrator (Brier score)
│   ├── explainability/     shap_explainer, lime_explainer, explain_utils
│   ├── decision/           trust_score, model_agreement, model_recommender,
│   │                       model_selector, ensemble_engine, decision_engine
│   ├── pipeline/           pipeline (9-stage), orchestrator
│   ├── research/           experiments, benchmark, ablation, trust_validation
│   └── utils/              fault_tolerance, cache_manager, experiment_tracker,
│                           model_saver, report_generator, result_store,
│                           logger, config, helpers, error_handler
├── api/                    FastAPI — 7 endpoints
├── app/                    Streamlit — 4 pages (upload/analysis/training/results)
├── config/settings.yaml
├── Dockerfile
├── docker-compose.yml
├── run.py                  CLI entry point
└── tests/test_pipeline.py  16 tests (all passing)
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload-dataset` | Upload CSV |
| GET | `/api/dataset-info` | Dataset preview |
| POST | `/api/analyze` | Full data analysis |
| POST | `/api/train` | 9-stage pipeline |
| GET | `/api/results` | Full results |
| GET | `/api/results/summary` | Decision summary |
| POST | `/api/predict` | Single prediction |

---

## Tech Stack

scikit-learn · SHAP · LIME · FastAPI · Streamlit · Plotly · joblib · scipy · Docker

---

## Limitations & Future Work

- Weight tuning validated on same 12 datasets used for evaluation (no held-out meta-test set)
- Synthetic datasets approximate but don't fully replicate real-world distribution diversity
- Results should be replicated on 20+ diverse UCI/OpenML datasets for publication
- Future: adaptive weight learning using meta-features, online drift re-training trigger

*EMMDS v2.0 — Masters Research Prototype*
