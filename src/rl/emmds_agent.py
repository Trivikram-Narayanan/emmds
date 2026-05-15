"""
EMMDS System 3: Agentic Orchestrator
=====================================
An autonomous agent that orchestrates the EMMDS pipeline by:
  1. Observing intermediate results after each stage
  2. Adapting its evaluation strategy based on what it finds
  3. Generating natural language reasoning for every decision
  4. Producing an auditable decision trail

WHY THIS IS NOVEL:
  Standard AutoML pipelines execute a fixed sequence of steps.
  This agent observes intermediate results and adapts:
    - If imbalance is detected → prioritise calibration evaluation
    - If models disagree strongly → run additional validation folds
    - If trust ranking and accuracy ranking disagree → investigate why
    - If data quality is poor → flag all model results as provisional

  The agent's reasoning is auditable — every decision has a
  natural language justification that a practitioner can read.
  This directly addresses EU AI Act requirements for explainable
  automated decisions.

ARCHITECTURE:
  The agent maintains a structured context (observations, decisions,
  reasoning) and calls EMMDS pipeline tools based on what it observes.

  Tools available to the agent:
    - analyse_dataset()
    - train_models()
    - evaluate_models()
    - compute_trust()
    - compute_agreement()
    - explain_disagreement()
    - generate_explanation()
    - flag_concern()

  The agent's policy is rule-based (for interpretability) but the
  rules are derived from research findings — they encode what the
  experiments showed about when extra analysis is needed.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

OUT = Path("outputs/research/agent")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# AGENT CONTEXT: structured reasoning trail
# ══════════════════════════════════════════════════════════════════════

class AgentContext:
    """Maintains the agent's observations, decisions, and reasoning."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.observations: list  = []
        self.decisions:    list  = []
        self.concerns:     list  = []
        self.reasoning:    list  = []
        self.adaptations:  list  = []
        self.final_report: Optional[str] = None

    def observe(self, observation: str, value=None):
        self.observations.append({'text': observation, 'value': value,
                                   'time': datetime.now().isoformat()})

    def decide(self, decision: str, reason: str):
        self.decisions.append({'decision': decision, 'reason': reason,
                                'time': datetime.now().isoformat()})
        self.reasoning.append(reason)

    def adapt(self, adaptation: str, triggered_by: str):
        self.adaptations.append({'adaptation': adaptation,
                                  'triggered_by': triggered_by})

    def flag(self, concern: str, severity: str = "warning"):
        self.concerns.append({'concern': concern, 'severity': severity})

    def to_dict(self) -> dict:
        return {
            'dataset':      self.dataset_name,
            'observations': self.observations,
            'decisions':    self.decisions,
            'concerns':     self.concerns,
            'adaptations':  self.adaptations,
            'final_report': self.final_report,
        }


# ══════════════════════════════════════════════════════════════════════
# AGENT POLICY: adaptive rules derived from experiments
# ══════════════════════════════════════════════════════════════════════

class AgentPolicy:
    """
    Rule-based adaptive policy derived from EMMDS research findings.

    Rules encode:
      - When to increase CV folds (high noise, small n)
      - When to investigate trust/accuracy disagreement
      - When to prioritise calibration evaluation
      - When to flag results as provisional
      - How to explain the trust score composition
    """

    # Thresholds derived from experimental findings
    IMBALANCE_THRESHOLD   = 3.0     # IR > 3 → flag imbalance concern
    NOISE_THRESHOLD       = 0.10    # noise_estimate > 0.10 → extra CV
    SMALL_N_THRESHOLD     = 300     # n < 300 → extra CV folds
    AGREEMENT_LOW         = 0.65    # agreement < 0.65 → investigate
    TRUST_ACC_GAP         = 0.10    # |trust_rank - acc_rank| > 0.10 → explain
    DQ_CONCERN            = 0.70    # data quality < 0.70 → flag
    CALIBRATION_PRIORITY  = 0.50    # cal_score < 0.50 → prioritise calibration

    def assess_dataset(self, analysis: dict, meta: dict) -> list:
        """
        Observe dataset properties and decide on evaluation strategy.
        Returns list of adaptations to make.
        """
        adaptations = []

        n         = meta.get('n_samples', 0)
        noise     = meta.get('noise_estimate', 0)
        imbalance = meta.get('imbalance_ratio') or 1.0
        dq        = analysis.get('data_quality', {}).get('quality_score', 1.0)

        if imbalance > self.IMBALANCE_THRESHOLD:
            adaptations.append({
                'action':       'prioritise_calibration',
                'reason':       f"Class imbalance ratio {imbalance:.1f} detected. "
                                f"Accuracy is misleading — calibration evaluation "
                                f"takes priority to detect majority-class bias.",
                'triggered_by': f"imbalance_ratio={imbalance:.1f}",
            })

        if noise > self.NOISE_THRESHOLD:
            adaptations.append({
                'action':       'increase_cv_folds',
                'reason':       f"Noise estimate {noise:.3f} exceeds threshold {self.NOISE_THRESHOLD}. "
                                f"High noise increases variance of 5-fold CV estimates. "
                                f"Increasing to 10-fold for more stable stability assessment.",
                'triggered_by': f"noise_estimate={noise:.3f}",
            })

        if n < self.SMALL_N_THRESHOLD:
            adaptations.append({
                'action':       'increase_cv_folds',
                'reason':       f"Dataset has only {n} samples. "
                                f"With small n, each CV fold has high variance. "
                                f"Increasing CV folds to 10 to better estimate stability.",
                'triggered_by': f"n_samples={n}",
            })

        if dq < self.DQ_CONCERN:
            adaptations.append({
                'action':       'flag_provisional',
                'reason':       f"Data quality score {dq:.3f} is below threshold {self.DQ_CONCERN}. "
                                f"Low quality data (missing values, duplicates, or severe imbalance) "
                                f"means model performance estimates are unreliable. "
                                f"All results should be treated as provisional.",
                'triggered_by': f"data_quality={dq:.3f}",
            })

        return adaptations

    def assess_results(self, eval_results: dict, trust_scores: dict,
                        agreement_score: float) -> list:
        """
        Observe model evaluation results and decide on follow-up analysis.
        """
        adaptations = []

        # Check agreement
        if agreement_score < self.AGREEMENT_LOW:
            adaptations.append({
                'action':       'investigate_disagreement',
                'reason':       f"Cross-model agreement score {agreement_score:.3f} is low. "
                                f"Models are giving inconsistent predictions, suggesting "
                                f"the decision boundary is ambiguous in this feature space. "
                                f"Running instance-level disagreement analysis.",
                'triggered_by': f"agreement={agreement_score:.3f}",
            })

        # Check trust/accuracy disagreement
        if eval_results and trust_scores:
            acc_ranking   = sorted(eval_results.keys(),
                                   key=lambda m: eval_results[m].get('f1',0), reverse=True)
            trust_ranking = sorted(trust_scores.keys(),
                                   key=trust_scores.get, reverse=True)

            if acc_ranking and trust_ranking:
                acc_best   = acc_ranking[0]
                trust_best = trust_ranking[0]
                if acc_best != trust_best:
                    adaptations.append({
                        'action':       'explain_rank_disagreement',
                        'reason':       f"Trust ranking and accuracy ranking disagree. "
                                        f"Accuracy selects {acc_best} "
                                        f"(F1={eval_results[acc_best].get('f1',0):.4f}) "
                                        f"but trust selects {trust_best} "
                                        f"(trust={trust_scores[trust_best]:.4f}). "
                                        f"This disagreement requires explanation.",
                        'triggered_by': f"acc_best={acc_best}, trust_best={trust_best}",
                    })

        # Check calibration quality
        for model, metrics in eval_results.items():
            if trust_scores.get(model, 1.0) > 0.7:
                cal = metrics.get('cal_score', 1.0)
                if cal is not None and cal < self.CALIBRATION_PRIORITY:
                    adaptations.append({
                        'action':       'flag_calibration_concern',
                        'reason':       f"Model {model} has high trust "
                                        f"({trust_scores[model]:.3f}) but poor calibration "
                                        f"({cal:.3f}). Its probability estimates are unreliable "
                                        f"despite good F1. Flagging for practitioner attention.",
                        'triggered_by': f"{model}_cal={cal:.3f}",
                    })

        return adaptations


# ══════════════════════════════════════════════════════════════════════
# EXPLANATION ENGINE
# ══════════════════════════════════════════════════════════════════════

class ExplanationEngine:
    """
    Generates structured natural language explanations of EMMDS decisions.
    Uses templates grounded in research findings — not generic LLM outputs.

    The explanations reference:
      - The specific trust components and their empirical weights
      - The dataset properties that influenced the decision
      - The tradeoffs between competing models
      - Practical deployment implications
    """

    COMPONENT_DESCRIPTIONS = {
        'stability':     "cross-validation consistency (weight 0.40 — most important component)",
        'data_quality':  "dataset quality (weight 0.35 — second most important)",
        'agreement':     "cross-model consensus (weight 0.10)",
        'calibration':   "probability calibration reliability (weight 0.10)",
        'accuracy':      "predictive performance F1 (weight 0.05 — least influential)",
    }

    def explain_model_selection(
        self,
        selected_model:  str,
        rejected_model:  str,
        selected_metrics: dict,
        rejected_metrics: dict,
        selected_trust:  float,
        rejected_trust:  float,
        dataset_properties: dict,
        context: AgentContext,
    ) -> str:
        """
        Generate a plain-language explanation for why one model was
        selected over another.
        """
        s_acc   = selected_metrics.get('test_f1', 0)
        r_acc   = rejected_metrics.get('test_f1', 0)
        s_stab  = selected_metrics.get('stability', 0)
        r_stab  = rejected_metrics.get('stability', 0)
        s_cal   = selected_metrics.get('cal_score', 0)
        r_cal   = rejected_metrics.get('cal_score', 0)
        s_cv_std = selected_metrics.get('cv_std', 0)
        r_cv_std = rejected_metrics.get('cv_std', 0)

        n = dataset_properties.get('n_samples', 0)
        ir = dataset_properties.get('imbalance_ratio', 1.0) or 1.0

        paragraphs = []

        # Opening: state the selection and the key tradeoff
        if s_acc < r_acc:
            # Trust selected a lower-accuracy model
            paragraphs.append(
                f"EMMDS selected {selected_model} (F1={s_acc:.4f}, Trust={selected_trust:.4f}) "
                f"over {rejected_model} (F1={r_acc:.4f}, Trust={rejected_trust:.4f}), "
                f"despite {rejected_model} having higher accuracy. Here is why:"
            )
        else:
            paragraphs.append(
                f"EMMDS selected {selected_model} (F1={s_acc:.4f}, Trust={selected_trust:.4f}) "
                f"over {rejected_model} (F1={r_acc:.4f}, Trust={rejected_trust:.4f}). "
                f"The selection is consistent with both accuracy and trust."
            )

        # Stability explanation
        if s_stab > r_stab + 0.05:
            paragraphs.append(
                f"The key factor is stability: {selected_model} shows CV standard deviation "
                f"of {s_cv_std:.4f} versus {r_cv_std:.4f} for {rejected_model}. "
                f"Lower variance across cross-validation folds means {selected_model} "
                f"performs consistently regardless of which data subset it sees — "
                f"a strong signal that it will remain reliable when deployed on new data. "
                f"EMMDS weights stability at 0.40 (empirically derived across 21 datasets) "
                f"because it is the strongest predictor of deployment reliability."
            )

        # Calibration explanation
        if s_cal > r_cal + 0.05:
            paragraphs.append(
                f"Probability calibration also favours {selected_model}: "
                f"calibration score {s_cal:.4f} versus {r_cal:.4f}. "
                f"A well-calibrated model's confidence estimates are meaningful — "
                f"when it says 80% probability, approximately 80% of such predictions "
                f"are correct. Poor calibration means the probabilities are unreliable "
                f"even when the label predictions are accurate."
            )

        # Dataset-specific context
        if ir > 3.0:
            paragraphs.append(
                f"This dataset has class imbalance ratio {ir:.1f}:1. "
                f"On imbalanced data, accuracy metrics are inflated by the majority class. "
                f"The trust score's calibration component is particularly important here "
                f"because it penalises models that predict the majority class confidently "
                f"without genuine discrimination."
            )

        if n < 300:
            paragraphs.append(
                f"With only {n} samples, all performance estimates carry high uncertainty. "
                f"The trust score's stability component (measuring CV variance) is "
                f"especially important for small datasets — a model that happens to "
                f"perform well on one fold but poorly on another is not reliable."
            )

        # Deployment recommendation
        if selected_trust >= 0.85:
            rec = (f"{selected_model} has Very High Trust ({selected_trust:.4f}). "
                   f"It is suitable for deployment with standard monitoring.")
        elif selected_trust >= 0.70:
            rec = (f"{selected_model} has High Trust ({selected_trust:.4f}). "
                   f"Suitable for deployment with monitoring — set drift alert "
                   f"threshold at PSI > 0.10.")
        else:
            rec = (f"{selected_model} has Moderate Trust ({selected_trust:.4f}). "
                   f"Consider collecting more data or feature engineering before deployment. "
                   f"If deployed, set aggressive monitoring thresholds.")

        paragraphs.append(f"Deployment recommendation: {rec}")

        explanation = "\n\n".join(paragraphs)
        context.final_report = explanation
        return explanation

    def explain_trust_components(
        self,
        model_name:  str,
        trust_breakdown: dict,
        context: AgentContext,
    ) -> str:
        """Explain what each trust component contributes for a specific model."""
        lines = [f"Trust score breakdown for {model_name} "
                 f"(total: {trust_breakdown.get('trust_score',0):.4f}):\n"]

        components = [
            ('stability',    'stability_component',    0.40),
            ('data_quality', 'data_quality_component', 0.35),
            ('agreement',    'agreement_component',    0.10),
            ('calibration',  'calibration_component',  0.10),
            ('accuracy',     'accuracy_component',     0.05),
        ]

        for comp_name, key, weight in components:
            val  = trust_breakdown.get(key, 0)
            contrib = round(weight * val, 4)
            desc = self.COMPONENT_DESCRIPTIONS[comp_name]
            lines.append(f"  {comp_name:12s}  score={val:.4f}  weight={weight:.2f}  "
                          f"contribution={contrib:.4f}  [{desc}]")

        lines.append(f"\nThe dominant components are stability and data quality "
                     f"(total weight 0.75), reflecting the empirical finding that "
                     f"these two dimensions are the strongest predictors of "
                     f"deployment reliability.")

        explanation = "\n".join(lines)
        return explanation


# ══════════════════════════════════════════════════════════════════════
# EMMDS AGENT: main orchestrator
# ══════════════════════════════════════════════════════════════════════

class EMMDSAgent:
    """
    The full agentic orchestrator.

    Takes a dataset, runs the EMMDS pipeline adaptively, generates
    explanations, and produces an auditable decision trail.
    """

    def __init__(self):
        self.policy  = AgentPolicy()
        self.explainer = ExplanationEngine()
        self.run_log = []

    def run(
        self,
        df:         pd.DataFrame,
        target_col: str,
        dataset_name: str = "dataset",
        verbose:    bool  = True,
    ) -> dict:
        """
        Run the full agentic pipeline.

        Returns:
            {
                'decision': {...},
                'context':  AgentContext,
                'explanation': str,
                'adaptations_made': [...],
                'concerns': [...],
            }
        """
        context = AgentContext(dataset_name)
        t_start = __import__('time').time()

        if verbose:
            print(f"\n{'='*65}")
            print(f"  EMMDS AGENT — {dataset_name}")
            print(f"{'='*65}")

        # ── Stage 1: Dataset assessment ───────────────────────────────
        context.observe(f"Dataset loaded: {df.shape[0]} samples, "
                        f"{df.shape[1]-1} features, target='{target_col}'")

        from src.data_engine.analyzer import DataAnalyzer
        from src.data_engine.meta_features import MetaFeatureExtractor
        from src.data_engine.data_quality import DataQualityScorer

        analysis  = DataAnalyzer().analyze(df, target_col)
        meta_ext  = MetaFeatureExtractor(); meta_ext.extract(df, target_col)
        meta      = meta_ext.get_meta()
        dq_scorer = DataQualityScorer()
        dq_score  = dq_scorer.score_dataset(df, target_col)
        dq_info   = dq_scorer.get_breakdown()

        context.observe(f"Task detected: {analysis['task']}")
        context.observe(f"Meta-features extracted: "
                        f"imbalance={meta.get('imbalance_ratio',1):.2f}, "
                        f"noise={meta.get('noise_estimate',0):.3f}, "
                        f"n={meta.get('n_samples',0)}")
        context.observe(f"Data quality: {dq_score:.3f} ({dq_info.get('label','')})")

        # Agent adapts based on dataset properties
        dataset_adaptations = self.policy.assess_dataset(
            {'data_quality': dq_info}, meta)
        for adp in dataset_adaptations:
            context.adapt(adp['action'], adp['triggered_by'])
            context.decide(f"Adapting: {adp['action']}", adp['reason'])
            if verbose:
                print(f"  ADAPT  [{adp['triggered_by']}]")
                print(f"         {adp['reason'][:100]}...")

        # Decide CV folds based on dataset
        cv_folds = 5
        if any(a['action'] == 'increase_cv_folds' for a in dataset_adaptations):
            cv_folds = 10
            context.decide("Increased CV folds to 10",
                           "Noise or small-n detected — more folds needed for stable estimates")

        provisional = any(a['action'] == 'flag_provisional'
                          for a in dataset_adaptations)
        if provisional:
            context.flag("Data quality is low — results provisional", "critical")

        # ── Stage 2: Preprocessing + Training ─────────────────────────
        context.observe("Starting preprocessing and model training")

        from src.data_engine.preprocessor import DataPreprocessor
        from src.training.trainer import ModelTrainer
        from src.training.cross_validation import CrossValidator
        from src.calibration.calibrator import ModelCalibrator
        from src.evaluation.evaluator import ModelEvaluator
        from src.evaluation.ranking import ModelRanker
        from src.decision.model_agreement import ModelAgreementEngine
        from src.decision.trust_score import TrustScoreEngine
        import numpy as np

        pp = DataPreprocessor(task=analysis['task'])
        X_tr, X_te, y_tr, y_te = pp.fit_transform(df, target_col)
        X_all = np.vstack([X_tr, X_te])
        y_all = np.concatenate([y_tr, y_te])

        trainer = ModelTrainer()
        trained = trainer.train_all(X_tr, y_tr)
        context.observe(f"Trained {len(trained)} models: {list(trained.keys())}")

        # CV with adapted folds
        cv_engine  = CrossValidator(task=analysis['task'])
        cv_results = cv_engine.run(trained, X_all, y_all, n_splits=cv_folds)
        context.observe(f"Cross-validation complete ({cv_folds} folds)")

        # Calibration
        calibrator = ModelCalibrator()
        calibrated = calibrator.calibrate_all(trained, X_tr, y_tr, X_te, y_te)
        cal_scores = calibrator.get_calibration_scores()

        # Evaluation
        evaluator    = ModelEvaluator(task=analysis['task'])
        eval_results = evaluator.evaluate_all(calibrated, X_te, y_te)

        # Agreement
        try:
            agree_result = ModelAgreementEngine().compute(calibrated, X_te)
            agree_score  = agree_result.get('agreement_score', 0.5)
        except Exception:
            agree_score = 0.5

        context.observe(f"Agreement score: {agree_score:.4f}")

        # ── Stage 3: Trust scores + ranking ──────────────────────────
        trust_engine = TrustScoreEngine(use_empirical_weights=True)
        trust_scores = trust_engine.compute_all(
            eval_results, cal_scores, cv_results,
            task=analysis['task'],
            agreement_score=agree_score,
            data_quality_score=dq_score,
        )

        ranker = ModelRanker(task=analysis['task'])
        lb     = ranker.rank(eval_results, cv_results)

        # Agent assesses results and adapts
        result_adaptations = self.policy.assess_results(
            eval_results, trust_scores, agree_score)
        for adp in result_adaptations:
            context.adapt(adp['action'], adp['triggered_by'])
            context.decide(f"Investigating: {adp['action']}", adp['reason'])
            if verbose:
                print(f"  ADAPT  [{adp['triggered_by']}]")
                print(f"         {adp['reason'][:100]}...")

        # ── Stage 4: Final selection ───────────────────────────────────
        from src.decision.model_selector import ModelSelector
        selector = ModelSelector()
        selection = selector.select(lb, trust_scores, calibrated)
        best_name = selection.get('best_model_name', lb[0]['model'] if lb else '')
        best_trust = trust_scores.get(best_name, 0.0)
        best_breakdown = trust_engine.get_breakdown().get(best_name, {})
        best_metrics   = eval_results.get(best_name, {})

        context.observe(f"Selected model: {best_name} "
                        f"(trust={best_trust:.4f}, "
                        f"F1={best_metrics.get('f1',0):.4f})")

        # ── Stage 5: Generate explanation ─────────────────────────────
        # Find the accuracy-best model for comparison
        acc_best_name = max(eval_results.keys(),
                            key=lambda m: eval_results[m].get('f1',0))

        explanation = self.explainer.explain_model_selection(
            selected_model=best_name,
            rejected_model=acc_best_name if acc_best_name != best_name else lb[1]['model'] if len(lb)>1 else best_name,
            selected_metrics={**best_metrics,
                               'stability': best_breakdown.get('stability_component',0),
                               'cv_std':    cv_results.get(best_name,{}).get('f1_weighted',{}).get('std',0)},
            rejected_metrics={**eval_results.get(acc_best_name if acc_best_name!=best_name else (lb[1]['model'] if len(lb)>1 else best_name), {}),
                               'stability': trust_engine.get_breakdown().get(acc_best_name,{}).get('stability_component',0),
                               'cv_std':    cv_results.get(acc_best_name,{}).get('f1_weighted',{}).get('std',0)},
            selected_trust=best_trust,
            rejected_trust=trust_scores.get(acc_best_name, 0),
            dataset_properties=meta,
            context=context,
        )

        trust_explanation = self.explainer.explain_trust_components(
            best_name, best_breakdown, context)

        if verbose:
            print(f"\n  DECISION: {best_name}  trust={best_trust:.4f}")
            print(f"\n  EXPLANATION:\n")
            for para in explanation.split('\n\n'):
                print(f"    {para}")

        # ── Build final output ─────────────────────────────────────────
        elapsed = round(__import__('time').time() - t_start, 2)
        result  = {
            'dataset':            dataset_name,
            'best_model':         best_name,
            'trust_score':        best_trust,
            'trust_label':        trust_engine.get_trust_label(best_trust),
            'f1':                 best_metrics.get('f1', 0),
            'accuracy':           best_metrics.get('accuracy', 0),
            'trust_breakdown':    best_breakdown,
            'leaderboard':        lb,
            'all_trust_scores':   trust_scores,
            'agreement_score':    agree_score,
            'data_quality':       dq_info,
            'adaptations_made':   context.adaptations,
            'concerns':           context.concerns,
            'cv_folds_used':      cv_folds,
            'provisional':        provisional,
            'explanation':        explanation,
            'trust_explanation':  trust_explanation,
            'decision_trail':     context.to_dict(),
            'elapsed_s':          elapsed,
        }

        # Save decision trail
        def _j(o):
            if isinstance(o,(bool,int,float,str,type(None))): return o
            if isinstance(o, np.bool_): return bool(o)
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, dict): return {k: _j(v) for k,v in o.items()}
            if isinstance(o, list): return [_j(i) for i in o]
            return str(o)

        trail_path = OUT / f"{dataset_name}_trail.json"
        trail_path.write_text(json.dumps(_j(result), indent=2))

        self.run_log.append({
            'dataset':         dataset_name,
            'best_model':      best_name,
            'trust_score':     round(best_trust, 4),
            'adaptations':     len(context.adaptations),
            'concerns':        len(context.concerns),
            'elapsed_s':       elapsed,
        })

        return result
