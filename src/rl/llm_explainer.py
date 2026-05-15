"""
EMMDS System 4: LLM Trust Explainer
=====================================
Uses a language model to generate natural language explanations
of trust-based model selection decisions.

Unlike System 3's rule-based explanations, this uses an LLM to:
  1. Synthesise multiple pieces of evidence into coherent narrative
  2. Adapt explanation complexity to the audience
  3. Answer follow-up questions about the decision
  4. Generate domain-specific interpretations

RESEARCH CONTRIBUTION:
  The research question is not "can an LLM explain things" —
  that is already known. The question is:
  "Do practitioners make better deployment decisions when given
   LLM-generated natural language explanations of trust scores
   compared to numeric scores alone?"

  This is evaluated by a practitioner study (designed here,
  to be conducted by the researcher with 10+ participants).

IMPLEMENTATION:
  Uses the Google Gemini API to generate contextual explanations
  from structured trust data.

  Prompt engineering follows chain-of-thought reasoning:
    1. Summarise the dataset properties
    2. Explain what each trust component means in this context
    3. Describe the tradeoffs between competing models
    4. Give a deployment recommendation with reasoning
    5. Flag any concerns the practitioner should investigate

FALLBACK:
  When API unavailable, uses high-quality structured templates
  that are more informative than simple string formatting.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

OUT = Path("outputs/research/genai")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════

class TrustExplainerPrompts:
    """Builds structured prompts for trust score explanation."""

    SYSTEM_PROMPT = """You are EMMDS-Explainer, an expert AI system that explains
machine learning model selection decisions to practitioners.

Your explanations must be:
1. ACCURATE: based only on the data provided, no invention
2. CLEAR: understandable by someone without ML expertise
3. ACTIONABLE: tell the practitioner what to do, not just what happened
4. HONEST: acknowledge uncertainty and limitations

The EMMDS Trust Score combines five components with empirically
derived weights (from meta-learning across 21 datasets):
  - Stability (0.40):     Cross-validation consistency — most important
  - Data Quality (0.35):  Dataset cleanliness and balance — second
  - Agreement (0.10):     Cross-model consensus
  - Calibration (0.10):   Probability reliability
  - Accuracy (0.05):      Raw F1 performance — least important

KEY RESEARCH FINDING: Accuracy weight is nearly zero because by
the time models reach evaluation, they all have reasonable accuracy.
What distinguishes deployment-ready from deployment-risky models is
consistency (stability) and data quality."""

    @staticmethod
    def build_explanation_prompt(
        dataset_name:    str,
        dataset_props:   dict,
        selected_model:  str,
        rejected_model:  str,
        selected_data:   dict,
        rejected_data:   dict,
        concerns:        list,
        adaptations:     list,
    ) -> str:
        """Build a complete explanation prompt."""

        props_str = f"""
Dataset: {dataset_name}
  Samples: {dataset_props.get('n_samples', '?')}
  Features: {dataset_props.get('n_features', '?')}
  Classes: {dataset_props.get('n_classes', '?')}
  Imbalance ratio: {dataset_props.get('imbalance_ratio', 1.0):.2f}
  Noise estimate: {dataset_props.get('noise_estimate', 0):.3f}
  Data quality score: {dataset_props.get('dq_score', '?')}"""

        selected_str = f"""
Selected model: {selected_model}
  Trust score:  {selected_data.get('trust', 0):.4f}
  F1:           {selected_data.get('f1', 0):.4f}
  Stability:    {selected_data.get('stability', 0):.4f}
  Calibration:  {selected_data.get('calibration', 0):.4f}
  CV std:       {selected_data.get('cv_std', 0):.4f}"""

        rejected_str = f"""
Alternative model: {rejected_model}
  Trust score:  {rejected_data.get('trust', 0):.4f}
  F1:           {rejected_data.get('f1', 0):.4f}
  Stability:    {rejected_data.get('stability', 0):.4f}
  Calibration:  {rejected_data.get('calibration', 0):.4f}
  CV std:       {rejected_data.get('cv_std', 0):.4f}"""

        concerns_str = ""
        if concerns:
            concerns_str = "\nConcerns identified:\n" + "\n".join(
                f"  - [{c['severity']}] {c['concern']}" for c in concerns)

        adaptations_str = ""
        if adaptations:
            adaptations_str = "\nAgent adaptations made:\n" + "\n".join(
                f"  - {a['adaptation']} (triggered by: {a['triggered_by']})"
                for a in adaptations)

        return f"""Please explain this model selection decision to a practitioner
who needs to deploy a machine learning model in production.

{props_str}

{selected_str}

{rejected_str}
{concerns_str}
{adaptations_str}

Generate an explanation that:
1. Starts with the bottom line (which model was selected and the one key reason why)
2. Explains the most important trust component that drove the decision
3. Describes the specific risk if the alternative model had been deployed instead
4. Gives a concrete monitoring recommendation
5. Notes any concerns the practitioner should investigate

Write in plain English. No bullet points. Maximum 300 words."""

    @staticmethod
    def build_qa_prompt(question: str, context_json: str) -> str:
        """Build a follow-up Q&A prompt."""
        return f"""A practitioner is asking a follow-up question about an EMMDS decision.

Decision context (JSON):
{context_json}

Practitioner question: {question}

Answer the question directly, referencing specific numbers from the
decision context. Be concise (100 words maximum)."""


# ══════════════════════════════════════════════════════════════════════
# LLM EXPLAINER
# ══════════════════════════════════════════════════════════════════════

class LLMTrustExplainer:
    """
    Generates natural language trust score explanations using an LLM.
    Falls back to structured templates when API is unavailable.
    """

    def __init__(self, use_api: bool = True):
        self.use_api    = use_api
        self.prompts    = TrustExplainerPrompts()
        self._api_available = False
        self.explanation_log = []

        if use_api:
            self._check_api()

    def _check_api(self) -> bool:
        """Test if Gemini API is available."""
        try:
            import google.generativeai as genai
            if os.environ.get("GOOGLE_API_KEY"):
                self._api_available = True
                return True
        except Exception:
            pass
        self._api_available = False
        return False

    def explain(
        self,
        dataset_name:   str,
        dataset_props:  dict,
        selected_model: str,
        rejected_model: str,
        selected_data:  dict,
        rejected_data:  dict,
        concerns:       list = None,
        adaptations:    list = None,
    ) -> dict:
        """
        Generate a natural language explanation.

        Returns:
            {
                'explanation': str,
                'source':      'llm' | 'template',
                'prompt':      str (the prompt used),
            }
        """
        prompt = self.prompts.build_explanation_prompt(
            dataset_name, dataset_props,
            selected_model, rejected_model,
            selected_data, rejected_data,
            concerns or [], adaptations or [],
        )

        if self.use_api and self._api_available:
            explanation = self._call_api(prompt)
            source = 'llm'
        else:
            explanation = self._template_explanation(
                dataset_name, dataset_props,
                selected_model, rejected_model,
                selected_data, rejected_data,
                concerns or [], adaptations or [],
            )
            source = 'template'

        result = {
            'explanation': explanation,
            'source':      source,
            'prompt':      prompt,
            'dataset':     dataset_name,
            'selected':    selected_model,
            'rejected':    rejected_model,
        }
        self.explanation_log.append(result)
        return result

    def answer_question(
        self,
        question:     str,
        context_json: str,
    ) -> str:
        """Answer a follow-up question about a decision."""
        if self.use_api and self._api_available:
            prompt = self.prompts.build_qa_prompt(question, context_json)
            return self._call_api(prompt)
        return self._template_answer(question, context_json)

    def _call_api(self, prompt: str) -> str:
        """Call the Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=self.prompts.SYSTEM_PROMPT
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._template_explanation.__doc__

    def _template_explanation(
        self,
        dataset_name, dataset_props,
        selected_model, rejected_model,
        selected_data, rejected_data,
        concerns, adaptations,
    ) -> str:
        """
        High-quality structured template explanation.
        More informative than generic LLM output when
        grounded in specific numbers.
        """
        s_trust = selected_data.get('trust', 0)
        r_trust = rejected_data.get('trust', 0)
        s_f1    = selected_data.get('f1', 0)
        r_f1    = rejected_data.get('f1', 0)
        s_stab  = selected_data.get('stability', 0)
        r_stab  = rejected_data.get('stability', 0)
        s_cal   = selected_data.get('calibration', 0)
        s_cv_std = selected_data.get('cv_std', 0)
        r_cv_std = rejected_data.get('cv_std', 0)
        n       = dataset_props.get('n_samples', 0)
        ir      = dataset_props.get('imbalance_ratio', 1.0) or 1.0

        lines = []

        # Bottom line
        if s_f1 < r_f1:
            lines.append(
                f"EMMDS selected {selected_model} over {rejected_model} "
                f"despite lower accuracy (F1 {s_f1:.4f} vs {r_f1:.4f}). "
                f"The reason: {selected_model} is substantially more consistent "
                f"across data subsets and therefore more trustworthy for deployment."
            )
        else:
            lines.append(
                f"EMMDS selected {selected_model} (F1={s_f1:.4f}, Trust={s_trust:.4f}) "
                f"over {rejected_model} (F1={r_f1:.4f}, Trust={r_trust:.4f})."
            )

        # Key component
        if s_stab > r_stab + 0.03:
            lines.append(
                f"The decisive factor is cross-validation stability: {selected_model} "
                f"had a CV standard deviation of {s_cv_std:.4f} versus {r_cv_std:.4f} "
                f"for {rejected_model}. A smaller CV standard deviation means the model "
                f"performs consistently no matter which portion of your data it sees — "
                f"a strong indicator it will remain reliable after deployment when it "
                f"encounters data that differs slightly from training."
            )

        # Calibration
        if s_cal > rejected_data.get('calibration', 0) + 0.05:
            lines.append(
                f"{selected_model}'s probability calibration score ({s_cal:.4f}) is "
                f"also better. Its confidence estimates are reliable — when it says "
                f"80% probability, approximately 80% of such cases are correct. "
                f"This matters if your application uses probability outputs for "
                f"decision thresholds or risk scoring."
            )

        # Dataset context
        if ir > 3.0:
            lines.append(
                f"Note: this dataset has {ir:.1f}:1 class imbalance. "
                f"On imbalanced data, accuracy can be misleadingly high "
                f"(a model predicting only the majority class gets {ir/(ir+1)*100:.0f}% accuracy). "
                f"The trust score's calibration component penalises this — making it "
                f"a more reliable selector than accuracy alone."
            )

        # Concerns
        if concerns:
            concern_texts = [c['concern'] for c in concerns]
            lines.append("⚠️ Concerns requiring attention: " +
                         "; ".join(concern_texts))

        # Monitoring recommendation
        if s_trust >= 0.85:
            mon = "Standard monitoring at 2-week intervals is sufficient."
        elif s_trust >= 0.70:
            mon = "Weekly monitoring recommended. Set PSI drift alert at threshold 0.10."
        else:
            mon = ("Increased monitoring required (daily checks). "
                   f"Trust score {s_trust:.3f} indicates elevated deployment risk. "
                   "Consider collecting more data before deploying.")

        lines.append(f"Monitoring recommendation: {mon}")

        return "\n\n".join(lines)

    def _template_answer(self, question: str, context_json: str) -> str:
        """Simple Q&A fallback."""
        return (f"Based on the decision context, the answer to '{question}' "
                f"requires examining the trust breakdown. The trust score "
                f"primarily reflects stability and data quality (combined weight 0.75). "
                f"For more detail, review the trust_breakdown field in the decision output.")


# ══════════════════════════════════════════════════════════════════════
# PRACTITIONER STUDY DESIGN
# ══════════════════════════════════════════════════════════════════════

class PractitionerStudyDesign:
    """
    Defines the evaluation methodology for testing whether LLM
    explanations improve practitioner deployment decisions.

    This is the research evaluation — it cannot be run programmatically
    but provides the complete study design for the thesis.
    """

    STUDY_DESIGN = {
        "title": "Does Natural Language Trust Explanation Improve "
                 "ML Deployment Decision Quality?",

        "hypothesis": {
            "H0": "Practitioners receiving numeric trust scores make deployment "
                  "decisions of equal quality to those receiving natural language "
                  "explanations (measured by deployment risk of chosen model).",
            "H1": "Practitioners receiving natural language explanations make "
                  "significantly better deployment decisions than those receiving "
                  "numeric scores alone (lower mean deployment risk of chosen model).",
        },

        "participants": {
            "n":          "≥10 (Masters/PhD students or industry practitioners)",
            "background": "Some ML experience but not AutoML experts",
            "groups":     "Randomly assigned to numeric or explanation condition",
        },

        "procedure": [
            "Participant receives 3 model selection scenarios (different datasets)",
            "Control group: sees numeric trust scores and F1 values only",
            "Treatment group: sees same numbers + natural language explanation",
            "Both groups choose which model to deploy",
            "Outcome: deployment risk of their chosen model",
            "Secondary: confidence rating (1-5) and explanation of their choice",
        ],

        "analysis": {
            "primary":   "Mann-Whitney U test on deployment risk (control vs treatment)",
            "secondary": "Thematic analysis of participant explanations",
            "effect":    "Cohen's d for deployment risk difference",
        },

        "scenarios": [
            {
                "id": 1,
                "type": "Trust beats accuracy",
                "description": "Two models: Model A has higher F1 but lower stability. "
                               "Model B has lower F1 but much better stability. "
                               "Trust selects B. Correct deployment choice: B.",
            },
            {
                "id": 2,
                "type": "Imbalanced dataset",
                "description": "Dataset with 10:1 imbalance. Model A has 94% accuracy "
                               "but poor calibration (majority-class bias). "
                               "Model B has 87% accuracy but good calibration. "
                               "Correct deployment choice: B.",
            },
            {
                "id": 3,
                "type": "Small dataset",
                "description": "150 samples. Model A shows high mean CV F1 but "
                               "σ=0.15 (highly variable). Model B shows lower mean "
                               "but σ=0.03 (stable). Correct deployment choice: B.",
            },
        ],

        "expected_outcome": (
            "Practitioners with natural language explanations will more frequently "
            "choose the trust-recommended model (Model B in all 3 scenarios), "
            "resulting in lower mean deployment risk compared to the control group "
            "who see only numeric scores."
        ),
    }

    def generate_study_materials(self) -> dict:
        """Generate the complete study materials as a structured dict."""
        return self.STUDY_DESIGN

    def print_study_design(self) -> None:
        """Print the full study design."""
        d = self.STUDY_DESIGN
        print("\n" + "="*65)
        print(f"  {d['title']}")
        print("="*65)
        print(f"\nHypotheses:")
        print(f"  H0: {d['hypothesis']['H0']}")
        print(f"  H1: {d['hypothesis']['H1']}")
        print(f"\nParticipants: {d['participants']['n']} ({d['participants']['background']})")
        print(f"\nScenarios (3 deployment decisions each):")
        for s in d['scenarios']:
            print(f"  {s['id']}. {s['type']}: {s['description'][:80]}...")
        print(f"\nAnalysis: {d['analysis']['primary']}")
        print(f"\nExpected: {d['expected_outcome'][:100]}...")
        print("="*65)
