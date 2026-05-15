"""
LLM Trust Score — Direction 4
================================
Extends the EMMDS trust framework to language model selection.

Background
----------
The EMMDS 5-component trust score was designed for tabular classifiers.
Language models require different reliability signals:
  • Consistency:    Does the model give the same answer to paraphrased questions?
  • Calibration:    Does the model's stated confidence match its accuracy?
  • Agreement:      Do multiple LLMs agree on the same output?
  • Contamination:  Has the benchmark appeared in the model's training data?
  • Helpfulness:    A proxy for usefulness (task completion rate).

Each component produces a score in [0, 1]. The composite is a weighted sum
using empirically motivated weights (analogous to EMMDS tabular weights).

Architecture
------------
                                    ┌─ ConsistencyScorer
                                    ├─ CalibrationScorer
  LLMTrustScore.evaluate(model_fn)─┤─ AgreementScorer
                                    ├─ ContaminationScorer
                                    └─ HelpfulnessScorer
                                              ↓
                               Weighted sum → LLMTrustResult

Model interface
---------------
Any callable:  model_fn(prompt: str) → {"text": str, "confidence": float (optional)}
Works offline with mock models — no API keys required.

Reference:
    Kadavath et al. (2022) "Language Models (Mostly) Know What They Know."
    Magar & Schwartz (2022) "Data Contamination: From Impurity to Incompetence."
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


ModelFn = Callable[[str], Dict]

# Composite weights (sum to 1.0)
LLM_WEIGHTS = {
    "consistency":    0.30,   # highest: unreliable outputs are unusable
    "calibration":    0.20,   # confidence-accuracy alignment
    "agreement":      0.20,   # cross-model consensus
    "contamination":  0.15,   # data leakage penalty (inverted)
    "helpfulness":    0.15,   # task completion rate
}


# ─────────────────────────────────────────────────────────────
# Consistency Scorer
# ─────────────────────────────────────────────────────────────

class ConsistencyScorer:
    """
    Measures output stability under prompt paraphrasing.

    For each (question, answer) pair, generates k paraphrased prompts
    and measures the fraction that produce the same answer.

    score = mean fraction of paraphrase answers matching the canonical answer.
    """

    PARAPHRASE_TEMPLATES = [
        "{}",
        "Please answer: {}",
        "Question: {} Answer:",
        "I'd like to know: {}",
        "Can you tell me {}?",
        "What is the answer to: {}?",
    ]

    def __init__(self, n_paraphrases: int = 5):
        self.n_paraphrases = min(n_paraphrases, len(self.PARAPHRASE_TEMPLATES) - 1)

    def score(
        self,
        model_fn: ModelFn,
        questions: List[str],
        canonical_answers: Optional[List[str]] = None,
    ) -> float:
        """
        Args:
            model_fn:           callable model
            questions:          list of base questions
            canonical_answers:  if None, uses model's answer to the first paraphrase
        """
        if not questions:
            return 1.0

        agreements = []
        templates = self.PARAPHRASE_TEMPLATES[:self.n_paraphrases + 1]

        for i, q in enumerate(questions):
            # Get canonical answer
            canonical = (canonical_answers[i] if canonical_answers
                         else self._call(model_fn, templates[0].format(q)))

            n_agree = 0
            for tmpl in templates[1:]:
                answer = self._call(model_fn, tmpl.format(q))
                if self._answers_match(canonical, answer):
                    n_agree += 1

            agreements.append(n_agree / (len(templates) - 1))

        return float(np.clip(np.mean(agreements), 0, 1))

    def _call(self, model_fn: ModelFn, prompt: str) -> str:
        try:
            result = model_fn(prompt)
            return str(result.get("text", "")).strip().lower()
        except Exception:
            return ""

    def _answers_match(self, a: str, b: str) -> bool:
        a, b = a.strip().lower(), b.strip().lower()
        if not a or not b:
            return False
        # Exact match or one contains the other (handles verbosity variation)
        return a == b or a in b or b in a


# ─────────────────────────────────────────────────────────────
# Calibration Scorer
# ─────────────────────────────────────────────────────────────

class CalibrationScorer:
    """
    Measures confidence-accuracy alignment via Brier score on factual Q&A.

    If the model returns a confidence score, use it directly.
    If not, use prompt-based elicitation: "How confident are you? (0-1)"

    score = 1 - Brier(confidence, correctness)
    """

    CONFIDENCE_ELICIT_TEMPLATE = (
        "On a scale of 0.0 to 1.0, how confident are you that your answer "
        "to '{}' was correct? Reply with only a number."
    )

    def score(
        self,
        model_fn: ModelFn,
        questions: List[str],
        correct_answers: List[str],
    ) -> float:
        if not questions:
            return 0.5

        confidences = []
        correctness = []

        for q, ans_gt in zip(questions, correct_answers):
            result = model_fn(q)
            text = str(result.get("text", "")).strip().lower()
            conf = result.get("confidence")

            if conf is None:
                conf_result = model_fn(self.CONFIDENCE_ELICIT_TEMPLATE.format(q))
                conf = self._parse_confidence(conf_result.get("text", ""))

            is_correct = float(ans_gt.strip().lower() in text or
                                text in ans_gt.strip().lower())
            confidences.append(float(np.clip(conf, 0, 1)))
            correctness.append(is_correct)

        if not confidences:
            return 0.5

        confidences = np.array(confidences)
        correctness = np.array(correctness)
        brier = float(np.mean((confidences - correctness) ** 2))
        return float(np.clip(1.0 - brier, 0, 1))

    def _parse_confidence(self, text: str) -> float:
        import re
        matches = re.findall(r"\d+\.?\d*", str(text))
        for m in matches:
            val = float(m)
            if 0.0 <= val <= 1.0:
                return val
            if 1.0 < val <= 100.0:
                return val / 100.0
        return 0.5


# ─────────────────────────────────────────────────────────────
# Agreement Scorer
# ─────────────────────────────────────────────────────────────

class AgreementScorer:
    """
    Cross-model consensus: fraction of questions on which all provided
    models agree with the focal model.

    If only one model is provided, returns 1.0 (no disagreement possible).
    """

    def score(
        self,
        focal_model_fn: ModelFn,
        peer_model_fns: List[ModelFn],
        questions: List[str],
    ) -> float:
        if not peer_model_fns or not questions:
            return 1.0

        agreements = []
        for q in questions:
            focal_ans = self._call(focal_model_fn, q)
            peer_agree = 0
            for peer_fn in peer_model_fns:
                peer_ans = self._call(peer_fn, q)
                if self._answers_match(focal_ans, peer_ans):
                    peer_agree += 1
            agreements.append(peer_agree / len(peer_model_fns))

        return float(np.clip(np.mean(agreements), 0, 1))

    def _call(self, fn: ModelFn, prompt: str) -> str:
        try:
            return str(fn(prompt).get("text", "")).strip().lower()
        except Exception:
            return ""

    def _answers_match(self, a: str, b: str) -> bool:
        a, b = a.strip().lower(), b.strip().lower()
        return bool(a and b and (a == b or a in b or b in a))


# ─────────────────────────────────────────────────────────────
# Contamination Scorer
# ─────────────────────────────────────────────────────────────

class ContaminationScorer:
    """
    Detects potential benchmark contamination via n-gram overlap detection.

    Method: present partial benchmark questions and check if the model
    completes them verbatim (suggesting memorisation rather than reasoning).

    score = 1 - contamination_rate  (higher = less contaminated = better)
    """

    COMPLETION_TEMPLATE = (
        "Complete this sentence exactly as it appears in a well-known "
        "benchmark dataset: '{}...'"
    )

    def score(
        self,
        model_fn: ModelFn,
        benchmark_items: List[Dict],  # [{"prefix": ..., "suffix": ...}, ...]
    ) -> float:
        if not benchmark_items:
            return 1.0

        contaminated = 0
        for item in benchmark_items:
            prefix = item.get("prefix", "")
            suffix = item.get("suffix", "").strip().lower()
            if not suffix:
                continue

            prompt = self.COMPLETION_TEMPLATE.format(prefix)
            result = model_fn(prompt)
            completion = str(result.get("text", "")).strip().lower()

            # Contamination: model completes the suffix almost verbatim
            if self._ngram_overlap(completion, suffix) > 0.7:
                contaminated += 1

        return float(np.clip(1.0 - contaminated / len(benchmark_items), 0, 1))

    def _ngram_overlap(self, pred: str, ref: str, n: int = 3) -> float:
        def _ngrams(text, n):
            tokens = text.split()
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        pred_ng = _ngrams(pred, n)
        ref_ng  = _ngrams(ref,  n)
        if not ref_ng:
            return 0.0
        return len(pred_ng & ref_ng) / len(ref_ng)


# ─────────────────────────────────────────────────────────────
# Helpfulness Scorer
# ─────────────────────────────────────────────────────────────

class HelpfulnessScorer:
    """
    Task completion rate: fraction of prompts where the model produces
    a non-empty, non-refusal response.

    Refusal patterns: "I can't", "I cannot", "I don't know", "N/A", "sorry".
    """

    REFUSAL_PATTERNS = [
        "i can't", "i cannot", "i don't know", "i do not know",
        "n/a", "not applicable", "sorry", "as an ai",
        "i'm unable", "i am unable", "no answer",
    ]

    def score(self, model_fn: ModelFn, prompts: List[str]) -> float:
        if not prompts:
            return 1.0

        completions = 0
        for prompt in prompts:
            try:
                result = model_fn(prompt)
                text = str(result.get("text", "")).strip().lower()
                if text and not any(r in text for r in self.REFUSAL_PATTERNS):
                    completions += 1
            except Exception:
                pass

        return float(completions / len(prompts))


# ─────────────────────────────────────────────────────────────
# LLM Trust Result
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMTrustResult:
    model_name: str
    consistency:   float
    calibration:   float
    agreement:     float
    contamination: float
    helpfulness:   float
    composite:     float
    weights:       Dict[str, float] = field(default_factory=lambda: dict(LLM_WEIGHTS))

    def trust_label(self) -> str:
        if self.composite >= 0.85:
            return "HIGH"
        if self.composite >= 0.70:
            return "MEDIUM"
        if self.composite >= 0.55:
            return "LOW"
        return "VERY_LOW"

    def deployment_decision(self, threshold: float = 0.70) -> str:
        return "DEPLOY" if self.composite >= threshold else "REJECT"

    def to_dict(self) -> Dict:
        return {
            "model_name":    self.model_name,
            "composite":     round(self.composite, 4),
            "trust_label":   self.trust_label(),
            "deployment":    self.deployment_decision(),
            "components": {
                "consistency":   round(self.consistency,   4),
                "calibration":   round(self.calibration,   4),
                "agreement":     round(self.agreement,     4),
                "contamination": round(self.contamination, 4),
                "helpfulness":   round(self.helpfulness,   4),
            },
            "weights": self.weights,
        }


# ─────────────────────────────────────────────────────────────
# LLM Trust Score Engine
# ─────────────────────────────────────────────────────────────

class LLMTrustScore:
    """
    Main entry point for LLM trust evaluation.

    Usage
    -----
    evaluator = LLMTrustScore()
    result = evaluator.evaluate(
        model_fn      = my_llm,
        model_name    = "my-language-model",
        questions     = [...],
        correct_answers = [...],
        peer_model_fns = [llm2, llm3],
        benchmark_items = [{"prefix": ..., "suffix": ...}, ...],
    )
    print(result.to_dict())
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or dict(LLM_WEIGHTS)
        self._consistency   = ConsistencyScorer()
        self._calibration   = CalibrationScorer()
        self._agreement     = AgreementScorer()
        self._contamination = ContaminationScorer()
        self._helpfulness   = HelpfulnessScorer()

    def evaluate(
        self,
        model_fn:        ModelFn,
        model_name:      str = "unknown",
        questions:       Optional[List[str]] = None,
        correct_answers: Optional[List[str]] = None,
        peer_model_fns:  Optional[List[ModelFn]] = None,
        benchmark_items: Optional[List[Dict]] = None,
        all_prompts:     Optional[List[str]] = None,
    ) -> LLMTrustResult:
        """
        Evaluate all trust components for a single LLM.

        Parameters
        ----------
        model_fn:        The focal model callable.
        model_name:      String identifier for reporting.
        questions:       Factual Q&A questions (used for consistency + calibration).
        correct_answers: Ground-truth answers matching `questions`.
        peer_model_fns:  Other models for agreement scoring.
        benchmark_items: [{"prefix": ..., "suffix": ...}] for contamination check.
        all_prompts:     General prompts for helpfulness scoring.
        """
        qs = questions or []
        ans = correct_answers or []

        c_score  = self._consistency.score(model_fn, qs)
        cal_score = self._calibration.score(model_fn, qs, ans) if ans else 0.5
        ag_score  = self._agreement.score(model_fn, peer_model_fns or [], qs)
        ct_score  = self._contamination.score(model_fn, benchmark_items or [])
        h_score   = self._helpfulness.score(model_fn, all_prompts or qs)

        composite = (
            self.weights["consistency"]   * c_score  +
            self.weights["calibration"]   * cal_score +
            self.weights["agreement"]     * ag_score  +
            self.weights["contamination"] * ct_score  +
            self.weights["helpfulness"]   * h_score
        )

        return LLMTrustResult(
            model_name    = model_name,
            consistency   = c_score,
            calibration   = cal_score,
            agreement     = ag_score,
            contamination = ct_score,
            helpfulness   = h_score,
            composite     = float(np.clip(composite, 0, 1)),
            weights       = dict(self.weights),
        )

    def compare_models(
        self,
        results: List[LLMTrustResult],
    ) -> Dict:
        """
        Rank multiple evaluated models by composite trust score.
        """
        ranked = sorted(results, key=lambda r: r.composite, reverse=True)
        return {
            "ranking": [
                {"rank": i + 1, **r.to_dict()}
                for i, r in enumerate(ranked)
            ],
            "best_model":    ranked[0].model_name if ranked else None,
            "best_composite": round(ranked[0].composite, 4) if ranked else None,
        }


# ─────────────────────────────────────────────────────────────
# Mock model factory (for offline smoke-testing)
# ─────────────────────────────────────────────────────────────

def make_mock_llm(
    accuracy: float = 0.80,
    confidence_bias: float = 0.0,
    consistency_rate: float = 0.85,
    refusal_rate: float = 0.05,
    seed: int = 0,
) -> ModelFn:
    """
    Returns a mock LLM callable for testing.

    Parameters
    ----------
    accuracy:         Fraction of factual questions answered correctly.
    confidence_bias:  Shift applied to confidence (positive → overconfident).
    consistency_rate: Fraction of paraphrase answers that match canonical.
    refusal_rate:     Fraction of prompts that trigger a refusal response.
    """
    rng = np.random.default_rng(seed)

    _ANSWER_POOL = ["paris", "london", "42", "newton", "darwin", "oxygen",
                    "1945", "python", "1.618", "einstein"]

    def _model(prompt: str) -> Dict:
        # Refusal check
        if rng.uniform() < refusal_rate:
            return {"text": "I cannot answer that question.", "confidence": 0.1}

        # Template-invariant hash: use only alphabetic words ≥5 chars
        # (strips template prefixes like "Please answer:", "Question:", etc.)
        import re
        core_words = sorted(set(w for w in re.findall(r'[a-z]{5,}', prompt.lower())
                                if w not in {"please", "answer", "question", "complete",
                                             "sentence", "scale", "reply", "number",
                                             "confident", "correct"}))
        core_key = " ".join(core_words[:4])  # first 4 significant words
        p_hash = hash(core_key) % 1000
        base_answer = _ANSWER_POOL[p_hash % len(_ANSWER_POOL)]

        if rng.uniform() < (1 - consistency_rate):
            alt_idx = (p_hash + 1) % len(_ANSWER_POOL)
            answer = _ANSWER_POOL[alt_idx]
        else:
            answer = base_answer

        # Accuracy: sometimes return wrong answer
        if rng.uniform() > accuracy:
            answer = _ANSWER_POOL[(p_hash + 3) % len(_ANSWER_POOL)]

        conf = float(np.clip(accuracy + confidence_bias + rng.normal(0, 0.05), 0.01, 0.99))
        return {"text": answer, "confidence": conf}

    return _model
