# EMMDS Agent Architecture

EMMDS exposes an agentic orchestration layer (`EMMDSAgent`) that wraps the full nine-stage pipeline and makes adaptive decisions based on intermediate results.

## EMMDSAgent

**Location:** `src/rl/emmds_agent.py`

The agent observes dataset meta-features and pipeline outputs and applies the following adaptive rules:

| Condition | Action |
|-----------|--------|
| imbalance_ratio > 3.0 | Increase calibration evaluation priority |
| noise_estimate > 0.10 OR n < 300 | Increase CV folds from 5 to 10 |
| data_quality < 0.70 | Flag all results as provisional |
| trust_ranking ≠ accuracy_ranking | Trigger explanation generation |

Every decision is logged with a natural language justification for auditability.

## TrustExplanationAgent

**Location:** `src/agentic/explanation_agent.py`

Generates deployment-oriented natural language explanations for each model selection decision:

- **Selection narrative**: Why this model was chosen over alternatives
- **Trust breakdown narrative**: What each trust component signals
- **Deployment advice**: What to monitor post-deployment
- **Counterfactual**: What would have changed if a different model was selected
- **Risk warnings**: Specific red flags from the trust breakdown
- **Confidence statement**: How certain the system is in its selection

Backend: Gemini 1.5 Flash (requires `GOOGLE_API_KEY`). Falls back to template-based generation when the API key is unavailable.

## DQL Deployment Agent

**Location:** `src/rl/dql_deployment_agent.py`

A Deep Q-Learning agent that learns optimal retraining timing policies under distribution drift. State space includes KS-statistic, PSI, observed F1, and training-time trust score. Evaluated against four baselines: always-continue, periodic retrain, threshold-F1, and random.
