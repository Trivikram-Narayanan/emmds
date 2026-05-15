# EMMDS Installation Guide

## Quick start (full system)

```bash
# 1. Clone / unzip the project
cd emmds

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Install all required dependencies
pip install -r requirements.txt

# 4. Run the demo to verify everything works
python run.py --test
```

## Optional packages (install for enhanced functionality)

```bash
# XGBoost + LightGBM (adds 2 more models to the registry)
pip install xgboost lightgbm

# OpenML (enables 80+ real dataset download for experiments)
pip install openml

# CTGAN (enables generative data augmentation for Phase 4)
pip install ctgan

# Gemini API (enables NL explanation generation for Phase 3)
pip install google-generativeai
```

## Launch the full system

```bash
# Streamlit UI only
python run.py --ui
# Opens at http://localhost:8501

# FastAPI backend only  
python run.py --api
# Docs at http://localhost:8000/docs

# Both simultaneously
python run.py --all

# Docker (builds everything automatically)
docker-compose up --build
# API: http://localhost:8000/docs
# UI:  http://localhost:8501
```

## Run research experiments

```bash
# Core experiments (Claims A-D)
python src/research/experiments.py

# Direction 1: Meta-weight learning (21+ datasets)
python src/research/direction1_meta_weights.py

# Direction 2: Trust score calibration
python src/research/direction2_calibration.py

# Direction 3: Distribution shift
python src/research/direction3_shift.py

# Phase 2: DQL deployment agent
python src/rl/dql_deployment_agent.py

# Phase 3: Explanation agent
python src/agentic/explanation_agent.py

# Phase 4: CTGAN augmentation
python src/genai/ctgan_augmentation.py
```

## Verify all modules load correctly

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.pipeline.pipeline import EMPipeline
from src.decision.trust_score import TrustScoreEngine
from src.rl.dql_deployment_agent import DQLAgent
from src.agentic.explanation_agent import TrustExplanationAgent
from src.genai.ctgan_augmentation import TabularAugmenter
print('All core modules OK')
"
```

## Environment requirements

- Python 3.11+
- 4GB RAM minimum (8GB recommended for full experiments)
- No GPU required (all models are CPU-based)
- Network access required for: OpenML datasets, Gemini API (optional)