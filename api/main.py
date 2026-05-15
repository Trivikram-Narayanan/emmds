"""
EMMDS FastAPI Backend
Main entry point. Mounts all route modules.
"""

import sys
from pathlib import Path

# Ensure project root is on path so `src.` imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import upload, analyze, train, results

app = FastAPI(
    title="EMMDS — Ensemble Multi-Model Decision System",
    description="AutoML + Explainability + Trust Scoring API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (allow Streamlit on localhost) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route modules ────────────────────────────────────────────────────
app.include_router(upload.router,  prefix="/api", tags=["Upload"])
app.include_router(analyze.router, prefix="/api", tags=["Analyze"])
app.include_router(train.router,   prefix="/api", tags=["Train"])
app.include_router(results.router, prefix="/api", tags=["Results"])


@app.get("/", tags=["Health"])
def root():
    return {
        "system": "EMMDS",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
