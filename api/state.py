"""
EMMDS API State
In-memory session state shared across route modules.
In production, replace with Redis or a database.
"""

import pandas as pd
from typing import Optional

# ── Session store (single-user, in-memory) ───────────────────────────

class SessionState:
    df: Optional[pd.DataFrame] = None
    filename: Optional[str] = None
    target_col: Optional[str] = None
    task: Optional[str] = None
    pipeline_result: Optional[dict] = None

    @classmethod
    def clear(cls):
        cls.df = None
        cls.filename = None
        cls.target_col = None
        cls.task = None
        cls.pipeline_result = None

    @classmethod
    def has_data(cls) -> bool:
        return cls.df is not None

    @classmethod
    def has_result(cls) -> bool:
        return cls.pipeline_result is not None


state = SessionState()
