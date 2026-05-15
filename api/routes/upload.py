"""
EMMDS Upload Route
POST /api/upload-dataset
Accepts CSV, Excel (.xlsx/.xls), JSON, and Parquet files.
"""

import io
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas.request_response import UploadResponse
from api.state import state

router = APIRouter()

_SUPPORTED = {
    ".csv":     "CSV",
    ".tsv":     "TSV",
    ".xlsx":    "Excel",
    ".xls":     "Excel (legacy)",
    ".json":    "JSON",
    ".parquet": "Parquet",
    ".feather": "Feather",
}


def _parse_file(filename: str, contents: bytes) -> pd.DataFrame:
    """Parse file bytes into a DataFrame based on extension."""
    ext = Path(filename).suffix.lower()

    if ext not in _SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(_SUPPORTED.keys())}"
            ),
        )

    try:
        if ext == ".csv":
            return pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if ext == ".tsv":
            return pd.read_csv(io.StringIO(contents.decode("utf-8")), sep="\t")
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(contents))
        if ext == ".json":
            # Try records orientation first, then default
            try:
                return pd.read_json(io.StringIO(contents.decode("utf-8")))
            except Exception:
                return pd.DataFrame(
                    pd.read_json(io.StringIO(contents.decode("utf-8")), orient="records")
                )
        if ext == ".parquet":
            return pd.read_parquet(io.BytesIO(contents))
        if ext == ".feather":
            return pd.read_feather(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse {_SUPPORTED[ext]} file: {e}",
        )


@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset in any supported format.
    Supported: CSV, TSV, Excel (.xlsx/.xls), JSON (records), Parquet, Feather.
    The dataframe is stored in memory for subsequent analyze/train calls.
    """
    contents = await file.read()
    df = _parse_file(file.filename, contents)

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded file produced an empty DataFrame.")

    if len(df.columns) < 2:
        raise HTTPException(
            status_code=422,
            detail="Dataset must have at least 2 columns (features + target).",
        )

    state.clear()
    state.df = df
    state.filename = file.filename

    ext = Path(file.filename).suffix.lower()
    return UploadResponse(
        status="success",
        filename=file.filename,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        column_names=list(df.columns),
        message=(
            f"{_SUPPORTED.get(ext, 'File')} '{file.filename}' uploaded successfully "
            f"({df.shape[0]:,} rows × {df.shape[1]} columns)."
        ),
    )


@router.get("/dataset-info")
def dataset_info():
    """Return basic info about the currently loaded dataset."""
    if not state.has_data():
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload first.")
    df = state.df
    # Column type summary
    type_summary = {
        "numeric": int((df.dtypes != "object").sum()),
        "categorical": int((df.dtypes == "object").sum()),
        "datetime": int(sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)),
    }
    return {
        "filename": state.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "type_summary": type_summary,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()},
        "sample": df.head(5).to_dict(orient="records"),
    }
