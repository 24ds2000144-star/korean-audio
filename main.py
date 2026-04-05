import base64
import io
import json
import os
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import speech_recognition as sr
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def parse_dataset_from_text(text: str) -> pd.DataFrame:
    """Try to parse a CSV-like dataset from transcribed text."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rows = []
    for line in lines:
        # replace common speech-to-text artifacts
        line = line.replace("，", ",").replace("、", ",").replace(" ", ",")
        parts = [p.strip() for p in line.split(",") if p.strip()]
        rows.append(parts)

    if not rows:
        return pd.DataFrame()

    # Try to use first row as header if it looks non-numeric
    def is_numeric(val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    if rows and not all(is_numeric(v) for v in rows[0]):
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.DataFrame(rows)

    # Convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def compute_stats(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def safe_mode(series):
        if series.empty:
            return None
        m = series.mode()
        return m.iloc[0] if not m.empty else None

    def allowed_values(series):
        unique = series.dropna().unique().tolist()
        return sorted(unique) if len(unique) <= 20 else None

    def value_range(series):
        if pd.api.types.is_numeric_dtype(series):
            return {"min": float(series.min()), "max": float(series.max())}
        unique = series.dropna().unique().tolist()
        return sorted(unique) if len(unique) <= 20 else None

    mean = {}
    std = {}
    variance = {}
    min_ = {}
    max_ = {}
    median = {}
    mode = {}
    range_ = {}
    allowed = {}
    val_range = {}

    for col in numeric_cols:
        s = df[col].dropna()
        mean[col] = float(s.mean()) if not s.empty else None
        std[col] = float(s.std()) if not s.empty else None
        variance[col] = float(s.var()) if not s.empty else None
        min_[col] = float(s.min()) if not s.empty else None
        max_[col] = float(s.max()) if not s.empty else None
        median[col] = float(s.median()) if not s.empty else None
        mode[col] = float(safe_mode(s)) if safe_mode(s) is not None else None
        range_[col] = float(s.max() - s.min()) if not s.empty else None
        allowed[col] = None
        val_range[col] = value_range(df[col])

    for col in categorical_cols:
        s = df[col].dropna()
        mean[col] = None
        std[col] = None
        variance[col] = None
        min_[col] = None
        max_[col] = None
        median[col] = None
        mode[col] = safe_mode(s)
        range_[col] = None
        allowed[col] = allowed_values(s)
        val_range[col] = value_range(df[col])

    # Correlation matrix (numeric only)
    correlation = []
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()
        for i, row_col in enumerate(corr_df.index):
            for j, col_col in enumerate(corr_df.columns):
                correlation.append({
                    "col1": row_col,
                    "col2": col_col,
                    "value": float(corr_df.iloc[i, j])
                })

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": min_,
        "max": max_,
        "median": median,
        "mode": mode,
        "range": range_,
        "allowed_values": allowed,
        "value_range": val_range,
        "correlation": correlation,
    }


@app.post("/analyze")
async def analyze_audio(request: AudioRequest):
    audio_bytes = base64.b64decode(request.audio_base64)

    # Save to temp file and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
        # Try Google STT (free, no key needed)
        text = recognizer.recognize_google(audio_data, language="ko-KR")
    except Exception as e:
        text = ""
    finally:
        os.unlink(tmp_path)

    df = parse_dataset_from_text(text)

    if df.empty:
        return {
            "rows": 0,
            "columns": [],
            "mean": {},
            "std": {},
            "variance": {},
            "min": {},
            "max": {},
            "median": {},
            "mode": {},
            "range": {},
            "allowed_values": {},
            "value_range": {},
            "correlation": [],
        }

    return compute_stats(df)


@app.get("/")
def root():
    return {"status": "ok"}