import base64
import os
import tempfile

import numpy as np
import pandas as pd
import librosa
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def compute_stats(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def safe_mode(series):
        m = series.mode()
        return m.iloc[0] if not m.empty else None

    def value_range(series):
        if pd.api.types.is_numeric_dtype(series):
            return {"min": float(series.min()), "max": float(series.max())}
        unique = series.dropna().unique().tolist()
        return sorted([str(v) for v in unique]) if len(unique) <= 20 else None

    def allowed_values(series):
        unique = series.dropna().unique().tolist()
        return sorted([str(v) for v in unique]) if len(unique) <= 20 else None

    mean, std, variance, min_, max_, median, mode, range_, allowed, val_range = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

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


def extract_features(y: np.ndarray, sr: int) -> pd.DataFrame:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    frames = mfccs.shape[1]
    data = {}
    for i in range(13):
        data[f"mfcc_{i+1}"] = mfccs[i][:frames]
    for i in range(12):
        data[f"chroma_{i+1}"] = chroma[i][:frames]
    data["spectral_centroid"] = spec_centroid[0][:frames]
    data["spectral_bandwidth"] = spec_bandwidth[0][:frames]
    data["spectral_rolloff"] = spec_rolloff[0][:frames]
    data["rms"] = rms[0][:frames]
    data["zcr"] = zcr[0][:frames]

    return pd.DataFrame(data)


async def process_audio(audio_id: str, audio_base64: str) -> dict:
    audio_bytes = base64.b64decode(audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
    except Exception:
        return {
            "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
            "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
            "allowed_values": {}, "value_range": {}, "correlation": []
        }
    finally:
        os.unlink(tmp_path)

    df = extract_features(y, sample_rate)
    return compute_stats(df)


@app.post("/analyze")
async def analyze_post(request: AudioRequest):
    return await process_audio(request.audio_id, request.audio_base64)


@app.get("/analyze")
async def analyze_get():
    return {"status": "ok", "message": "Send POST request with audio_id and audio_base64"}


@app.post("/")
async def root_post(request: Request):
    body = await request.json()
    return await process_audio(body.get("audio_id", ""), body.get("audio_base64", ""))


@app.get("/")
def root_get():
    return {"status": "ok"}