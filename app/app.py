# app/app.py
import os
import json
import joblib
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model" / "model.joblib"
META_PATH = HERE / "model" / "meta.json"

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first.")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta not found at {META_PATH}.")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta

model, meta = load_artifacts()

# Rehydrate feature lists
CATEGORICAL_COLS = meta["categorical_cols"]
NUMERIC_COLS = meta["numeric_cols"]
TEXT_COL = meta["text_col"]
ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS + [TEXT_COL]

def make_dataframe(payload):
    """
    Accepts dict (single) or list[dict] (batch).
    Returns a pandas DataFrame with required columns, filling defaults.
    """
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Payload must be an object or an array of objects.")

    df = pd.DataFrame(records)

    # Ensure all required columns exist
    for c in CATEGORICAL_COLS:
        if c not in df.columns: df[c] = "Unknown"
        df[c] = df[c].astype(str).fillna("Unknown")

    for c in NUMERIC_COLS:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if TEXT_COL not in df.columns:
        df[TEXT_COL] = "No essay provided"
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("No essay provided")

    # Order columns consistently
    return df[ALL_COLS]

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_PATH.name,
        "version": meta.get("version", "unknown")
    })

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        X = make_dataframe(payload)

        # predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            # Fall back to decision_function if needed
            scores = model.decision_function(X)
            probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        preds = (probs >= 0.5).astype(int)

        # shape output like input
        if isinstance(payload, list):
            out = [{"approved": int(p), "probability": float(pr)} for p, pr in zip(preds, probs)]
        else:
            out = {"approved": int(preds[0]), "probability": float(probs[0])}

        return jsonify({"ok": True, "result": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
