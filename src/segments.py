"""Assign students to segments and attach the suggested teacher action.

Used by the Streamlit app (and any notebook) so everyone gets the same answer:

    from src.data_loader import load_clean
    from src.features import build_features
    from src.segments import load_bundle, predict_segment

    bundle = load_bundle()
    result = predict_segment(build_features(load_clean()), bundle)

The model is produced by the last cell of `notebooks/03_clustering.ipynb`
(`models/segmentation.joblib`).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.data_loader import ROOT

MODEL_PATH = ROOT / "models" / "segmentation.joblib"
INSUFFICIENT = "بيانات غير كافية"

# Order matters: 1 = look at first. Names come from the model (English, measured behaviour).
PRIORITY = {
    "Sharply Declining": 1,
    "Struggling": 2,
    "Rising": 3,
    "Solid Regulars": 4,
    "Top Performers": 5,
}

# Suggestions for the teacher. They are starting points, not tested interventions.
ACTIONS = {
    "Sharply Declining": "درجاته بتنزل بسرعة لكن لسه بيحضر. اتواصل معاه ومع ولي أمره بسرعة واسأل عن السبب.",
    "Struggling": "ضعيف في الدرجات والواجب والحضور. حصص تقوية ومتابعة أسبوعية للواجب، وابدأ بالأقل حضوراً.",
    "Rising": "درجاته بتطلع. شجّعه وخلّي الاتجاه يستمر.",
    "Solid Regulars": "مستواه ثابت وفوق المتوسط. متابعة عادية وسؤال أصعب من وقت لتاني.",
    "Top Performers": "متفوق ومستقر. مسائل تحدٍّ، أو خلّيه يشرح لزمايله.",
    INSUFFICIENT: "درجاته مش كفاية لحساب الاتجاه (أقل من 3 شهور). استنى لحد ما تتجمع بيانات أكتر.",
}


def load_bundle(path: str | Path = MODEL_PATH) -> dict:
    """Load the saved model. Warns if scikit-learn differs from the version it was saved with."""
    bundle = joblib.load(path)
    saved = bundle.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(
            f"model saved with scikit-learn {saved}, running {sklearn.__version__}: "
            "pin the same version in requirements.txt",
            RuntimeWarning,
            stacklevel=2,
        )
    return bundle


def predict_segment(feat: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Segment + suggested action per student.

    `feat` is the output of `src.features.build_features` (one row per student).
    Students flagged `insufficient_data`, or with a missing model feature, get
    "بيانات غير كافية" instead of a segment, because their numbers are not reliable.
    """
    cols = bundle["features"]
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise KeyError(f"feature columns missing: {missing}")

    flagged = feat["insufficient_data"] if "insufficient_data" in feat.columns else pd.Series(False, index=feat.index)
    insufficient = flagged.astype(bool) | feat[cols].isna().any(axis=1)

    segment = pd.Series(INSUFFICIENT, index=feat.index, dtype="object")
    ok = ~insufficient
    if ok.any():
        ids = bundle["pipeline"].predict(feat.loc[ok, cols])
        segment[ok] = [bundle["cluster_to_segment"][int(c)] for c in ids]

    out = pd.DataFrame({
        "segment": segment,
        "priority": segment.map(PRIORITY).fillna(99).astype(int),
        "action": segment.map(ACTIONS),
        "insufficient_data": insufficient,
    })
    return out.sort_values(["priority"], kind="stable")