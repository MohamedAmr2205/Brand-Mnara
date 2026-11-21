"""Next-month grade forecast per student, for the Streamlit app.

Same student x month table and features as `notebooks/models.ipynb` (everything is
computed from the months BEFORE the one being predicted, so there is no leakage).
The app fits a Ridge model once at start-up and predicts the coming month for every
student, then reports its error on held-out students next to the "last month" baseline,
so the teacher can see how much to trust the number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import HW_MAP
from src import rules

STATIC_CAT = ["gender", "school_type", "track", "group_id"]
NUM = ["month", "last_month_avg", "prev_all_avg", "prev2_month_avg", "trend", "prev_std",
       "n_prev_exams", "prior_attendance", "last_attendance", "prior_hw", "last_hw"]
TARGET = "pct_mean"
SEED = 42


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _monthly(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per student per month, plus one empty 'next month' row per student."""
    rec = clean["Records"].dropna(subset=["student_id", "session_id"]).copy()
    rec["percentage"] = _num(rec["percentage"])
    ses = clean["Sessions"][["session_id", "group_id", "month"]].copy()
    ses["month"] = _num(ses["month"])
    df = rec[["student_id", "session_id", "attendance", "homework", "percentage"]].merge(ses, on="session_id", how="left")
    df = df.dropna(subset=["month"])
    df["hw_score"] = df["homework"].map(HW_MAP)
    df["attended"] = rules.attended(df["attendance"]).astype(float)

    mm = (df.groupby(["student_id", "month"])
            .agg(pct_mean=("percentage", "mean"), pct_sum=("percentage", "sum"), n_scored=("percentage", "count"),
                 attend_rate=("attended", "mean"), hw_mean=("hw_score", "mean"), group_id=("group_id", "last"))
            .reset_index())

    nxt = mm.sort_values("month").groupby("student_id").tail(1)[["student_id", "month", "group_id"]].copy()
    nxt["month"] = nxt["month"] + 1
    nxt["is_next"] = True
    mm["is_next"] = False
    mm = pd.concat([mm, nxt], ignore_index=True)
    mm[["pct_sum", "n_scored"]] = mm[["pct_sum", "n_scored"]].fillna(0)
    mm = mm.sort_values(["student_id", "month"]).reset_index(drop=True)

    mm["_cs"] = mm.groupby("student_id")["pct_sum"].cumsum()
    mm["_cn"] = mm.groupby("student_id")["n_scored"].cumsum()
    g = mm.groupby("student_id")
    mm["prev_all_avg"] = g["_cs"].shift(1) / g["_cn"].shift(1).replace(0, np.nan)
    mm["n_prev_exams"] = g["_cn"].shift(1)
    mm["last_month_avg"] = g["pct_mean"].transform(lambda s: s.shift(1).ffill())
    mm["prev2_month_avg"] = g["pct_mean"].shift(2)
    mm["trend"] = g["pct_mean"].shift(1) - g["pct_mean"].shift(2)
    mm["prev_std"] = g["pct_mean"].transform(lambda s: s.shift(1).expanding().std())
    mm["prior_attendance"] = g["attend_rate"].transform(lambda s: s.shift(1).expanding().mean())
    mm["last_attendance"] = g["attend_rate"].shift(1)
    mm["prior_hw"] = g["hw_mean"].transform(lambda s: s.shift(1).expanding().mean())
    mm["last_hw"] = g["hw_mean"].shift(1)

    stu, grp = clean["Students"], clean["Groups"]
    s_cols = ["student_id"] + [c for c in STATIC_CAT if c in stu.columns and c != "group_id"]
    g_cols = ["group_id"] + [c for c in STATIC_CAT if c in grp.columns and c not in s_cols and c != "group_id"]
    mm = mm.merge(stu[s_cols], on="student_id", how="left").merge(grp[g_cols], on="group_id", how="left")
    cat = [c for c in STATIC_CAT if c in mm.columns]
    for c in cat:
        mm[c] = mm[c].astype("object").where(mm[c].notna(), np.nan)
    return mm


def _pipeline(cat: list[str]) -> Pipeline:
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cats = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", num, NUM), ("cat", cats, cat)])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=10.0))])


def next_month_forecast(clean: dict[str, pd.DataFrame], test_share: float = 0.2) -> tuple[pd.DataFrame, dict]:
    """Predicted average for each student's coming month, and the model's honest error.

    Returns (per-student table, metrics). Percentages are fractions (0-1); the
    metrics are in points out of 100.
    """
    mm = _monthly(clean)
    cat = [c for c in STATIC_CAT if c in mm.columns]
    feats = NUM + cat
    hist = mm[~mm["is_next"] & mm[TARGET].notna() & mm["prev_all_avg"].notna()]

    # held-out students, to say how far off the forecast usually is
    ids = hist["student_id"].drop_duplicates().sample(frac=1.0, random_state=SEED)
    test_ids = set(ids.iloc[: int(len(ids) * test_share)])
    tr, te = hist[~hist["student_id"].isin(test_ids)], hist[hist["student_id"].isin(test_ids)]
    model = _pipeline(cat).fit(tr[feats], tr[TARGET])
    pred_te = np.clip(model.predict(te[feats]), 0, 1)
    base_te = te["last_month_avg"].fillna(te["prev_all_avg"])
    metrics = {
        "mae": float(np.mean(np.abs(pred_te - te[TARGET])) * 100),
        "baseline_mae": float(np.mean(np.abs(base_te - te[TARGET])) * 100),
        "n_test_students": len(test_ids),
    }

    # refit on everyone, then predict the coming month
    model = _pipeline(cat).fit(hist[feats], hist[TARGET])
    nxt = mm[mm["is_next"] & mm["prev_all_avg"].notna()].copy()
    nxt["predicted"] = np.clip(model.predict(nxt[feats]), 0, 1)
    nxt["change"] = nxt["predicted"] - nxt["last_month_avg"]
    out = nxt.set_index("student_id")[["month", "last_month_avg", "prev_all_avg", "predicted", "change"]]
    return out, metrics
