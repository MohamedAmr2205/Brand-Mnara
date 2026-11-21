"""Student-level features, built once and shared by the notebook and the Streamlit app.

Input : the cleaned tables (`clean` dict from `src.data_loader.load_clean`, or any dict
        with the same `Records`, `Sessions` and `Payments` tables).
Output: one row per student, indexed by `student_id`.

Nothing here should be re-implemented in a notebook or in the app: if the feature
definition changes, it changes here and everything stays consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import rules

HW_MAP = {"ممتاز": 5, "جيد جدا": 4, "جيد": 3, "مقبول": 2, "ضعيف": 1, "لم يسلم": 0}

# features used for the final segmentation (academic only)
ACADEMIC = ["att_rate", "avg_pct", "pct_slope", "hw_avg", "std_pct"]

# a slope needs grades in at least this many different months
MIN_MONTHS_FOR_SLOPE = 3


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _records(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attendance records joined with their session (month, session number)."""
    rec = clean["Records"].dropna(subset=["student_id", "session_id"]).copy()
    for c in ["exam_score", "max_exam_score", "percentage"]:
        rec[c] = _num(rec[c])

    ses = clean["Sessions"][["session_id", "session_number", "month"]].copy()
    ses["session_number"] = _num(ses["session_number"])
    ses["month"] = _num(ses["month"])
    rec = rec.merge(ses, on="session_id", how="left")

    rec["attended"] = rules.attended(rec["attendance"]).astype(int)  # present or late
    rec["is_late"] = rules.is_late(rec["attendance"]).astype(int)
    rec["excused"] = (rec["attendance"] == "excused").astype(int)
    rec["hw_score"] = rec["homework"].map(HW_MAP)
    return rec


def _monthly_slope(months: pd.Series, pct: pd.Series) -> float:
    """Slope of the monthly mean percentage (change per month). NaN if < 3 months."""
    if len(months) < MIN_MONTHS_FOR_SLOPE:
        return np.nan
    return float(np.polyfit(months.values, pct.values, 1)[0])


def attendance_features(rec: pd.DataFrame) -> pd.DataFrame:
    g = rec.groupby("student_id")
    return pd.DataFrame({
        "n_sessions": g.size(),
        "att_rate": g["attended"].mean(),                          # present or late, of all sessions
        "excused_rate": g["excused"].mean(),                       # excused, of all sessions
        "late_rate": g["is_late"].sum() / g["attended"].sum(),     # late, of attended sessions only
    })


def performance_features(rec: pd.DataFrame) -> pd.DataFrame:
    ex = rec.dropna(subset=["percentage"])
    by_student = ex.groupby("student_id")["percentage"]

    monthly = ex.groupby(["student_id", "month"])["percentage"].mean().reset_index()
    slope = {sid: _monthly_slope(d["month"], d["percentage"])
             for sid, d in monthly.groupby("student_id")}

    perf = pd.DataFrame({
        "avg_pct": by_student.mean(),
        "std_pct": by_student.std(),
        "pct_slope": pd.Series(slope),
    })

    # homework: attended sessions only (absent rows are filled with "لم يسلم" = 0 by the cleaning,
    # counting them would count the absence twice)
    perf["hw_avg"] = rec[rec["attended"] == 1].groupby("student_id")["hw_score"].mean()
    return perf


def payment_features(clean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pay = clean["Payments"].dropna(subset=["student_id"]).copy()
    pay["due_date"] = pd.to_datetime(pay["due_date"], errors="coerce")
    pay["paid_date"] = pd.to_datetime(pay["paid_date"], errors="coerce")
    pay["delay_days"] = (pay["paid_date"] - pay["due_date"]).dt.days   # NaN when never paid

    gp = pay.groupby("student_id")
    return pd.DataFrame({
        "n_payments": gp.size(),
        "unpaid_rate": gp["status"].apply(lambda s: (s == "unpaid").mean()),
        "partial_rate": gp["status"].apply(lambda s: (s == "partial").mean()),
        "paid_late_rate": gp["status"].apply(lambda s: (s == "paid_late").mean()),
        "exempt_rate": gp["status"].apply(lambda s: (s == "exempt").mean()),
        "avg_delay_days": gp["delay_days"].mean(),   # only over payments that were actually paid
    })


def build_features(clean: dict[str, pd.DataFrame], fill_slope: bool = True) -> pd.DataFrame:
    """One row per student with attendance, performance and payment features.

    `insufficient_data` is True for students whose grades cover fewer than 3 months
    (no slope can be computed). With `fill_slope=True` their slope is set to 0 ("no trend"),
    which is what the notebook used when fitting the model; the app should still show
    "insufficient data" for them instead of trusting the segment.
    """
    rec = _records(clean)
    feat = attendance_features(rec).join(performance_features(rec)).join(payment_features(clean))

    feat["insufficient_data"] = feat["pct_slope"].isna() | feat["avg_pct"].isna()
    if fill_slope:
        feat["pct_slope"] = feat["pct_slope"].fillna(0)
    return feat