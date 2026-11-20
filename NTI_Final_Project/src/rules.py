"""Business rules shared by every notebook, the Streamlit app and the bots.

Rules live here once; nothing else should redefine them.
"""
from __future__ import annotations

import pandas as pd

# --- attendance -----------------------------------------------------------
# Team decision (2026-11-20): a late student counts as attended.
# The live dashboard used to count only 'present'; lateness is kept as its own
# signal (see `is_late`) so it can still be used as a feature.
ATTENDED = ("present", "late")
NOT_ATTENDED = ("absent", "excused")
ATTENDANCE_VALUES = ATTENDED + NOT_ATTENDED


def attended(att: pd.Series) -> pd.Series:
    """True when the student showed up (present or late)."""
    return att.isin(ATTENDED)


def is_late(att: pd.Series) -> pd.Series:
    return att.eq("late")


# --- risk flags ------------------------------------------------------------
# "At risk" (from the dataset README / live dashboard):
#   attendance rate < 80%  AND  average exam percentage < 75%
AT_RISK_ATTENDANCE_MAX = 0.80
AT_RISK_AVG_MAX = 0.75

# "Needs follow-up": threshold still OPEN (checklist: tune it so it does not
# flag a quarter of the students; the raw dataset rule flags ~26%).
# Set in the EDA phase once agreed with the team.
NEEDS_ATTENTION = None


def at_risk(attendance_rate: pd.Series, avg_percentage: pd.Series) -> pd.Series:
    """Boolean flag. `avg_percentage` is a fraction (0-1)."""
    return (attendance_rate < AT_RISK_ATTENDANCE_MAX) & (avg_percentage < AT_RISK_AVG_MAX)


# --- payments ---------------------------------------------------------------
PAYMENT_STATUSES = ("paid", "paid_late", "partial", "unpaid", "exempt")
UNPAID_LIKE = ("unpaid", "partial")
