"""Compare cleaned tables with the clean workbook and the dirty_issue_log.

The clean workbook and the log are NEVER used to clean -- only to score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = {"Students": "student_id", "Groups": "group_id", "Sessions": "session_id", "Records": "record_id", "Payments": "payment_id"}


def load_reference(xlsx_path) -> dict[str, pd.DataFrame]:
    """Clean workbook as the scoring reference (phones read as text so leading zeros survive)."""
    out = {}
    for n in KEYS:
        dtype = {"phone": str, "parent_phone": str} if n == "Students" else None
        out[n] = pd.read_excel(xlsx_path, sheet_name=n, dtype=dtype)
    return out


def _norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp,)):
        return v.normalize().isoformat()
    if hasattr(v, "hour") and not hasattr(v, "year"):  # datetime.time
        return v.strftime("%H:%M")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return round(float(v), 3)
    return str(v).strip()


def _col_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series([_norm(x) == _norm(y) for x, y in zip(a, b)], index=a.index)


def compare_table(name: str, cleaned: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    k = KEYS[name]
    c = cleaned.set_index(k)
    o = oracle.set_index(k)
    common = o.index.intersection(c.index)
    rows = []
    for col in o.columns:
        if col not in c.columns:
            continue
        eq = _col_equal(c.loc[common, col], o.loc[common, col])
        rows.append({"table": name, "column": col, "match_pct": round(100 * eq.mean(), 2), "mismatches": int((~eq).sum())})
    return pd.DataFrame(rows)


def table_report(cleaned: dict[str, pd.DataFrame], oracle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = [compare_table(n, cleaned[n], oracle[n]) for n in KEYS]
    shapes = pd.DataFrame(
        [{"table": n, "rows_clean": len(oracle[n]), "rows_cleaned": len(cleaned[n]), "same_rows": len(oracle[n]) == len(cleaned[n])} for n in KEYS]
    )
    return pd.concat(out, ignore_index=True), shapes


def issue_report(cleaned: dict[str, pd.DataFrame], oracle: dict[str, pd.DataFrame], log: pd.DataFrame, dirty: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of logged issues that are fixed. A fixed cell == the value in the clean workbook."""
    res = []
    for (t, issue), g in log.groupby(["table", "issue"]):
        k = KEYS[t]
        if issue == "duplicate_row":
            dup_after = cleaned[t][k].duplicated().sum()
            dup_before = dirty[t][k].duplicated().sum()
            res.append({"table": t, "issue": issue, "n": len(g), "fixed": int(dup_before - dup_after)})
            continue
        c = cleaned[t].set_index(k)
        o = oracle[t].set_index(k)
        fixed = 0
        for key, col in zip(g["key"], g["column"]):
            if key in c.index and key in o.index and col in c.columns:
                cv = c.loc[key, col]
                ov = o.loc[key, col]
                if isinstance(cv, pd.Series):
                    cv = cv.iloc[0]
                fixed += int(_norm(cv) == _norm(ov))
        res.append({"table": t, "issue": issue, "n": len(g), "fixed": fixed})
    df = pd.DataFrame(res)
    df["fixed_pct"] = (100 * df["fixed"] / df["n"]).round(1)
    return df
