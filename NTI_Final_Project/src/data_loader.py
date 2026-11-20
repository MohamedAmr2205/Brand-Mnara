"""One place that knows how to read the project data (so every notebook agrees)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE_COLS = {
    "Students": ["join_date", "leave_date"],
    "Groups": ["start_date"],
    "Sessions": ["date"],
    "Records": ["date"],
    "Payments": ["due_date", "paid_date"],
}


def load_clean(folder: str | Path = ROOT / "data" / "clean") -> dict[str, pd.DataFrame]:
    """Cleaned tables. Phones/ids are read as text so leading zeros survive."""
    out = {}
    for name, dates in DATE_COLS.items():
        df = pd.read_csv(Path(folder) / f"{name}_clean.csv", dtype={"phone": str, "parent_phone": str}, parse_dates=dates)
        out[name] = df
    return out


def load_ground_truth(folder: str | Path = ROOT / "data" / "ground_truth") -> pd.DataFrame:
    """Hidden truth -- for EVALUATION only, never a feature."""
    return pd.read_csv(Path(folder) / "ground_truth_students.csv")
