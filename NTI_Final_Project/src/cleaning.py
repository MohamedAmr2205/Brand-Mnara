"""Cleaning pipeline for the dirty Students-Management dataset.

Principles
----------
* Every fix is derived from the data itself (other tables, neighbouring rows,
  deterministic id patterns) -- never copied from `dirty_issue_log.csv` or the
  clean workbook. Those two are used only to *evaluate* the result.
* Values that cannot be recovered honestly (a missing phone number, a missing
  student name, a missing exam note) stay NaN instead of being invented.
* Every decision is recorded in `DECISIONS` and printed in the cleaning notebook.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import rules

TABLES = ["Students", "Groups", "Sessions", "Records", "Payments"]
DAY_NAMES = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]  # weekday() 0..6

DECISIONS: list[tuple[str, str]] = []


def _log(table: str, text: str) -> None:
    DECISIONS.append((table, text))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _blank_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA).astype(object)
        df[c] = df[c].where(df[c].notna(), np.nan)
    return df


def load_dirty(folder: str | Path) -> dict[str, pd.DataFrame]:
    """Read the dirty CSVs as strings (blank/whitespace -> NaN)."""
    folder = Path(folder)
    out = {}
    for t in TABLES:
        raw = pd.read_csv(folder / f"{t}_dirty.csv", dtype=str, keep_default_na=False)
        out[t] = _blank_to_nan(raw)
    return out


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iat[0] if len(s) else np.nan


def _id_num(s: pd.Series) -> pd.Series:
    return s.str.extract(r"(\d+)")[0].astype(float)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def clean_sessions(s: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    s = s.drop_duplicates().copy()
    _log("Sessions", "حذف الصفوف المكررة تمامًا")
    s["n"] = _id_num(s["session_id"])
    per_group = int(round(len(s) / s["group_id"].nunique()))  # 36 sessions per group
    # group_id: session ids are numbered in blocks of `per_group` per group
    guess = s["n"].map(lambda n: f"G{int((n - 1) // per_group + 1):02d}")
    s["group_id"] = s["group_id"].fillna(guess)
    _log("Sessions", f"group_id الناقص = من ترقيم session_id (كل مجموعة {per_group} حصة متتالية)")
    # date: every record carries the session date
    rec_dates = rec.dropna(subset=["session_id", "date"]).groupby("session_id")["date"].agg(_mode)
    s["date"] = pd.to_datetime(s["date"].fillna(s["session_id"].map(rec_dates)))
    _log("Sessions", "date الناقص = تاريخ السجلات المربوطة بنفس الحصة")
    s["session_number"] = pd.to_numeric(s["session_number"]).fillna((s["n"] - 1) % per_group + 1).astype(int)
    s["month"] = pd.to_numeric(s["month"]).fillna(np.ceil(s["session_number"] / 4)).astype(int)
    _log("Sessions", "session_number = ترتيب الحصة داخل المجموعة، month = ceil(session_number/4)")
    wd = s["date"].dt.weekday
    group_wd = wd.groupby(s["group_id"]).agg(_mode)
    makeup = wd != s["group_id"].map(group_wd)
    s["session_type"] = s["session_type"].fillna(pd.Series(np.where(makeup, "makeup", "regular"), index=s.index))
    _log("Sessions", "session_type الناقص = makeup لو يوم الحصة مختلف عن يوم المجموعة المعتاد، غير كده regular")
    return s.drop(columns="n").sort_values("session_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
_GRADE_PATTERNS = [("اولى|أولى", "أولى ثانوي"), ("تانية|ثانية", "تانية ثانوي"), ("تالتة|ثالثة", "تالتة ثانوي")]
_GRADE_SHORT = {"أولى ثانوي": "أولى", "تانية ثانوي": "تانية", "تالتة ثانوي": "تالتة"}


def _grade_from_name(name):
    if isinstance(name, str):
        for pat, grade in _GRADE_PATTERNS:
            if re.search(pat, name):
                return grade
    return np.nan


def _hour_to_time(h: int):
    # observed schedule: 10 -> AM, 12 -> noon, 2..8 -> PM
    if h in (9, 10, 11):
        return pd.Timestamp(2000, 1, 1, h).time()
    return pd.Timestamp(2000, 1, 1, 12 if h == 12 else h + 12).time()


def clean_groups(g: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    g = g.drop_duplicates().copy()
    _log("Groups", "حذف الصف المكرر")
    g["time"] = pd.to_datetime(g["time"], format="%I:%M:%S %p", errors="coerce").dt.time
    g["monthly_fee"] = pd.to_numeric(g["monthly_fee"])
    g["start_date"] = pd.to_datetime(g["start_date"])
    # grade / track from the group name
    g["grade"] = g["grade"].fillna(g["group_name"].map(_grade_from_name))
    track_guess = g["group_name"].map(lambda n: ("لغات" if "لغات" in n else "عام") if isinstance(n, str) else np.nan)
    g["track"] = g["track"].fillna(track_guess)
    _log("Groups", "grade و track الناقصين = من اسم المجموعة (أولى/تانية/تالتة، وكلمة لغات)")
    # day: from the name, otherwise from the weekday of its regular sessions
    def day_from_name(n):
        if isinstance(n, str):
            for d in DAY_NAMES:
                if d in n:
                    return d
        return np.nan

    sess_day = sess[sess["session_type"] == "regular"].groupby("group_id")["date"].agg(
        lambda d: DAY_NAMES[int(d.dt.weekday.mode().iat[0])]
    )
    g["day"] = g["day"].fillna(g["group_name"].map(day_from_name)).fillna(g["group_id"].map(sess_day))
    # time: hour is the last number in the name
    def time_from_name(n):
        m = re.search(r"(\d+)\s*$", n) if isinstance(n, str) else None
        return _hour_to_time(int(m.group(1))) if m else np.nan

    g["time"] = g["time"].where(g["time"].notna(), g["group_name"].map(time_from_name))
    _log("Groups", "day = من الاسم أو من يوم حصصها العادية؛ time = من آخر رقم في الاسم (10 صباحًا، 12 ظهرًا، 2-8 مساءً)")
    fee = g.dropna(subset=["monthly_fee"]).groupby(["grade", "track"])["monthly_fee"].agg(_mode)
    miss = g["monthly_fee"].isna()
    g.loc[miss, "monthly_fee"] = [fee.get((a, b), np.nan) for a, b in zip(g.loc[miss, "grade"], g.loc[miss, "track"])]
    _log("Groups", "monthly_fee الناقص = السعر الشائع لنفس (السنة، المسار) في باقي المجموعات")
    first_date = sess.groupby("group_id")["date"].min()
    g["start_date"] = g["start_date"].fillna(g["group_id"].map(first_date))
    _log("Groups", "start_date الناقص = تاريخ أول حصة للمجموعة")

    def build_name(r):
        lang = "لغات " if r["track"] == "لغات" else ""
        t = r["time"]
        hr = (t.hour % 12) or 12
        return f"مجموعة {lang}{_GRADE_SHORT.get(r['grade'], '')} {r['day']} {hr}"

    miss_name = g["group_name"].isna()
    g.loc[miss_name, "group_name"] = g[miss_name].apply(build_name, axis=1)
    _log("Groups", "group_name الناقص = إعادة بناء بنفس نمط الأسماء (الأسماء الأصلية مش موحدة، فالبناء تقريبي)")
    return g.sort_values("group_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
_ATT_FIX = {"present": "present", "absent": "absent", "late": "late", "excused": "excused"}


def clean_records(r: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    r = r.drop_duplicates(subset="record_id").copy()
    _log("Records", "حذف الصفوف المكررة (نفس record_id)")
    r["rid"] = _id_num(r["record_id"])
    r = r.sort_values("rid").reset_index(drop=True)
    for c in ["exam_score", "max_exam_score", "percentage"]:
        r[c] = pd.to_numeric(r[c])
    for c in ["student_session_number", "group_month", "group_session", "student_month", "student_session"]:
        r[c] = pd.to_numeric(r[c])
    r["date"] = pd.to_datetime(r["date"])
    r["attendance"] = r["attendance"].str.lower()
    _log("Records", "توحيد attendance (lowercase + شيل المسافات) وhomework (شيل المسافات)")
    sess_i = sess.set_index("session_id")

    # --- session_id: neighbours share the session, else date + student's groups
    known = r.dropna(subset=["student_id", "session_id"])
    stud_groups = known.assign(g=known["session_id"].map(sess_i["group_id"])).groupby("student_id")["g"].agg(set)
    for i in r.index[r["session_id"].isna()]:
        prev = r.at[i - 1, "session_id"] if i > 0 else np.nan
        nxt = r.at[i + 1, "session_id"] if i + 1 < len(r) else np.nan
        if isinstance(prev, str) and prev == nxt:
            r.at[i, "session_id"] = prev
            continue
        stu, d = r.at[i, "student_id"], r.at[i, "date"]
        if isinstance(stu, str) and pd.notna(d):
            cand = sess[(sess["date"] == d) & (sess["group_id"].isin(stud_groups.get(stu, set())))]["session_id"].tolist()
            if len(cand) == 1:
                r.at[i, "session_id"] = cand[0]
            elif len(cand) > 1:
                pick = [c for c in cand if c in (prev, nxt)]
                if len(pick) == 1:
                    r.at[i, "session_id"] = pick[0]
    _log("Records", "session_id الناقص = لو السجل اللي قبله وبعده في نفس الحصة، أو من (تاريخ + مجموعة الطالب)")

    # --- student_id: roster of the session's group, ordered by student id
    known = r.dropna(subset=["student_id", "session_id"])
    sess_students = known.groupby("session_id")["student_id"].agg(set)
    roster = known.assign(g=known["session_id"].map(sess_i["group_id"])).groupby("g")["student_id"].agg(set)
    sess_by_group = sess.groupby("group_id")["session_id"].apply(list)
    for i in r.index[r["student_id"].isna()]:
        s = r.at[i, "session_id"]
        if not isinstance(s, str):
            continue
        g = sess_i.at[s, "group_id"]
        cand = set(roster.get(g, set())) - set(sess_students.get(s, set()))
        prev = r.at[i - 1, "student_id"] if i > 0 and r.at[i - 1, "session_id"] == s else np.nan
        nxt = r.at[i + 1, "student_id"] if i + 1 < len(r) and r.at[i + 1, "session_id"] == s else np.nan
        cand = {c for c in cand if (not isinstance(prev, str) or c > prev) and (not isinstance(nxt, str) or c < nxt)}
        if len(cand) > 1:  # continuity: must also appear in the neighbouring sessions of the group
            order = sess_by_group[g]
            k = order.index(s)
            neighbours = [order[j] for j in (k - 1, k + 1) if 0 <= j < len(order)]
            cont = {c for c in cand if all(c in sess_students.get(n, set()) for n in neighbours)}
            cand = cont or cand
        if len(cand) == 1:
            r.at[i, "student_id"] = next(iter(cand))
    _log("Records", "student_id الناقص = الطالب الوحيد من قايمة المجموعة اللي ملوش سجل في الحصة وبين جاريه في الترتيب")

    # --- date & session-derived columns
    r["date"] = r["date"].fillna(r["session_id"].map(sess_i["date"]))
    r["group_session"] = r["group_session"].fillna(r["session_id"].map(sess_i["session_number"]))
    r["group_month"] = r["group_month"].fillna(r["session_id"].map(sess_i["month"]))
    _log("Records", "date وgroup_session وgroup_month الناقصين = من جدول Sessions")

    # --- exam triangle (score, max, percentage)
    max_by_session = r.dropna(subset=["max_exam_score"]).groupby("session_id")["max_exam_score"].agg(_mode)
    bad = (r["exam_score"] < 0) | (r["exam_score"] > r["max_exam_score"])
    n_bad = int(bad.sum())
    r.loc[bad, "exam_score"] = np.nan
    _log("Records", f"درجات خارج النطاق (<0 أو > الدرجة النهائية): {n_bad} -> تتحسب من percentage × max")
    took_exam = r["exam_score"].notna() | r["percentage"].notna() | r["max_exam_score"].notna()
    r.loc[took_exam, "max_exam_score"] = r.loc[took_exam, "max_exam_score"].fillna(r.loc[took_exam, "session_id"].map(max_by_session))
    r["max_exam_score"] = r["max_exam_score"].fillna((r["exam_score"] / r["percentage"]).round())
    r["exam_score"] = r["exam_score"].fillna((r["percentage"] * r["max_exam_score"]).round())
    r["percentage"] = r["percentage"].fillna((r["exam_score"] / r["max_exam_score"]).round(3))
    _log("Records", "exam_score / max_exam_score / percentage: لو واحد ناقص يتحسب من الاتنين التانيين أو من درجة الحصة")

    # --- attendance
    has_exam = r[["exam_score", "max_exam_score", "percentage"]].notna().any(axis=1)
    note_map = r.dropna(subset=["attendance", "attendance_notes"]).groupby("attendance_notes")["attendance"].agg(_mode)
    miss = r["attendance"].isna()
    guess = r["attendance_notes"].map(note_map)
    guess = guess.where(~has_exam, np.where(r["attendance_notes"].eq("تأخر عن الميعاد"), "late", "present"))
    guess = guess.fillna("absent")
    r["attendance"] = r["attendance"].fillna(guess)
    _log("Records", "attendance الناقص = لو فيه درجة امتحان فالطالب حضر (late لو سبب 'تأخر عن الميعاد')؛ غير كده من سبب الغياب")

    # --- homework: student's usual value (absent rows -> 'لم يسلم')
    hw_mode = r.dropna(subset=["homework"]).groupby("student_id")["homework"].agg(_mode)
    glob = _mode(r["homework"])
    fill = r["student_id"].map(hw_mode).fillna(glob)
    fill = fill.where(r["attendance"].isin(rules.ATTENDED), "لم يسلم")
    r["homework"] = r["homework"].fillna(fill)
    _log("Records", "homework الناقص = القيمة الأكتر تكرارًا للطالب (أو 'لم يسلم' لو غايب) -- تقدير، مش استرجاع")

    # --- student counters
    ok = r["student_id"].notna()
    order = r[ok].sort_values(["student_id", "date", "rid"])
    n = order.groupby("student_id").cumcount() + 1
    r.loc[order.index, "_n"] = n
    r["student_session_number"] = r["student_session_number"].fillna(r["_n"])
    r["student_month"] = r["student_month"].fillna(np.ceil(r["student_session_number"] / 4))
    r["student_session"] = r["student_session"].fillna((r["student_session_number"] - 1) % 4 + 1)
    _log("Records", "student_session_number = رقم السجل للطالب بترتيب التاريخ؛ student_month = ceil(n/4)؛ student_session = موقعه في الشهر (1-4)")
    r = r.drop(columns=["_n"])
    # notes & attendance_notes are free text -> never invented
    _log("Records", "notes و attendance_notes الناقصين: مش بنخمنهم (نص حر)، بيفضلوا NaN")
    for c in ["student_session_number", "group_month", "group_session", "student_month", "student_session"]:
        r[c] = r[c].astype("Int64")
    return r.drop(columns="rid").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------
def _norm_phone(x):
    if not isinstance(x, str):
        return np.nan
    d = re.sub(r"\D", "", x)
    if d.startswith("0020"):
        d = "0" + d[4:]
    elif d.startswith("20") and len(d) == 12:
        d = "0" + d[2:]
    if len(d) == 10 and d.startswith("1"):
        d = "0" + d
    return d if len(d) == 11 and d.startswith("01") else np.nan


def clean_students(s: pd.DataFrame, rec: pd.DataFrame, sess: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    s = s.drop_duplicates().copy()
    _log("Students", "حذف الصفوف المكررة")
    _log("Students", "شيل المسافات الزايدة من الأسماء وباقي النصوص")
    for c in ["phone", "parent_phone"]:
        s[c] = s[c].map(_norm_phone)
    _log("Students", "توحيد التليفونات: أرقام بس، 11 رقم بيبدأ بـ 01 (شيل المسافات والعلامات و+20)")
    for c in ["join_date", "leave_date"]:
        s[c] = pd.to_datetime(s[c])
    # group: group of the student's latest record (matches 'Students.group_id = latest' for transfers)
    k = rec.dropna(subset=["student_id", "session_id", "date"]).copy()
    k["g"] = k["session_id"].map(sess.set_index("session_id")["group_id"])
    latest = k.sort_values("date").groupby("student_id")["g"].last()
    s["group_id"] = s["group_id"].fillna(s["student_id"].map(latest))
    _log("Students", "group_id الناقص = مجموعة آخر سجل حضور للطالب")
    s["grade"] = s["grade"].fillna(s["group_id"].map(groups.set_index("group_id")["grade"]))
    _log("Students", "grade الناقص = سنة مجموعته")
    s["status"] = s["status"].fillna(pd.Series(np.where(s["leave_date"].notna(), "dropped", "active"), index=s.index))
    _log("Students", "status الناقص = dropped لو فيه leave_date غير كده active")
    _log("Students", "name / phone / parent_phone / gender / school_type / area / join_date الناقصين: مفيش مصدر موثوق، بيفضلوا NaN")
    return s.sort_values("student_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------
_STATUS_MAP = {
    "paid": "paid", "unpaid": "unpaid", "exempt": "exempt", "exempted": "exempt",
    "late": "paid_late", "paid_late": "paid_late", "partial": "partial", "partially paid": "partial",
}


def clean_payments(p: pd.DataFrame, students: pd.DataFrame, sess: pd.DataFrame) -> pd.DataFrame:
    p = p.drop_duplicates(subset="payment_id").copy()
    _log("Payments", "حذف الصفوف المكررة (نفس payment_id)")
    p["pid"] = _id_num(p["payment_id"])
    p = p.sort_values("pid").reset_index(drop=True)
    p["status"] = p["status"].str.lower().map(_STATUS_MAP).where(p["status"].notna(), np.nan)
    _log("Payments", "توحيد status: Paid->paid، UNPAID->unpaid، exempted->exempt، late->paid_late، partially paid->partial")
    _log("Payments", "شيل المسافات الزايدة من method")
    for c in ["amount_due", "amount_paid"]:
        p[c] = pd.to_numeric(p[c])
    for c in ["due_date", "paid_date"]:
        p[c] = pd.to_datetime(p[c])
    p["m"] = pd.to_numeric(p["month_label"].str.extract(r"(\d+)")[0])

    # month: neighbours in the same block share the month
    for i in p.index[p["m"].isna()]:
        prev = p.at[i - 1, "m"] if i > 0 else np.nan
        nxt = p.at[i + 1, "m"] if i + 1 < len(p) else np.nan
        if pd.notna(prev) and prev == nxt:
            p.at[i, "m"] = prev
    grp = students.set_index("student_id")["group_id"]
    first_of_month = sess[sess["session_number"] % 4 == 1].copy()
    first_of_month["m"] = (first_of_month["session_number"] - 1) // 4 + 1
    due_lookup = first_of_month.set_index(["group_id", "m"])["date"]
    for i in p.index[p["m"].isna()]:  # boundary rows: use due_date
        stu, d = p.at[i, "student_id"], p.at[i, "due_date"]
        if isinstance(stu, str) and pd.notna(d) and stu in grp.index:
            cand = first_of_month[(first_of_month["group_id"] == grp[stu]) & (first_of_month["date"] == d)]["m"]
            if len(cand) == 1:
                p.at[i, "m"] = cand.iat[0]
    p["month_label"] = p["month_label"].fillna(p["m"].map(lambda v: f"شهر {int(v)}" if pd.notna(v) else np.nan))
    _log("Payments", "month_label الناقص = زي جيرانه في نفس الدفعة (الجداول مرتبة شهر ثم طالب)، أو من due_date")

    # student_id: block order is month-major, students ascending
    month_students = p.dropna(subset=["student_id", "m"]).groupby("m")["student_id"].agg(set)
    for i in p.index[p["student_id"].isna()]:
        m = p.at[i, "m"]
        if pd.isna(m):
            continue
        here = month_students.get(m, set())
        around = month_students.get(m - 1, set()) | month_students.get(m + 1, set())
        cand = around - here
        prev = p.at[i - 1, "student_id"] if i > 0 and p.at[i - 1, "m"] == m else np.nan
        nxt = p.at[i + 1, "student_id"] if i + 1 < len(p) and p.at[i + 1, "m"] == m else np.nan
        cand = {c for c in cand if (not isinstance(prev, str) or c > prev) and (not isinstance(nxt, str) or c < nxt)}
        if len(cand) == 1:
            p.at[i, "student_id"] = next(iter(cand))
    _log("Payments", "student_id الناقص = الطالب الوحيد اللي ملوش دفعة في الشهر وبين جاريه في الترتيب")

    # amount_due / due_date
    usual_due = p[p["amount_due"] > 0].groupby("student_id")["amount_due"].agg(_mode)
    p["amount_due"] = p["amount_due"].fillna(
        pd.Series(np.where(p["status"].eq("exempt"), 0.0, p["student_id"].map(usual_due)), index=p.index)
    )
    _log("Payments", "amount_due الناقص = 0 لو معفي، غير كده قيمة الطالب المعتادة (بتراعي الخصومات)")
    stu_group = p["student_id"].map(grp)
    due_fill = [due_lookup.get((g, m), pd.NaT) if pd.notna(m) and isinstance(g, str) else pd.NaT for g, m in zip(stu_group, p["m"])]
    p["due_date"] = p["due_date"].fillna(pd.Series(due_fill, index=p.index))
    _log("Payments", "due_date الناقص = تاريخ أول حصة في شهر الدفع لمجموعة الطالب")

    # status: derive from amounts/dates when missing
    paid_rows = p[p["status"] == "paid"]
    grace = (paid_rows["paid_date"] - paid_rows["due_date"]).dt.days.max()
    miss = p["status"].isna()
    late_days = (p["paid_date"] - p["due_date"]).dt.days

    def derive(i):
        due, paid, pdt = p.at[i, "amount_due"], p.at[i, "amount_paid"], p.at[i, "paid_date"]
        if pd.notna(due) and due == 0:
            return "exempt"
        if pd.isna(pdt) and (pd.isna(paid) or paid == 0):
            return "unpaid"
        if pd.notna(due) and pd.notna(paid) and 0 < paid < due:
            return "partial"
        if pd.notna(pdt):
            return "paid_late" if late_days[i] > grace else "paid"
        return np.nan

    for i in p.index[miss]:
        p.at[i, "status"] = derive(i)
    _log("Payments", f"status الناقص = من المبالغ والتواريخ (معفي لو due=0، غير مدفوع لو مفيش تاريخ سداد، paid_late لو التأخير أكتر من {int(grace)} يوم)")

    # amount_paid
    fill = p["status"].map({"paid": None, "paid_late": None, "unpaid": 0.0, "exempt": 0.0})
    fill = fill.where(fill.notna(), np.where(p["status"].isin(["paid", "paid_late"]), p["amount_due"], np.nan))
    p["amount_paid"] = p["amount_paid"].fillna(pd.Series(fill, index=p.index).astype(float))
    _log("Payments", "amount_paid الناقص = المبلغ كامل لو paid/paid_late، 0 لو unpaid/exempt؛ partial مش بنخمنه")

    # method: student's usual method for rows that were actually paid
    usual_m = p.dropna(subset=["method"]).groupby("student_id")["method"].agg(_mode)
    glob_m = _mode(p["method"])
    need = p["method"].isna() & p["status"].isin(["paid", "paid_late", "partial"])
    p.loc[need, "method"] = p.loc[need, "student_id"].map(usual_m).fillna(glob_m)
    _log("Payments", "method الناقص في دفعات اتدفعت فعلًا = طريقة الدفع المعتادة للطالب -- تقدير")
    _log("Payments", "paid_date الناقص في دفعات اتدفعت: مفيش مصدر، بيفضل NaT")
    return p.drop(columns=["pid", "m"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def clean_all(dirty: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    DECISIONS.clear()
    rec0 = dirty["Records"].drop_duplicates(subset="record_id")
    sess = clean_sessions(dirty["Sessions"], rec0)
    groups = clean_groups(dirty["Groups"], sess)
    rec = clean_records(dirty["Records"], sess)
    students = clean_students(dirty["Students"], rec, sess, groups)
    pay = clean_payments(dirty["Payments"], students, sess)
    return {"Students": students, "Groups": groups, "Sessions": sess, "Records": rec, "Payments": pay}


def save_clean(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}_clean.csv", index=False, encoding="utf-8-sig")
