"""EDA question bank -> one Plotly figure + one takeaway sentence per question.

Every question is a function `qN(ctx) -> (figure, takeaway)`; `QUESTIONS` is the registry
(id, axis, priority, title). The same functions feed the notebook, the HTML report and
the Streamlit pages, so nothing is written twice.

Attendance follows the team rule in rules.py: late counts as attended.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import rules

C = dict(midnight="#071A3D", deep="#0D2A5E", blue="#174EA6", cyan="#00B8E6", mint="#2FB89A", mist="#E8EFF3",
         amber="#E0A100", red="#D64550", green="#1B8A63", grey="#8A97AB")
SEQ = [C["blue"], C["mint"], C["amber"], C["red"], C["deep"], C["cyan"], C["grey"]]
FONT = "IBM Plex Sans Arabic, Tahoma, Arial, sans-serif"
RAMADAN = (pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19"))
HW_SCORE = {"ممتاز": 5, "جيد جدا": 4, "جيد": 3, "مقبول": 2, "ضعيف": 1, "لم يسلم": 0}
GRADES = ["أولى ثانوي", "تانية ثانوي", "تالتة ثانوي"]
LEVELS = [("ممتاز", 0.90, C["green"]), ("جيد", 0.75, C["blue"]), ("متوسط", 0.60, C["amber"]), ("ضعيف", 0.0, C["red"])]
STATUS_AR = {"paid": "مدفوع", "paid_late": "متأخر", "partial": "جزئي", "unpaid": "غير مدفوع", "exempt": "معفي"}
STATUS_COL = {"paid": C["green"], "paid_late": C["amber"], "partial": C["cyan"], "unpaid": C["red"], "exempt": C["grey"]}

QUESTIONS: dict[int, dict] = {}
_FUNCS: dict[int, callable] = {}
pc = lambda x: f"{x * 100:.1f}%"
eg = lambda x: f"{x:,.0f} ج"


def q(n, axis, prio, title):
    def deco(fn):
        QUESTIONS[n] = dict(id=n, axis=axis, prio=prio, title=title)
        _FUNCS[n] = fn
        return fn
    return deco


def fin(fig, title, h=430, legend=True):
    fig.update_layout(title=dict(text=title, x=1, xanchor="right", font=dict(size=16, color=C["midnight"])),
                      height=h, template="plotly_white", font=dict(family=FONT, size=12), colorway=SEQ,
                      margin=dict(l=50, r=30, t=64, b=50), showlegend=legend,
                      legend=dict(orientation="h", y=-0.18, x=1, xanchor="right"), hoverlabel=dict(font_family=FONT))
    return fig


def _label(S, ids):
    return [f"{S.at[i, 'name'] if isinstance(S.at[i, 'name'], str) else i} ({i})" for i in ids]


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
def build(t: dict) -> SimpleNamespace:
    st, gr, se, rc, pay = (t[k].copy() for k in ["Students", "Groups", "Sessions", "Records", "Payments"])
    gr["hour"] = pd.to_datetime(gr["time"], format="%H:%M:%S", errors="coerce").dt.hour
    r = rc.dropna(subset=["student_id"]).merge(se[["session_id", "group_id", "session_type"]], on="session_id", how="left")
    r = r.merge(gr[["group_id", "group_name", "grade", "track", "hour", "monthly_fee"]], on="group_id", how="left")
    r = r.merge(st[["student_id", "name", "school_type", "area", "status", "group_id"]].rename(columns={"group_id": "cur_group"}),
                on="student_id", how="left")
    r = r.sort_values(["student_id", "date", "record_id"]).reset_index(drop=True)
    r["att"] = r.attendance.isin(rules.ATTENDED)
    r["present"] = r.attendance.eq("present")
    r["late"] = r.attendance.eq("late")
    r["is_exam"] = r.group_session % 4 == 0
    r["hw_score"] = r.homework.map(HW_SCORE)
    r["submitted"] = r.homework.ne("لم يسلم")
    r["week"] = r.date.dt.to_period("W").dt.start_time
    r["ym"] = r.date.dt.to_period("M").dt.to_timestamp()
    # consecutive-miss runs
    block = r.att.groupby(r.student_id).cumsum()
    r["run"] = (~r.att).astype(int).groupby([r.student_id, block]).cumsum()

    sc = r[r.percentage.notna()].copy()
    S = st.set_index("student_id").copy()
    g = r.groupby("student_id")
    S["n"] = g.size()
    S["att_rate"], S["late_rate"] = g.att.mean(), g.late.mean()
    S["avg"] = sc.groupby("student_id").percentage.mean()
    S["wk_avg"] = sc[~sc.is_exam].groupby("student_id").percentage.mean()
    S["ex_avg"] = sc[sc.is_exam].groupby("student_id").percentage.mean()
    S["std"] = sc.groupby("student_id").percentage.std()
    att_rows = r[r.att]
    S["hw_rate"] = att_rows.groupby("student_id").submitted.mean()
    S["hw_mean"] = att_rows.groupby("student_id").hw_score.mean()
    sc = sc.assign(x=sc.student_session_number / 4.0)
    a = sc.assign(xy=sc.x * sc.percentage, xx=sc.x ** 2).groupby("student_id").agg(
        n=("x", "size"), sx=("x", "sum"), sy=("percentage", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum"))
    S["slope"] = ((a.n * a.sxy - a.sx * a.sy) / (a.n * a.sxx - a.sx ** 2)).where(a.n >= 12)
    S["max_run"] = g.run.max()
    S["cur_run"] = g.run.last()
    S["last_month"] = g.group_month.max()
    S["first_session"] = g.group_session.min()
    rec4 = r.groupby("student_id").tail(4)
    S["att_recent"] = rec4.groupby("student_id").att.mean()
    S["pct_recent"] = rec4[rec4.percentage.notna()].groupby("student_id").percentage.mean()
    S["at_risk"] = rules.at_risk(S.att_rate, S.avg)
    S["gap"] = np.maximum(S.att_rate - rules.AT_RISK_ATTENDANCE_MAX, S.avg - rules.AT_RISK_AVG_MAX)

    P = pay.merge(st[["student_id", "name", "group_id", "status"]].rename(columns={"status": "stu_status"}), on="student_id", how="left")
    P = P.merge(gr[["group_id", "group_name", "grade", "monthly_fee"]], on="group_id", how="left")
    P["m"] = P.month_label.str.extract(r"(\d+)")[0].astype(float)
    P["days_late"] = (P.paid_date - P.due_date).dt.days
    P["overdue"] = P.status.isin(["unpaid", "partial"])
    P["owed"] = np.where(P.overdue, (P.amount_due - P.amount_paid.fillna(0)).clip(lower=0), 0.0)
    pg = P.groupby("student_id")
    S["unpaid_m"] = P[P.overdue].groupby("student_id").size()
    S["owed"] = pg.owed.sum()
    S["typ_due"] = P[P.amount_due > 0].groupby("student_id").amount_due.median()
    S[["unpaid_m", "owed"]] = S[["unpaid_m", "owed"]].fillna(0)

    G = gr.set_index("group_id").copy()
    gs = S.groupby("group_id")
    G["size"] = gs.size()
    G["att"] = r.groupby("group_id").att.mean()
    G["avg"] = sc.groupby("group_id").percentage.mean()
    G["hw_rate"] = att_rows.groupby("group_id").submitted.mean()
    pp = P.groupby("group_id")
    G["due"], G["paid"] = pp.amount_due.sum(), pp.amount_paid.sum()
    G["collection"] = G.paid / G.due
    G["expected"] = pp.monthly_fee.sum()
    G["dropped"] = gs.status.apply(lambda s: (s == "dropped").sum())
    G["risk_n"] = gs.at_risk.sum()
    G["risk_share"] = G.risk_n / G["size"]
    G["nm"] = G.index + " · " + G.group_name

    as_of = r.date.max()
    return SimpleNamespace(r=r, sc=sc, S=S, G=G, P=P, st=st, gr=gr, se=se, as_of=as_of)


# ---------------------------------------------------------------------------
# 1 · overview
# ---------------------------------------------------------------------------
@q(1, "نظرة عامة", "P1", "الطلاب والمجموعات متوزعين إزاي على السنين التلات؟")
def q1(c):
    a, b = c.S.groupby("grade").size(), c.gr.groupby("grade").size()
    fig = go.Figure([go.Bar(name="طلاب", x=GRADES, y=[a.get(x, 0) for x in GRADES], text=[a.get(x, 0) for x in GRADES], marker_color=C["blue"]),
                     go.Bar(name="مجموعات", x=GRADES, y=[b.get(x, 0) for x in GRADES], text=[b.get(x, 0) for x in GRADES], marker_color=C["mint"])])
    fig.update_traces(textposition="outside")
    top = a.idxmax()
    return fin(fig, "الطلاب والمجموعات حسب السنة", 380), (
        f"{top} هي الأكبر بـ {a.max()} طالب ({pc(a.max() / a.sum())}). متوسط حجم المجموعة {a.sum() / b.sum():.0f} طالب، "
        f"فوزّع وقتك وأسيستنتك على السنة الأكبر الأول.")


@q(2, "نظرة عامة", "P1", "إيه المؤشرات العامة (عدد الطلاب، الحضور، المتوسط، التحصيل، المتسربين)؟")
def q2(c):
    S, P, r = c.S, c.P, c.r
    att, avg = r.att.mean(), c.sc.percentage.mean()
    col = P.amount_paid.sum() / P.amount_due.sum()
    drop = (S.status == "dropped").sum()
    kp = [(len(S), "طالب", ""), (att * 100, "نسبة الحضور", "%"), (avg * 100, "متوسط الدرجات", "%"),
          (col * 100, "نسبة التحصيل", "%"), (drop, f"متسربين ({drop / len(S) * 100:.1f}%)", "")]
    fig = make_subplots(rows=1, cols=5, specs=[[{"type": "indicator"}] * 5])
    for i, (v, t, s) in enumerate(kp, 1):
        fig.add_trace(go.Indicator(mode="number", value=v, number=dict(suffix=s, valueformat=",.1f" if s else ",d", font=dict(color=C["blue"])),
                                   title=dict(text=t)), 1, i)
    return fin(fig, "المؤشرات العامة للسنة", 230, False), (
        f"حضور {pc(att)} ومتوسط درجات {pc(avg)} وتحصيل {pc(col)}، و{drop} طالب سابوا. "
        f"الأرقام كويسة عمومًا، فركّز على الشريحة الضعيفة والمتأخرين في الدفع بدل المتابعة العامة.")


@q(3, "نظرة عامة", "P3", "حجم المجموعات متوزع إزاي؟")
def q3(c):
    fig = go.Figure(go.Histogram(x=c.G["size"], nbinsx=14, marker_color=C["blue"]))
    fig.update_xaxes(title="عدد الطلاب في المجموعة"); fig.update_yaxes(title="عدد المجموعات")
    return fin(fig, "توزيع حجم المجموعات", 360, False), (
        f"المجموعات بين {int(c.G['size'].min())} و{int(c.G['size'].max())} طالب (متوسط {c.G['size'].mean():.0f}). "
        f"المجموعات الأكبر محتاجة انتباه أكتر في الحصة، والأصغر تتدمج لو الفرق كبير.")


@q(4, "نظرة عامة", "P2", "الطلاب متوزعين على الأحياء ونوع المدرسة إزاي؟")
def q4(c):
    ar = c.S.area.fillna("غير معروف").value_counts().head(12).iloc[::-1]
    sch = c.S.school_type.fillna("غير معروف").value_counts().iloc[::-1]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("أكتر 12 حي", "نوع المدرسة"), column_widths=[0.62, 0.38])
    fig.add_trace(go.Bar(x=ar.values, y=ar.index, orientation="h", marker_color=C["blue"], text=ar.values), 1, 1)
    fig.add_trace(go.Bar(x=sch.values, y=sch.index, orientation="h", marker_color=C["mint"], text=sch.values), 1, 2)
    return fin(fig, "الأحياء ونوع المدرسة", 420, False), (
        f"{ar.index[-1]} أكبر حي ({ar.iloc[-1]} طالب) و{sch.index[-1]} أكتر نوع مدرسة ({pc(sch.iloc[-1] / sch.sum())}). "
        f"اختار أماكن إعلانك وتوصيل أولياء الأمور حسب الأحياء الأكبر.")


@q(5, "نظرة عامة", "P1", "نسبة المتسربين مقابل النشطين كام؟")
def q5(c):
    v = c.S.status.value_counts()
    fig = go.Figure(go.Pie(labels=["نشط" if k == "active" else "متسرب" for k in v.index], values=v.values, hole=0.6,
                           marker_colors=[C["blue"] if k == "active" else C["red"] for k in v.index]))
    return fin(fig, "النشطين مقابل المتسربين", 360), (
        f"{v.get('dropped', 0)} طالب متسرب ({pc(v.get('dropped', 0) / v.sum())}) و{v.get('active', 0)} نشط. "
        f"النسبة صغيرة لكن كل متسرب بيكلف إيراد شهري، شوف أسباب التسرب في المحور السادس.")


# ---------------------------------------------------------------------------
# 2 · attendance
# ---------------------------------------------------------------------------
def _season(fig, c, y0=None):
    fig.add_vrect(x0=RAMADAN[0], x1=RAMADAN[1], fillcolor=C["amber"], opacity=0.14, line_width=0,
                  annotation_text="رمضان", annotation_position="top left")
    wk = set(c.se.date.dt.to_period("W").dt.start_time)
    for w in pd.date_range(c.r.week.min(), c.r.week.max(), freq="W-MON"):
        if w not in wk:
            fig.add_vrect(x0=w, x1=w + pd.Timedelta(days=6), fillcolor=C["grey"], opacity=0.22, line_width=0,
                          annotation_text="إجازة", annotation_position="top left")


@q(7, "الحضور", "P1", "الحضور الأسبوعي طول السنة إيه شكله، وإيه أثر الإجازات ورمضان والامتحانات؟")
def q7(c):
    w = c.r.groupby("week").agg(att=("att", "mean"), exam=("attendance_notes", lambda x: (x == "امتحانات مدرسة").sum()))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=w.index, y=w.exam, name="غياب بسبب امتحانات المدرسة", marker_color=C["amber"], opacity=0.35), secondary_y=True)
    fig.add_trace(go.Scatter(x=w.index, y=w.att, name="نسبة الحضور", mode="lines+markers", line=dict(color=C["blue"], width=3)), secondary_y=False)
    _season(fig, c)
    fig.update_yaxes(tickformat=".0%", secondary_y=False, title="الحضور"); fig.update_yaxes(title="غياب امتحانات", secondary_y=True)
    ram = w[(w.index >= RAMADAN[0] - pd.Timedelta(days=6)) & (w.index <= RAMADAN[1])].att.mean()
    oth = w.drop(w[(w.index >= RAMADAN[0] - pd.Timedelta(days=6)) & (w.index <= RAMADAN[1])].index).att.mean()
    lo = w.att.idxmin()
    return fin(fig, "الحضور الأسبوعي", 450), (
        f"أقل أسبوع حضور بدأ {lo:%d/%m} ({pc(w.att.min())}). في رمضان الحضور {pc(ram)} مقابل {pc(oth)} باقي السنة. "
        f"الإجازات الرمادية مفيهاش حصص أصلًا. قبل أسابيع الامتحانات ذكّر الأهالي بالمواعيد.")


@q(10, "الحضور", "P2", "إيه أكتر أسباب الغياب، وكام منها بعذر؟")
def q10(c):
    d = c.r[c.r.attendance.isin(rules.NOT_ATTENDED) & c.r.attendance_notes.notna()]
    t = d.groupby(["attendance_notes", "attendance"]).size().unstack(fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for col, name, colr in [("excused", "بعذر", C["mint"]), ("absent", "بدون عذر", C["red"])]:
        if col in t:
            fig.add_trace(go.Bar(y=t.index, x=t[col], name=name, orientation="h", marker_color=colr))
    fig.update_layout(barmode="stack")
    ex = (d.attendance == "excused").mean()
    return fin(fig, "أسباب الغياب", 400), (
        f"أكتر سبب «{t.sum(axis=1).idxmax()}» ({int(t.sum(axis=1).max())} حالة)، و{pc(ex)} من الغياب بعذر. "
        f"الغياب بدون عذر (نسيان وتعب) تقدر تقلله برسالة تذكير قبل الحصة بيوم.")


@q(11, "الحضور", "P1", "توزيع نسبة حضور الطالب، وكام طالب تحت 80% و85%؟")
def q11(c):
    S = c.S
    fig = go.Figure(go.Histogram(x=S.att_rate * 100, nbinsx=30, marker_color=C["blue"]))
    for x, col in [(80, C["red"]), (85, C["amber"])]:
        fig.add_vline(x=x, line_dash="dash", line_color=col, annotation_text=f"{x}%")
    fig.update_xaxes(title="نسبة حضور الطالب %"); fig.update_yaxes(title="عدد الطلاب")
    n80, n85 = (S.att_rate < 0.8).sum(), (S.att_rate < 0.85).sum()
    return fin(fig, "توزيع نسبة حضور الطلاب", 400, False), (
        f"{n80} طالب تحت 80% و{n85} تحت 85%. ابدأ بالـ {n80} بمكالمة ولي الأمر، والـ {n85 - n80} اللي بين الحدّين بتنبيه على الجروب.")


@q(12, "الحضور", "P2", "مين أطول طلاب الغياب المتتالي؟")
def q12(c):
    t = c.S.sort_values("max_run", ascending=False).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=t.max_run, y=_label(c.S, t.index), orientation="h", marker_color=C["red"], text=t.max_run))
    return fin(fig, "أطول غياب متتالي (عدد حصص) — أعلى 15", 520, False), (
        f"أطول غياب {int(t.max_run.iloc[-1])} حصص متتالية. لو الغياب متبرر (مرض/سفر) جهّز له ملخص، ولو لأ اتصل بولي الأمر من أول حصتين.")


@q(13, "الحضور", "P3", "الغياب بيتجمع في أسابيع معينة؟")
def q13(c):
    r = c.r
    top = (~r.att).groupby(r.student_id).sum().sort_values(ascending=False).head(40).index
    m = r[r.student_id.isin(top)].assign(miss=lambda d: (~d.att).astype(int)).pivot_table(index="student_id", columns="week", values="miss", aggfunc="sum", fill_value=0).loc[top]
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=_label(c.S, m.index), colorscale=[[0, "#F2F6FA"], [1, C["red"]]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    wk = (~r.att).groupby(r.week).mean()
    return fin(fig, "الغياب: أكتر 40 طالب × الأسابيع", 760, False), (
        f"أعلى أسبوع غياب بدأ {wk.idxmax():%d/%m} ({pc(wk.max())} من السجلات). الغياب موزع طول السنة مش متركز، يعني المشكلة في الأشخاص مش في أسابيع بعينها.")


@q(14, "الحضور", "P3", "التأخير (late) بيزيد في أنهي ميعاد؟")
def q14(c):
    h = c.r.groupby("hour").late.mean().dropna()
    lab = [f"{int(x) % 12 or 12} {'ص' if x < 12 else 'م'}" for x in h.index]
    fig = go.Figure(go.Bar(x=lab, y=h.values, marker_color=[C["red"] if v == h.max() else C["blue"] for v in h.values], text=[pc(v) for v in h.values]))
    fig.update_yaxes(tickformat=".0%")
    return fin(fig, "نسبة التأخير حسب ميعاد الحصة", 380, False), (
        f"أعلى تأخير في ميعاد {lab[int(np.argmax(h.values))]} ({pc(h.max())}). فكّر تأجّل بداية الحصة دي أو تحط تحذير قبلها.")


@q(17, "الحضور", "P1", "أنهي مجموعات حضورها الأسوأ والأفضل؟")
def q17(c):
    G = c.G.sort_values("att")
    col = [C["red"] if i < 5 else C["green"] if i >= len(G) - 5 else C["blue"] for i in range(len(G))]
    fig = go.Figure(go.Bar(x=G.att, y=G.nm, orientation="h", marker_color=col))
    fig.update_xaxes(tickformat=".0%", range=[G.att.min() - 0.02, 1])
    return fin(fig, "ترتيب المجموعات حسب الحضور", 900, False), (
        f"الأضعف {G.nm.iloc[0]} ({pc(G.att.iloc[0])}) والأقوى {G.nm.iloc[-1]} ({pc(G.att.iloc[-1])}). "
        f"اسأل أضعف 5 مجموعات عن الميعاد واليوم قبل ما تتهم الطلاب.")


@q(18, "الحضور", "P2", "هل أغلب الغياب بيجي من قلة من الطلاب؟")
def q18(c):
    ab = (~c.r.att).groupby(c.r.student_id).sum().sort_values(ascending=False)
    cum = ab.cumsum() / ab.sum()
    x = np.arange(1, len(ab) + 1) / len(ab) * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x[:150], y=ab.values[:150], name="غياب الطالب", marker_color=C["blue"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=cum.values * 100, name="التراكمي %", line=dict(color=C["red"])), secondary_y=True)
    fig.update_xaxes(title="% من الطلاب (من الأكتر غيابًا)"); fig.update_yaxes(title="% تراكمي", secondary_y=True)
    t10, t20 = cum.iloc[int(len(ab) * 0.1) - 1], cum.iloc[int(len(ab) * 0.2) - 1]
    return fin(fig, "باريتو الغياب", 420), (
        f"أكتر 10% من الطلاب عليهم {pc(t10)} من الغياب وأكتر 20% عليهم {pc(t20)}. ركّز مكالماتك على الفئة الصغيرة دي بدل الكل.")


# ---------------------------------------------------------------------------
# 3 · scores
# ---------------------------------------------------------------------------
@q(19, "الدرجات والأداء", "P1", "الدرجات (الاختبار الأسبوعي والامتحان الشهري) متوزعة إزاي؟")
def q19(c):
    sc = c.sc
    fig = make_subplots(rows=1, cols=2, subplot_titles=("اختبار أسبوعي", "امتحان شهري"))
    for i, (flag, col) in enumerate([(False, C["blue"]), (True, C["mint"])], 1):
        fig.add_trace(go.Histogram(x=sc[sc.is_exam == flag].percentage * 100, nbinsx=25, marker_color=col), 1, i)
    fig.update_xaxes(title="الدرجة %")
    return fin(fig, "توزيع الدرجات", 380, False), (
        f"متوسط الأسبوعي {pc(sc[~sc.is_exam].percentage.mean())} والشهري {pc(sc[sc.is_exam].percentage.mean())}. "
        f"الذيل الأيسر في الرسمتين هو الطلاب اللي محتاجين مراجعة قبل الامتحان القادم.")


@q(20, "الدرجات والأداء", "P1", "متوسط الدرجات بيتغير شهر بشهر لكل سنة؟")
def q20(c):
    d = c.sc.groupby(["group_month", "grade"]).percentage.mean().reset_index()
    fig = go.Figure()
    for gname, col in zip(GRADES, [C["mint"], C["blue"], C["amber"]]):
        x = d[d.grade == gname]
        fig.add_trace(go.Scatter(x=x.group_month, y=x.percentage, name=gname, mode="lines+markers", line=dict(color=col, width=3)))
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="شهر الدراسة", dtick=1)
    ch = d.groupby("grade").apply(lambda x: x.sort_values("group_month").percentage.iloc[-1] - x.sort_values("group_month").percentage.iloc[0])
    return fin(fig, "متوسط الدرجات بالشهر", 400), (
        f"التغيّر من أول شهر لآخر شهر: " + "، ".join(f"{k} {v * 100:+.1f} نقطة" for k, v in ch.items()) +
        ". السنة اللي بتنزل محتاجة مراجعة في الشرح مش بس متابعة.")


@q(21, "الدرجات والأداء", "P2", "الامتحان الشهري أصعب من الاختبار الأسبوعي؟")
def q21(c):
    sc = c.sc
    fig = go.Figure()
    for flag, name, col in [(False, "اختبار أسبوعي", C["blue"]), (True, "امتحان شهري", C["mint"])]:
        fig.add_trace(go.Box(y=sc[sc.is_exam == flag].percentage, name=name, marker_color=col, boxmean=True))
    fig.update_yaxes(tickformat=".0%")
    d = sc[sc.is_exam].percentage.mean() - sc[~sc.is_exam].percentage.mean()
    return fin(fig, "الأسبوعي مقابل الشهري", 400, False), (
        f"الشهري {'أصعب' if d < 0 else 'أسهل'} بفرق {abs(d) * 100:.1f} نقطة في المتوسط. "
        f"{'جهّز مراجعة مجمّعة قبل الشهري.' if d < 0 else 'مفيش ضغط زيادة في الشهري، حافظ على نظام المراجعة.'}")


@q(22, "الدرجات والأداء", "P1", "أنهي مجموعات الأقوى والأضعف؟")
def q22(c):
    S = c.S.dropna(subset=["avg"])
    order = S.groupby("group_id").avg.mean().sort_values().index
    fig = go.Figure(go.Box(x=S.group_id, y=S.avg, marker_color=C["blue"], boxpoints=False))
    fig.update_xaxes(categoryorder="array", categoryarray=list(order)); fig.update_yaxes(tickformat=".0%")
    g = c.G.avg.sort_values()
    return fin(fig, "متوسط درجات الطلاب لكل مجموعة (من الأضعف للأقوى)", 430, False), (
        f"الأضعف {c.G.nm[g.index[0]]} ({pc(g.iloc[0])}) والأقوى {c.G.nm[g.index[-1]]} ({pc(g.iloc[-1])}). "
        f"الفرق {((g.iloc[-1] - g.iloc[0]) * 100):.1f} نقطة، راجع طريقة الشرح في أضعف المجموعات.")


@q(24, "الدرجات والأداء", "P1", "الحضور بيأثر على الدرجات فعلًا؟")
def q24(c):
    S = c.S.dropna(subset=["avg"])
    k, b = np.polyfit(S.att_rate, S.avg, 1)
    xs = np.linspace(S.att_rate.min(), 1, 20)
    fig = go.Figure([go.Scatter(x=S.att_rate, y=S.avg, mode="markers", marker=dict(color=C["blue"], opacity=0.45, size=6),
                                text=_label(S, S.index), name="طالب"),
                     go.Scatter(x=xs, y=k * xs + b, mode="lines", line=dict(color=C["red"], width=3), name="خط الاتجاه")])
    fig.update_xaxes(tickformat=".0%", title="نسبة الحضور"); fig.update_yaxes(tickformat=".0%", title="متوسط الدرجات")
    r_ = S.att_rate.corr(S.avg)
    return fin(fig, "الحضور مقابل الدرجات", 440), (
        f"الارتباط {r_:.2f}: كل 10 نقاط حضور زيادة تقابلها {k * 10:.1f} نقطة درجات تقريبًا. "
        f"الحضور من أقوى المؤثرات على الدرجة، فمتابعة الغياب بتحسّن النتيجة.")


@q(25, "الدرجات والأداء", "P1", "الواجب بيأثر على الدرجات؟")
def q25(c):
    d = c.sc[c.sc.att]
    order = ["ممتاز", "جيد جدا", "جيد", "مقبول", "ضعيف", "لم يسلم"]
    fig = go.Figure()
    for o in order:
        fig.add_trace(go.Box(y=d[d.homework == o].percentage, name=o, boxmean=True, marker_color=C["red"] if o == "لم يسلم" else C["blue"]))
    fig.update_yaxes(tickformat=".0%")
    m = d.groupby("homework").percentage.mean()
    return fin(fig, "الدرجة حسب تقدير الواجب", 430, False), (
        f"اللي واجبه «ممتاز» متوسطه {pc(m['ممتاز'])} واللي «لم يسلم» {pc(m['لم يسلم'])} (فرق {(m['ممتاز'] - m['لم يسلم']) * 100:.1f} نقطة). "
        f"الواجب مؤشر مبكر، وتتبّع «لم يسلم» بيكشف المتراجعين قبل الامتحان.")


@q(26, "الدرجات والأداء", "P1", "مين أكتر 10 طلاب تحسنًا وأكتر 10 تراجعًا؟")
def q26(c):
    S = c.S.dropna(subset=["slope"]).sort_values("slope")
    t = pd.concat([S.head(10), S.tail(10)])
    fig = go.Figure(go.Bar(x=t.slope * 100, y=_label(S, t.index), orientation="h", marker_color=[C["red"] if v < 0 else C["green"] for v in t.slope]))
    fig.update_xaxes(title="التغير في الدرجة (نقطة / شهر)")
    return fin(fig, "أكتر 10 تراجعًا وأكتر 10 تحسنًا", 600, False), (
        f"أكبر تراجع {S.slope.iloc[0] * 100:.1f} نقطة/شهر وأكبر تحسن {S.slope.iloc[-1] * 100:+.1f}. "
        f"كلّم المتراجعين الأسبوع ده، وشجّع المتحسنين لأنهم أحسن إعلان ليك عند أهاليهم.")


@q(27, "الدرجات والأداء", "P3", "مين أكتر الطلاب تذبذبًا في درجاته؟")
def q27(c):
    t = c.S.sort_values("std", ascending=False).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["std"] * 100, y=_label(c.S, t.index), orientation="h", marker_color=C["amber"]))
    fig.update_xaxes(title="الانحراف المعياري (نقطة)")
    return fin(fig, "أكتر 15 طالب تذبذبًا", 520, False), (
        f"أعلى تذبذب {t['std'].iloc[-1] * 100:.1f} نقطة. التذبذب العالي غالبًا ظروف مش مستوى، اسألهم عن اللي بيحصل في الحصص الضعيفة.")


@q(28, "الدرجات والأداء", "P1", "الطلاب موزعين إزاي على مستويات (ممتاز، جيد، متوسط، ضعيف) في كل مجموعة؟")
def q28(c):
    S = c.S.dropna(subset=["avg"]).copy()
    S["lvl"] = pd.cut(S.avg, [0, 0.60, 0.75, 0.90, 1.01], labels=["ضعيف", "متوسط", "جيد", "ممتاز"], right=False)
    t = S.groupby(["group_id", "lvl"], observed=False).size().unstack(fill_value=0)
    t = t.div(t.sum(axis=1), axis=0).sort_values(["ضعيف", "ممتاز"], ascending=[True, False])
    fig = go.Figure()
    for name, _, col in LEVELS:
        fig.add_trace(go.Bar(x=t.index, y=t[name], name=name, marker_color=col))
    fig.update_layout(barmode="stack"); fig.update_yaxes(tickformat=".0%")
    tot = S.lvl.value_counts(normalize=True)
    return fin(fig, "مستويات الطلاب في كل مجموعة (ممتاز ≥90، جيد ≥75، متوسط ≥60، ضعيف <60)", 440), (
        f"على مستوى السنتر: ممتاز {pc(tot['ممتاز'])} وجيد {pc(tot['جيد'])} ومتوسط {pc(tot['متوسط'])} وضعيف {pc(tot['ضعيف'])}. "
        f"المجموعات اللي فيها شريحة «ضعيف» كبيرة (يمين الرسمة) أولى بحصص تقوية.")


@q(29, "الدرجات والأداء", "P2", "خريطة أداء المجموعات شهر بشهر")
def q29(c):
    m = c.sc.pivot_table(index="group_id", columns="group_month", values="percentage", aggfunc="mean")
    m = m.loc[m.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=m.values, x=m.columns, y=m.index, colorscale="RdYlGn", colorbar=dict(tickformat=".0%")))
    fig.update_xaxes(title="شهر الدراسة", dtick=1)
    d = m.iloc[:, -1] - m.iloc[:, 0]
    return fin(fig, "متوسط الدرجات: مجموعة × شهر", 800, False), (
        f"أكتر مجموعة اتحسنت {d.idxmax()} ({d.max() * 100:+.1f} نقطة) وأكتر واحدة نزلت {d.idxmin()} ({d.min() * 100:+.1f}). "
        f"الصفوف الحمرا المستمرة محتاجة تدخل، والنزول المفاجئ في شهر معين يعني حاجة اتغيرت فيه.")


@q(32, "الدرجات والأداء", "P2", "مين الطلاب اللي درجتهم انهارت فجأة في حصة واحدة؟")
def q32(c):
    sc = c.sc[c.sc.att].copy()
    sc["usual"] = sc.groupby("student_id").percentage.transform("median")
    sc["drop"] = sc.usual - sc.percentage
    ev = sc[sc["drop"] >= 0.30].sort_values("drop", ascending=False)
    top = ev.drop_duplicates("student_id").head(12)
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "xy"}]], row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Table(header=dict(values=["الطالب", "التاريخ", "درجته", "مستواه المعتاد", "الهبوط"], fill_color=C["midnight"], font=dict(color="white")),
                           cells=dict(values=[_label(c.S, top.student_id), top.date.dt.strftime("%d/%m"), top.percentage.map(pc), top.usual.map(pc), (top["drop"] * 100).round(0).astype(int).astype(str) + " نقطة"])), 1, 1)
    fig.add_trace(go.Scatter(x=ev.date, y=ev["drop"] * 100, mode="markers", marker=dict(color=C["red"], opacity=0.6), text=_label(c.S, ev.student_id)), 2, 1)
    fig.update_yaxes(title="الهبوط (نقطة)", row=2, col=1)
    return fin(fig, "حصص انهارت فيها الدرجة (هبوط ≥ 30 نقطة عن مستوى الطالب)", 700, False), (
        f"{ev.student_id.nunique()} طالب عندهم حصة انهيار واحدة على الأقل ({len(ev)} حالة). غالبًا ظرف مؤقت، اسأل ولي الأمر قبل ما تحكم إن مستواه نزل.")


# ---------------------------------------------------------------------------
# 4 · homework
# ---------------------------------------------------------------------------
@q(33, "الواجب", "P2", "نسبة تقديرات الواجب ونسبة «لم يسلم»")
def q33(c):
    v = c.r[c.r.att].homework.value_counts(normalize=True).reindex(list(HW_SCORE)[::1])
    fig = go.Figure(go.Bar(x=v.index, y=v.values, text=[pc(x) for x in v.values], marker_color=[C["red"] if k == "لم يسلم" else C["blue"] for k in v.index]))
    fig.update_yaxes(tickformat=".0%")
    return fin(fig, "تقديرات الواجب (في الحصص اللي الطالب حضرها)", 380, False), (
        f"«لم يسلم» {pc(v['لم يسلم'])} من الحصص اللي اتحضرت. لو الرقم عالي في مجموعة معينة اطلب الواجب بدري واربطه بدرجة.")


@q(34, "الواجب", "P2", "التزام الطالب بالواجب بيتغير مع الشهور؟")
def q34(c):
    d = c.r[c.r.att].groupby("group_month").agg(sub=("submitted", "mean"), mean=("hw_score", "mean"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.index, y=d["sub"], name="نسبة التسليم", mode="lines+markers", line=dict(color=C["blue"], width=3)))
    fig.add_trace(go.Scatter(x=d.index, y=d["mean"], name="متوسط التقدير (0-5)", mode="lines+markers", line=dict(color=C["mint"], width=3)), secondary_y=True)
    fig.update_yaxes(tickformat=".0%", secondary_y=False); fig.update_xaxes(title="شهر الدراسة", dtick=1)
    return fin(fig, "الواجب شهر بشهر", 380), (
        f"التسليم من {pc(d['sub'].iloc[0])} في أول شهر لـ {pc(d['sub'].iloc[-1])} في آخر شهر. " +
        ("الالتزام بينزل مع الوقت، فحافظ على التحفيز في النص التاني من السنة." if d['sub'].iloc[-1] < d['sub'].iloc[0] else "الالتزام ثابت أو بيتحسن."))


@q(35, "الواجب", "P2", "أنهي مجموعات مقصرة في الواجب؟")
def q35(c):
    G = c.G.sort_values("hw_rate")
    fig = go.Figure(go.Bar(x=G.hw_rate, y=G.nm, orientation="h", marker_color=[C["red"] if i < 5 else C["blue"] for i in range(len(G))]))
    fig.update_xaxes(tickformat=".0%", range=[G.hw_rate.min() - 0.02, 1])
    return fin(fig, "نسبة تسليم الواجب لكل مجموعة", 900, False), (
        f"أضعف مجموعة {G.nm.iloc[0]} ({pc(G.hw_rate.iloc[0])}) وأقواها {G.nm.iloc[-1]} ({pc(G.hw_rate.iloc[-1])}). ابدأ بتفعيل متابعة الواجب في الخمسة الحمر.")


@q(36, "الواجب", "P3", "الواجب مرتبط بالحضور؟")
def q36(c):
    S = c.S.dropna(subset=["hw_rate"]).copy()
    S["b"] = pd.cut(S.att_rate, [0, .7, .8, .9, .95, 1.01], labels=["<70%", "70-80%", "80-90%", "90-95%", "95%+"])
    d = S.groupby("b", observed=True).hw_rate.mean()
    fig = go.Figure(go.Bar(x=d.index.astype(str), y=d.values, text=[pc(v) for v in d.values], marker_color=C["blue"]))
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="شريحة الحضور")
    return fin(fig, "تسليم الواجب حسب شريحة الحضور", 380, False), (
        f"اللي حضوره 95%+ بيسلم {pc(d.iloc[-1])} واللي تحت 70% بيسلم {pc(d.iloc[0])}. اللي بيغيب بيسلم أقل، فتواصل معاه بالواجب مع المتابعة.")


# ---------------------------------------------------------------------------
# 5 · payments
# ---------------------------------------------------------------------------
@q(37, "الدفع والتحصيل", "P1", "نسبة التحصيل الشهرية والمبلغ المحصل مقابل المستحق")
def q37(c):
    d = c.P.groupby("m").agg(due=("amount_due", "sum"), paid=("amount_paid", "sum"))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=d.index, y=d.due, name="مستحق", marker_color=C["mist"], marker_line_color=C["blue"], marker_line_width=1))
    fig.add_trace(go.Bar(x=d.index, y=d.paid, name="محصّل", marker_color=C["blue"]))
    fig.add_trace(go.Scatter(x=d.index, y=d.paid / d.due, name="نسبة التحصيل", line=dict(color=C["red"], width=3)), secondary_y=True)
    fig.update_yaxes(tickformat=".0%", secondary_y=True, range=[0.7, 1.02]); fig.update_xaxes(title="شهر الفاتورة", dtick=1); fig.update_layout(barmode="overlay")
    r = d.paid / d.due
    return fin(fig, "المستحق والمحصل شهريًا", 420), (
        f"التحصيل بين {pc(r.min())} (شهر {int(r.idxmin())}) و{pc(r.max())}، والإجمالي {pc(d.paid.sum() / d.due.sum())}. "
        f"المتأخر {eg(d.due.sum() - d.paid.sum())}، اجمعه بمتابعة أول الشهر مش آخره.")


@q(38, "الدفع والتحصيل", "P1", "حالات الدفع (مدفوع، متأخر، جزئي، غير مدفوع، معفي) كل شهر")
def q38(c):
    t = c.P.groupby(["m", "status"]).size().unstack(fill_value=0)
    fig = go.Figure([go.Bar(x=t.index, y=t[s], name=STATUS_AR[s], marker_color=STATUS_COL[s]) for s in STATUS_AR if s in t])
    fig.update_layout(barmode="stack"); fig.update_xaxes(title="شهر الفاتورة", dtick=1)
    un = (t["unpaid"] / t.sum(axis=1))
    return fin(fig, "حالات الدفع شهريًا", 420), (
        f"غير المدفوع بين {pc(un.min())} و{pc(un.max())} من فواتير الشهر، وأعلاه شهر {int(un.idxmax())}. أي ارتفاع في الأحمر لازم يتتبّع فورًا.")


@q(39, "الدفع والتحصيل", "P1", "المتأخرات بالشهور: كام طالب متأخر شهر وشهرين وتلاتة+ وبكام؟")
def q39(c):
    S = c.S[c.S.unpaid_m > 0].copy()
    S["b"] = S.unpaid_m.clip(upper=3).astype(int).map({1: "شهر", 2: "شهرين", 3: "3+ شهور"})
    d = S.groupby("b").agg(n=("owed", "size"), owed=("owed", "sum")).reindex(["شهر", "شهرين", "3+ شهور"]).fillna(0)
    fig = go.Figure(go.Bar(x=d.index, y=d.n, marker_color=[C["amber"], C["red"], C["deep"]], text=[f"{int(n)} طالب<br>{eg(o)}" for n, o in zip(d.n, d.owed)], textposition="outside"))
    fig.update_yaxes(title="عدد الطلاب")
    return fin(fig, "أعمار المتأخرات", 420, False), (
        f"{len(S)} طالب عليهم متأخرات بإجمالي {eg(S.owed.sum())}، منهم {int(d.n['3+ شهور'])} طالب متأخر 3 شهور أو أكتر ({eg(d.owed['3+ شهور'])}). "
        f"ابدأ بشريحة 3+ لأنها الأكبر والأخطر.")


@q(45, "الدفع والتحصيل", "P2", "الخصومات والإعفاءات بتكلفنا كام؟")
def q45(c):
    P = c.P.dropna(subset=["monthly_fee"]).copy()
    P["cost"] = (P.monthly_fee - P.amount_due).clip(lower=0)
    ratio = (P.cost / P.monthly_fee).round(2)
    P["type"] = np.select([ratio >= 0.99, ratio.between(0.45, 0.55), ratio.between(0.15, 0.25)], ["إعفاء كامل", "منحة 50%", "خصم إخوات 20%"], "أخرى")
    d = P[P.cost > 0].groupby("type").agg(cost=("cost", "sum"), n=("student_id", "nunique")).sort_values("cost", ascending=False)
    fig = go.Figure(go.Bar(x=d.index, y=d.cost, text=[f"{eg(v)}<br>{n} طالب" for v, n in zip(d.cost, d.n)], marker_color=C["blue"], textposition="outside"))
    return fin(fig, "تكلفة الخصومات والإعفاءات على مدار السنة", 400, False), (
        f"الخصومات والإعفاءات كلفت {eg(d.cost.sum())} ({pc(d.cost.sum() / P.monthly_fee.sum())} من الإيراد المتوقع). راجع الإعفاء الكامل كل فترة وتأكد إن الحالة لسه مستحقة.")


@q(46, "الدفع والتحصيل", "P1", "الدخل الفعلي مقابل الدخل المتوقع كل شهر")
def q46(c):
    P = c.P
    d = P.groupby("m").agg(exp=("monthly_fee", "sum"), due=("amount_due", "sum"), paid=("amount_paid", "sum"))
    fig = go.Figure([go.Bar(x=d.index, y=d.exp, name="متوقع (سعر كامل)", marker_color=C["mist"], marker_line_color=C["blue"], marker_line_width=1),
                     go.Bar(x=d.index, y=d.due, name="مستحق بعد الخصومات", marker_color=C["cyan"]),
                     go.Bar(x=d.index, y=d.paid, name="فعلي", marker_color=C["blue"])])
    fig.update_xaxes(title="شهر الفاتورة", dtick=1); fig.update_layout(barmode="group")
    return fin(fig, "الدخل المتوقع والمستحق والفعلي", 420), (
        f"المتوقع {eg(d.exp.sum())} والفعلي {eg(d.paid.sum())} (الفجوة {eg(d.exp.sum() - d.paid.sum())} = {pc(1 - d.paid.sum() / d.exp.sum())}). "
        f"جزء من الفجوة خصومات مقصودة، والباقي متأخرات تقدر تحصّلها.")


@q(47, "الدفع والتحصيل", "P1", "الدخل الضايع بسبب التسرب والمتأخرات بالجنيه")
def q47(c):
    S = c.S
    dr = S[S.status == "dropped"]
    lost = ((9 - dr.last_month).clip(lower=0) * dr.typ_due.fillna(0)).sum()
    un = c.P[c.P.status == "unpaid"].owed.sum()
    pa = c.P[c.P.status == "partial"].owed.sum()
    fig = go.Figure(go.Bar(x=["التسرب (شهور فاضلة)", "غير مدفوع", "دفع جزئي (الباقي)"], y=[lost, un, pa], marker_color=[C["deep"], C["red"], C["amber"]],
                           text=[eg(lost), eg(un), eg(pa)], textposition="outside"))
    return fin(fig, "الدخل الضايع", 400, False), (
        f"الضايع {eg(lost + un + pa)} في السنة: تسرب {eg(lost)} ومتأخرات {eg(un + pa)}. "
        f"المتأخرات قابلة للتحصيل، والتسرب بيتقلل بالتدخل المبكر.")


# ---------------------------------------------------------------------------
# 6 · retention
# ---------------------------------------------------------------------------
@q(49, "التسرب والاحتفاظ", "P1", "منحنى الاحتفاظ: كام % من الطلاب لسه معانا بعد كل شهر؟")
def q49(c):
    S = c.S
    n = len(S)
    dr = S[S.status == "dropped"]
    ret = [1 - (dr.last_month <= m).sum() / n for m in range(0, 10)]
    fig = go.Figure(go.Scatter(x=list(range(0, 10)), y=ret, mode="lines+markers", line=dict(color=C["blue"], width=3, shape="hv"), fill="tozeroy", fillcolor="rgba(23,78,166,0.1)"))
    fig.update_yaxes(tickformat=".0%", range=[max(0, min(ret) - 0.05), 1.01]); fig.update_xaxes(title="شهر الدراسة", dtick=1)
    return fin(fig, "منحنى الاحتفاظ بالطلاب", 380, False), (
        f"بعد 9 شهور لسه معانا {pc(ret[-1])} من الطلاب. المنحنى بيبيّن الشهور اللي النزول فيها أسرع، وده وقت التدخل.")


@q(50, "التسرب والاحتفاظ", "P1", "الطلاب سابوا في أنهي شهر؟")
def q50(c):
    d = c.S[c.S.status == "dropped"].last_month.value_counts().sort_index().reindex(range(1, 10), fill_value=0)
    fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[C["red"] if v == d.max() else C["blue"] for v in d.values], text=d.values))
    fig.update_xaxes(title="آخر شهر حضره الطالب", dtick=1)
    return fin(fig, "التسرب حسب الشهر", 380, False), (
        f"أكتر شهر تسرب فيه الشهر {int(d.idxmax())} ({int(d.max())} طالب). راقب طلاب الشهر اللي قبله بشكل خاص.")


@q(51, "التسرب والاحتفاظ", "P1", "إيه العلامات المبكرة قبل ما الطالب يسيب (حضور ودرجات وواجب آخر 8 حصص)؟")
def q51(c):
    r, S = c.r, c.S
    dr = set(S[S.status == "dropped"].index)
    d = r[r.student_id.isin(dr)].copy()
    d["off"] = d.groupby("student_id").cumcount(ascending=False) + 1
    d = d[d.off <= 8]
    base = r[~r.student_id.isin(dr)]
    mets = [("الحضور", "att", base.att.mean()), ("الدرجات", "percentage", base.percentage.mean()), ("الواجب", "submitted", base[base.att].submitted.mean())]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[0] for m in mets])
    out = []
    for i, (nm, col, b) in enumerate(mets, 1):
        dd = d[d.att] if col == "submitted" else d
        y = dd.groupby("off")[col].mean().sort_index(ascending=False)
        fig.add_trace(go.Scatter(x=-y.index, y=y.values, mode="lines+markers", line=dict(color=C["red"], width=3)), 1, i)
        fig.add_hline(y=b, line_dash="dash", line_color=C["grey"], row=1, col=i)
        fig.update_yaxes(tickformat=".0%", row=1, col=i)
        out.append((nm, y.values[-1], b))
    fig.update_xaxes(title="الحصة قبل آخر حضور")
    pm = (S[S.status == "dropped"].unpaid_m >= 2).mean(), (S[S.status != "dropped"].unpaid_m >= 2).mean()
    return fin(fig, "آخر 8 حصص قبل التسرب (الخط المتقطع = متوسط باقي الطلاب)", 400, False), (
        "في آخر حصة: " + "، ".join(f"{n} {pc(v)} مقابل {pc(b)}" for n, v, b in out) +
        f". (تأخر الدفع شهرين+: {pc(pm[0])} للمتسربين مقابل {pc(pm[1])} للباقي، يعني مش إشارة قوية.) نزول الحضور والواجب في آخر حصص هو الإنذار المبكر الحقيقي.")


# ---------------------------------------------------------------------------
# 7 · groups
# ---------------------------------------------------------------------------
@q(54, "المجموعات", "P1", "كارت تقييم لكل مجموعة (حضور، متوسط، تحصيل، متسربين، خطر)")
def q54(c):
    G = c.G
    cols = [("att", "الحضور", 1, pc), ("avg", "المتوسط", 1, pc), ("collection", "التحصيل", 1, pc), ("dropped", "متسربين", -1, lambda v: f"{int(v)}"), ("risk_n", "في خطر", -1, lambda v: f"{int(v)}")]
    z = pd.DataFrame({k: (G[k] - G[k].min()) / (G[k].max() - G[k].min()) * (1 if s > 0 else -1) + (0 if s > 0 else 1) for k, _, s, _ in cols})
    order = z.mean(axis=1).sort_values().index
    fig = go.Figure(go.Heatmap(z=z.loc[order].values, x=[n for _, n, _, _ in cols], y=G.loc[order, "nm"], colorscale="RdYlGn", showscale=False,
                               text=[[f(G.at[i, k]) for k, _, _, f in cols] for i in order], texttemplate="%{text}"))
    return fin(fig, "كارت تقييم المجموعات (الأحمر أضعف، مرتبة من الأضعف)", 900, False), (
        f"أضعف مجموعة إجمالًا {G.nm[order[0]]} وأقواها {G.nm[order[-1]]}. الصفوف الحمرا في أكتر من عمود هي أولويتك في الأسبوع.")


@q(56, "المجموعات", "P2", "إيراد كل مجموعة كام؟")
def q56(c):
    G = c.G.sort_values("paid")
    fig = go.Figure([go.Bar(y=G.nm, x=G.expected, name="متوقع", orientation="h", marker_color=C["mist"], marker_line_color=C["blue"], marker_line_width=1),
                     go.Bar(y=G.nm, x=G.paid, name="محصّل", orientation="h", marker_color=C["blue"])])
    fig.update_layout(barmode="overlay")
    return fin(fig, "إيراد كل مجموعة (السنة كلها)", 900), (
        f"أعلى مجموعة إيرادًا {G.nm.iloc[-1]} ({eg(G.paid.iloc[-1])}) وأقلها {G.nm.iloc[0]} ({eg(G.paid.iloc[0])}). المجموعات الأقل غالبًا أصغر أو سعرها أقل أو تحصيلها ضعيف.")


@q(58, "المجموعات", "P3", "الملتحقين متأخر بيلحقوا بالمستوى؟")
def q58(c):
    late = set(c.S[c.S.first_session >= 5].index)
    d = c.sc.assign(lj=c.sc.student_id.isin(late)).groupby(["lj", "student_month"]).percentage.mean().reset_index()
    fig = go.Figure()
    for flag, name, col in [(False, "طلاب من البداية", C["blue"]), (True, f"ملتحقين متأخر ({len(late)})", C["red"])]:
        x = d[d.lj == flag]
        fig.add_trace(go.Scatter(x=x.student_month, y=x.percentage, name=name, mode="lines+markers", line=dict(color=col, width=3)))
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(title="شهر الطالب (من أول يوم له)", dtick=1)
    return fin(fig, "الملتحقين متأخر مقابل الباقي", 400), (
        f"{len(late)} طالب التحقوا بعد الشهر الأول. متوسطهم في أول شهر ليهم {pc(d[d.lj].percentage.iloc[0])} مقابل {pc(d[~d.lj].percentage.iloc[0])}. "
        f"لو الفرق بيضيق مع الوقت يبقى الاستقبال كويس، ولو لأ جهّز لهم حصة تأسيس.")


@q(59, "المجموعات", "P2", "أحسن وأسوأ شهر في السنة (حضور ودفع)؟")
def q59(c):
    a = c.r[c.r.ym < "2026-07-01"].groupby("ym").att.mean()
    P = c.P.assign(ym=c.P.due_date.dt.to_period("M").dt.to_timestamp()).groupby("ym").agg(p=("amount_paid", "sum"), d=("amount_due", "sum"))
    pr = (P.p / P.d).dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("الحضور", "التحصيل (حسب شهر الاستحقاق)"))
    for i, s in enumerate([a, pr], 1):
        fig.add_trace(go.Bar(x=s.index.strftime("%m/%Y"), y=s.values, marker_color=[C["green"] if v == s.max() else C["red"] if v == s.min() else C["blue"] for v in s.values],
                            text=[pc(v) for v in s.values]), 1, i)
        fig.update_yaxes(tickformat=".0%", range=[s.min() - 0.03, 1.0], row=1, col=i)
    return fin(fig, "أحسن وأسوأ شهر", 400, False), (
        f"أحسن حضور {a.idxmax():%m/%Y} ({pc(a.max())}) وأسوأ {a.idxmin():%m/%Y} ({pc(a.min())}). أحسن تحصيل {pr.idxmax():%m/%Y} وأسوأ {pr.idxmin():%m/%Y}. "
        f"جهّز حملة متابعة قبل الأشهر الضعيفة.")


# ---------------------------------------------------------------------------
# 8 · creative / teacher-facing
# ---------------------------------------------------------------------------
def call_list(c) -> pd.DataFrame:
    """Students the teacher should phone this week (rule-based, transparent reasons)."""
    S = c.S[c.S.status == "active"].copy()
    rules_ = {"في خطر": (S.at_risk, 3), "غايب آخر حصتين": (S.cur_run >= 2, 2),
              "حضور آخر 4 ≤50%": (S.att_recent <= 0.5, 2), "درجات نازلة 10+ نقاط": ((S.avg - S.pct_recent) >= 0.10, 2),
              "متأخر دفع شهرين+": (S.unpaid_m >= 2, 2), "على حافة الخطر": ((S.gap >= 0) & (S.gap <= 0.05), 1)}
    flags = pd.DataFrame({k: v[0].fillna(False).astype(bool) for k, v in rules_.items()})
    S["score"] = sum(flags[k].astype(int) * rules_[k][1] for k in rules_)
    S["why"] = flags.apply(lambda row: " + ".join(k for k in flags.columns if row[k]), axis=1)
    return S[S.score > 0].sort_values(["score", "att_rate"], ascending=[False, True])


def _table(df, cols, widths=None):
    return go.Table(header=dict(values=cols, fill_color=C["midnight"], font=dict(color="white", size=13), align="right"),
                    cells=dict(values=[df[k] for k in df.columns], fill_color="#F6F9FB", align="right", height=28), columnwidth=widths)


@q(61, "أسئلة إبداعية", "P1", "مين الطلاب اللي لازم أكلم أولياء أمورهم الأسبوع ده؟ (قائمة اتصال)")
def q61(c):
    cl = call_list(c)
    t = cl.head(20)
    df = pd.DataFrame({"الطالب": t.name.fillna(pd.Series(t.index, index=t.index)), "المجموعة": t.group_id, "تليفون ولي الأمر": t.parent_phone.fillna("غير مسجل"),
                       "السبب": t.why, "الأولوية": np.where(t.score >= 6, "عاجل", np.where(t.score >= 4, "مهم", "متابعة"))})
    fig = go.Figure(_table(df, list(df.columns), [150, 70, 110, 330, 70]))
    return fin(fig, "قائمة الاتصال لولي الأمر (أعلى 20 أولوية)", 880, False), (
        f"{len(cl)} طالب ليهم سبب للمكالمة، منهم {(cl.score >= 6).sum()} عاجل. اتصل بالعاجل الأول وسجل النتيجة عشان القايمة الجاية تتحدّث.")


def _scatter_table(c, mask, df_cols, title, x, y, xl, yl, label):
    S = c.S.dropna(subset=["avg"])
    sel = S[mask(S)]
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "xy"}], [{"type": "table"}]], row_heights=[0.58, 0.42], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=S[x], y=S[y], mode="markers", marker=dict(color=C["grey"], opacity=0.4, size=5), name="باقي الطلاب", text=_label(S, S.index)), 1, 1)
    fig.add_trace(go.Scatter(x=sel[x], y=sel[y], mode="markers", marker=dict(color=C["red"], size=8), name=label, text=_label(S, sel.index)), 1, 1)
    t = sel.sort_values(df_cols[2]).head(10)
    fig.add_trace(go.Table(header=dict(values=["الطالب", "المجموعة", xl, yl], fill_color=C["midnight"], font=dict(color="white")),
                           cells=dict(values=[_label(S, t.index), t.group_id, t[x].map(pc) if x != "hw_mean" else t[x].round(1), t[y].map(pc)], fill_color="#F6F9FB", align="right")), 2, 1)
    fig.update_xaxes(title=xl, row=1, col=1); fig.update_yaxes(title=yl, row=1, col=1)
    if x != "hw_mean":
        fig.update_xaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return fin(fig, title, 760), sel


@q(62, "أسئلة إبداعية", "P1", "مين الطلاب الموهوبين اللي غيابهم بيضيّع مستواهم؟")
def q62(c):
    fig, sel = _scatter_table(c, lambda S: (S.avg >= 0.85) & (S.att_rate < 0.85), ["", "", "att_rate"], "موهوبين وغيابهم عالي (درجات ≥85% وحضور <85%)", "att_rate", "avg", "الحضور", "متوسط الدرجات", "موهوب بيغيب")
    return fig, (f"{len(sel)} طالب درجاتهم عالية رغم الغياب، يعني قدرتهم أعلى من نتيجتهم. كلّم أولياء أمورهم عن الحضور فهم أسهل مكسب في المستوى.")


@q(63, "أسئلة إبداعية", "P1", "مين المجتهدين (واجب ممتاز) بس درجتهم ضعيفة ومحتاجين شرح مختلف؟")
def q63(c):
    S = c.S.dropna(subset=["avg", "hw_mean"])
    h, a = S.hw_mean.quantile(0.75), S.avg.quantile(0.25)
    fig, sel = _scatter_table(c, lambda S: (S.hw_mean >= h) & (S.avg <= a), ["", "", "avg"], "مجتهدين بس درجتهم ضعيفة (واجب في أعلى ربع ودرجات في أقل ربع)", "hw_mean", "avg", "متوسط تقدير الواجب (0-5)", "متوسط الدرجات", "مجتهد ودرجاته ضعيفة")
    return fig, (f"{len(sel)} طالب بيذاكروا وبيسلموا لكن درجاتهم ضعيفة. المشكلة في طريقة الفهم مش الجهد، جرّب معاهم شرح مختلف أو حصة فردية قصيرة.")


@q(64, "أسئلة إبداعية", "P1", "مين الطلاب على حافة الخطر (قريبين من الحد)؟")
def q64(c):
    S = c.S[(c.S.status == "active") & (c.S.gap >= 0) & (c.S.gap <= 0.05)].sort_values("gap").head(20)
    df = pd.DataFrame({"الطالب": _label(c.S, S.index), "المجموعة": S.group_id, "الحضور": S.att_rate.map(pc), "الدرجات": S.avg.map(pc), "المسافة عن الخطر (نقطة)": (S.gap * 100).round(1)})
    n = ((c.S.status == "active") & (c.S.gap >= 0) & (c.S.gap <= 0.05)).sum()
    fig = go.Figure(_table(df, list(df.columns)))
    return fin(fig, "طلاب على حافة الخطر (الحد: حضور 80% ودرجات 75%)", 700, False), (
        f"{n} طالب نشط على بعد 5 نقاط أو أقل من تصنيف «في خطر». رسالة تشجيع واحدة قبل ما يوقعوا أوفر من علاج بعد الوقوع.")


@q(65, "أسئلة إبداعية", "P2", "الأداء نزل في رمضان أو الامتحانات؟ (أثر المواسم)")
def q65(c):
    w = c.sc.groupby("week").percentage.mean()
    fig = go.Figure(go.Scatter(x=w.index, y=w.values, mode="lines+markers", line=dict(color=C["blue"], width=3)))
    _season(fig, c)
    fig.update_yaxes(tickformat=".0%")
    ram = w[(w.index >= RAMADAN[0] - pd.Timedelta(days=6)) & (w.index <= RAMADAN[1])].mean()
    oth = w.drop(w[(w.index >= RAMADAN[0] - pd.Timedelta(days=6)) & (w.index <= RAMADAN[1])].index).mean()
    return fin(fig, "متوسط الدرجات الأسبوعي والمواسم", 420, False), (
        f"في رمضان متوسط الدرجات {pc(ram)} مقابل {pc(oth)} باقي السنة (فرق {(ram - oth) * 100:+.1f} نقطة). " +
        ("الأداء بيتأثر برمضان، خفف حمل الواجب وقتها." if ram < oth - 0.01 else "مفيش أثر واضح لرمضان على الدرجات."))


@q(66, "أسئلة إبداعية", "P3", "الملاحظات اللي بكتبها بتتكرر على إيه؟")
def q66(c):
    d = c.sc.dropna(subset=["notes"]).groupby("notes").agg(n=("notes", "size"), p=("percentage", "mean")).sort_values("n")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(y=d.index, x=d.n, orientation="h", marker_color=C["blue"], name="التكرار"))
    fig.add_trace(go.Scatter(y=d.index, x=d.p, mode="markers", marker=dict(color=C["red"], size=11), name="متوسط الدرجة"), secondary_y=True)
    fig.update_yaxes(showticklabels=False, secondary_y=True)
    return fin(fig, "الملاحظات المكتوبة وارتباطها بالدرجة", 430), (
        f"أكتر ملاحظة «{d.index[-1]}» ({int(d.n.iloc[-1])} مرة). الملاحظات بتتبع الدرجة غالبًا، فاستخدمها كفلتر سريع للمتابعة بدل ما تقرا كل الحصص.")


@q(67, "أسئلة إبداعية", "P1", "طالب بالنسبة لزمايله: ترتيبه في مجموعته (percentile)")
def q67(c, student_id=None):
    S = c.S.dropna(subset=["avg"])
    if student_id is None:
        student_id = S[S.at_risk].sort_values("avg").index[0]
    g = S[S.group_id == S.at[student_id, "group_id"]].sort_values("avg")
    pct = (g.avg < g.at[student_id, "avg"]).mean() * 100
    fig = go.Figure(go.Bar(x=list(range(len(g))), y=g.avg, marker_color=[C["red"] if i == student_id else C["blue"] for i in g.index], hovertext=_label(S, g.index)))
    fig.update_yaxes(tickformat=".0%"); fig.update_xaxes(showticklabels=False, title="طلاب المجموعة مرتبين")
    return fin(fig, f"ترتيب {g.at[student_id, 'name']} داخل مجموعته", 400, False), (
        f"الطالب أحسن من {pct:.0f}% من زمايله (متوسطه {pc(g.at[student_id, 'avg'])} ومتوسط المجموعة {pc(g.avg.mean())}). "
        f"في الداشبورد تقدر تختار أي طالب وتشوف ترتيبه قدام ولي الأمر.")


def group_flags(c) -> pd.DataFrame:
    G = c.G
    bad = pd.DataFrame({"الحضور": G.att, "الدرجات": G.avg, "التحصيل": G.collection, "الواجب": G.hw_rate, "نسبة الخطر": -G.risk_share})
    flag = bad.rank(pct=True).le(1 / 3).sum(axis=1)
    return G.assign(nflags=flag).sort_values("nflags", ascending=False)


@q(71, "أسئلة إبداعية", "P1", "مجموعات محتاجة تدخل (كل مؤشراتها نازلة)")
def q71(c):
    G = group_flags(c)
    t = G[G.nflags >= 3].head(15)
    df = pd.DataFrame({"المجموعة": t.nm, "مؤشرات نازلة (من 5)": t.nflags.astype(int), "الحضور": t.att.map(pc), "الدرجات": t.avg.map(pc), "التحصيل": t.collection.map(pc), "الواجب": t.hw_rate.map(pc), "في خطر": t.risk_n.astype(int)})
    fig = go.Figure(_table(df, list(df.columns)))
    return fin(fig, "مجموعات محتاجة تدخل (3+ مؤشرات في أضعف ثلث)", 560, False), (
        f"{len(G[G.nflags >= 3])} مجموعة في أضعف ثلث على 3 مؤشرات أو أكتر. افتح معاهم حصة مراجعة وتواصل مع أولياء الأمور كمجموعة.")


@q(72, "أسئلة إبداعية", "P1", "ملخص الأسبوع للمدرس: 3 قرارات (مين، مجموعة، فلوس)")
def q72(c):
    cl = call_list(c)
    s = cl.iloc[0]
    g = group_flags(c).iloc[0]
    top = c.S.sort_values("owed", ascending=False).iloc[0]
    cards = [("مين تكلم الأسبوع ده", f"{s['name']}<br>{s['why'].replace(' + ', '<br>')}", C["red"]),
             ("أنهي مجموعة تتدخل فيها", f"{g['nm']}<br>{int(g['nflags'])} مؤشرات من 5<br>في أضعف ثلث", C["amber"]),
             ("فلوس تحصّلها", f"{top['name']}: {eg(top['owed'])}<br>إجمالي المتأخرات {eg(c.S.owed.sum())}", C["blue"])]
    fig = go.Figure()
    for i, (t, b, col) in enumerate(cards):
        x0 = i / 3 + 0.01
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0, x1=x0 + 0.31, y0=0.05, y1=0.95, fillcolor=C["mist"], line=dict(color=col, width=3))
        fig.add_annotation(xref="paper", yref="paper", x=x0 + 0.155, y=0.86, text=f"<b>{t}</b>", showarrow=False, font=dict(size=15, color=col))
        fig.add_annotation(xref="paper", yref="paper", x=x0 + 0.155, y=0.45, text=b, showarrow=False, font=dict(size=13, color=C["midnight"]), align="center")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fin(fig, "ملخص الأسبوع: 3 قرارات", 360, False), (
        "ثلاث قرارات تكفي أسبوعك: مكالمة واحدة بأولوية، مجموعة واحدة تتدخل فيها، ومبلغ واحد تلاحقه. باقي القوائم في الداشبورد.")


# ---------------------------------------------------------------------------
def run(ctx, ids=None) -> dict[int, tuple]:
    out = {}
    for n in (ids or sorted(_FUNCS)):
        out[n] = _FUNCS[n](ctx)
    return out
