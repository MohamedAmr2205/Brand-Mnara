"""منارة · Mnara — داشبورد المدرس وبوابة الطالب وولي الأمر.

    streamlit run app/app.py

Login:
  - Teacher: password from Streamlit secrets (TEACHER_PASSWORD).
  - Student / parent: student code + the parent's (or student's) full phone number.
Everything is read from data/clean and the shared src/ modules, so the numbers match
the notebooks.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

# the app can live in the repo root or in app/: find the folder that holds src/
ROOT = Path(__file__).resolve().parent
while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import eda, rules
from src.data_loader import load_clean
from src.features import build_features
from src.forecast import next_month_forecast
from src.segments import INSUFFICIENT, load_bundle, predict_segment

st.set_page_config(page_title="منارة | Mnara", page_icon="🔦", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# brand
# ---------------------------------------------------------------------------
MID, DEEP, BLUE, CYAN, MINT, MIST = "#071A3D", "#0D2A5E", "#174EA6", "#00D4FF", "#73D9C2", "#E8EFF3"
RED, AMBER, GREEN, GREY, SOFT, BLUE_L = "#F0656A", "#FFB020", "#2FBF8F", "#8A97AB", "#A9BCD3", "#5B8DEF"

SEG = {
    "Sharply Declining": ("متراجع بسرعة", RED),
    "Struggling": ("محتاج دعم", AMBER),
    "Rising": ("في صعود", MINT),
    "Solid Regulars": ("ثابت ومنتظم", BLUE_L),
    "Top Performers": ("متفوق", GREEN),
    INSUFFICIENT: ("بيانات غير كافية", GREY),
}
SEG_FOR_STUDENT = {
    "Sharply Declining": "درجاتك نزلت الفترة اللي فاتت. المدرس هيتابع معاك، ومتترددش تسأل في الحصة.",
    "Struggling": "محتاج تركيز أكتر على الحضور والواجب. كل حصة بتفرق.",
    "Rising": "درجاتك بتطلع. كمّل كده!",
    "Solid Regulars": "مستواك ثابت وكويس. حاول تزوّد شوية في الامتحانات.",
    "Top Performers": "مستواك ممتاز. استمر!",
    INSUFFICIENT: "لسه البيانات قليلة، والصورة هتبقى أوضح بعد كام أسبوع.",
}
ATT_AR = {"present": "حاضر", "late": "متأخر", "absent": "غايب", "excused": "غايب بعذر"}
ATT_COL = {"present": GREEN, "late": AMBER, "absent": RED, "excused": GREY}
PAY_AR = eda.STATUS_AR
PAY_COL = {"paid": GREEN, "paid_late": AMBER, "partial": AMBER, "unpaid": RED, "exempt": GREY}
MAX_TRIES = 5

LOGO = (
    '<svg viewBox="0 0 40 84" width="{w}" height="{h}" aria-hidden="true"><defs><linearGradient id="{gid}" x1="0" y1="1" x2="0" y2="0">'
    '<stop offset="0" stop-color="#00D4FF" stop-opacity=".95"/><stop offset=".55" stop-color="#00D4FF" stop-opacity=".35"/>'
    '<stop offset="1" stop-color="#00D4FF" stop-opacity="0"/></linearGradient></defs>'
    '<polygon points="13.5,56 26.5,56 36,0 4,0" fill="url(#{gid})"/>'
    '<rect x="9" y="57" width="22" height="22" rx="7" transform="rotate(45 20 68)" fill="#E8EFF3"/></svg>'
)


def logo(w=22, gid="lg"):
    return LOGO.format(w=w, h=round(w * 2.1), gid=gid)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-baseweb="tab"], .stApp [data-baseweb="select"] div {
  font-family: 'IBM Plex Sans Arabic', Tahoma, sans-serif !important; }
.stApp { direction: rtl; background: radial-gradient(120% 70% at 50% -10%, #0D2A5E 0%, #071A3D 58%) fixed; }
.stApp::before { content: ""; position: fixed; top: 0; left: 50%; width: 560px; height: 80vh; transform: translateX(-50%);
  background: linear-gradient(180deg, rgba(0,212,255,.26), rgba(0,212,255,.05) 60%, rgba(0,212,255,0));
  clip-path: polygon(45% 0, 55% 0, 100% 100%, 0 100%); filter: blur(8px); pointer-events: none; z-index: 0; }
.stApp::after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0; opacity: .55;
  background-image: radial-gradient(1.2px 1.2px at 12% 18%, #E8EFF3 50%, transparent 51%), radial-gradient(1px 1px at 72% 12%, #E8EFF3 50%, transparent 51%),
    radial-gradient(1.4px 1.4px at 86% 44%, #73D9C2 50%, transparent 51%), radial-gradient(1px 1px at 32% 66%, #E8EFF3 50%, transparent 51%),
    radial-gradient(1.2px 1.2px at 56% 82%, #00D4FF 50%, transparent 51%), radial-gradient(1px 1px at 8% 88%, #E8EFF3 50%, transparent 51%),
    radial-gradient(1px 1px at 44% 30%, #E8EFF3 50%, transparent 51%), radial-gradient(1.2px 1.2px at 94% 76%, #E8EFF3 50%, transparent 51%);
  background-size: 520px 520px; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { position: relative; z-index: 1; max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { direction: ltr; }
.stTextInput input { direction: rtl; text-align: right; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: rgba(13,42,94,.55); border: 1px solid rgba(232,239,243,.12); border-radius: 99px;
  padding: 8px 18px; height: auto; color: #A9BCD3; }
.stTabs [aria-selected="true"] { background: #E8EFF3 !important; color: #06142E !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stButton > button, .stFormSubmitButton > button { border-radius: 99px; font-weight: 600; min-height: 44px; }
.stFormSubmitButton > button { background: #00D4FF; color: #04142F; border: 0; }
[data-testid="stForm"] { background: rgba(13,42,94,.55); border: 1px solid rgba(232,239,243,.12); border-radius: 22px; padding: 22px; }
[data-testid="stExpander"] { background: rgba(13,42,94,.4); border-radius: 16px; }

.mn-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
.mn-brand { display: flex; align-items: center; gap: 12px; direction: ltr; }
.mn-brand b { display: block; font-size: 24px; line-height: 1; color: #E8EFF3; }
.mn-brand small { display: block; font-size: 15px; color: #00D4FF; font-weight: 600; text-align: left; }
.mn-asof { font-size: 14px; color: #8EA3C0; }
.mn-eye { color: #73D9C2; font-weight: 600; font-size: 15px; margin: 0 0 4px; }
.mn-h1 { font-size: clamp(28px, 4vw, 44px); font-weight: 700; line-height: 1.2; margin: 0 0 6px; color: #E8EFF3; }
.mn-lead { color: #A9BCD3; font-size: 17px; margin: 0 0 18px; }
.mn-sec { font-size: 22px; font-weight: 700; margin: 26px 0 12px; color: #E8EFF3; }
.mn-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 6px 0 8px; }
.mn-kpi { background: rgba(13,42,94,.75); border: 1px solid rgba(232,239,243,.08); border-radius: 18px; padding: 16px 18px; position: relative; overflow: hidden; }
.mn-kpi::before { content: ""; position: absolute; inset: 0 0 auto auto; width: 4px; height: 100%; background: var(--c, #73D9C2); }
.mn-kpi .v { font-size: 30px; font-weight: 700; line-height: 1.1; color: #E8EFF3; }
.mn-kpi .l { font-size: 14px; color: #A9BCD3; margin-top: 4px; }
.mn-kpi .s { font-size: 12.5px; color: #7F93AE; margin-top: 2px; }
.mn-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
.mn-card { background: rgba(13,42,94,.75); border: 1px solid rgba(232,239,243,.08); border-radius: 20px; padding: 18px 20px; }
.mn-card h4 { margin: 0 0 6px; font-size: 17px; color: var(--c, #73D9C2); }
.mn-card p { margin: 0; color: #C9D7E6; font-size: 15px; line-height: 1.6; }
.mn-card .big { font-size: 19px; font-weight: 700; color: #E8EFF3; margin-bottom: 4px; }
.mn-list { display: flex; flex-direction: column; gap: 8px; }
.mn-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; background: rgba(13,42,94,.7);
  border: 1px solid rgba(232,239,243,.07); border-radius: 16px; padding: 12px 16px; }
.mn-row b { color: #E8EFF3; font-size: 16px; }
.mn-row small { display: block; color: #93A7C2; font-size: 13.5px; line-height: 1.5; }
.mn-row .end { display: flex; align-items: center; gap: 10px; flex: none; }
.mn-row a { color: #00D4FF; text-decoration: none; font-size: 14px; direction: ltr; }
.chip { display: inline-block; font-size: 12.5px; font-weight: 600; padding: 3px 11px; border-radius: 99px; color: var(--c);
  background: color-mix(in srgb, var(--c) 16%, transparent); border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); white-space: nowrap; }
.mn-note { font-size: 13px; color: #7F93AE; margin-top: 8px; }
.mn-say { border-right: 3px solid #00D4FF; padding: 10px 14px; background: rgba(0,212,255,.06); border-radius: 10px; color: #E8EFF3; margin: 8px 0 4px; }
.mn-login { max-width: 460px; margin: 4vh auto 0; text-align: center; }
.mn-login .lock { display: flex; flex-direction: column; align-items: center; gap: 6px; margin-bottom: 14px; }
.mn-login .lock b { font-size: 46px; line-height: 1; direction: ltr; }
.mn-login .lock span { font-size: 30px; color: #00D4FF; font-weight: 700; }
@media (max-width: 640px) {
  .mn-row { flex-direction: column; align-items: flex-start; }
  .mn-kpi .v { font-size: 26px; }
  .stApp::before { width: 340px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def e(x) -> str:
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


def pc(x, d=0) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def egp(x) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f} ج"


def chip(text, color) -> str:
    return f'<span class="chip" style="--c:{color}">{e(text)}</span>'


def kpis(items) -> None:
    cells = "".join(
        f'<div class="mn-kpi" style="--c:{c}"><div class="v">{e(v)}</div><div class="l">{e(l)}</div>'
        + (f'<div class="s">{e(s)}</div>' if s else "") + "</div>"
        for v, l, s, c in items)
    st.markdown(f'<div class="mn-kpis">{cells}</div>', unsafe_allow_html=True)


def section(title) -> None:
    st.markdown(f'<div class="mn-sec">{e(title)}</div>', unsafe_allow_html=True)


def dark(fig: go.Figure, h: int | None = None) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="IBM Plex Sans Arabic, Tahoma, sans-serif", color=MIST, size=13),
                      margin=dict(l=40, r=20, t=50, b=40), hoverlabel=dict(font_family="IBM Plex Sans Arabic"))
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title=dict(font=dict(color=MIST, size=16), x=1, xanchor="right"))
    fig.update_xaxes(gridcolor="rgba(232,239,243,.08)", zerolinecolor="rgba(232,239,243,.12)")
    fig.update_yaxes(gridcolor="rgba(232,239,243,.08)", zerolinecolor="rgba(232,239,243,.12)")

    def _tbl(t):
        if t.type == "table":
            t.update(header=dict(fill_color=DEEP, font=dict(color=MIST, size=13), line_color=MID),
                     cells=dict(fill_color="#0B2350", font=dict(color="#C9D7E6", size=12.5), line_color=MID))
    fig.for_each_trace(_tbl)
    if h:
        fig.update_layout(height=h)
    return fig


def show_fig(fig) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg)


def show_df(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", (s or "").translate(_DIGITS))
    if d.startswith("0020"):
        d = d[4:]
    if d.startswith("20") and len(d) == 12:
        d = d[2:]
    if len(d) == 10 and d.startswith("1"):
        d = "0" + d
    return d


def norm_code(s: str) -> str:
    s = (s or "").translate(_DIGITS).strip().upper().replace(" ", "")
    m = re.fullmatch(r"S?0*(\d{1,4})", s)
    return f"S{int(m.group(1)):04d}" if m else s


def secret(key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets[key]
    except Exception:
        return default


# ---------------------------------------------------------------------------
# data (loaded once per server)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="بنجهّز الداتا…")
def load():
    clean = load_clean()
    ctx = eda.build(clean)
    seg = predict_segment(build_features(clean), load_bundle())
    fc, fc_metrics = next_month_forecast(clean)
    S = ctx.S.join(seg[["segment", "priority", "action"]]).join(fc[["predicted", "change", "last_month_avg"]])
    S["segment"] = S["segment"].fillna(INSUFFICIENT)
    ctx.S = S
    return ctx, fc_metrics


ctx, FC = load()
S = ctx.S
AS_OF = ctx.as_of


@st.cache_resource(show_spinner=False)
def qfig(n: int):
    fig, takeaway = eda._FUNCS[n](ctx)
    return dark(fig), takeaway


def eda_block(ids, note: bool = True) -> None:
    """Render EDA questions (same functions as notebooks/02_eda.ipynb)."""
    if note:
        st.markdown('<p class="mn-note">الرسومات دي من نفس أسئلة الـ EDA، ومحسوبة على كل الطلاب والمجموعات مش على الفلتر.</p>', unsafe_allow_html=True)
    for n in ids:
        q = eda.QUESTIONS[n]
        fig, takeaway = qfig(n)
        st.markdown(f'<div class="mn-sec" style="font-size:19px">{e(q["title"])}</div><div class="mn-say">{e(takeaway)}</div>',
                    unsafe_allow_html=True)
        show_fig(fig)


def topbar(title_right: str = "") -> None:
    st.markdown(
        f'<div class="mn-top"><div class="mn-brand">{logo(22, "tb")}<div><b>Mnara</b><small>منارة</small></div></div>'
        f'<div class="mn-asof">{e(title_right)}آخر تحديث للبيانات: {AS_OF:%d/%m/%Y}</div></div>', unsafe_allow_html=True)


def logout_button(key: str) -> None:
    if st.button("خروج", key=key):
        for k in ("role", "sid"):
            st.session_state.pop(k, None)
        st.rerun()


# ---------------------------------------------------------------------------
# login
# ---------------------------------------------------------------------------
def login() -> None:
    st.markdown(
        f'<div class="mn-login"><div class="lock">{logo(40, "lg1")}<b>Mnara</b><span>منارة</span></div>'
        '<p class="mn-lead">نوّر على طلابك.</p></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        t_student, t_teacher = st.tabs(["الطالب وولي الأمر", "المدرس"])
        tries = st.session_state.get("tries", 0)
        with t_student:
            with st.form("student_login"):
                code = st.text_input("كود الطالب", placeholder="مثال: S0001")
                phone = st.text_input("رقم موبايل ولي الأمر أو الطالب", placeholder="01xxxxxxxxx")
                ok = st.form_submit_button("دخول", disabled=tries >= MAX_TRIES)
            if tries >= MAX_TRIES:
                st.error("محاولات كتير غلط. كلّم المدرس عشان يتأكد من الكود والرقم.")
            elif ok:
                sid, ph = norm_code(code), norm_phone(phone)
                row = S.loc[sid] if sid in S.index else None
                valid = row is not None and len(ph) == 11 and ph in {norm_phone(str(row.get("parent_phone", ""))), norm_phone(str(row.get("phone", "")))}
                if valid:
                    st.session_state.update(role="student", sid=sid, tries=0)
                    st.rerun()
                else:
                    st.session_state["tries"] = tries + 1
                    st.error("الكود أو الرقم مش مظبوط. اتأكد منهم وجرّب تاني.")
            st.markdown('<p class="mn-note">الكود والرقم موجودين عند المدرس. الطالب بيشوف بياناته هو بس.</p>', unsafe_allow_html=True)
        with t_teacher:
            with st.form("teacher_login"):
                pw = st.text_input("الباسورد", type="password")
                ok_t = st.form_submit_button("دخول", disabled=tries >= MAX_TRIES)
            if ok_t:
                if pw and pw == secret("TEACHER_PASSWORD", "mnara2026"):
                    st.session_state.update(role="teacher", tries=0)
                    st.rerun()
                else:
                    st.session_state["tries"] = tries + 1
                    st.error("الباسورد غلط.")


# ---------------------------------------------------------------------------
# one student (shared by the teacher's student page and the student portal)
# ---------------------------------------------------------------------------
def student_view(sid: str, teacher: bool) -> None:
    s = S.loc[sid]
    seg_ar, seg_c = SEG.get(s["segment"], SEG[INSUFFICIENT])
    grp = ctx.G.loc[s["group_id"]] if s["group_id"] in ctx.G.index else None
    first = str(s["name"]).split()[0] if isinstance(s["name"], str) else sid

    st.markdown(
        f'<p class="mn-eye">{e(grp["group_name"] if grp is not None else s["group_id"])} · {e(s.get("grade", ""))}</p>'
        f'<h1 class="mn-h1">{"" if teacher else "أهلاً، "}{e(s["name"] if teacher else first)}</h1>'
        f'<div>{chip(seg_ar, seg_c)} {chip("في خطر", RED) if bool(s.get("at_risk")) else ""}'
        f' {chip("ساب المجموعة", GREY) if s.get("status") == "dropped" else ""}</div>', unsafe_allow_html=True)
    msg = s["action"] if teacher else SEG_FOR_STUDENT.get(s["segment"], "")
    if isinstance(msg, str) and msg:
        st.markdown(f'<div class="mn-say">{e(msg)}</div>', unsafe_allow_html=True)

    g_avg = S.loc[S.group_id == s["group_id"], "avg"]
    rank = (g_avg < s["avg"]).mean() if pd.notna(s["avg"]) and len(g_avg) else np.nan
    pred = s.get("predicted")
    ch = s.get("change")
    kpis([
        (pc(s["att_rate"]), "الحضور", f"آخر 4 حصص: {pc(s['att_recent'])}", GREEN if s["att_rate"] >= rules.AT_RISK_ATTENDANCE_MAX else RED),
        (pc(s["avg"]), "متوسط الدرجات", f"آخر 4 حصص: {pc(s['pct_recent'])}", GREEN if s["avg"] >= rules.AT_RISK_AVG_MAX else AMBER),
        (pc(s["hw_rate"]), "سلّم الواجب", f"متوسط التقدير {s['hw_mean']:.1f} من 5" if pd.notna(s["hw_mean"]) else "", MINT),
        (pc(pred), "الدرجة المتوقعة الشهر الجاي", (f"{ch * 100:+.1f} نقطة عن آخر شهر" if pd.notna(ch) else ""), CYAN),
        (pc(rank), "أحسن من زمايله في المجموعة", "حسب متوسط الدرجات", BLUE_L),
    ] + ([(egp(s["owed"]), "متأخرات الدفع", f"{int(s['unpaid_m'])} شهر", RED if s["owed"] > 0 else GREEN)] if teacher or s["owed"] > 0 else []))

    # monthly chart: student vs group
    r = ctx.r[ctx.r.student_id == sid]
    sm = r.groupby("ym").agg(pct=("percentage", "mean"), att=("att", "mean"))
    gm = ctx.sc[ctx.sc.group_id == s["group_id"]].groupby("ym").percentage.mean()
    fig = go.Figure()
    fig.add_bar(x=sm.index, y=sm.att, name="الحضور", marker_color="rgba(115,217,194,.35)", hovertemplate="%{y:.0%}")
    fig.add_scatter(x=gm.index, y=gm.values, name="متوسط المجموعة", mode="lines", line=dict(color=GREY, dash="dot", width=2), hovertemplate="%{y:.0%}")
    fig.add_scatter(x=sm.index, y=sm.pct, name="درجاته" if teacher else "درجاتك", mode="lines+markers",
                    line=dict(color=CYAN, width=3), marker=dict(size=8), hovertemplate="%{y:.0%}")
    if pd.notna(pred) and len(sm):
        nxt = sm.index.max() + pd.offsets.MonthBegin(1)
        fig.add_scatter(x=[sm.index.max(), nxt], y=[sm.pct.iloc[-1], pred], name="المتوقع", mode="lines+markers",
                        line=dict(color=CYAN, width=2, dash="dash"), marker=dict(size=9, symbol="diamond"), hovertemplate="%{y:.0%}")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(title="الدرجات والحضور شهر بشهر", legend=dict(orientation="h", y=-0.2, x=1, xanchor="right"))
    show_fig(dark(fig, 380))

    c1, c2 = st.columns(2)
    with c1:
        section("آخر الحصص")
        last = r.sort_values("date").tail(8).iloc[::-1]
        rows = "".join(
            f'<div class="mn-row"><div><b>{x.date:%d/%m}</b><small>الواجب: {e(x.homework)}'
            + (f" · الدرجة {x.exam_score:g} من {x.max_exam_score:g}" if pd.notna(x.exam_score) and pd.notna(x.max_exam_score) else "")
            + f'</small></div><div class="end">{chip(ATT_AR.get(x.attendance, x.attendance), ATT_COL.get(x.attendance, GREY))}</div></div>'
            for x in last.itertuples())
        st.markdown(f'<div class="mn-list">{rows}</div>', unsafe_allow_html=True)
    with c2:
        section("الدفع")
        p = ctx.P[ctx.P.student_id == sid].sort_values("m", ascending=False)
        rows = "".join(
            f'<div class="mn-row"><div><b>{e(x.month_label)}</b><small>مدفوع {egp(x.amount_paid if pd.notna(x.amount_paid) else 0)} من {egp(x.amount_due)}</small></div>'
            f'<div class="end">{chip(PAY_AR.get(x.status, x.status), PAY_COL.get(x.status, GREY))}</div></div>'
            for x in p.itertuples())
        st.markdown(f'<div class="mn-list">{rows or "<p class=mn-note>مفيش مدفوعات مسجلة.</p>"}</div>', unsafe_allow_html=True)

    if teacher:
        st.markdown(
            f'<p class="mn-note">كود الطالب: {e(sid)} · موبايل ولي الأمر: <span dir="ltr">{e(s.get("parent_phone", "غير مسجل"))}</span></p>',
            unsafe_allow_html=True)
    st.markdown('<p class="mn-note">الدرجة المتوقعة تقدير من أداء الشهور اللي فاتت، مش حكم نهائي.</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# teacher pages
# ---------------------------------------------------------------------------
def page_today(A: pd.DataFrame) -> None:
    cl = eda.call_list(ctx)
    cl = cl[cl.index.isin(A.index)]
    P = ctx.P[ctx.P.student_id.isin(A.index)]
    coll = P.amount_paid.sum() / P.amount_due.sum() if P.amount_due.sum() else np.nan
    risk = int(A.at_risk.sum())
    follow = int(((cl.score > 0) & ~cl.at_risk).sum())
    kpis([
        (f"{risk}", "في خطر", "حضور أقل من 80% ودرجات أقل من 75%", RED),
        (f"{follow}", "محتاجين متابعة", "ليهم سبب للمكالمة", AMBER),
        (f"{len(A) - risk - follow}", "تمام", "مفيش علامات قلق", GREEN),
        (pc(A.att_rate.mean()), "الحضور", "المتأخر محسوب حاضر", MINT),
        (pc(A.avg.mean()), "متوسط الدرجات", "", BLUE_L),
        (pc(coll), "نسبة التحصيل", f"متأخرات {egp(A.owed.sum())}", CYAN),
    ])

    section("ملخص الأسبوع: 3 قرارات")
    cards = []
    if len(cl):
        s0 = cl.iloc[0]
        cards.append((RED, "مين تكلّمه الأول", e(s0["name"]), e(s0["why"])))
    gf = eda.group_flags(ctx)
    gf = gf[gf.index.isin(A.group_id.unique())]
    if len(gf):
        g0 = gf.iloc[0]
        cards.append((AMBER, "أنهي مجموعة تتدخل فيها", e(g0["nm"]), f"{int(g0['nflags'])} مؤشرات من 5 في أضعف ثلث"))
    if A.owed.sum() > 0:
        top = A.sort_values("owed", ascending=False).iloc[0]
        cards.append((CYAN, "فلوس تحصّلها", f"{e(top['name'])}: {egp(top['owed'])}", f"إجمالي المتأخرات {egp(A.owed.sum())}"))
    st.markdown('<div class="mn-grid">' + "".join(
        f'<div class="mn-card" style="--c:{c}"><h4>{t}</h4><div class="big">{b}</div><p>{s}</p></div>' for c, t, b, s in cards) + "</div>",
        unsafe_allow_html=True)

    section("مين محتاجك النهارده؟")
    n = st.slider("عدد الطلاب", 5, 40, 12, key="today_n")
    rows = []
    for sid, x in cl.head(n).iterrows():
        lvl, col = ("عاجل", RED) if x.score >= 6 else (("مهم", AMBER) if x.score >= 4 else ("متابعة", BLUE_L))
        ph = x.get("parent_phone")
        tel = f'<a href="tel:{e(ph)}">{e(ph)}</a>' if isinstance(ph, str) and ph else ""
        rows.append(f'<div class="mn-row"><div><b>{e(x["name"])}</b><small>{e(x.group_id)} · {e(x.why)}</small></div>'
                    f'<div class="end">{tel}{chip(lvl, col)}</div></div>')
    st.markdown(f'<div class="mn-list">{"".join(rows) or "<p class=mn-note>مفيش حد محتاج تدخل دلوقتي.</p>"}</div>', unsafe_allow_html=True)


def page_students(A: pd.DataFrame) -> None:
    opts = A.sort_values("name").index.tolist()
    if not opts:
        st.info("مفيش طلاب في الفلتر ده.")
        return
    sid = st.selectbox("اختار طالب (اكتب الاسم أو الكود)", opts,
                       format_func=lambda i: f"{S.at[i, 'name']} · {i} · {S.at[i, 'group_id']}", key="stu_pick")
    student_view(sid, teacher=True)


def page_segments(A: pd.DataFrame) -> None:
    counts = A.segment.value_counts()
    order = [k for k in SEG if k in counts.index]
    st.markdown('<div class="mn-grid">' + "".join(
        f'<div class="mn-card" style="--c:{SEG[k][1]}"><h4>{SEG[k][0]}</h4><div class="big">{counts[k]} طالب</div>'
        f'<p>{e(A.loc[A.segment == k, "action"].iloc[0])}</p></div>' for k in order) + "</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for k in order:
        d = A[A.segment == k]
        fig.add_scatter(x=d.att_rate, y=d.avg, mode="markers", name=SEG[k][0], marker=dict(color=SEG[k][1], size=7, opacity=.8),
                        text=d.name, hovertemplate="%{text}<br>الحضور %{x:.0%} · الدرجات %{y:.0%}<extra></extra>")
    fig.add_hline(y=rules.AT_RISK_AVG_MAX, line=dict(color=RED, dash="dot", width=1))
    fig.add_vline(x=rules.AT_RISK_ATTENDANCE_MAX, line=dict(color=RED, dash="dot", width=1))
    fig.update_xaxes(tickformat=".0%", title="الحضور"); fig.update_yaxes(tickformat=".0%", title="متوسط الدرجات")
    fig.update_layout(title="كل طالب نقطة: الحضور قدام الدرجات", legend=dict(orientation="h", y=-0.22, x=1, xanchor="right"))
    show_fig(dark(fig, 460))

    pick = st.selectbox("اعرض طلاب شريحة", order, format_func=lambda k: SEG[k][0], key="seg_pick")
    d = A[A.segment == pick].sort_values("avg")
    show_df(pd.DataFrame({"الطالب": d.name, "الكود": d.index, "المجموعة": d.group_id, "الحضور %": (d.att_rate * 100).round(0),
                          "الدرجات %": (d.avg * 100).round(0), "المتوقع %": (d.predicted * 100).round(0)}))
    st.markdown('<p class="mn-note">الشرايح من موديل KMeans على الحضور والدرجات والاتجاه والواجب والتذبذب. الإجراءات مقترحات مش تدخلات مجرّبة.</p>', unsafe_allow_html=True)


def page_forecast(A: pd.DataFrame) -> None:
    kpis([
        (f"{FC['mae']:.1f}", "متوسط خطأ التوقع (نقطة من 100)", f"على {FC['n_test_students']} طالب ما شافهمش الموديل", CYAN),
        (f"{FC['baseline_mae']:.1f}", "خطأ «زي الشهر اللي فات»", "المقارنة اللي لازم الموديل يكسبها", GREY),
        (f"{int((A.change <= -0.05).sum())}", "متوقع ينزلوا 5 نقاط أو أكتر", "", RED),
        (f"{int((A.change >= 0.05).sum())}", "متوقع يطلعوا 5 نقاط أو أكتر", "", GREEN),
    ])
    section("أكتر الطلاب المتوقع درجاتهم تنزل")
    d = A.dropna(subset=["predicted"]).sort_values("change").head(25)
    show_df(pd.DataFrame({"الطالب": d.name, "الكود": d.index, "المجموعة": d.group_id, "آخر شهر %": (d.last_month_avg * 100).round(0),
                          "المتوقع %": (d.predicted * 100).round(0), "الفرق (نقطة)": (d.change * 100).round(1)}))
    st.markdown('<p class="mn-note">موديل Ridge على جدول (طالب × شهر)، وكل الـ features من الشهور اللي قبل الشهر المتوقع بس.</p>', unsafe_allow_html=True)


def page_groups(A: pd.DataFrame) -> None:
    G = eda.group_flags(ctx)
    G = G[G.index.isin(A.group_id.unique())]
    fig = go.Figure(go.Scatter(x=G.att, y=G.avg, mode="markers+text", text=G.index, textposition="top center",
                               marker=dict(size=np.clip(G["size"].fillna(10), 8, 40), color=G.collection, colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                                           showscale=True, colorbar=dict(title="التحصيل", tickformat=".0%")),
                               customdata=G.nm, hovertemplate="%{customdata}<br>الحضور %{x:.0%} · الدرجات %{y:.0%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="الحضور"); fig.update_yaxes(tickformat=".0%", title="متوسط الدرجات")
    fig.update_layout(title="المجموعات: الحجم = عدد الطلاب، واللون = التحصيل")
    show_fig(dark(fig, 460))
    show_df(pd.DataFrame({"المجموعة": G.nm, "الطلاب": G["size"], "الحضور %": (G.att * 100).round(0), "الدرجات %": (G.avg * 100).round(0),
                          "الواجب %": (G.hw_rate * 100).round(0), "التحصيل %": (G.collection * 100).round(0), "في خطر": G.risk_n,
                          "سابوا": G.dropped, "مؤشرات ضعيفة (من 5)": G.nflags}))
    with st.expander("تحليلات المجموعات من الـ EDA", expanded=False):
        eda_block([54, 71, 17, 22, 29, 35, 56])


def page_money(A: pd.DataFrame) -> None:
    P = ctx.P[ctx.P.student_id.isin(A.index)]
    m = P.groupby("m").agg(due=("amount_due", "sum"), paid=("amount_paid", "sum")).sort_index()
    kpis([
        (egp(P.amount_paid.sum()), "اتحصّل", f"من {egp(P.amount_due.sum())}", GREEN),
        (egp(A.owed.sum()), "متأخرات", f"{int((A.owed > 0).sum())} طالب", RED),
        (f"{int((A.unpaid_m >= 2).sum())}", "متأخرين شهرين أو أكتر", "", AMBER),
    ])
    fig = go.Figure()
    fig.add_bar(x=[f"شهر {int(i)}" for i in m.index], y=m.due, name="المستحق", marker_color="rgba(91,141,239,.35)")
    fig.add_bar(x=[f"شهر {int(i)}" for i in m.index], y=m.paid, name="المحصّل", marker_color=MINT)
    fig.update_layout(barmode="overlay", title="المحصّل قدام المستحق كل شهر", legend=dict(orientation="h", y=-0.2, x=1, xanchor="right"))
    show_fig(dark(fig, 380))
    section("أكبر المتأخرات")
    d = A[A.owed > 0].sort_values("owed", ascending=False).head(25)
    show_df(pd.DataFrame({"الطالب": d.name, "الكود": d.index, "المجموعة": d.group_id, "شهور متأخرة": d.unpaid_m.astype(int),
                          "المبلغ (ج)": d.owed.round(0), "موبايل ولي الأمر": d.parent_phone}))
    with st.expander("تحليلات الدفع والتحصيل من الـ EDA", expanded=False):
        eda_block([37, 38, 39, 46, 47, 45])


def page_insights() -> None:
    axes = list(dict.fromkeys(q["axis"] for q in eda.QUESTIONS.values()))
    st.markdown(f'<p class="mn-lead">كل أسئلة الـ EDA ({len(eda.QUESTIONS)} سؤال)، مقسمة على {len(axes)} محاور. اختار محور وهتظهر كل أسئلته.</p>',
                unsafe_allow_html=True)
    ax = st.radio("المحور", axes, horizontal=True, key="ins_axis", label_visibility="collapsed")
    ids = [q["id"] for q in eda.QUESTIONS.values() if q["axis"] == ax]
    only_p1 = st.toggle("الأسئلة الأساسية بس (P1)", value=False, key="ins_p1")
    if only_p1:
        ids = [n for n in ids if eda.QUESTIONS[n]["prio"] == "P1"]
    eda_block(ids, note=False)


def teacher() -> None:
    topbar("نسخة المدرس · ")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        grades = ["كل السنين"] + [g for g in eda.GRADES if g in set(S.grade.dropna())]
        gsel = st.selectbox("السنة", grades, key="f_grade")
    base = S[S.status == "active"]
    if gsel != "كل السنين":
        base = base[base.grade == gsel]
    with c2:
        groups = ["كل المجموعات"] + sorted(base.group_id.dropna().unique().tolist())
        grsel = st.selectbox("المجموعة", groups, key="f_group",
                             format_func=lambda g: g if g == "كل المجموعات" else ctx.G.at[g, "nm"] if g in ctx.G.index else g)
    if grsel != "كل المجموعات":
        base = base[base.group_id == grsel]
    with c3:
        st.write("")
        logout_button("lo_t")
    if secret("TEACHER_PASSWORD") is None:
        st.caption("الباسورد الافتراضي شغال. غيّره من Secrets بـ TEACHER_PASSWORD قبل ما تشارك اللينك.")

    tabs = st.tabs(["النهارده", "الطلاب", "الشرايح", "الدرجة المتوقعة", "المجموعات", "الفلوس", "التحليلات"])
    pages = [page_today, page_students, page_segments, page_forecast, page_groups, page_money]
    for tab, fn in zip(tabs, pages):
        with tab:
            fn(base)
    with tabs[-1]:
        page_insights()


def student() -> None:
    topbar()
    sid = st.session_state["sid"]
    student_view(sid, teacher=False)
    st.write("")
    logout_button("lo_s")


# ---------------------------------------------------------------------------
role = st.session_state.get("role")
if role == "teacher":
    teacher()
elif role == "student" and st.session_state.get("sid") in S.index:
    student()
else:
    login()
