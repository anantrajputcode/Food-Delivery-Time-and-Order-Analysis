"""
DeliverIQ — Food Delivery Time Prediction
Predict smarter. Deliver faster.

Single-file Streamlit app. Uses the existing trained model
(models/best_delivery_model.pkl) and the project dataset. Nothing is retrained.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # charts degrade gracefully; `pip install plotly` to enable
    HAS_PLOTLY = False

# =========================================================
# 1. CONFIG
# =========================================================

st.set_page_config(
    page_title="DeliverIQ | Food Delivery Time Prediction",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_delivery_model.pkl"
EVAL_PATH = BASE_DIR / "models" / "model_evaluation.txt"
DATA_PATH = BASE_DIR / "dataset" / "food_delivery.csv"
CLEAN_DATA_PATH = BASE_DIR / "dataset" / "cleaned_food_delivery.csv"

FEATURES = [
    "distance_km", "order_hour", "Delivery_person_Age", "Delivery_person_Ratings",
    "Road_traffic_density", "Weather_conditions", "Type_of_order", "Type_of_vehicle",
    "Vehicle_condition", "multiple_deliveries", "Festival", "City",
]

# Previously reported results (verified against model_evaluation.txt when parseable)
REPORTED = {"MAE": 3.22, "RMSE": 4.09, "R2": 0.8107}

PAGES = ["Predict", "Analytics", "Performance", "About", "Stack"]

ACCENT = "#22d3ee"
BLUE = "#3b82f6"

# =========================================================
# 2. STYLE
# =========================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #0a0f1c; --panel: #111827; --panel-2: #0f1626; --line: #1f2a3d;
  --text: #e6edf7; --muted: #8b9ab3; --accent: #22d3ee; --blue: #3b82f6;
}
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { max-width: 1180px; padding-top: 1.4rem; padding-bottom: 4rem; }

/* ---- top bar ---- */
.topbar { display:flex; align-items:center; justify-content:space-between;
  padding: 6px 0 14px 0; border-bottom: 1px solid var(--line); margin-bottom: 18px; }
.brand { display:flex; align-items:center; gap:12px; }
.logo { width:34px; height:34px; border-radius:9px;
  background: linear-gradient(135deg, var(--blue), var(--accent));
  display:flex; align-items:center; justify-content:center;
  color:#04121f; font-weight:800; font-size:16px; }
.brand-name { font-size:19px; font-weight:700; letter-spacing:-0.01em; }
.brand-name span { color: var(--accent); }
.brand-tag { font-size:12px; color: var(--muted); margin-top:1px; }
.status { display:flex; align-items:center; gap:8px; font-size:12.5px; color: var(--muted);
  border:1px solid var(--line); background: var(--panel); padding:6px 12px; border-radius:999px; }
.dot { width:8px; height:8px; border-radius:50%; background:#34d399; box-shadow:0 0 0 3px rgba(52,211,153,.15); }
.dot.off { background:#f87171; box-shadow:0 0 0 3px rgba(248,113,113,.15); }

/* ---- nav (segmented control / radio) ---- */
[data-testid="stSegmentedControl"] { width:100%; }
[data-testid="stSegmentedControl"] button { font-weight:600; font-size:13.5px; padding:8px 18px; }
div[role="radiogroup"][aria-orientation="horizontal"] { gap:6px; }

/* ---- hero ---- */
.hero { padding: 34px 0 22px 0; }
.eyebrow { color: var(--accent); font-size:12px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { font-size: 40px; line-height:1.12; font-weight:800; letter-spacing:-0.025em; margin:10px 0 12px 0; color:#f4f8ff; }
.hero p { color: var(--muted); font-size:16px; max-width:640px; margin:0; line-height:1.55; }

/* ---- cards ---- */
.kpi { background: var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px; height:100%; }
.kpi .l { color: var(--muted); font-size:12px; font-weight:500; letter-spacing:.04em; text-transform:uppercase; }
.kpi .v { font-size:28px; font-weight:700; margin-top:6px; letter-spacing:-0.02em; }
.kpi .s { color: var(--muted); font-size:12px; margin-top:4px; }
.group-title { font-size:14px; font-weight:600; margin: 2px 0 2px 0; display:flex; align-items:center; gap:10px; }
.group-title i { width:3px; height:15px; border-radius:2px; background: var(--accent); display:inline-block; }
.group-sub { color: var(--muted); font-size:12.5px; margin-bottom:10px; }
.section { font-size:20px; font-weight:700; letter-spacing:-0.01em; margin: 28px 0 4px 0; }
.section-sub { color: var(--muted); font-size:14px; margin-bottom:14px; }
.note { background: var(--panel-2); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:12px 16px; color:#b7c3d8; font-size:13.5px; line-height:1.55; }

div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > .group-title),
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--panel); border-color: var(--line);
}
div[data-testid="stForm"] { border:none; padding:0; background: transparent; }
label, [data-testid="stWidgetLabel"] p { color:#c3cee0 !important; font-size:13px !important; font-weight:500 !important; }

div[data-baseweb="select"] > div, .stNumberInput input {
  background: #0b1322 !important; border-color: var(--line) !important; border-radius:9px !important; }

/* ---- primary button ---- */
.stFormSubmitButton > button, .stButton > button {
  width:100%; height:52px; border-radius:11px; font-weight:700; font-size:15px; border:none; color:#04121f;
  background: linear-gradient(135deg, var(--accent), #60a5fa); transition: transform .12s ease, filter .12s ease; }
.stFormSubmitButton > button:hover, .stButton > button:hover { filter:brightness(1.08); transform: translateY(-1px); color:#04121f; }

/* ---- result ---- */
.result { background: linear-gradient(160deg, #0f1d33 0%, #0c1626 100%); border:1px solid #234060;
  border-radius:18px; padding:30px 28px; margin-top:6px; }
.result .l { color: var(--accent); font-size:12px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.result .v { font-size:64px; font-weight:800; letter-spacing:-0.04em; line-height:1.05; margin:10px 0 2px 0; color:#f4f8ff; }
.result .v small { font-size:22px; font-weight:600; color: var(--muted); margin-left:8px; letter-spacing:0; }
.result .i { color:#b7c3d8; font-size:14.5px; line-height:1.55; margin-top:10px; }
.bar { position:relative; height:8px; border-radius:6px; background:#16233a; margin:20px 0 8px 0; }
.bar .fill { position:absolute; left:0; top:0; bottom:0; border-radius:6px; background: linear-gradient(90deg, var(--blue), var(--accent)); }
.bar .pin { position:absolute; top:-5px; width:4px; height:18px; border-radius:2px; background:#fff; transform:translateX(-2px); }
.bar-l { display:flex; justify-content:space-between; color: var(--muted); font-size:11.5px; }
.disclaimer { color: var(--muted); font-size:12.5px; margin-top:16px; }
.empty { border:1px dashed #28374f; border-radius:18px; padding:42px 24px; text-align:center; color: var(--muted); font-size:14px; }
.empty b { display:block; color:#cfd9ea; font-size:16px; margin-bottom:6px; }

/* ---- feature / stack tiles ---- */
.tile { background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; height:100%; }
.tile .n { font-size:13.5px; font-weight:600; color:#f1f5fb; }
.tile .c { font-size:11px; color: var(--accent); font-family: ui-monospace, monospace; margin:2px 0 6px 0; }
.tile .d { font-size:12.5px; color: var(--muted); line-height:1.5; }
.step { background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; text-align:center; height:100%; }
.step .k { color: var(--accent); font-weight:700; font-size:12px; letter-spacing:.1em; }
.step .t { font-weight:600; margin:6px 0 4px 0; font-size:14px; }
.step .d { color: var(--muted); font-size:12.5px; line-height:1.5; }

@media (max-width: 768px) {
  .hero h1 { font-size: 28px; } .result .v { font-size: 46px; }
  .topbar { flex-direction:column; align-items:flex-start; gap:10px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# 3. LOADING (cached)
# =========================================================


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def _find_target(df: pd.DataFrame) -> str | None:
    for key in ("time_taken", "delivery_time", "time taken", "target"):
        for c in df.columns:
            if key in c.lower():
                return c
    return None


@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, str | None, list[str]]:
    """
    Load the cleaned dataset when present (it should hold the engineered features);
    otherwise fall back to the raw CSV. Returns (df, target_column, notes).
    Analytics-only fallbacks (distance/hour derived here) never touch the model.
    """
    notes: list[str] = []
    path = CLEAN_DATA_PATH if CLEAN_DATA_PATH.exists() else DATA_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # strip whitespace in text columns (the raw CSV has padded strings)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip().replace({"NaN": pd.NA, "nan": pd.NA}).astype(object)
            df[c] = df[c].where(df[c].notna(), np.nan)

    target = _find_target(df)
    if target is not None and not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = pd.to_numeric(df[target].astype(str).str.extract(r"(\d+)")[0], errors="coerce")

    if "distance_km" not in df.columns and {
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
    } <= set(df.columns):
        notes.append("distance_km derived with the haversine formula for charts only.")
        lat1, lon1, lat2, lon2 = (np.radians(pd.to_numeric(df[c], errors="coerce")) for c in (
            "Restaurant_latitude", "Restaurant_longitude",
            "Delivery_location_latitude", "Delivery_location_longitude"))
        a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))
        df = df[df["distance_km"] > 0.05]

    if "order_hour" not in df.columns and "Time_Orderd" in df.columns:
        notes.append("order_hour derived from Time_Orderd for charts only.")
        df["order_hour"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce").dt.hour

    for c in ("Delivery_person_Age", "Delivery_person_Ratings"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df, target, notes


@st.cache_data(show_spinner=False)
def read_eval_file() -> dict[str, float]:
    """Parse MAE / RMSE / R2 from models/model_evaluation.txt if possible."""
    out: dict[str, float] = {}
    if not EVAL_PATH.exists():
        return out
    text = EVAL_PATH.read_text(errors="ignore")
    for key, pat in {
        "MAE": r"\bMAE\b\D{0,12}([0-9]*\.?[0-9]+)",
        "RMSE": r"\bRMSE\b\D{0,12}([0-9]*\.?[0-9]+)",
        "R2": r"\bR\s*(?:²|2|\^2)[^0-9\-]{0,16}(-?[0-9]*\.?[0-9]+)",
    }.items():
        m = re.search(pat, text, re.I)
        if m:
            out[key] = float(m.group(1))
    return out


def options_for(df: pd.DataFrame, col: str) -> list:
    """Valid dropdown values straight from the data (never invented)."""
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


def pick(options: list, preferred) -> int:
    for i, o in enumerate(options):
        if str(o).lower() == str(preferred).lower():
            return i
    return 0


def hour_label(h: int) -> str:
    return f"{(h % 12) or 12}:00 {'AM' if h < 12 else 'PM'}"


def esc(x) -> str:
    return html.escape(str(x))


# ---- load everything, fail softly ----
model, model_err = None, None
try:
    model = load_model()
except Exception as e:  # noqa: BLE001 - surfaced to user below
    model_err = e

df, TARGET, DATA_NOTES, data_err = None, None, [], None
try:
    df, TARGET, DATA_NOTES = load_dataset()
except Exception as e:  # noqa: BLE001
    data_err = e

# =========================================================
# 4. HEADER + NAVIGATION
# =========================================================

status_ok = model is not None
st.markdown(
    f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">D</div>
    <div>
      <div class="brand-name">Deliver<span>IQ</span></div>
      <div class="brand-tag">Predict smarter. Deliver faster.</div>
    </div>
  </div>
  <div class="status"><span class="dot {'' if status_ok else 'off'}"></span>
    {'Model loaded · Random Forest' if status_ok else 'Model unavailable'}</div>
</div>
""",
    unsafe_allow_html=True,
)

if hasattr(st, "segmented_control"):
    choice = st.segmented_control("Navigation", PAGES, default=PAGES[0], label_visibility="collapsed")
    page = choice or PAGES[0]
else:
    page = st.radio("Navigation", PAGES, horizontal=True, label_visibility="collapsed")

if model_err is not None:
    st.error(
        f"Could not load the model at `{MODEL_PATH.name}`: {model_err}\n\n"
        "If the file is only a few hundred bytes it is a Git LFS pointer — run `git lfs pull`. "
        "Otherwise check that your scikit-learn version matches the one used to train it."
    )
if data_err is not None:
    st.error(f"Could not load the dataset: {data_err}")

# =========================================================
# 5. PAGES
# =========================================================


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
            f'<div class="s">{esc(sub)}</div></div>')


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="section">{esc(title)}</div><div class="section-sub">{esc(sub)}</div>',
                unsafe_allow_html=True)


def round_bars(fig) -> None:
    """Rounded bar corners need Plotly >= 6; silently skip on older versions."""
    try:
        fig.update_traces(marker_cornerradius=4)
    except Exception:  # noqa: BLE001
        pass


def style_fig(fig, height: int = 340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#b7c3d8", size=12),
        title=dict(font=dict(size=14, color="#e6edf7"), x=0.0),
        xaxis=dict(gridcolor="#1a2438", zerolinecolor="#1a2438"),
        yaxis=dict(gridcolor="#1a2438", zerolinecolor="#1a2438"),
        showlegend=False,
    )
    return fig


VIS_DIR = BASE_DIR / "visualizations"


def saved_figures(exclude: tuple[str, ...] = ()) -> list[Path]:
    """PNG/JPG charts saved from the notebook, sorted by filename (01_, 02_, ...)."""
    if not VIS_DIR.exists():
        return []
    return [p for p in sorted(VIS_DIR.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and not any(x in p.name.lower() for x in exclude)]


def fig_caption(p: Path) -> str:
    return re.sub(r"^\d+_", "", p.stem).replace("_", " ").capitalize()


def render_gallery(paths: list[Path], per_row: int = 2) -> None:
    for i in range(0, len(paths), per_row):
        cols = st.columns(per_row)
        for col, p in zip(cols, paths[i:i + per_row]):
            with col:
                with st.container(border=True):
                    st.image(str(p), width="stretch")
                    st.caption(fig_caption(p))


# ---------------------------------------------------------
# PAGE: PREDICT
# ---------------------------------------------------------
def page_predict() -> None:
    st.markdown(
        """
<div class="hero">
  <div class="eyebrow">Delivery time estimation</div>
  <h1>Know your delivery time<br>before it arrives.</h1>
  <p>Estimate food delivery duration using machine learning and real-world delivery factors.</p>
</div>""",
        unsafe_allow_html=True,
    )

    if model is None or df is None:
        st.warning("The model and dataset must both load before predictions are available.")
        return

    opt = {c: options_for(df, c) for c in (
        "City", "Road_traffic_density", "Weather_conditions", "Type_of_order",
        "Type_of_vehicle", "Vehicle_condition", "multiple_deliveries", "Festival")}
    missing = [c for c, v in opt.items() if not v]
    if missing:
        st.error(f"No valid category values found in the dataset for: {', '.join(missing)}")
        return

    age_lo = int(df["Delivery_person_Age"].min()) if "Delivery_person_Age" in df and df["Delivery_person_Age"].notna().any() else 18
    age_hi = int(df["Delivery_person_Age"].max()) if "Delivery_person_Age" in df and df["Delivery_person_Age"].notna().any() else 60

    form_col, result_col = st.columns([1.15, 1], gap="large")

    with form_col:
        with st.form("predict_form", border=False):
            with st.container(border=True):
                st.markdown('<div class="group-title"><i></i>Delivery</div>'
                            '<div class="group-sub">Where and when the order is placed.</div>',
                            unsafe_allow_html=True)
                distance = st.number_input("Distance, restaurant to customer (km)", 0.1, 50.0, 8.0, 0.1,
                                           help="Straight-line distance between restaurant and delivery location.")
                a, b = st.columns(2)
                order_hour = a.select_slider("Order time", options=list(range(24)), value=20,
                                             format_func=hour_label,
                                             help="Hour of day the order was placed (0–23).")
                city = b.selectbox("City category", opt["City"], index=pick(opt["City"], "Urban"))

            with st.container(border=True):
                st.markdown('<div class="group-title"><i></i>Delivery partner</div>'
                            '<div class="group-sub">Who is delivering and with what.</div>',
                            unsafe_allow_html=True)
                a, b = st.columns(2)
                age = a.number_input("Partner age (years)", age_lo, age_hi, min(max(28, age_lo), age_hi))
                rating = b.slider("Partner rating (1–5)", 1.0, 5.0, 4.5, 0.1)
                a, b = st.columns(2)
                vehicle = a.selectbox("Vehicle type", opt["Type_of_vehicle"])
                vcond = b.selectbox("Vehicle condition", opt["Vehicle_condition"],
                                    help="Condition code as recorded in the dataset.")

            with st.container(border=True):
                st.markdown('<div class="group-title"><i></i>Traffic &amp; environment</div>'
                            '<div class="group-sub">Conditions on the road.</div>',
                            unsafe_allow_html=True)
                a, b, c = st.columns(3)
                traffic = a.selectbox("Traffic density", opt["Road_traffic_density"],
                                      index=pick(opt["Road_traffic_density"], "Low"))
                weather = b.selectbox("Weather", opt["Weather_conditions"],
                                      index=pick(opt["Weather_conditions"], "Sunny"))
                festival = c.selectbox("Festival", opt["Festival"],
                                       index=pick(opt["Festival"], "No"))

            with st.container(border=True):
                st.markdown('<div class="group-title"><i></i>Order</div>'
                            '<div class="group-sub">What is being delivered.</div>',
                            unsafe_allow_html=True)
                a, b = st.columns(2)
                otype = a.selectbox("Order type", opt["Type_of_order"])
                multi = b.selectbox("Multiple deliveries", opt["multiple_deliveries"],
                                    help="Number of deliveries handled together, as recorded in the dataset.")

            submitted = st.form_submit_button("Predict delivery time", width="stretch")

    with result_col:
        if not submitted and "last_result" not in st.session_state:
            st.markdown('<div class="empty"><b>Your estimate will appear here</b>'
                        'Fill in the details and press “Predict delivery time”.</div>',
                        unsafe_allow_html=True)
        if submitted:
            sample = pd.DataFrame([{
                "distance_km": float(distance), "order_hour": int(order_hour),
                "Delivery_person_Age": age, "Delivery_person_Ratings": float(rating),
                "Road_traffic_density": traffic, "Weather_conditions": weather,
                "Type_of_order": otype, "Type_of_vehicle": vehicle,
                "Vehicle_condition": vcond, "multiple_deliveries": multi,
                "Festival": festival, "City": city,
            }])
            try:
                expected = list(getattr(model, "feature_names_in_", FEATURES))
                absent = [c for c in expected if c not in sample.columns]
                if absent:
                    raise ValueError(f"Model expects features not provided by the app: {absent}")
                sample = sample[expected]
                pred = float(model.predict(sample)[0])
                if not np.isfinite(pred):
                    raise ValueError("The model returned a non-finite prediction.")
                st.session_state["last_result"] = (pred, sample)
            except Exception as e:  # noqa: BLE001
                st.session_state.pop("last_result", None)
                st.error(f"Prediction failed: {e}")

        if "last_result" in st.session_state:
            pred, sample = st.session_state["last_result"]
            render_result(pred, sample)


def render_result(pred: float, sample: pd.DataFrame) -> None:
    interp, bar = "", ""
    if df is not None and TARGET and df[TARGET].notna().any():
        t = df[TARGET].dropna()
        med, lo, hi = float(t.median()), float(t.min()), float(t.max())
        pct = float((t < pred).mean() * 100)
        diff = pred - med
        direction = "above" if diff > 0 else "below"
        interp = (f"That is about {abs(diff):.0f} min {direction} the historical median of {med:.0f} min — "
                  f"longer than roughly {pct:.0f}% of past deliveries in the dataset.")
        pos = min(max((pred - lo) / (hi - lo) * 100, 0), 100) if hi > lo else 50
        bar = (f'<div class="bar"><div class="fill" style="width:{pos:.1f}%"></div>'
               f'<div class="pin" style="left:{pos:.1f}%"></div></div>'
               f'<div class="bar-l"><span>{lo:.0f} min fastest</span><span>{hi:.0f} min slowest</span></div>')

    st.markdown(
        f"""
<div class="result">
  <div class="l">Estimated delivery time</div>
  <div class="v">{pred:.0f}<small>minutes</small></div>
  <div class="i">{esc(interp)}</div>
  {bar}
  <div class="disclaimer">This is a machine-learning estimate based on historical patterns,
  not a guaranteed delivery time.</div>
</div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Inputs sent to the model"):
        st.dataframe(sample.T.rename(columns={0: "value"}).astype(str), width="stretch")


# ---------------------------------------------------------
# PAGE: ANALYTICS
# ---------------------------------------------------------
def page_analytics() -> None:
    section("Analytics", "Visualizations from the exploratory analysis in your notebook.")
    figs = saved_figures(exclude=("importance",))
    if figs:
        render_gallery(figs)
    else:
        st.info(f"No images found. Looked in: {VIS_DIR}  (exists: {VIS_DIR.exists()}). "
                "Put .png/.jpg files there. If they are Git LFS files, run `git lfs pull`.")


# ---------------------------------------------------------
# PAGE: PERFORMANCE
# ---------------------------------------------------------
def final_estimator(m):
    return m.steps[-1][1] if hasattr(m, "steps") else m


def feature_importance() -> pd.DataFrame | None:
    """Built-in (impurity-based) importances, folded back onto the 12 input features."""
    if model is None:
        return None
    est = final_estimator(model)
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        return None
    names = None
    if hasattr(model, "steps") and len(model.steps) > 1:
        try:
            names = list(model[:-1].get_feature_names_out())
        except Exception:  # noqa: BLE001
            names = None
    if names is None:
        names = list(getattr(est, "feature_names_in_", getattr(model, "feature_names_in_", [])))
    if len(names) != len(imp):
        return None
    agg: dict[str, float] = {}
    for n, v in zip(names, imp):
        owner = max((f for f in FEATURES if f in n), key=len, default=n)
        agg[owner] = agg.get(owner, 0.0) + float(v)
    return pd.DataFrame({"Feature": list(agg), "Importance": list(agg.values())}).sort_values("Importance")


def page_performance() -> None:
    section("Model performance", "How well the Random Forest generalises to unseen orders.")
    verified = read_eval_file()
    ok = bool(verified) and all(abs(verified.get(k, 1e9) - REPORTED[k]) < 0.01 for k in REPORTED)
    c = st.columns(3)
    c[0].markdown(kpi("MAE", f"{REPORTED['MAE']:.2f} min", "average absolute error"), unsafe_allow_html=True)
    c[1].markdown(kpi("RMSE", f"{REPORTED['RMSE']:.2f} min", "penalises large errors"), unsafe_allow_html=True)
    c[2].markdown(kpi("R²", f"{REPORTED['R2']:.4f}", "variance explained"), unsafe_allow_html=True)
    st.caption("Values match models/model_evaluation.txt." if ok
               else "Previously reported results — could not be automatically verified against models/model_evaluation.txt.")

    st.write("")
    st.markdown(
        """<div class="note">
<b>MAE</b> — on average, predictions differ from the actual delivery time by about 3.22 minutes.<br>
<b>RMSE</b> — the square root of the mean squared error; larger mistakes are penalised more heavily.<br>
<b>R²</b> — the model explains about 81.07% of the variance in delivery time on the test set, relative to always predicting the mean.
This is <b>not</b> accuracy, and it does not mean 81% of predictions are correct.
</div>""", unsafe_allow_html=True)

    section("Feature importance", "Which inputs the model relies on most.")
    fi = feature_importance()
    if fi is None:
        st.info("Feature importances could not be extracted from this model object.")
    elif HAS_PLOTLY:
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[ACCENT])
        round_bars(fig)
        st.plotly_chart(style_fig(fig, 420), width="stretch", config={"displayModeBar": False})
    else:
        st.bar_chart(fi.set_index("Feature"))
    st.caption("Built-in Random Forest (impurity-based) importance, with one-hot columns summed per original feature. "
               "It shows reliance, not causation.")

    section("Actual vs predicted & residuals")
    st.markdown(
        """<div class="note">These charts need the model's predictions on its original held-out test set.
They are not displayed here because that test data is not stored in the repository, and re-creating the split
or scoring training rows would give misleading results. To enable them, save <code>y_test</code> and the predictions
from the notebook (e.g. <code>reports/test_predictions.csv</code>) and this page can plot them.</div>""",
        unsafe_allow_html=True)

    fi_png = VIS_DIR / "feature_importance.png"
    extra = [p for p in saved_figures() if re.search(r"actual|predict|residual", p.name, re.I)]
    shown = ([fi_png] if fi_png.exists() and fi is None else []) + extra
    if shown:
        st.caption("Saved from your notebook:")
        render_gallery(shown)

    pred_file = BASE_DIR / "reports" / "test_predictions.csv"
    if pred_file.exists() and HAS_PLOTLY:
        tp = pd.read_csv(pred_file)
        if {"actual", "predicted"} <= set(tp.columns):
            tp["residual"] = tp["actual"] - tp["predicted"]
            a, b = st.columns(2)
            fig = px.scatter(tp, x="actual", y="predicted", opacity=0.35, title="Actual vs predicted",
                             color_discrete_sequence=[ACCENT])
            lo, hi = tp[["actual", "predicted"]].min().min(), tp[["actual", "predicted"]].max().max()
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#f87171", dash="dash")))
            a.plotly_chart(style_fig(fig, 380), width="stretch")
            fig = px.histogram(tp, x="residual", nbins=40, title="Residuals (actual − predicted)",
                               color_discrete_sequence=[BLUE])
            b.plotly_chart(style_fig(fig, 380), width="stretch")


# ---------------------------------------------------------
# PAGE: ABOUT
# ---------------------------------------------------------
FEATURE_INFO = [
    ("distance_km", "Distance between restaurant and customer, engineered from coordinates. Longer trips generally take longer."),
    ("order_hour", "Hour the order was placed (0–23). Captures rush hours and demand."),
    ("Delivery_person_Age", "Age of the delivery partner in years."),
    ("Delivery_person_Ratings", "Partner rating on roughly a 1–5 scale."),
    ("Road_traffic_density", "Road congestion level at order time."),
    ("Weather_conditions", "Weather during the delivery."),
    ("Type_of_order", "Category of food ordered."),
    ("Type_of_vehicle", "Vehicle used for delivery."),
    ("Vehicle_condition", "Vehicle condition code as recorded in the dataset."),
    ("multiple_deliveries", "Number of deliveries handled together on the same trip."),
    ("Festival", "Whether the delivery took place during a festival."),
    ("City", "City category (e.g. metropolitan, urban, semi-urban)."),
]


def page_about() -> None:
    section("About the model", "A supervised regression model that estimates delivery time in minutes.")
    steps = [
        ("01", "Inputs", "12 order, partner and condition features."),
        ("02", "Many trees", "Each decision tree is trained on a random sample of historical orders."),
        ("03", "Each predicts", "Every tree outputs its own delivery-time estimate."),
        ("04", "Average", "The forest averages the estimates into one final prediction."),
    ]
    cols = st.columns(4)
    for col, (k, t, d) in zip(cols, steps):
        col.markdown(f'<div class="step"><div class="k">{k}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    a, b, c = st.columns(3)
    a.markdown(kpi("Learning type", "Supervised", "learns from labelled history"), unsafe_allow_html=True)
    b.markdown(kpi("Problem type", "Regression", "predicts a number"), unsafe_allow_html=True)
    c.markdown(kpi("Algorithm", "Random Forest", "Scikit-learn Regressor"), unsafe_allow_html=True)

    section("The 12 input features")
    cols = st.columns(3)
    for i, (n, d) in enumerate(FEATURE_INFO):
        cols[i % 3].markdown(f'<div class="tile"><div class="n">{esc(n.replace("_", " "))}</div>'
                             f'<div class="c">{esc(n)}</div><div class="d">{esc(d)}</div></div>',
                             unsafe_allow_html=True)
        if i % 3 == 2:
            cols = st.columns(3)

    section("Limitations", "Read estimates with these in mind.")
    st.markdown(
        """<div class="note">
• Traffic can change unexpectedly after the order is placed.<br>
• Restaurant preparation time is not a direct input.<br>
• Distance is a straight-line measure; real road distance can be longer.<br>
• Weather and order demand can shift during delivery.<br>
• Predictions reflect historical patterns and are estimates, not guarantees.
</div>""", unsafe_allow_html=True)


# ---------------------------------------------------------
# PAGE: STACK
# ---------------------------------------------------------
STACK = [
    ("Python", "Core language", "Data processing, training, evaluation and the app."),
    ("Pandas", "Data manipulation", "Loading, cleaning and feature engineering of the dataset."),
    ("NumPy", "Numerical computing", "Array maths and distance calculations."),
    ("Matplotlib", "Visualization", "Notebook charts and saved figures."),
    ("Seaborn", "Statistical plots", "Heatmaps and distribution plots in the analysis."),
    ("Scikit-learn", "Machine learning", "Training, splitting, preprocessing and metrics."),
    ("Random Forest Regressor", "Algorithm", "Averages many decision trees to predict minutes."),
    ("Joblib", "Serialization", "Saves and loads the trained model file."),
    ("Streamlit", "Web app", "Interactive interface for this application."),
    ("Git", "Version control", "Tracks project history."),
    ("GitHub", "Hosting", "Hosts the repository."),
    ("Git LFS", "Large files", "Stores the ~104 MB model outside normal Git history."),
]
if HAS_PLOTLY:
    STACK.insert(9, ("Plotly", "Interactive charts", "Powers the Analytics and Performance charts."))


def page_stack() -> None:
    section("Technology stack", "The tools behind DeliverIQ.")
    for i in range(0, len(STACK), 4):
        cols = st.columns(4)
        for col, (n, c, d) in zip(cols, STACK[i:i + 4]):
            col.markdown(f'<div class="tile"><div class="n">{esc(n)}</div><div class="c">{esc(c)}</div>'
                         f'<div class="d">{esc(d)}</div></div>', unsafe_allow_html=True)
        st.write("")


# =========================================================
# 6. ROUTER
# =========================================================

{
    "Predict": page_predict,
    "Analytics": page_analytics,
    "Performance": page_performance,
    "About": page_about,
    "Stack": page_stack,
}[page]()

st.markdown('<div style="text-align:center;color:#5b6a84;font-size:12px;margin-top:48px;">'
            'DeliverIQ · Estimates are not guarantees of delivery time.</div>', unsafe_allow_html=True)