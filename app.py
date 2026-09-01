"""
app.py  –  ChurnIQ: AI-Powered Customer Retention Dashboard
============================================================
Two modes:
  Tab 1 – Single Customer Prediction  (manual input + Gemini recommendation)
  Tab 2 – Batch Customer Analysis     (CSV upload + bulk ML + Gemini for top-N)

ML model: churn_artifacts.pkl  (XGBoost pipeline, optimal threshold, metrics)
Gemini:   gemini_service.py    (business recommendations only, never re-predicts)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import custom_transformers          # needed so joblib can deserialise IQRCapper
import gemini_service as gs

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ | AI Retention Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:          #0b0f19;
    --card:        rgba(22, 28, 48, 0.85);
    --border:      rgba(255,255,255,0.08);
    --primary:     #3b82f6;
    --glow:        rgba(59,130,246,0.35);
    --text:        #f8fafc;
    --muted:       #94a3b8;
    --green:       #10b981;
    --amber:       #f59e0b;
    --red:         #ef4444;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(30,58,138,.18) 0, transparent 55%),
        radial-gradient(ellipse at 80% 0%, rgba(79,70,229,.1)  0, transparent 55%);
    color: var(--text);
}

/* ── Hide default chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Top nav ── */
.top-nav {
    display:flex; justify-content:space-between; align-items:center;
    padding: 1.25rem 2rem;
    background: rgba(11,15,25,.85);
    backdrop-filter: blur(14px);
    border-bottom: 1px solid var(--border);
    border-radius: 0 0 14px 14px;
    margin-bottom: 1.75rem;
}
.brand { font-size:1.45rem; font-weight:800; letter-spacing:-.5px; }
.brand-accent {
    background: linear-gradient(135deg,#60a5fa,#6366f1);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.status-pill {
    display:flex; align-items:center; gap:.45rem;
    font-size:.78rem; font-weight:700; letter-spacing:.8px;
    color: var(--green);
    background: rgba(16,185,129,.1);
    padding: .35rem .9rem;
    border-radius:50px; border: 1px solid rgba(16,185,129,.22);
}
.pulse {
    width:8px; height:8px; background:var(--green);
    border-radius:50%;
    box-shadow:0 0 7px var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow:0 0 0 0   rgba(16,185,129,.5); }
    70%  { box-shadow:0 0 0 7px rgba(16,185,129,0);  }
    100% { box-shadow:0 0 0 0   rgba(16,185,129,0);  }
}

/* ── Hero ── */
.hero { text-align:center; padding:.5rem 1rem 2.5rem; }
.hero h1 {
    font-size:3rem; font-weight:800; letter-spacing:-1px; margin-bottom:.75rem;
    background: linear-gradient(to right,#fff,#94a3b8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { font-size:1.1rem; color:var(--muted); max-width:580px; margin:0 auto; line-height:1.65; }

/* ── Section headers ── */
.sec-hdr {
    font-size:.88rem; font-weight:700; letter-spacing:1.2px;
    text-transform:uppercase; color:#cbd5e1;
    border-bottom:1px solid var(--border);
    padding-bottom:.6rem; margin-bottom:1.25rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
}

/* ── KPI row ── */
.kpi-grid {
    display:grid; grid-template-columns:repeat(auto-fit, minmax(130px,1fr));
    gap:.9rem; margin-bottom:1.5rem;
}
.kpi {
    background:rgba(255,255,255,.03);
    border:1px solid var(--border);
    border-radius:12px; padding:1rem;
    text-align:center;
}
.kpi-val { font-size:1.55rem; font-weight:700; }
.kpi-lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; margin-top:.2rem; }

/* ── Risk badges ── */
.badge {
    display:inline-block; padding:.38rem 1.1rem;
    border-radius:50px; font-weight:700; font-size:.95rem; letter-spacing:.6px;
}
.badge-low  { background:rgba(16,185,129,.12); color:var(--green); border:1px solid rgba(16,185,129,.25); }
.badge-med  { background:rgba(245,158,11,.12); color:var(--amber); border:1px solid rgba(245,158,11,.25); }
.badge-high { background:rgba(239,68,68,.12);  color:var(--red);   border:1px solid rgba(239,68,68,.25);  }

/* ── Result card ── */
.result-card {
    background:rgba(11,15,25,.85);
    backdrop-filter:blur(16px);
    border:1px solid var(--border);
    border-radius:18px; padding:2.25rem;
    text-align:center; margin:1.5rem 0;
    position:relative; overflow:hidden;
}
.result-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
}
.rc-high::before { background:var(--red); }
.rc-med::before  { background:var(--amber); }
.rc-low::before  { background:var(--green); }

.prob-num  { font-size:4rem; font-weight:800; line-height:1; }
.prob-lbl  { font-size:.85rem; color:var(--muted); letter-spacing:2px; text-transform:uppercase; margin:.4rem 0 1.5rem; }

/* ── Rec card ── */
.rec-card {
    background:rgba(59,130,246,.07);
    border:1px solid rgba(59,130,246,.18);
    border-radius:14px; padding:1.4rem;
    margin-top:1.25rem;
}
.rec-section-lbl { font-size:.72rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:var(--primary); margin-bottom:.25rem; }
.rec-value { color:#e2e8f0; line-height:1.55; margin-bottom:1rem; }

/* ── Feature bar ── */
.feat-row { display:flex; align-items:center; margin-bottom:.7rem; }
.feat-nm  { width:160px; text-align:right; padding-right:.9rem; font-size:.85rem; color:#cbd5e1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.feat-bg  { flex-grow:1; height:7px; background:rgba(255,255,255,.09); border-radius:4px; overflow:hidden; }
.feat-fill{ height:100%; background:linear-gradient(90deg,#3b82f6,#6366f1); border-radius:4px; }

/* ── Upload area ── */
.upload-zone {
    border: 2px dashed rgba(59,130,246,.35);
    border-radius: 14px; padding: 2.5rem;
    text-align:center; background:rgba(59,130,246,.04);
}
.upload-zone p { color:var(--muted); margin:.35rem 0; }

/* ── Progress bar override ── */
div[data-testid="stProgress"] > div { border-radius:8px; }

/* ── Button override ── */
.stButton>button {
    background:linear-gradient(135deg,#2563eb,#4f46e5) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important; font-weight:700 !important;
    font-size:1rem !important; letter-spacing:.6px !important;
    padding:.75rem 1.5rem !important;
    box-shadow:0 4px 18px -3px var(--glow) !important;
    transition:all .25s ease !important;
    width:100% !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px -3px rgba(59,130,246,.55) !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"]  { gap:1rem; background:transparent; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]       { background:transparent; border:none; color:var(--muted); font-weight:600; font-size:.9rem; padding:.55rem 1.1rem; border-radius:8px 8px 0 0; }
.stTabs [aria-selected="true"]     { background:rgba(59,130,246,.12) !important; color:var(--primary) !important; border-bottom:2px solid var(--primary) !important; }

/* ── Input style ── */
div[data-baseweb="select"]>div { background:rgba(15,23,42,.6)!important; border-color:rgba(255,255,255,.1)!important; border-radius:8px!important; }
.stNumberInput>div>div>div { background:rgba(15,23,42,.6)!important; }
input[type="number"] { color:#fff!important; }

/* ── Dataframe ── */
.stDataFrame { border-radius:10px; overflow:hidden; }

/* ── Footer ── */
.custom-footer {
    text-align:center; padding:2.5rem 0; color:var(--muted);
    font-size:.85rem; border-top:1px solid var(--border); margin-top:3rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

# ── Model columns expected by the pipeline ──────────────────────────────────
REQUIRED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

OPTIONAL_ID_COLS = {"customerid", "customer name", "customername", "email", "phone"}


def risk_label(prob: float, threshold: float) -> tuple[str, str, str]:
    """Return (risk_level, badge_class, card_class) tuple."""
    high_thresh = min(0.85, threshold * 1.6)
    if prob >= high_thresh:
        return "High Risk", "badge-high", "rc-high"
    elif prob >= threshold:
        return "Medium Risk", "badge-med", "rc-med"
    else:
        return "Low Risk", "badge-low", "rc-low"


def prediction_label(prob: float, threshold: float) -> str:
    return "Likely to Churn" if prob >= threshold else "Not Likely to Churn"


def color_for_risk(risk: str) -> str:
    return {"High Risk": "var(--red)", "Medium Risk": "var(--amber)", "Low Risk": "var(--green)"}.get(risk, "#fff")


def priority_color(priority: str) -> str:
    return {"High": "var(--red)", "Medium": "var(--amber)", "Low": "var(--green)"}.get(priority, "#fff")


@st.cache_resource(show_spinner=False)
def load_artifacts():
    try:
        return joblib.load("churn_artifacts.pkl")
    except FileNotFoundError:
        return None


def render_recommendation(rec: dict):
    """Render a Gemini recommendation inside a styled card."""
    priority = rec.get("priority", "Medium")
    p_color = priority_color(priority)
    st.markdown(f"""
    <div class="rec-card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <span style="font-weight:700;font-size:1rem;color:#e2e8f0;">🤖 AI Business Recommendation</span>
            <span class="badge {'badge-high' if priority=='High' else 'badge-med' if priority=='Medium' else 'badge-low'}"
                  style="font-size:.78rem;">Priority: {priority}</span>
        </div>
        <div class="rec-section-lbl">Recommended Action</div>
        <div class="rec-value">{rec.get('recommended_action','—')}</div>
        <div class="rec-section-lbl">Reason</div>
        <div class="rec-value">{rec.get('reason','—')}</div>
        <div class="rec-section-lbl">Retention Strategy</div>
        <div class="rec-value">{rec.get('retention_strategy','—')}</div>
        <div class="rec-section-lbl">Suggested Next Step</div>
        <div class="rec-value" style="margin-bottom:0">{rec.get('suggested_next_step','—')}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_bars(feat_df: pd.DataFrame | None):
    if feat_df is None or feat_df.empty:
        return
    max_imp = feat_df["Importance"].max()
    bars = ""
    for _, row in feat_df.iterrows():
        pct = (row["Importance"] / max_imp) * 100
        name = (row["Feature"]
                .replace("InternetService_", "Net: ")
                .replace("PaymentMethod_", "Pay: ")
                .replace("Contract_", "Ctr: "))[:20]
        bars += f"""
        <div class="feat-row">
            <div class="feat-nm">{name}</div>
            <div class="feat-bg"><div class="feat-fill" style="width:{pct:.1f}%"></div></div>
        </div>"""
    st.markdown(bars, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# App entry point
# ══════════════════════════════════════════════════════════════════════════════

# ── Load artifacts ────────────────────────────────────────────────────────────
artifacts = load_artifacts()

# ── Top nav ───────────────────────────────────────────────────────────────────
gemini_ok = gs.is_gemini_available()
model_ok  = artifacts is not None
status_text  = "MODEL ONLINE" if model_ok else "MODEL OFFLINE"
status_color = "var(--green)" if model_ok else "var(--red)"

st.markdown(f"""
<div class="top-nav">
    <div class="brand">⚡ Churn<span class="brand-accent">IQ</span>
        <span style="font-size:.7rem;font-weight:400;color:var(--muted);margin-left:.6rem;">AI Retention Platform</span>
    </div>
    <div style="display:flex;gap:.75rem;align-items:center;">
        {'<div class="status-pill"><div class="pulse"></div>GEMINI CONNECTED</div>' if gemini_ok else
         '<div class="status-pill" style="color:var(--muted);background:rgba(255,255,255,.05);border-color:rgba(255,255,255,.1);">GEMINI OFFLINE</div>'}
        <div class="status-pill" style="{'color:var(--green);background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.22)' if model_ok else 'color:var(--red);background:rgba(239,68,68,.1);border-color:rgba(239,68,68,.22)'};">
            <div class="pulse" style="background:{status_color};box-shadow:0 0 7px {status_color};"></div>
            {status_text}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Predict Customer Churn Before It Happens</h1>
    <p>Advanced XGBoost analytics + Gemini AI business intelligence — identify at-risk customers
       and deploy precision retention strategies before churn occurs.</p>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error("⚠️ `churn_artifacts.pkl` not found. Run `python train_model.py` first.", icon="🚨")
    st.stop()

pipeline  = artifacts["pipeline"]
threshold = artifacts["optimal_threshold"]
metrics   = artifacts["metrics"]
feat_names = artifacts.get("feature_names", [])

# Global feature importances
try:
    xgb_model  = pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_
    feat_df_global = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(8)
        .reset_index(drop=True)
    )
except Exception:
    feat_df_global = None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["👤  Single Customer", "📊  Batch Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Single Customer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Input form ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="sec-hdr">👤 Personal Profile</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        with r1c1: gender        = st.selectbox("Gender", ["Female", "Male"], key="s_gender")
        with r1c2: senior_citizen = st.selectbox("Senior Citizen", [0, 1],
                                                  format_func=lambda x: "Yes" if x else "No", key="s_senior")
        r2c1, r2c2 = st.columns(2)
        with r2c1: partner    = st.selectbox("Partner",    ["Yes", "No"], key="s_partner")
        with r2c2: dependents = st.selectbox("Dependents", ["Yes", "No"], key="s_dep")

        st.markdown('<br><div class="sec-hdr">💳 Financial</div>', unsafe_allow_html=True)
        r3c1, r3c2 = st.columns(2)
        with r3c1: monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0,  key="s_mc")
        with r3c2: total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0, 1000.0, step=50.0, key="s_tc")

    with col_r:
        st.markdown('<div class="sec-hdr">📑 Account & Contract</div>', unsafe_allow_html=True)
        r4c1, r4c2 = st.columns(2)
        with r4c1: tenure   = st.number_input("Tenure (months)", 0, 72, 12, key="s_tenure")
        with r4c2: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="s_contract")
        r5c1, r5c2 = st.columns(2)
        with r5c1: payment_method   = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], key="s_pm")
        with r5c2: paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="s_pb")

        st.markdown('<br><div class="sec-hdr">🌐 Services</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        INET_OPTS  = ["Yes", "No", "No internet service"]
        PHONE_OPTS = ["No", "Yes", "No phone service"]
        with sc1:
            phone_service = st.selectbox("Phone",     ["Yes", "No"],  key="s_phone")
            online_sec    = st.selectbox("Security",  INET_OPTS,      key="s_sec")
            streaming_tv  = st.selectbox("Stream TV", INET_OPTS,      key="s_tv")
        with sc2:
            multi_lines   = st.selectbox("Multi Lines", PHONE_OPTS,   key="s_ml")
            online_backup = st.selectbox("Backup",      INET_OPTS,    key="s_bk")
            streaming_mov = st.selectbox("Stream Film",  INET_OPTS,   key="s_mov")
        with sc3:
            internet_svc  = st.selectbox("Internet",   ["DSL", "Fiber optic", "No"], key="s_inet")
            device_prot   = st.selectbox("Protection",  INET_OPTS,    key="s_dp")
            tech_support  = st.selectbox("Support",     INET_OPTS,    key="s_ts")

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, cta_col, _ = st.columns([1, 1.5, 1])
    with cta_col:
        analyze = st.button("⚡  ANALYZE CHURN RISK", key="btn_single")

    # ── Prediction ────────────────────────────────────────────────────────────
    if analyze:
        input_data = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_service, "MultipleLines": multi_lines,
            "InternetService": internet_svc, "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup, "DeviceProtection": device_prot,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov, "Contract": contract,
            "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        with st.spinner("Running ML inference…"):
            try:
                prob       = float(pipeline.predict_proba(input_data)[0][1])
                risk, bdg, rcc = risk_label(prob, threshold)
                pred_label = prediction_label(prob, threshold)
                risk_color = color_for_risk(risk)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card {rcc}">
            <span class="badge {bdg}" style="margin-bottom:1.2rem;">{risk}</span><br>
            <div class="prob-num" style="color:{risk_color};">{prob:.1%}</div>
            <div class="prob-lbl">Churn Probability</div>
            <div style="font-size:1.1rem;font-weight:600;color:#e2e8f0;margin-bottom:.5rem;">
                Prediction: {pred_label}
            </div>
            <div style="font-size:.85rem;color:var(--muted);">
                Decision threshold: {threshold:.2f} &nbsp;|&nbsp;
                Confidence: {"High" if abs(prob - threshold) > 0.25 else "Medium" if abs(prob - threshold) > 0.1 else "Low"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gemini recommendation ─────────────────────────────────────────────
        if gemini_ok:
            with st.spinner("Generating AI business recommendation…"):
                rec = gs.get_recommendation(
                    customer_attrs=input_data.iloc[0].to_dict(),
                    churn_prob=prob, risk_level=risk, prediction=pred_label,
                )
            render_recommendation(rec)
        else:
            st.info("ℹ️ Add your `GEMINI_API_KEY` to `.env` to enable AI business recommendations.", icon="🔑")

    # ── Model info / global features ──────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">⚙️ Model Diagnostics</div>', unsafe_allow_html=True)
    mi_col, fi_col = st.columns([3, 2], gap="large")

    with mi_col:
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi"><div class="kpi-val">XGBoost</div><div class="kpi-lbl">Algorithm</div></div>
            <div class="kpi"><div class="kpi-val">{threshold:.2f}</div><div class="kpi-lbl">Threshold</div></div>
            <div class="kpi"><div class="kpi-val">{metrics['f1']:.3f}</div><div class="kpi-lbl">F1-Score</div></div>
            <div class="kpi"><div class="kpi-val">{metrics['roc_auc']:.3f}</div><div class="kpi-lbl">ROC-AUC</div></div>
            <div class="kpi"><div class="kpi-val">{metrics['pr_auc']:.3f}</div><div class="kpi-lbl">PR-AUC</div></div>
            <div class="kpi"><div class="kpi-val">{metrics['recall']:.3f}</div><div class="kpi-lbl">Recall</div></div>
        </div>
        """, unsafe_allow_html=True)

    with fi_col:
        st.markdown('<p style="color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:.9rem;">Top Churn Drivers</p>', unsafe_allow_html=True)
        render_feature_bars(feat_df_global)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="upload-zone">
        <p style="font-size:1.05rem;font-weight:600;color:#e2e8f0;">📂 Upload Customer CSV for Batch Analysis</p>
        <p>CSV must include the required model features. Optional columns like <code>customerID</code> are preserved in the report.</p>
        <p style="font-size:.8rem;">Required columns: gender · SeniorCitizen · Partner · Dependents · tenure · PhoneService · MultipleLines · InternetService ·
        OnlineSecurity · OnlineBackup · DeviceProtection · TechSupport · StreamingTV · StreamingMovies · Contract · PaperlessBilling · PaymentMethod · MonthlyCharges · TotalCharges</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], key="batch_upload", label_visibility="collapsed")

    if uploaded is not None:
        # ── 1. Load & validate ─────────────────────────────────────────────────
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        if raw_df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        present_cols_lower = {c.lower(): c for c in raw_df.columns}
        missing = [f for f in REQUIRED_FEATURES if f not in raw_df.columns]
        extra   = [c for c in raw_df.columns
                   if c not in REQUIRED_FEATURES and c.lower() not in OPTIONAL_ID_COLS]

        if missing:
            st.error(f"❌ Missing required columns ({len(missing)}): `{'`, `'.join(missing)}`")
            st.stop()

        if extra:
            st.warning(f"⚠️ Unexpected columns (will be ignored for ML): `{'`, `'.join(extra)}`")

        # Detect identifier column
        id_col = None
        for cname in ["customerID", "customerid", "customer_id", "CustomerID"]:
            if cname in raw_df.columns:
                id_col = cname
                break

        # ── 2. ML prediction for ALL customers ────────────────────────────────
        X_batch = raw_df[REQUIRED_FEATURES].copy()
        with st.spinner(f"Running ML predictions for {len(X_batch)} customers…"):
            try:
                probs = pipeline.predict_proba(X_batch)[:, 1]
            except Exception as e:
                st.error(f"ML prediction error: {e}")
                st.stop()

        risk_labels_list = [risk_label(p, threshold)[0] for p in probs]
        pred_labels_list = [prediction_label(p, threshold) for p in probs]

        results_df = raw_df.copy()
        if id_col:
            results_df = results_df[[id_col] + [c for c in results_df.columns if c != id_col]]
        results_df["Churn Probability"] = (probs * 100).round(1).astype(str) + "%"
        results_df["Churn Probability (raw)"] = probs.round(4)
        results_df["Risk Level"]        = risk_labels_list
        results_df["Prediction"]        = pred_labels_list
        results_df["Confidence"]        = [
            "High" if abs(p - threshold) > 0.25 else
            "Medium" if abs(p - threshold) > 0.1 else "Low"
            for p in probs
        ]

        # ── 3. KPI summary ────────────────────────────────────────────────────
        n_total  = len(results_df)
        n_high   = (results_df["Risk Level"] == "High Risk").sum()
        n_med    = (results_df["Risk Level"] == "Medium Risk").sum()
        n_low    = (results_df["Risk Level"] == "Low Risk").sum()
        avg_prob = probs.mean()
        n_churn  = (results_df["Prediction"] == "Likely to Churn").sum()

        st.markdown(f"""
        <div class="kpi-grid" style="margin-top:1.5rem;">
            <div class="kpi"><div class="kpi-val">{n_total:,}</div><div class="kpi-lbl">Total Customers</div></div>
            <div class="kpi"><div class="kpi-val" style="color:var(--red);">{n_high:,}</div><div class="kpi-lbl">High Risk</div></div>
            <div class="kpi"><div class="kpi-val" style="color:var(--amber);">{n_med:,}</div><div class="kpi-lbl">Medium Risk</div></div>
            <div class="kpi"><div class="kpi-val" style="color:var(--green);">{n_low:,}</div><div class="kpi-lbl">Low Risk</div></div>
            <div class="kpi"><div class="kpi-val">{avg_prob:.1%}</div><div class="kpi-lbl">Avg Churn Prob</div></div>
            <div class="kpi"><div class="kpi-val">{n_churn:,}</div><div class="kpi-lbl">Predicted Churners</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4. Charts ─────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        chart_style = {
            "figure.facecolor": "#0b0f19", "axes.facecolor": "#0b0f19",
            "axes.edgecolor": "none", "axes.labelcolor": "#94a3b8",
            "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
            "text.color": "#f8fafc", "grid.color": "rgba(255,255,255,0.05)",
        }
        plt.rcParams.update(chart_style)

        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown('<div class="sec-hdr">Risk Distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.2))
            risk_counts = results_df["Risk Level"].value_counts()
            colors      = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#10b981"}
            for rl, cnt in risk_counts.items():
                ax.bar(rl, cnt, color=colors.get(rl, "#3b82f6"), width=0.55, zorder=3)
                ax.text(list(risk_counts.index).index(rl), cnt + max(risk_counts) * 0.01,
                        str(cnt), ha="center", va="bottom", fontsize=9, color="#f8fafc")
            ax.set_ylabel("Customers"); ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with ch2:
            st.markdown('<div class="sec-hdr">Churn Probability Distribution</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 3.2))
            ax2.hist(probs, bins=30, color="#3b82f6", alpha=0.8, edgecolor="none", zorder=3)
            ax2.axvline(threshold, color="#ef4444", linewidth=1.5, linestyle="--", label=f"Threshold {threshold:.2f}")
            ax2.set_xlabel("Churn Probability"); ax2.set_ylabel("Count")
            ax2.legend(fontsize=8); ax2.grid(axis="y", zorder=0); ax2.set_axisbelow(True)
            fig2.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

        # ── 5. Top-N table ────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr">🔴 Highest Risk Customers</div>', unsafe_allow_html=True)
        top_n = st.slider("Show top N highest-risk customers", 5, min(100, n_total), min(20, n_total), key="top_n_slider")
        top_df = (
            results_df
            .sort_values("Churn Probability (raw)", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        display_cols = ([id_col] if id_col else []) + ["Churn Probability", "Risk Level", "Prediction", "Confidence"]
        st.dataframe(top_df[display_cols], use_container_width=True, hide_index=True)

        # ── 6. Filterable full table ───────────────────────────────────────────
        with st.expander("🔍 Filter & Browse All Customers"):
            filter_risk = st.multiselect("Filter by Risk Level",
                ["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk", "Medium Risk", "Low Risk"], key="filt_risk")
            filter_pred = st.multiselect("Filter by Prediction",
                ["Likely to Churn", "Not Likely to Churn"],
                default=["Likely to Churn", "Not Likely to Churn"], key="filt_pred")

            filtered = results_df[
                (results_df["Risk Level"].isin(filter_risk)) &
                (results_df["Prediction"].isin(filter_pred))
            ].reset_index(drop=True)

            st.markdown(f"<p style='color:var(--muted);font-size:.85rem;'>{len(filtered)} customers match filters</p>", unsafe_allow_html=True)
            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

        # ── 7. Gemini batch recommendations ───────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-hdr">🤖 AI Business Recommendations</div>', unsafe_allow_html=True)

        if not gemini_ok:
            st.info("ℹ️ Add your `GEMINI_API_KEY` to `.env` to enable AI recommendations.", icon="🔑")
        else:
            gem_scope = st.radio(
                "Generate AI recommendations for:",
                ["Top 10 Highest Risk", "Top 20 Highest Risk", "All High Risk", "High + Medium Risk"],
                horizontal=True, key="gem_scope",
            )

            scope_df = results_df.sort_values("Churn Probability (raw)", ascending=False)
            if gem_scope == "Top 10 Highest Risk":
                scope_df = scope_df.head(10)
            elif gem_scope == "Top 20 Highest Risk":
                scope_df = scope_df.head(20)
            elif gem_scope == "All High Risk":
                scope_df = scope_df[scope_df["Risk Level"] == "High Risk"]
            else:
                scope_df = scope_df[scope_df["Risk Level"].isin(["High Risk", "Medium Risk"])]

            scope_df = scope_df.reset_index(drop=True)
            st.markdown(f"<p style='color:var(--muted);font-size:.85rem;'>Will call Gemini for <strong style='color:#e2e8f0;'>{len(scope_df)}</strong> customers.</p>", unsafe_allow_html=True)

            if st.button("🚀 Generate AI Recommendations", key="btn_gemini_batch"):
                progress = st.progress(0, text="Calling Gemini API…")
                batch_recs = []

                for i, (_, row) in enumerate(scope_df.iterrows()):
                    prob_val = row["Churn Probability (raw)"]
                    attrs    = {c: row[c] for c in REQUIRED_FEATURES if c in row}
                    rec      = gs.get_recommendation(
                        customer_attrs=attrs,
                        churn_prob=prob_val,
                        risk_level=row["Risk Level"],
                        prediction=row["Prediction"],
                    )
                    batch_recs.append(rec)
                    progress.progress((i + 1) / len(scope_df),
                                      text=f"Generating… {i+1}/{len(scope_df)}")

                progress.empty()
                st.success(f"✅ Generated recommendations for {len(scope_df)} customers.")

                # Assemble combined report df
                rec_df = pd.DataFrame(batch_recs)
                if id_col:
                    rec_df.insert(0, id_col, scope_df[id_col].values)
                rec_df.insert(1 if id_col else 0, "Churn Probability", scope_df["Churn Probability"].values)
                rec_df.insert(2 if id_col else 1, "Risk Level",        scope_df["Risk Level"].values)
                rec_df.insert(3 if id_col else 2, "Prediction",        scope_df["Prediction"].values)

                st.dataframe(rec_df, use_container_width=True, hide_index=True)

                # Store in session for export
                st.session_state["batch_rec_df"] = rec_df

        # ── 8. Downloads ───────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr" style="margin-top:2rem;">📥 Export Reports</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            csv_pred = results_df.drop(columns=["Churn Probability (raw)"], errors="ignore").to_csv(index=False)
            st.download_button(
                label="⬇️  Download ML Prediction Report (CSV)",
                data=csv_pred,
                file_name="churniq_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_pred",
            )

        with dl2:
            rec_data = st.session_state.get("batch_rec_df")
            if rec_data is not None:
                csv_rec = rec_data.to_csv(index=False)
                st.download_button(
                    label="⬇️  Download AI Recommendation Report (CSV)",
                    data=csv_rec,
                    file_name="churniq_ai_recommendations.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_rec",
                )
            else:
                st.button("⬇️  Download AI Report (generate first)", disabled=True, use_container_width=True, key="dl_rec_dis")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    <strong>ChurnIQ</strong> &bull; AI-Powered Customer Retention Platform &bull;
    ML predictions are provided for decision-support only and should be reviewed by a human retention specialist.
</div>
""", unsafe_allow_html=True)
