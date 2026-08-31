import streamlit as st
import pandas as pd
import joblib
import numpy as np
import custom_transformers

# =========================
# Page Configuration & Aesthetics
# =========================
st.set_page_config(
    page_title="ChurnIQ | AI Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --bg-color: #0b0f19;
        --card-bg: rgba(26, 32, 53, 0.7);
        --card-border: rgba(255, 255, 255, 0.08);
        --primary: #3b82f6;
        --primary-glow: rgba(59, 130, 246, 0.4);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --risk-low: #10b981;
        --risk-med: #f59e0b;
        --risk-high: #ef4444;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main Background */
    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(at 0% 0%, rgba(15, 23, 42, 1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, rgba(30, 58, 138, 0.15) 0, transparent 50%), 
            radial-gradient(at 100% 0%, rgba(15, 23, 42, 1) 0, transparent 50%);
        color: var(--text-main);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Top Navigation */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--card-border);
        margin-bottom: 2rem;
        border-radius: 0 0 16px 16px;
    }
    .brand {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-accent {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--risk-low);
        background: rgba(16, 185, 129, 0.1);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--risk-low);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--risk-low);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Hero Section */
    .hero {
        text-align: center;
        margin-bottom: 3rem;
        padding: 0 1rem;
    }
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.15rem;
        color: var(--text-muted);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid var(--card-border);
        padding-bottom: 0.75rem;
    }
    
    /* Input Styling overrides */
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    input[type="number"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    .stNumberInput > div > div > div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: none !important;
    }
    
    /* Primary CTA Button */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
        box-shadow: 0 4px 20px -2px var(--primary-glow) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px -2px rgba(59, 130, 246, 0.6) !important;
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    }
    
    /* Result Card Styles */
    .result-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .risk-high::before { background: var(--risk-high); }
    .risk-med::before { background: var(--risk-med); }
    .risk-low::before { background: var(--risk-low); }
    
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        letter-spacing: 1px;
    }
    .badge-high { background: rgba(239, 68, 68, 0.1); color: var(--risk-high); border: 1px solid rgba(239, 68, 68, 0.2); }
    .badge-med { background: rgba(245, 158, 11, 0.1); color: var(--risk-med); border: 1px solid rgba(245, 158, 11, 0.2); }
    .badge-low { background: rgba(16, 185, 129, 0.1); color: var(--risk-low); border: 1px solid rgba(16, 185, 129, 0.2); }
    
    .prob-value {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .prob-label {
        color: var(--text-muted);
        font-size: 1.1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 2rem;
    }
    
    /* Model Info Section */
    .model-info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .model-metric {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--card-border);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 3rem 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        border-top: 1px solid var(--card-border);
        margin-top: 4rem;
    }
    
    /* Feature Importance Bars */
    .feat-bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    .feat-name {
        width: 150px;
        text-align: right;
        padding-right: 1rem;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .feat-bar-bg {
        flex-grow: 1;
        height: 8px;
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        overflow: hidden;
    }
    .feat-bar-fill {
        height: 100%;
        background: var(--primary);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model Artifacts
# =========================
@st.cache_resource
def load_artifacts():
    try:
        return joblib.load("churn_artifacts.pkl")
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

# =========================
# UI Layout
# =========================

# Navigation Header
st.markdown("""
<div class="top-nav">
    <div class="brand">⚡ Churn<span class="brand-accent">IQ</span></div>
    <div class="status-indicator">
        <div class="status-dot"></div>
        MODEL ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero">
    <h1>Predict Customer Churn Before It Happens</h1>
    <p>Leverage our advanced XGBoost predictive analytics engine to assess customer risk profiles in real-time, interpret key drivers, and proactively deploy retention strategies.</p>
</div>
""", unsafe_allow_html=True)


if artifacts is None:
    st.error("Model artifacts not found. Please ensure 'churn_artifacts.pkl' is present.")
    st.stop()

pipeline = artifacts["pipeline"]
threshold = artifacts["optimal_threshold"]
metrics = artifacts["metrics"]
feature_names = artifacts["feature_names"]

MODEL_INPUT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

TOTAL_CHARGES_RATIO_MIN = 0.69
TOTAL_CHARGES_RATIO_MAX = 1.58


def estimated_total_charges(tenure_months, monthly_charge):
    return round(float(tenure_months) * float(monthly_charge), 2)


def get_churn_probability(model, input_data):
    probabilities = model.predict_proba(input_data)[0]
    classifier = model.named_steps.get("classifier")
    classes = getattr(classifier, "classes_", getattr(model, "classes_", None))

    if classes is None:
        raise ValueError("The loaded model does not expose class labels for probability mapping.")

    class_list = list(classes)
    if 1 in class_list:
        churn_index = class_list.index(1)
    elif "Yes" in class_list:
        churn_index = class_list.index("Yes")
    elif True in class_list:
        churn_index = class_list.index(True)
    else:
        raise ValueError(f"Could not identify the positive churn class from model classes: {class_list}")

    return float(probabilities[churn_index]), class_list


def build_result_summary(churn_probability, decision_threshold):
    high_risk_threshold = min(0.9, max(decision_threshold + 0.15, decision_threshold * 1.5))

    if churn_probability >= high_risk_threshold:
        return {
            "prediction_text": "Likely to Churn",
            "risk_level": "High Risk",
            "badge_class": "badge-high",
            "card_class": "risk-high",
            "color_hex": "#ef4444",
            "action": "Prioritize retention outreach. Consider a targeted incentive, service review, or direct follow-up from the retention team."
        }

    if churn_probability >= decision_threshold:
        return {
            "prediction_text": "At Risk",
            "risk_level": "Medium Risk",
            "badge_class": "badge-med",
            "card_class": "risk-med",
            "color_hex": "#f59e0b",
            "action": "Monitor engagement closely and consider a personalized offer or support check-in before the customer escalates to high risk."
        }

    return {
        "prediction_text": "Likely to Stay",
        "risk_level": "Low Risk",
        "badge_class": "badge-low",
        "card_class": "risk-low",
        "color_hex": "#10b981",
        "action": "No immediate retention action is required. Continue normal engagement and maintain service quality."
    }

# Attempt to extract feature importances
try:
    xgb_model = pipeline.named_steps["classifier"]
    importances = xgb_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(5)
except:
    feat_df = None


# --- Input Form Area ---
with st.container():
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">👤 Personal Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: gender = st.selectbox("Gender", ["Female", "Male"])
        with c2: senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
        c3, c4 = st.columns(2)
        with c3: partner = st.selectbox("Partner", ["Yes", "No"])
        with c4: dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        st.markdown('<br><div class="section-header">💳 Financial Data</div>', unsafe_allow_html=True)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=130.0, value=70.0, step=5.0)

    with col2:
        st.markdown('<div class="section-header">📑 Account & Subscriptions</div>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7: tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
        with c8: contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
        c9, c10 = st.columns(2)
        with c9: payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with c10: paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        st.markdown('<br><div class="section-header">🌐 Subscribed Services</div>', unsafe_allow_html=True)
        service_parent_1, service_parent_2 = st.columns(2)
        with service_parent_1:
            internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        with service_parent_2:
            phone_service = st.selectbox("Phone", ["Yes", "No"])

        service_1, service_2, service_3 = st.columns(3)
        internet_disabled = internet_service == "No"
        internet_options = ["No internet service"] if internet_disabled else ["Yes", "No"]
        internet_help = (
            "Automatically set because Internet Service is No."
            if internet_disabled
            else "Choose whether this internet add-on is active."
        )

        with service_1:
            online_security = st.selectbox("Security", internet_options, disabled=internet_disabled, help=internet_help)
            streaming_tv = st.selectbox("Stream TV", internet_options, disabled=internet_disabled, help=internet_help)
        with service_2:
            online_backup = st.selectbox("Backup", internet_options, disabled=internet_disabled, help=internet_help)
            streaming_movies = st.selectbox("Stream Movies", internet_options, disabled=internet_disabled, help=internet_help)
        with service_3:
            device_protection = st.selectbox("Protection", internet_options, disabled=internet_disabled, help=internet_help)
            tech_support = st.selectbox("Support", internet_options, disabled=internet_disabled, help=internet_help)

        if internet_disabled:
            st.caption("Internet add-ons are automatically set to 'No internet service' because this customer has no internet plan.")

        phone_disabled = phone_service == "No"
        multiple_line_options = ["No phone service"] if phone_disabled else ["No", "Yes"]
        multiple_lines = st.selectbox(
            "Multiple Lines",
            multiple_line_options,
            disabled=phone_disabled,
            help="Automatically set because Phone Service is No." if phone_disabled else "Choose whether the phone plan has multiple lines."
        )
        if phone_disabled:
            st.caption("Multiple Lines is automatically set to 'No phone service' because this customer has no phone plan.")

st.markdown('<br><div class="section-header">💰 Total Charges Consistency</div>', unsafe_allow_html=True)
calculated_total_charges = estimated_total_charges(tenure, monthly_charges)

if tenure == 0:
    total_charges = np.nan
    st.info("Tenure is 0 months, so Total Charges is treated as missing/new-customer history. The model pipeline will handle it with its trained imputer.")
else:
    total_mode = st.radio(
        "Total Charges",
        ["Use calculated estimate", "Enter known historical total"],
        horizontal=True,
        help="Total Charges should stay connected to tenure and monthly charges. Use the estimate unless you have the actual historical billing total."
    )

    if total_mode == "Use calculated estimate":
        total_charges = calculated_total_charges
        st.metric("Estimated Total Charges ($)", f"${total_charges:,.2f}")
        st.caption("Calculated from the selected tenure and monthly charge. Switch modes only if you know the customer's actual historical total.")
    else:
        lower_total = max(0.0, calculated_total_charges * TOTAL_CHARGES_RATIO_MIN)
        upper_total = min(9000.0, max(calculated_total_charges * TOTAL_CHARGES_RATIO_MAX, monthly_charges))
        total_charges = st.number_input(
            "Known Historical Total Charges ($)",
            min_value=float(lower_total),
            max_value=float(upper_total),
            value=float(calculated_total_charges),
            step=50.0,
            key=f"known_total_{tenure}_{monthly_charges}",
            help="Limited to the observed dataset relationship between Total Charges, Tenure, and Monthly Charges."
        )
        st.caption(f"Allowed range: ${lower_total:,.2f} to ${upper_total:,.2f}, based on the dataset's observed Total Charges relationship.")

# --- CTA ---
st.markdown("<br><br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_button = st.button("Analyze Churn Risk")


# --- Prediction Logic & Result ---
if predict_button:
    # Build dataframe directly from raw inputs - Pipeline handles the rest!
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    }, columns=MODEL_INPUT_COLUMNS)
    
    # Inference using the model's actual positive-class probability and optimized threshold.
    prob, model_classes = get_churn_probability(pipeline, input_data)
    confidence = max(prob, 1 - prob)
    result = build_result_summary(prob, threshold)

    # Render Result
    st.html(f"""
    <div class="result-card {result["card_class"]}">
        <div class="risk-badge {result["badge_class"]}">
            {result["risk_level"]}
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: #fff;">
            Prediction: {result["prediction_text"]}
        </div>
        <div class="prob-value" style="color: {result["color_hex"]};">{prob:.1%}</div>
        <div class="prob-label">Estimated Churn Probability</div>

        <div style="color: #e2e8f0; font-size: 1rem; line-height: 1.8; margin-bottom: 1.75rem;">
            <strong>Risk Level:</strong> {result["risk_level"]}<br>
            <strong>Decision Threshold:</strong> {threshold:.2f}<br>
            <strong>Model Confidence:</strong> {confidence:.1%}
        </div>
        
        <div style="max-width: 600px; margin: 0 auto; color: var(--text-muted); line-height: 1.6; padding-top: 1rem; border-top: 1px solid var(--card-border);">
            <strong>Suggested Business Action:</strong><br> {result["action"]}
        </div>
    </div>
    """)


# --- Model Information & Global Interpretability ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ System Architecture & Global Insights</div>', unsafe_allow_html=True)

mc1, mc2 = st.columns([3, 2], gap="large")

with mc1:
    st.markdown("""
    <p style="color: var(--text-muted); margin-bottom: 1.5rem;">
    Our machine learning pipeline uses XGBoost combined with a dynamically optimized classification threshold to prioritize F1-score over raw accuracy, maximizing business value by balancing Precision and Recall.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="model-info-grid">
        <div class="model-metric">
            <div class="metric-val">XGBoost</div>
            <div class="metric-lbl">Core Algorithm</div>
        </div>
        <div class="model-metric">
            <div class="metric-val">{threshold:.2f}</div>
            <div class="metric-lbl">Optimal Threshold</div>
        </div>
        <div class="model-metric">
            <div class="metric-val">{metrics['f1']:.3f}</div>
            <div class="metric-lbl">F1-Score</div>
        </div>
        <div class="model-metric">
            <div class="metric-val">{metrics['roc_auc']:.3f}</div>
            <div class="metric-lbl">ROC-AUC</div>
        </div>
        <div class="model-metric">
            <div class="metric-val">{metrics['pr_auc']:.3f}</div>
            <div class="metric-lbl">PR-AUC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with mc2:
    if feat_df is not None:
        st.markdown("<p style='color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;'>Top 5 Global Churn Factors</p>", unsafe_allow_html=True)
        max_imp = feat_df['Importance'].max()
        html_bars = ""
        for _, row in feat_df.iterrows():
            pct = (row['Importance'] / max_imp) * 100
            name = row['Feature'].replace('InternetService_', 'Net: ').replace('PaymentMethod_', 'Pay: ').replace('Contract_', 'Contract: ')
            if len(name) > 18:
                name = name[:16] + ".."
            html_bars += f"""
            <div class="feat-bar-container">
                <div class="feat-name">{name}</div>
                <div class="feat-bar-bg">
                    <div class="feat-bar-fill" style="width: {pct}%;"></div>
                </div>
            </div>
            """
        st.markdown(html_bars, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="custom-footer">
    <strong>ChurnIQ</strong> &bull; Machine Learning Customer Analytics &bull; Capstone Project Deployment
</div>
""", unsafe_allow_html=True)
