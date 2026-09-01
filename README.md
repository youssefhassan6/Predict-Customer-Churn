# ChurnIQ — AI-Powered Customer Retention Platform

> **Predict customer churn before it happens.**  
> Advanced XGBoost analytics + Gemini AI business intelligence — identify at-risk customers and deploy precision retention strategies before churn occurs.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [File Structure](#file-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Team](#team)
- [License](#license)

---

## Overview

**ChurnIQ** is an end-to-end AI platform that helps telecom companies predict and prevent customer churn. It combines a production-grade **XGBoost** machine learning pipeline with **Google Gemini** generative AI to deliver both accurate predictions and actionable business recommendations.

The platform supports two operational modes:
- **Single Customer Analysis** — Manual input with instant prediction + AI recommendation
- **Batch Analysis** — CSV upload for bulk scoring with risk dashboards and export reports

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Dual Prediction Modes** | Single customer (form-based) and batch (CSV upload) |
| **XGBoost ML Engine** | Optimized pipeline with threshold tuning and cross-validation |
| **3-Tier Risk Classification** | Low / Medium / High risk with color-coded badges |
| **Gemini AI Recommendations** | Auto-generated, prioritized retention strategies per customer |
| **Interactive Dashboards** | Risk distribution charts, probability histograms, KPI grids |
| **Export Reports** | Downloadable CSV for predictions and AI recommendations |
| **Dark Glassmorphism UI** | Modern, responsive Streamlit interface with custom CSS |
| **PII Protection** | Automatic stripping of sensitive fields before Gemini API calls |
| **Graceful Degradation** | App works without Gemini API key (recommendations disabled) |

---

## Architecture

```
+------------------+      +------------------+      +------------------+
|   Streamlit UI   | <--> |  Python Backend  | <--> |   XGBoost ML     |
|   (app.py)       |      |  (app.py logic)  |      |   Pipeline       |
+------------------+      +------------------+      +------------------+
         |                                               |
         v                                               v
+------------------+                          +------------------+
|  Gemini Service  |                          |  churn_artifacts |
|  (gemini_service)|                          |     .pkl       |
+------------------+                          +------------------+
```

**Data Flow:**
1. User inputs customer data (manual or CSV)
2. XGBoost pipeline predicts churn probability
3. Risk level is computed from optimized threshold
4. Gemini generates business recommendation (if API key configured)
5. Results displayed with charts, tables, and export options

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit + Custom CSS (Dark Theme / Glassmorphism) |
| **ML Framework** | scikit-learn, XGBoost, pandas, numpy |
| **Gen AI** | Google Gemini 2.0 Flash (via `google-genai` SDK) |
| **Visualization** | matplotlib, seaborn |
| **Data Storage** | CSV (in-memory) |
| **Environment** | python-dotenv |

---

## ML Pipeline

### 1. Data Preprocessing
- **Missing values**: Median imputation (numeric), mode imputation (categorical)
- **Outlier handling**: Custom `IQRCapper` transformer (IQR rule with capping)
- **Skewness correction**: Log1p transformation on charge variables
- **Encoding**: One-Hot Encoding (multi-category), Ordinal Encoding (binary)
- **Scaling**: StandardScaler on numeric features

### 2. Model Training
- **Algorithm**: XGBoost Classifier with `scale_pos_weight` for class imbalance
- **Hyperparameter tuning**: GridSearchCV with Stratified 3-Fold CV
- **Optimization metric**: F1-score
- **Threshold optimization**: Custom threshold search (not fixed 0.5) via cross-validation

### 3. Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | ~0.76 |
| Precision | ~0.54 |
| Recall | ~0.75 |
| F1-Score | ~0.63 |
| ROC-AUC | ~0.83 |
| PR-AUC | ~0.60 |

### 4. Feature Importance
Top churn drivers identified by the model:
1. **Contract type** (Month-to-month = highest risk)
2. **Tenure** (shorter = higher risk)
3. **MonthlyCharges / TotalCharges**
4. **InternetService** (Fiber optic)
5. **PaymentMethod** (Electronic check)

---

## File Structure

```
churniq/
├── app.py                          # Main Streamlit application
├── train_model.py                  # ML pipeline training script
├── gemini_service.py               # Gemini AI recommendation service
├── custom_transformers.py          # Custom sklearn transformer (IQRCapper)
├── customer_churn_prediction_professional.py  # EDA & model comparison notebook
├── churn_artifacts.pkl             # Serialized model, threshold, metrics, features
├── customer_churn.csv              # Dataset (7,043 customers)
├── .env                            # Environment variables (GEMINI_API_KEY)
└── README.md                       # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone <repo-url>
cd churniq
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn joblib python-dotenv google-genai
```

### 4. Configure Gemini API (Optional)
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
> The app works without the API key — Gemini recommendations will be disabled and a fallback message shown.

### 5. Train the model (if `churn_artifacts.pkl` is missing)
```bash
python train_model.py
```

### 6. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

### Tab 1 — Single Customer Prediction
1. Fill in customer profile, financial, account, and service details
2. Click **"Analyze Churn Risk"**
3. View churn probability, risk badge, prediction, and confidence level
4. If Gemini is connected, an AI business recommendation card appears
5. Model diagnostics (KPIs + feature importance bars) shown below

### Tab 2 — Batch Analysis
1. Upload a CSV file with required columns (see below)
2. View instant KPI summary (Total, High/Med/Low risk, avg probability)
3. Explore interactive charts (Risk Distribution + Probability Histogram)
4. Browse top-N highest risk customers
5. Filter full dataset by risk level and prediction
6. Generate Gemini AI recommendations for selected scope
7. Export both prediction and recommendation reports as CSV

### Required CSV Columns
```
gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
```
Optional ID columns: `customerID`, `customer name`, `email`, `phone`

---

## Screenshots

> Add your screenshots here:
> - `assets/screenshot_single.png` — Single customer prediction result
> - `assets/screenshot_batch.png` — Batch analysis dashboard
> - `assets/screenshot_gemini.png` — Gemini recommendation card

---

## Team

| Name | Role |
|------|------|
| **Youssef** | Business Problem, Dataset, Conclusion & Future Work |
| **Nabil** | Data Preprocessing, XGBoost Model, Threshold Optimization, Architecture |
| **Ali** | Streamlit UI/UX, Single & Batch Demo, Gemini Integration, Results |

---

## License

This project is for educational and demonstration purposes.  
ML predictions are provided for **decision-support only** and should be reviewed by a human retention specialist before taking action.

---

<p align="center">
  <strong>ChurnIQ</strong> &bull; AI-Powered Customer Retention Platform &bull; Built with ❤️ by Youssef, Nabil & Ali
</p>
