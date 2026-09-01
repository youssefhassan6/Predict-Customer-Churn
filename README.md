# 🔮ChurnIQ — AI-Powered Customer Retention Platform

> **Predict customer churn before it happens.**

Advanced XGBoost analytics + Gemini AI business intelligence — identify at-risk customers and deploy precision retention strategies before churn occurs.

## 🚀 Live Demo

**[Open ChurnIQ Live Demo](http://customer-churn-prediction-6.streamlit.app/)**

> **Demo Note:** The live demo is currently deployed **without Gemini AI integration**, so AI-generated retention recommendations are disabled in the deployed version. The full project supports Gemini when a valid `GEMINI_API_KEY` is configured locally.

---

## 📚 Table of Contents

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

**ChurnIQ** is an end-to-end AI platform that helps telecom companies predict and prevent customer churn. It combines a production-grade **XGBoost** machine learning pipeline with **Google Gemini** generative AI to deliver accurate predictions and actionable business recommendations.

The platform supports two operational modes:

- **Single Customer Analysis** — Manual input with instant prediction and AI recommendation
- **Batch Analysis** — CSV upload for bulk scoring with risk dashboards and export reports

---

## Key Features

| Feature | Description |
|---|---|
| **Dual Prediction Modes** | Single customer form-based prediction and batch CSV analysis |
| **XGBoost ML Engine** | Optimized machine learning pipeline with threshold tuning and cross-validation |
| **3-Tier Risk Classification** | Low / Medium / High risk with color-coded indicators |
| **Gemini AI Recommendations** | Auto-generated, prioritized customer retention strategies |
| **Interactive Dashboards** | Risk distribution charts, probability histograms, KPI cards, and customer tables |
| **Batch Processing** | Upload multiple customers and score them in one operation |
| **Export Reports** | Download prediction and recommendation results as CSV |
| **Dark Glassmorphism UI** | Modern responsive Streamlit interface with custom CSS |
| **PII Protection** | Sensitive fields are stripped before Gemini API calls |
| **Graceful Degradation** | The application remains usable without a Gemini API key |

---

## Architecture

```text
+------------------+      +------------------+      +------------------+
|   Streamlit UI   | <--> |  Python Backend  | <--> |   XGBoost ML     |
|     (app.py)     |      |   App Logic      |      |    Pipeline      |
+------------------+      +------------------+      +------------------+
          |                                              |
          v                                              v
+------------------+                          +------------------------+
|  Gemini Service  |                          |   churn_artifacts.pkl  |
| gemini_service.py|                          | Model + Threshold +    |
+------------------+                          | Metrics + Feature Data |
                                             +------------------------+
```

### Data Flow

1. User enters customer information manually or uploads a CSV file.
2. The XGBoost pipeline predicts the probability of churn.
3. The optimized threshold is used to determine the prediction and risk level.
4. Gemini generates a business recommendation when the API is configured.
5. Results are displayed through dashboards, tables, visualizations, and exportable reports.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS |
| **Machine Learning** | scikit-learn, XGBoost, pandas, NumPy |
| **Generative AI** | Google Gemini via `google-genai` |
| **Visualization** | Matplotlib, Seaborn |
| **Model Serialization** | Joblib / Pickle |
| **Environment Management** | python-dotenv |
| **Data Storage** | CSV / in-memory processing |

---

## ML Pipeline

### 1. Data Preprocessing

- **Missing values:** Median imputation for numerical variables and mode imputation for categorical variables
- **Outlier handling:** Custom `IQRCapper` transformer using the IQR rule
- **Skewness correction:** Log1p transformation on charge-related variables
- **Encoding:** One-Hot Encoding for multi-category variables and Ordinal Encoding for binary variables
- **Scaling:** `StandardScaler` for numerical features

### 2. Model Training

- **Algorithm:** XGBoost Classifier
- **Class imbalance:** `scale_pos_weight`
- **Hyperparameter tuning:** `GridSearchCV`
- **Cross-validation:** Stratified 3-Fold CV
- **Optimization metric:** F1-score
- **Threshold optimization:** Custom threshold search instead of using a fixed 0.5 cutoff

### 3. Test Set Performance

| Metric | Value |
|---|---:|
| **Accuracy** | ~0.76 |
| **Precision** | ~0.54 |
| **Recall** | ~0.75 |
| **F1-Score** | ~0.63 |
| **ROC-AUC** | ~0.83 |
| **PR-AUC** | ~0.60 |

### 4. Top Churn Drivers

1. **Contract Type** — Month-to-month customers show the highest risk
2. **Tenure** — Shorter-tenure customers are more likely to churn
3. **Monthly Charges / Total Charges**
4. **Internet Service** — Fiber optic customers are a notable risk segment
5. **Payment Method** — Electronic check is associated with higher churn risk

---

## File Structure

```text
churniq/
├── app.py
├── train_model.py
├── gemini_service.py
├── custom_transformers.py
├── customer_churn_prediction_professional.py
├── churn_artifacts.pkl
├── customer_churn.csv
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

### File Responsibilities

- `app.py` — Main Streamlit application
- `train_model.py` — Model training and artifact generation
- `gemini_service.py` — Gemini AI recommendation service
- `custom_transformers.py` — Custom `IQRCapper` transformer
- `churn_artifacts.pkl` — Serialized model, threshold, metrics, and feature metadata
- `customer_churn.csv` — Telecom churn dataset
- `.env` — Local environment variables such as the Gemini API key

> **Security:** Never commit a real API key to GitHub. Keep secrets inside `.env` and add `.env` to `.gitignore`.

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip
- Git

### 1. Clone the Repository

```bash
git clone <repo-url>
cd churniq
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn joblib python-dotenv google-genai
```

### 4. Configure Gemini AI (Optional)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

Gemini is optional. The application can still perform churn prediction and batch analysis without the API key.

### 5. Train the Model

Run this only when `churn_artifacts.pkl` is missing or you want to retrain:

```bash
python train_model.py
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Usage

### Tab 1 — Single Customer Prediction

1. Enter the customer's profile, account, service, and financial information.
2. Click **Analyze Churn Risk**.
3. View the churn probability, prediction, risk level, and confidence information.
4. When Gemini is configured, view the AI-powered retention recommendation.
5. Review model diagnostics and feature importance.

### Tab 2 — Batch Analysis

1. Upload a CSV containing the required customer columns.
2. View KPI summaries such as total customers, high/medium/low-risk counts, and average churn probability.
3. Explore the risk distribution and probability histogram.
4. Review the highest-risk customers.
5. Filter customers by risk level and prediction.
6. Generate Gemini recommendations when the service is configured.
7. Export prediction and recommendation reports as CSV files.

### Required CSV Columns

```text
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

Optional identification columns:

```text
customerID
customer name
email
phone
```

> **Important:** If your uploaded CSV contains a `Churn` column, it should be treated as the **actual target / ground-truth label**, not as an input feature for the prediction model.

---



## Team

| Name | 
|---|
| **Youssef** |
| **Nabil** | 
| **Ali** | 

---

## License

This project is for educational and demonstration purposes.

ML predictions are provided for **decision-support only** and should be reviewed by a human retention specialist before taking action.

---

<p align="center">
  <strong>ChurnIQ</strong> • AI-Powered Customer Retention Platform • Built with ❤️ by Youssef, Nabil & Ali
</p>
