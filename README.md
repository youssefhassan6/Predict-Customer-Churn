# 🔮 ChurnIQ — AI-Powered Customer Retention Platform

A production-style **Customer Churn Prediction & Retention Intelligence platform** built with **XGBoost**, **scikit-learn**, **Streamlit**, and **Google Gemini AI**.

ChurnIQ is designed to identify customers who are most likely to leave, quantify their churn risk, and generate actionable retention recommendations. The platform supports both **single-customer analysis** and **batch CSV processing**, making it suitable for interactive analysis as well as larger customer portfolios.

---

## 🎯 Project Overview

Customer churn is not only a machine learning problem — it is a business problem.

The objective of ChurnIQ is to move beyond a simple **Churn / No Churn** prediction and provide a complete workflow:

**Customer Data → ML Prediction → Risk Scoring → Customer Segmentation → AI Recommendations → Retention Action**

The system combines a carefully engineered ML pipeline with an AI recommendation layer to transform predictions into business-oriented insights.

---

# 🚀 Key Features

### 🤖 Machine Learning

* End-to-end **scikit-learn Pipeline**
* **XGBoost** as the final production model
* Missing-value handling
* IQR-based outlier capping
* Skewness-aware transformations
* Feature scaling
* Ordinal and One-Hot encoding
* Stratified train/test split
* Cross-validation with `StratifiedKFold`
* Hyperparameter tuning with `GridSearchCV`
* Class-imbalance handling using XGBoost's `scale_pos_weight`
* F1-score optimization instead of relying on accuracy alone
* Cross-validation based classification threshold optimization
* Prevention of preprocessing and threshold-selection leakage

### 📊 Customer Risk Intelligence

For every customer, the system provides:

* Churn probability
* Churn prediction
* Risk level: **Low / Medium / High**
* Confidence indicator
* Ranked customer risk profile

### 📂 Batch Customer Analysis

Upload a CSV containing multiple customers and generate a complete churn analysis report.

The batch dashboard includes:

* Total customers analyzed
* High / Medium / Low risk counts
* Average churn probability
* Risk distribution
* Churn probability histogram
* Top-N highest-risk customers
* Filterable customer table
* Exportable prediction report

### 🧠 Gemini AI Recommendations

Google Gemini is integrated as an additional intelligence layer.

Instead of returning only:

> "This customer is likely to churn."

ChurnIQ can generate a business-oriented recommendation based on the customer's available profile and ML prediction.

Example:

> High churn risk — consider a retention offer, service-quality intervention, or contract upgrade before the customer becomes inactive.

Gemini can be used for:

* Individual customer recommendations
* Top 10 highest-risk customers
* Top 20 highest-risk customers
* All High-Risk customers
* High + Medium risk segments

AI recommendations remain separate from the ML prediction itself, allowing the predictive model to operate independently.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Customer Dataset  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Imputation          │
                    │ IQR Capping         │
                    │ Transformations     │
                    │ Encoding / Scaling  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      XGBoost        │
                    │  Churn Prediction   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Risk Classification │
                    │ Probability         │
                    │ Threshold Decision   │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌───────────────────┐    ┌────────────────────┐
        │ Streamlit Dashboard│    │   Gemini AI Layer  │
        │ Visualization      │    │ Recommendations    │
        │ Filtering          │    │ Retention Actions  │
        └─────────┬─────────┘    └──────────┬─────────┘
                  │                         │
                  └────────────┬────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Retention Insights  │
                    │ & Actionable Report │
                    └─────────────────────┘
```

---

# 📁 Project Structure

```text
Final/
│
├── app.py                      # Streamlit dashboard
├── train_model.py              # Model training pipeline
├── gemini_service.py           # Gemini API integration
├── custom_transformers.py      # Custom serialization-safe transformers
│
├── customer_churn.csv          # Source dataset
├── churn_artifacts.pkl         # Generated trained model artifacts
│
├── requirements.txt             # Python dependencies
├── .env                         # API key configuration (NOT committed)
├── .gitignore                   # Git exclusions
└── README.md
```

---

# ⚙️ Setup & Installation

## 1. Clone the Repository

```bash
git clone https://github.com/youssefhassan6/Predict-Customer-Churn.git
cd Predict-Customer-Churn
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 3. Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

The application can still run without Gemini.

When the API key is unavailable:

* ML predictions remain fully functional
* Churn probabilities are still generated
* Risk classification remains available
* Gemini recommendations are disabled

Never commit `.env` to Git.

---

# 🧠 4. Train the Model

Run:

```bash
python train_model.py
```

The training process generates:

```text
churn_artifacts.pkl
```

The artifact contains the production model pipeline and the information required for inference, including the optimized classification threshold and evaluation metrics.

---

# ▶️ 5. Run the Dashboard

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

---

# 🧪 Machine Learning Methodology

## Data Split

The dataset is divided using:

```text
80% Training
20% Testing
```

with:

```python
stratify=y
random_state=42
```

This preserves the class distribution between training and testing sets.

---

## Preprocessing Pipeline

The production preprocessing workflow includes:

```text
Missing Value Imputation
        ↓
IQR Outlier Capping
        ↓
Skewness Transformation
        ↓
Feature Encoding
        ↓
Feature Scaling
        ↓
XGBoost
```

### Numerical Features

```text
Median Imputation
→ IQR Capping
→ Log / Skewness Transformation
→ StandardScaler
```

### Binary Features

```text
OrdinalEncoder
```

### Multi-Class Categorical Features

```text
OneHotEncoder
```

The entire preprocessing workflow is encapsulated inside the ML pipeline.

---

# 🛡️ Data Leakage Prevention

Data leakage is treated as a critical concern in the project.

All preprocessing operations are fitted **only on the training data** and then applied to validation and test data.

This includes:

* Imputation statistics
* Outlier-capping boundaries
* Transformation parameters
* Scaling parameters
* Categorical encoders

The classification threshold is also selected using **cross-validation out-of-fold probabilities**, rather than optimizing directly on the final test set.

The test set is therefore kept isolated for final performance evaluation.

---

# ⚖️ Handling Class Imbalance

Churn datasets commonly contain fewer churned customers than retained customers.

Instead of using SMOTE in the final production pipeline, ChurnIQ uses:

```python
scale_pos_weight
```

inside XGBoost.

This allows the model to assign greater importance to the minority churn class while keeping preprocessing inside the model's normal training workflow.

SMOTE was explored during the experimentation stage but was not selected for the final production pipeline.

---

# 🎯 Model Selection

The original experimentation phase evaluated multiple algorithms, including:

* Logistic Regression
* K-Nearest Neighbors
* Support Vector Machine
* Decision Tree-based models
* Random Forest
* XGBoost

After comparison, **XGBoost** was selected as the final production model based on its overall performance and suitability for the dataset.

---

# 📈 Evaluation Strategy

Because churn is an imbalanced classification problem, accuracy alone is not sufficient.

The project evaluates:

| Metric    | Purpose                                             |
| --------- | --------------------------------------------------- |
| Accuracy  | Overall classification correctness                  |
| Precision | How many predicted churners actually churn          |
| Recall    | How many actual churners were identified            |
| F1-Score  | Balance between Precision and Recall                |
| ROC-AUC   | Overall ranking/discrimination ability              |
| PR-AUC    | More informative view for imbalanced classification |

The training process focuses primarily on **F1-score** while still reporting the other metrics.

---

# 👤 Mode 1 — Single Customer Prediction

The dashboard allows users to manually enter a customer's profile.

The system returns:

```text
Churn Probability
        ↓
Risk Level
        ↓
Prediction
        ↓
Confidence
        ↓
Gemini Business Recommendation
```

Example:

```text
Prediction: Likely to Churn

Churn Probability: 82.4%

Risk Level: High Risk
```

When Gemini is enabled, the platform can additionally generate a recommended retention strategy.

---

# 📊 Mode 2 — Batch CSV Prediction

The second workflow allows users to upload a CSV containing multiple customers instead of entering them individually.

## Required Model Features

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

Additional customer information can also be included.

Examples:

```text
customerID
customer_name
email
phone
address
```

These optional fields can be preserved for reporting purposes but are not required by the ML model.

---

# 📋 Batch Analysis Output

After uploading the dataset, ChurnIQ generates:

### KPI Dashboard

* Total customers
* High-risk customers
* Medium-risk customers
* Low-risk customers
* Average churn probability

### Visual Analytics

* Risk distribution chart
* Churn probability histogram

### Customer Intelligence

* Top-N highest-risk customers
* Full filterable customer table
* Risk-level filtering
* Prediction filtering
* Customer ranking by churn probability

### Export

The system can generate downloadable CSV reports containing the model predictions and available AI recommendations.

---

# 🧠 Gemini Batch Intelligence

Gemini recommendations can be generated for selected customer groups.

| Scope         | Description                                     |
| ------------- | ----------------------------------------------- |
| Top 10        | 10 customers with the highest churn probability |
| Top 20        | 20 customers with the highest churn probability |
| All High Risk | Customers classified as high risk               |
| High + Medium | Customers classified as likely churners         |

For batch analysis, Gemini is called **per customer** within the selected scope.

Because API calls consume tokens and may be subject to rate limits, smaller scopes such as **Top 10 or Top 20** are recommended for free-tier usage.

---

# 🔒 Privacy & Security

The project follows several basic security practices:

* `.env` is excluded through `.gitignore`
* API credentials are not hard-coded into source files
* PII fields such as email, phone number, and customer name are excluded from Gemini requests
* Only relevant customer features and model results are sent to Gemini
* Uploaded CSV data is processed during the active Streamlit session rather than being intentionally stored by the application

> Never commit an actual API key to GitHub.

---

# 💸 Gemini API Usage

Gemini is an optional intelligence layer, not a dependency for the core ML prediction system.

The application is designed so that:

```text
Gemini Available
       ↓
ML Prediction + AI Recommendation

Gemini Unavailable
       ↓
ML Prediction Only
```

For larger batches, API usage can increase significantly because recommendations are generated per customer.

For this reason, batch AI generation is intentionally scoped to selected customer groups.

---

# ⚠️ Limitations

The current project has several limitations that should be understood before treating it as a production business system.

### Dataset Limitation

The project is based on a synthetic / benchmark Telco Customer Churn dataset rather than continuously collected real-world production data.

A production deployment would require validation against real customer behavior.

### Temporal Limitation

The data represents customer information at a snapshot in time.

A longitudinal approach using customer history could capture churn triggers and behavioral changes more effectively.

### Evaluation Limitation

The current evaluation relies on a single held-out test split.

A stronger production validation strategy could include:

* Nested Cross-Validation
* Repeated Cross-Validation
* Time-based validation when temporal data is available
* External validation on unseen real-world data

### Explainability Limitation

Individual SHAP-based explanations are not currently included in the production version.

This is a planned enhancement to provide customer-level explanations such as:

```text
Why is this customer considered high risk?
```

---

# 🔮 Future Improvements

Planned enhancements include:

### Explainable AI

Integrate **SHAP** to provide:

* Global feature importance
* Individual customer explanations
* SHAP waterfall plots
* Top positive and negative churn drivers

### Customer Lifetime Value

Integrate **CLV estimation** so the platform can prioritize customers based not only on churn probability, but also on their expected business value.

This would enable a stronger decision framework:

```text
Churn Probability
        +
Customer Lifetime Value
        ↓
Retention Priority
```

### API Layer

Expose the prediction pipeline through **FastAPI** or **Flask** so external services can consume predictions programmatically.

Example:

```text
CRM / Backend
      ↓
FastAPI
      ↓
Churn Model
      ↓
Prediction + Probability
```

This would allow integration with:

* CRM systems
* Automated retention workflows
* Customer support systems
* Notification services
* Business intelligence platforms

### Monitoring & MLOps

Future production improvements could include:

* Data drift detection
* Model performance monitoring
* Prediction logging
* Automated retraining
* Model versioning
* Experiment tracking

---

# 🏆 Project Goal

ChurnIQ is designed to demonstrate how a machine learning project can evolve from:

```text
Notebook
   ↓
Model
   ↓
Production Pipeline
   ↓
Interactive Dashboard
   ↓
Batch Analytics
   ↓
AI Recommendations
   ↓
Business Retention Intelligence
```

The goal is not simply to predict **who may churn**, but to help answer the more useful business question:

> **Who is at risk, why should we care, and what action should we take?**

---

# 👥 Authors

**Youssef Hassan**
**Ali Mohamed**
**Nabil Sultan**

Developed as the Capstone Project for the **Machine Learning & Data Analysis Track**.
