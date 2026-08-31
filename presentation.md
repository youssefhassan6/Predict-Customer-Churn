# Customer Churn Prediction
## Machine Learning Capstone Project

---

### Slide 1: Problem Statement and Motivation
- **The Problem:** Customer attrition (churn) directly impacts company revenue. It is significantly more expensive to acquire a new customer than to retain an existing one.
- **The Motivation:** By predicting which customers are at high risk of churning, the business can proactively target them with retention strategies (discounts, personalized support), thereby saving revenue.
- **Business Goal:** Move from reactive customer service to proactive retention using Machine Learning.

---

### Slide 2: Dataset Description & Preprocessing Pipeline
- **Dataset Overview:** 7,043 customers, 21 features (demographics, services, account details).
- **Target Variable:** `Churn` (Yes/No). Highly imbalanced (~73% No, 27% Yes).
- **Data Cleaning:** Dropped ID columns, converted `TotalCharges` to numeric, mapped target to binary (0/1).
- **Preventing Data Leakage:** Implemented a Scikit-Learn `Pipeline` to ensure all imputation (median/mode) and transformations are strictly learned on the *training set* and mapped to the test set.

---

### Slide 3: Feature Engineering & Preprocessing
- **Handling Outliers:** Applied Interquartile Range (IQR) capping on numerical features to limit extreme values without discarding data.
- **Addressing Skewness:** Applied logarithmic transformation (`log1p`) on `TotalCharges` and `MonthlyCharges` to normalize the distribution for linear models.
- **Encoding:** Used `OrdinalEncoder` for binary variables and `OneHotEncoder` for multi-class categorical variables.
- **Scaling:** Used `StandardScaler` to ensure features like Tenure and Charges are on the same scale, critical for distance-based and linear models.

---

### Slide 4: Methodology and Technical Approach
- **Experimentation Phase:** Evaluated multiple models in the original notebook, including Logistic Regression, KNN, Support Vector Machines (SVM), Decision Trees, Random Forest, and **XGBoost**.
- **Handling Imbalance During Experiments:** SMOTE (Synthetic Minority Over-sampling Technique) was explored during experimentation to reduce majority-class bias without leaking synthetic samples into validation data.
- **Final Production Choice:** XGBoost achieved the best overall performance, so the final training and deployment pipeline focuses on XGBoost rather than re-training every experimental model.
- **Model Tuning:** Conducted hyperparameter tuning using `GridSearchCV` with Stratified Cross-Validation, specifically optimizing for **F1-Score**.

---

### Slide 5: Model Evaluation Metrics and Results
- **Why Accuracy is Misleading:** With 73% of customers staying, a "dumb" model that always predicts "No Churn" gets 73% accuracy but 0% recall for churners.
- **Evaluation Focus:** Evaluated primarily on **F1-score**, Precision (avoiding false alarms), Recall (catching as many churners as possible), and **ROC-AUC**.
- **Results:** 
  - XGBoost achieved the best overall balance of Precision, Recall, F1-score, and ROC-AUC during experimentation.
  - The final production script therefore trains and tunes XGBoost as the selected model.
  - The `Pipeline` ensures real-world test results accurately reflect production performance without data leakage.

---

### Slide 6: System Architecture and Deployment
- **End-to-End Pipeline:** All preprocessing steps and the selected XGBoost model are combined into serialized `churn_artifacts.pkl` artifacts.
- **Application Interface:** Built an interactive web application using **Streamlit**.
- **Deployment Process:** The Streamlit app loads the unified pipeline. A user inputs raw categorical and numerical strings on the dashboard, and the pipeline handles encoding, scaling, and inference seamlessly.
- **Production Imbalance Handling:** The final XGBoost implementation tunes `scale_pos_weight` instead of using SMOTE in the deployed pipeline.

---

### Slide 7: Challenges and Solutions
- **Challenge:** Data Leakage in initial EDA script (imputing and capping the whole dataset before splitting).
- **Solution:** Rebuilt the architecture using a Scikit-Learn `ColumnTransformer` and `Pipeline` to strictly isolate the train and test environments.
- **Challenge:** Models were biased towards predicting "No Churn".
- **Solution:** Explored SMOTE during experimentation, then used XGBoost's `scale_pos_weight` tuning in the final production pipeline to handle class imbalance.

---

### Slide 8: Future Work and Improvements
- **Model Explainability:** Integrate SHAP values into the Streamlit dashboard to explain *why* a specific customer is predicted to churn.
- **Temporal Data:** Move from static snapshot data to time-series data to predict *when* a customer might churn.
- **Business Integration:** Create a batch-processing script to score the entire customer database weekly and push the top 5% highest-risk customers directly to the CRM for the retention team.
