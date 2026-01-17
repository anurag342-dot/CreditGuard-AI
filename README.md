# 🏦 CreditGuard AI - Intelligent Credit Risk System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-guard-ai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)

**CreditGuard AI** is a machine learning-powered financial assessment tool designed to predict loan default risk in real-time. By analyzing 20+ financial behaviors (including Income, Debt-to-Income Ratio, and Payment History), it classifies applicants into **Good**, **Standard**, or **Poor** risk categories.

🔗 **[Live Demo: Click Here to Launch App](https://credit-guard-ai.streamlit.app)**
*(Note: If the app is sleeping, click "Wake Up" and wait 30 seconds)*

---

## 🎯 Key Features

* **Real-Time Prediction Engine:** Utilizes a trained **XGBoost Classifier** to deliver instant credit scores with probability confidence levels.
* **🛡️ Hybrid Decision Logic (Safety Layer):**
    * Unlike standard "black box" models, this system includes a **Hard-Rule Override** layer.
    * *Example:* If an applicant has a payment delay > 90 days, the system automatically flags them as "High Risk" regardless of the model's output, preventing costly false positives.
* **Interactive Dashboard:** A user-friendly interface built with Streamlit for loan officers to input data and visualize risk factors dynamically.
* **Robust Data Handling:** Handles outliers and skewed financial data using industry-standard scaling techniques.

---

## 🧠 Machine Learning Pipeline

The backend intelligence was developed manually using a rigorous Data Science workflow:

1.  **Data Preprocessing:**
    * **Cleaning:** Removed garbage values and handled missing data.
    * **Encoding:** Applied `LabelEncoder` for categorical variables (e.g., Occupation, Loan Type).
    * **Scaling:** Used `RobustScaler` to normalize financial data while minimizing the impact of extreme outliers.
2.  **Feature Engineering:**
    * Implemented `SelectFromModel` to identify the most impactful financial indicators, reducing dimensionality and improving speed.
3.  **Class Balancing:**
    * Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to fix dataset imbalance, ensuring the model learns to identify "Poor" credit risks just as well as "Good" ones.
4.  **Model Selection:**
    * Trained multiple models (Random Forest, Decision Tree, Bagging).
    * Selected **XGBoost** for its superior performance (Accuracy: ~87%) and handling of non-linear financial patterns.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** XGBoost, Scikit-Learn, Imbalanced-learn
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend Interface:** Streamlit (Rapid prototyping assisted by AI tooling)
* **Deployment:** Streamlit Community Cloud

---
## 🚀 How to Run Locally

If you want to run this dashboard on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anurag342-dot/CreditGuard-AI.git](https://github.com/anurag342-dot/CreditGuard-AI.git)
    cd CreditGuard-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run interface.py
    ```

---

## 🔮 Future Improvements

* **Explainability:** Integrate SHAP (SHapley Additive exPlanations) plots to show *why* a specific user was rejected.
* **Database Integration:** Connect to a SQL database to save applicant history.
* **API Deployment:** Wrap the model in FastAPI/Flask for integration with banking software.

---

### 👨‍💻 Author

**Anurag**
* *Aspiring Machine Learning Engineer & Developer*
* [GitHub Profile](https://github.com/anurag342-dot)
## 📂 Project Structure

```text
CreditGuard-AI/
├── interface.py            # Main application script (Streamlit Frontend)
├── xgb_credit_model.pkl    # Trained XGBoost Model object
├── robust_scaler.pkl       # Fitted Scaler for data normalization
├── feature_selector.pkl    # Feature selection logic
├── requirements.txt        # Python dependencies for cloud deployment
└── README.md               # Project documentation
