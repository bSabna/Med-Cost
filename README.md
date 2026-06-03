# Medicare Part D - Drug Cost Prediction Engine

A machine learning web application that predicts the total cost of prescription drugs in Medicare Part D using Support Vector Regression (SVR). This tool turns historical government provider datasets into a dynamic forecasting engine to assist healthcare clinics and auditors with budget planning and anomaly detection.

## Live Demo
You can interact with the live deployed web application here: 
 [Live Streamlit App](https://med-cost-dejpnk2hmriig2tgznlqkw.streamlit.app/)

---

##  Dataset Overview
The model is trained on the official **Medicare Part D Prescribers by Provider and Drug** dataset from the Centers for Medicare & Medicaid Services (CMS). 

Key features utilized:
* **`Tot_Clms`**: Total number of prescriptions filled.
* **`Tot_30day_Fills`**: Number of standard 30-day fills.
* **`Tot_Day_Suply`**: Total days of medication supplied.
* **`Tot_Benes`**: Number of unique beneficiaries served.
* **`Brnd_Name`**: The brand name of the prescription drug.
* **`Tot_Drug_Cst`** *(Target)*: The total dollar amount spent.

---

## The Machine Learning Pipeline
To handle heavily skewed healthcare financial data and missing records, the backend Python script implements an end-to-end data science architecture:

1. **Exploratory Data Analysis**: Includes Pearson correlation for numerical variables and One-way ANOVA tests to confirm the statistical significance of drug brand categories.
2. **Log Transformation**: Applied `np.log1p` on the target variable to stabilize variance and counteract extreme pricing outliers.
3. **Data Splitting**: 80/20 train-test split to strictly evaluate generalization and prevent data leakage.
4. **Target Encoding**: Out-of-fold brand mapping to mathematically represent high-cardinality text data safely.
5. **Preprocessing Pipeline**: Missing data handling via **Median Imputation** and feature scaling using **StandardScaler** (vital for distance-based models).
6. **SVR Modeling**: A Support Vector Regression (SVR) engine optimized via **5-Fold Grid Search Cross-Validation** over hyperparameter spaces ($C$, $\gamma$, and RBF kernel settings).

---

## Technology
* **Language:** Python
* **Libraries:** scikit-learn, Pandas, NumPy, SciPy, Joblib
* **Interface & Deployment:** Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

##  Local Setup & Installation

If you want to run this project locally on your machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/bSabna/Med-Cost.git](https://github.com/bSabna/Med-Cost.git)
   cd Med-Cost