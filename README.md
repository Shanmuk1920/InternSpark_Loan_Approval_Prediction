# 🏦 Loan Approval Prediction Using Machine Learning

## 📌 Project Overview

This project was developed as **Task 2 of the InternSpark Data Science Internship**. It uses supervised machine learning techniques to predict loan approval outcomes based on applicant information.

The project focuses on data preprocessing, handling missing values, addressing class imbalance, training classification models, and evaluating their performance using multiple metrics.

## 🎯 Objectives

* Analyze applicant information and loan approval outcomes.
* Handle missing values in numerical and categorical features.
* Apply feature scaling and categorical encoding.
* Train and compare Logistic Regression and Random Forest models.
* Evaluate performance using classification metrics.
* Interpret results from a business perspective.

## 📂 Dataset

**Dataset:** Loan Approval Prediction Case Study

**Source:** [Kaggle — Loan Approval Prediction Case Study](https://www.kaggle.com/datasets/bhanupratapbiswas/loan-approval-prediction-case-study)

The dataset contains **614 records and 13 columns**, including applicant details, financial information, and loan approval status.

### Target Variable

The target column is `Loan_Status`.

* `Y` — Loan approved
* `N` — Loan not approved

### Dataset Summary

| Description           | Value |
| --------------------- | ----: |
| Total records         |   614 |
| Total columns         |    13 |
| Approved applications |   422 |
| Rejected applications |   192 |
| Duplicate rows        |     0 |

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Environment:** Google Colab / Jupyter Notebook
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Saving:** Joblib

## ⚙️ Project Workflow

### 1. Data Loading and Exploration

* Loaded the dataset into a Pandas DataFrame.
* Examined dataset dimensions, column types, and missing values.
* Checked for duplicate records.
* Analyzed the distribution of loan approvals and rejections.

### 2. Data Preprocessing

* Removed `Loan_ID` from the input features.
* Separated input features (`X`) and the target (`y`).
* Filled missing numerical values using median imputation.
* Filled missing categorical values using most-frequent-value imputation.
* Applied StandardScaler to numerical features.
* Applied OneHotEncoder to categorical features.
* Split the dataset into training and testing sets using an 80:20 ratio.
* Used stratified sampling to preserve class proportions.

### 3. Model Training

Two supervised classification models were trained:

**Logistic Regression**

* Used as a baseline classification model.
* Configured with balanced class weights.

**Random Forest Classifier**

* Used to learn more complex relationships between applicant features.
* Configured with 200 decision trees and balanced class weights.

Scikit-learn pipelines were used to combine preprocessing and model training.

### 4. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices
* ROC curves

## 📊 Model Performance

The following results were obtained during the initial model evaluation:

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |               82.9% |         82.9% |
| Precision |               87.2% |         84.0% |
| Recall    |               88.2% |         92.9% |
| F1-score  |               87.7% |         88.3% |
| ROC-AUC   |               85.5% |         77.3% |

### Results Interpretation

* Both models achieved an accuracy of 82.9%.
* Logistic Regression achieved higher precision and ROC-AUC in the reported results.
* Random Forest achieved higher recall and F1-score.
* Random Forest may be worth further evaluation when recall and F1-score are priorities.

The final model selection should consider the confusion matrices, the costs of incorrect predictions, fairness, and validation results rather than accuracy alone.

*Note: These are the initial reported results and should be verified against the notebook's saved evaluation outputs before final submission.*

## 💼 Business Insights

A loan approval prediction model could help financial institutions analyze applications and prioritize cases for further review.

Important considerations include:

* False approvals may increase financial risk.
* False rejections may prevent eligible applicants from accessing credit.
* Precision and recall should be considered together.
* The prediction threshold should be selected based on validation results, business costs, and fairness considerations.
* Predictions should support human review rather than replace responsible lending decisions.

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Harshi1904/InternSpark_Loan_Approval_Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd InternSpark_Loan_Approval_Prediction
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Step 4: Open the Notebook

```bash
jupyter notebook InternSpark_Loan_Approval_Prediction.ipynb
```

Alternatively, open the notebook in Google Colab.

### Step 5: Load the Dataset

Download the dataset from Kaggle and upload the CSV file when prompted by the notebook.

### Step 6: Run the Notebook

Execute the cells in order to perform data exploration, preprocessing, model training, and evaluation.

## 📁 Project Structure

```text
InternSpark_Loan_Approval_Prediction/
│
├── InternSpark_Loan_Approval_Prediction.ipynb
├── README.md
├── Loan_Approval_Prediction_Report.pdf
└── loan_approval_model.pkl
```

*The report and saved model should be included only if you have generated those files. The model file is optional.*

## ⚠️ Limitations and Future Improvements

* Evaluate additional models and compare their performance.
* Experiment with SMOTE or undersampling using training data only.
* Tune model hyperparameters through cross-validation.
* Validate the reported metrics and investigate the ROC-AUC difference between the models.
* Select a decision threshold using validation data and business requirements.
* Assess fairness across relevant applicant groups.
* Test the model on independent data before considering real-world use.

## 👨‍💻 Author

**Gali Shanmuk**

* GitHub: [Shanmuk1920](https://github.com/Shanmuk1920)

---

**Project developed as part of the InternSpark Data Science Internship — Task 2.**

## 🏁 Conclusion

This project demonstrates an end-to-end introductory machine learning workflow for loan approval prediction, including data exploration, preprocessing, model training, and performance evaluation.

The comparison of Logistic Regression and Random Forest highlights the importance of evaluating multiple metrics and considering business requirements when selecting a classification model.
