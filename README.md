# Employee Attrition Prediction System

## 📌 Project Overview

The Employee Attrition Prediction System is a Machine Learning project that predicts whether an employee is likely to leave an organization.

The system analyzes employee-related factors such as age, department, monthly income, years at the company, job satisfaction, work-life balance, performance rating, overtime, and distance from home.

The project was developed as part of the **Machine Learning Internship Program at Infobharat Interns**.

---

## 🎯 Problem Statement

Employee attrition can create significant challenges for organizations, including increased recruitment costs, productivity loss, and workload on existing employees.

This project uses Machine Learning to identify employees who may be at risk of attrition and provides an estimated attrition probability.

---

## 🎯 Objectives

- Predict whether an employee is likely to leave the organization.
- Analyze factors associated with employee attrition.
- Perform exploratory data analysis on employee information.
- Handle categorical and numerical features appropriately.
- Address class imbalance using class weights.
- Compare multiple Machine Learning models.
- Evaluate models using multiple performance metrics.
- Provide a simple prediction interface using Streamlit.
- Save and reuse the trained Machine Learning model.

---

## 📊 Dataset

The dataset contains **2,000 employee records** and the following attributes:

| Feature | Description |
|---|---|
| EmployeeID | Unique employee identifier |
| Age | Employee age |
| Department | Employee department |
| MonthlyIncome | Monthly salary |
| YearsAtCompany | Number of years at the company |
| JobSatisfaction | Job satisfaction rating from 1–5 |
| WorkLifeBalance | Work-life balance rating from 1–5 |
| PerformanceRating | Performance rating from 1–5 |
| OverTime | Whether the employee works overtime |
| DistanceFromHome | Distance between home and workplace |
| Attrition | Whether the employee left the organization |

### Target Distribution

- No Attrition: 1,630 employees
- Attrition: 370 employees

The dataset therefore contains an imbalanced target variable, so class weights were used during model training.

---

## 🔍 Exploratory Data Analysis

The project includes analysis of:

- Attrition rate by department
- Monthly income versus attrition
- Correlation between numerical features
- Attrition based on overtime
- Attrition based on job satisfaction
- Feature importance from the trained Random Forest model

### Required Visualizations

The generated visualizations are stored in the `visuals/` folder.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Removed `EmployeeID` from model features because it is only an identifier.
2. Separated input features and target variable.
3. Converted:
   - `No` → 0
   - `Yes` → 1
4. Numerical features were standardized using `StandardScaler`.
5. Categorical features were encoded using `OneHotEncoder`.
6. Class imbalance was handled using balanced class weights.
7. The preprocessing steps were included in the Machine Learning pipeline to avoid data leakage.

---

## 🤖 Machine Learning Models

The following models were evaluated:

### 1. Logistic Regression

Logistic Regression was used as a classification model and provided strong recall and ROC-AUC performance.

### 2. Random Forest

Random Forest was used to capture non-linear relationships between employee attributes.

### 3. Tuned Random Forest

GridSearchCV was used to tune important Random Forest hyperparameters.

---

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7125 | 0.3504 | 0.6486 | 0.4550 | 0.7901 |
| Random Forest | 0.8075 | 0.4706 | 0.3243 | 0.3840 | 0.7125 |
| Tuned Random Forest | 0.7775 | 0.4074 | 0.4459 | 0.4258 | 0.7443 |

For this project, **Logistic Regression was selected as the final model** because it provided the highest recall, F1-score, and ROC-AUC among the evaluated models.

---

## 🏗️ Project Structure

```text
Employee-Attrition-Prediction/
│
├── data/
│   └── employee_data.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── visuals/
│   ├── attrition_by_department.png
│   ├── monthly_income_vs_attrition.png
│   ├── correlation_heatmap.png
│   ├── attrition_by_overtime.png
│   ├── attrition_by_job_satisfaction.png
│   ├── logistic_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── feature_importance.png
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── generate_dataset.py
├── requirements.txt
└── README.md