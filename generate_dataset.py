import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of employees
n = 2000

# Employee ID
employee_id = np.arange(1001, 1001 + n)

# Age
age = np.random.randint(21, 61, n)

# Department
departments = np.random.choice(
    ["Sales", "Engineering", "HR", "Marketing", "Finance"],
    size=n,
    p=[0.25, 0.30, 0.10, 0.20, 0.15]
)

# Monthly income in ₹
monthly_income = np.random.randint(20000, 120001, n)

# Years at company
years_at_company = np.random.randint(0, 21, n)

# Job satisfaction: 1 to 5
job_satisfaction = np.random.randint(1, 6, n)

# Work-life balance: 1 to 5
work_life_balance = np.random.randint(1, 6, n)

# Performance rating: 1 to 5
performance_rating = np.random.randint(1, 6, n)

# Overtime
overtime = np.random.choice(
    ["Yes", "No"],
    size=n,
    p=[0.30, 0.70]
)

# Distance from home in km
distance_from_home = np.random.randint(1, 51, n)


# -----------------------------
# Generate Attrition
# -----------------------------

# Start with a base probability
attrition_score = np.full(n, -2.5)

# Factors increasing attrition
attrition_score += np.where(overtime == "Yes", 1.2, 0)
attrition_score += np.where(job_satisfaction <= 2, 0.9, 0)
attrition_score += np.where(work_life_balance <= 2, 0.7, 0)
attrition_score += np.where(years_at_company <= 2, 0.7, 0)
attrition_score += np.where(distance_from_home >= 30, 0.5, 0)

# Factors reducing attrition
attrition_score -= np.where(job_satisfaction >= 4, 0.5, 0)
attrition_score -= np.where(work_life_balance >= 4, 0.4, 0)
attrition_score -= np.where(years_at_company >= 10, 0.5, 0)

# Income effect
attrition_score += np.where(monthly_income < 30000, 0.4, 0)
attrition_score -= np.where(monthly_income > 80000, 0.3, 0)

# Convert score into probability
attrition_probability = 1 / (1 + np.exp(-attrition_score))

# Generate Yes/No attrition
attrition = np.where(
    np.random.random(n) < attrition_probability,
    "Yes",
    "No"
)


# -----------------------------
# Create DataFrame
# -----------------------------

df = pd.DataFrame({
    "EmployeeID": employee_id,
    "Age": age,
    "Department": departments,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "PerformanceRating": performance_rating,
    "OverTime": overtime,
    "DistanceFromHome": distance_from_home,
    "Attrition": attrition
})


# Save dataset
file_path = "data/employee_data.csv"

df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print(f"File saved at: {file_path}")
print(f"Number of employees: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

print("\nAttrition percentage:")
print(df["Attrition"].value_counts(normalize=True) * 100)