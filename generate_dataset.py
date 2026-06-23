"""
generate_dataset.py
--------------------
Generates a realistic synthetic employee dataset and saves it as
data/employee_data.csv

Features:
    age              - employee age (years)
    education_level  - 0: High School, 1: Bachelors, 2: Masters, 3: PhD
    experience        - years of professional experience
    income           - current annual income (INR)
    loan_amount      - outstanding loan amount (INR)

Target:
    salary_category  - "High" or "Low"

This replaces the original 5-row dummy dataset with a larger,
more realistic dataset so the trained model generalizes properly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 1000

age = np.random.randint(21, 65, N_SAMPLES)
education_level = np.random.choice([0, 1, 2, 3], size=N_SAMPLES, p=[0.2, 0.45, 0.25, 0.10])
experience = np.clip(age - np.random.randint(21, 25, N_SAMPLES), 0, None)

# Income depends on age, education and experience (with noise)
base_income = (
    20000
    + experience * 2200
    + education_level * 15000
    + np.random.normal(0, 8000, N_SAMPLES)
)
income = np.clip(base_income, 15000, None).round(-2)

# Loan amount loosely tied to income (with noise)
loan_amount = np.clip(
    income * np.random.uniform(0.05, 0.4, N_SAMPLES) + np.random.normal(0, 3000, N_SAMPLES),
    0,
    None,
).round(-2)

df = pd.DataFrame({
    "age": age,
    "education_level": education_level,
    "experience": experience,
    "income": income,
    "loan_amount": loan_amount,
})

# Define salary category based on income threshold (median split with a
# small random flip to make the classification problem non-trivial)
threshold = df["income"].median()
df["salary_category"] = np.where(df["income"] >= threshold, "High", "Low")

# Introduce a small amount of label noise (5%) to mimic real-world data
flip_idx = df.sample(frac=0.05, random_state=42).index
df.loc[flip_idx, "salary_category"] = df.loc[flip_idx, "salary_category"].map(
    {"High": "Low", "Low": "High"}
)

df.to_csv("data/employee_data.csv", index=False)
print(f"Saved {len(df)} rows to data/employee_data.csv")
print(df["salary_category"].value_counts())
