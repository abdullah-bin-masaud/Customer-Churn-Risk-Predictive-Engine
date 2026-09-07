"""
Generates synthetic banking/telecom customer records with realistic churn dynamics and class imbalance.
"""
import random
import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent


def generate_churn_data(num_samples: int = 1500):
    rows = []
    countries = ["Germany", "France", "Spain"]
    genders = ["Male", "Female"]

    for i in range(10001, 10001 + num_samples):
        credit_score = int(random.gauss(650, 80))
        credit_score = max(350, min(850, credit_score))

        country = random.choice(countries)
        gender = random.choice(genders)
        age = int(random.gauss(38, 10))
        age = max(18, min(80, age))

        tenure = random.randint(0, 10)
        balance = round(max(0.0, random.gauss(75000, 40000)), 2)
        num_products = random.choices([1, 2, 3, 4], weights=[0.5, 0.45, 0.04, 0.01])[0]
        has_credit_card = 1 if random.random() > 0.3 else 0
        is_active_member = 1 if random.random() > 0.45 else 0
        estimated_salary = round(random.uniform(25000, 180000), 2)

        # Churn probability heuristic
        risk_score = 0.0
        if age > 45: risk_score += 0.3
        if balance > 100000 and country == "Germany": risk_score += 0.2
        if is_active_member == 0: risk_score += 0.25
        if num_products == 1: risk_score += 0.15
        if num_products >= 3: risk_score += 0.4
        if credit_score < 500: risk_score += 0.2

        prob = min(0.9, max(0.05, risk_score + random.gauss(0, 0.15)))
        churned = 1 if random.random() < prob else 0

        rows.append([
            i, credit_score, country, gender, age, tenure, balance,
            num_products, has_credit_card, is_active_member, estimated_salary, churned
        ])

    csv_path = DATA_DIR / "customer_churn.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CustomerID", "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
        ])
        writer.writerows(rows)

    churn_rate = sum(r[-1] for r in rows) / len(rows) * 100
    print(f"[*] Generated {len(rows)} customer records at {csv_path}")
    print(f"[*] Churn class distribution: {churn_rate:.1f}% Churned (Exited=1)")


if __name__ == "__main__":
    generate_churn_data()
