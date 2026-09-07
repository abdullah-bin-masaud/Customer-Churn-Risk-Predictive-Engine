"""
Real-Time Customer Churn Risk Inference Module.
Takes a customer profile and returns churn probability and risk categorization.
"""
import sys
from pathlib import Path

try:
    import pandas as pd
    import joblib
except ImportError:
    pd = None

MODEL_DIR = Path(__file__).parent.parent / "models"


def assess_customer_risk(customer_data: dict) -> dict:
    """Evaluates churn risk tier for a given customer profile."""
    rf = joblib.load(MODEL_DIR / "churn_random_forest.pkl")
    features = joblib.load(MODEL_DIR / "features.pkl")

    df = pd.DataFrame([customer_data])
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    churn_prob = float(rf.predict_proba(df)[0, 1])

    if churn_prob > 0.65:
        risk_tier = "CRITICAL / HIGH RISK"
        recommended_action = "Deploy immediate retention incentive & dedicated account outreach"
    elif churn_prob > 0.35:
        risk_tier = "MODERATE RISK"
        recommended_action = "Schedule engagement follow-up; offer loyalty program enrolment"
    else:
        risk_tier = "LOW RISK"
        recommended_action = "Standard service tier; healthy engagement profile"

    return {
        "churn_probability": round(churn_prob * 100, 2),
        "risk_category": risk_tier,
        "recommended_action": recommended_action
    }


if __name__ == "__main__":
    sample_customer = {
        "CreditScore": 580,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "Tenure": 2,
        "Balance": 125000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 65000.0
    }
    print("Evaluating Sample Customer Profile:")
    res = assess_customer_risk(sample_customer)
    print(f"Churn Probability : {res['churn_probability']}%")
    print(f"Risk Assessment   : {res['risk_category']}")
    print(f"Recommended Action: {res['recommended_action']}")
