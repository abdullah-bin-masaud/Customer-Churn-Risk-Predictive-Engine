"""
Model Training and Optimization Pipeline.
Trains Random Forest and Logistic Regression with class-weight compensation for imbalance.
"""
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    import joblib
except ImportError:
    pd = None

DATA_PATH = Path(__file__).parent.parent / "data" / "customer_churn.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"


def train():
    if pd is None:
        print("[!] Required ML libraries not installed.")
        return

    print("==================================================================")
    print("  CUSTOMER CHURN & RISK PREDICTIVE ANALYTICS ENGINE")
    print("==================================================================")

    if not DATA_PATH.exists():
        from data.generate_churn_dataset import generate_churn_data
        generate_churn_data()

    df = pd.read_csv(DATA_PATH)

    # Preprocessing
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    features = [c for c in df.columns if c not in ["CustomerID", "Exited"]]
    X = df[features]
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline: Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)

    # 2. Optimized Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_probs)

    print(f"[*] Logistic Regression Baseline ROC-AUC: {lr_auc:.4f}")
    print(f"[*] Random Forest Classifier ROC-AUC    : {rf_auc:.4f}")
    print("\n[Classification Report - Random Forest (Target: Churn)]")
    print(classification_report(y_test, rf_preds, target_names=["Retained", "Churned"]))

    # Save artifacts
    joblib.dump(rf, MODEL_DIR / "churn_random_forest.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(features, MODEL_DIR / "features.pkl")
    print(f"[*] Saved serialized model and artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    train()
