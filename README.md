# Customer Churn & Default Risk Predictive Analytics Engine

A machine learning classification pipeline to predict customer churn, identify flight-risk accounts, and surface key behavioral drivers using ensemble decision trees with class-imbalance compensation.

## Pipeline Architecture
```
[ Customer Features (Age, Products, Activity, Balance) ]
                         |
                         v
          [ One-Hot Encoding & Scaling ]
                         |
                         v
     [ Class-Weighted Random Forest Classifier ]
            /                        \
           v                          v
[ Churn Probability (0-100%) ]  [ Risk Tier Classification ]
```

## Key Technical Features
- **Class Imbalance Mitigation:** Tuned `class_weight='balanced'` to prevent bias toward majority retention class.
- **Metric Optimization:** Evaluated against **ROC-AUC** and **Precision-Recall** rather than raw accuracy.
- **Explainability:** Feature importance scoring highlights age, activity status, and number of products as primary risk drivers.
- **Inference CLI:** Real-time scoring function providing probability scores and retention action recommendations.

## Quickstart
```bash
pip install -r requirements.txt
python models/train_churn_model.py
python models/predict.py
```
