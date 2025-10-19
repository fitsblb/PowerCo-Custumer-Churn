
"""
Production Inference Template for PowerCo Churn Model
Usage: Load model + calibrator, score new customers
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Load artifacts
MODEL_PATH = Path("Model")
model = pickle.load(open(MODEL_PATH / "xgboost_best_model.pkl", "rb"))
calibrator = pickle.load(open(MODEL_PATH / "isotonic_calibrator_best.pkl", "rb"))

# Load metadata
import json
metadata = json.load(open(MODEL_PATH / "model_metadata.json", "r"))

THRESHOLD = metadata['optimal_threshold']
FEATURES = metadata['features']

def predict_churn(X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Score new customers for churn risk.

    Args:
        X_new: DataFrame with 20 features (must match training features)

    Returns:
        DataFrame with customer_id, churn_probability, predicted_churn, recommended_action
    """

    # Validate features
    missing_features = set(FEATURES) - set(X_new.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Score
    y_proba_raw = model.predict_proba(X_new[FEATURES])[:, 1]
    y_proba_calibrated = np.clip(calibrator(y_proba_raw), 0, 1)

    # Apply threshold
    y_pred_binary = (y_proba_calibrated >= THRESHOLD).astype(int)

    # Output
    results = pd.DataFrame({
        'customer_id': X_new['id'],
        'churn_probability': y_proba_calibrated,
        'predicted_churn': y_pred_binary,
        'action': ['Contact: Price Lock Offer' if pred == 1 else 'Monitor' for pred in y_pred_binary]
    })

    return results

# Example usage
if __name__ == "__main__":
    # Load new customer data (must have 20 engineered features)
    X_new = pd.read_csv("new_customer_data.csv")

    # Predict
    predictions = predict_churn(X_new)

    # Export
    predictions.to_csv("churn_predictions.csv", index=False)
    print(f"Predictions saved. Contacts needed for {predictions['predicted_churn'].sum()} customers")
