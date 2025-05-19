import numpy as np
import pandas as pd
import joblib, os
from typing import Any, Tuple
import pickle
import sklearn

def predict(model, X : Any) -> Tuple[float, float]:
    """
    Predict the target variable using the trained model.

    Args:
        X (Any): X can be a DataFrame or a JSON object with the same structure as the training data.

    Returns:
        Tuple[float, float]: Predicted log value and predicted value.
    """
    try:
        X_df = pd.DataFrame([X])
    except ValueError as e:
        print(f"Error converting input to DataFrame: {e}")
        return None, None
    
    y_pred = model.predict(X_df)
    return np.exp(y_pred[0])