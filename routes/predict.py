import numpy as np
import joblib, os
from typing import Any, Tuple

def predict(X : Any) -> Tuple[float, float]:
    """
    Predict the target variable using the trained model.

    Args:
        X (Any): X can be a DataFrame or a JSON object with the same structure as the training data.

    Returns:
        Tuple[float, float]: Predicted log value and predicted value.
    """
    # Load the model
    try:
        model = joblib.load('../model_params/model.pkl')
    except FileNotFoundError:
        return None, None
    
    # Make predictions
    y_pred = model.predict(X)
    
    return np.exp(y_pred)