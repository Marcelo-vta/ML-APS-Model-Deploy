from flask import Flask, request
from flask_cors import CORS
from routes.predict import *
import os

BASE_URL = '/api/v1'

app = Flask("Green Thumb")
CORS(app)

#-------------------------------- ROUTES --------------------------------

#--------------------------------- PREDICT ---------------------------------

@app.route(BASE_URL + '/predict', methods=['POST'])
def prediction():
    """
    Predict the target variable using the trained model.

    Input must be a JSON object with the same structure as the training data.
    """
    log_pred, pred = predict(request.json)
    if log_pred is None or pred is None:
        return {'error': 'Invalid input data'}, 400
    return {'log_value' : log_pred, 'value' : pred}, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

