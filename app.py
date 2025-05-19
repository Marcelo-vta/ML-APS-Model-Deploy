from flask import Flask, request
from flask_cors import CORS
from routes.predict import *
import os

try:
    MODEL = joblib.load('./model_params/model.pkl')
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure the model is available at the specified path.")

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
    # print(request.json)
    pred = predict(MODEL, request.json)
    if pred is None:
        return {'error': 'Invalid input data'}, 400
    return {'predicted_value' : float(pred)}, 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)

