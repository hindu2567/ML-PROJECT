from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model (adjust path as necessary)
model = pickle.load(open('models/fraud_detection_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        amount = float(request.form['amount'])
        time = int(request.form['time'])
        location = int(request.form['location'])
        device_info = int(request.form['device_info'])

        # Prepare the data for prediction (you can modify this based on your model's input)
        features = np.array([[amount, time, location, device_info]])

        # Predict using the model
        prediction = model.predict(features)

        if prediction == 1:
            result = 'Fraudulent Transaction Detected!'
        else:
            result = 'Transaction is Normal'

        return render_template('result.html', prediction_result=result)

if __name__ == "_main_":
    app.run(debug=True)
