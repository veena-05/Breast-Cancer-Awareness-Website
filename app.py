from flask import Flask, render_template, request
import numpy as np
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from waitress import serve
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load and preprocess data
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Prevention page
@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

# Symptoms page
@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

# Breast Cancer Classification page
@app.route('/classification')
def classification():
    return render_template('index.html', feature_names=breast_cancer_dataset.feature_names)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect 30 feature inputs from the form
        input_data = []
        for field in breast_cancer_dataset.feature_names:
            field_value = request.form.get(field)
            if field_value is None or field_value.strip() == '':
                raise ValueError(f"Input missing for field: {field}")
            try:
                input_data.append(float(field_value))  # Convert form input to float
            except ValueError:
                raise ValueError(f"Invalid input for field: {field}")

        # Convert input to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Log the input data
        app.logger.debug(f"Input data: {input_data_reshaped}")

        # Make prediction
        prediction = model.predict(input_data_reshaped)
        result = "Benign" if prediction[0] == 1 else "Malignant"

        # Return result
        return render_template('result.html', result=result)
    except ValueError as ve:
        # Log the error and return a friendly message
        app.logger.error(f"ValueError: {ve}")
        return f"Error: {ve}"
    except Exception as e:
        # Log the error and return a general error message
        app.logger.error(f"Exception: {e}")
        return "An unexpected error occurred. Please try again."

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5001)
