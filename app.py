from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = 'lr.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = {
        'Amount': request.form['Amount'],
        'Time': request.form['Time'],
        'Age': request.form['Age'],
        'TransactionType': request.form['TransactionType'],
        'Location': request.form['Location'],
        'CardType': request.form['CardType'],
        'MerchantCategory': request.form['MerchantCategory']
    }

    # Convert to DataFrame (same structure as model training data)
    df = pd.DataFrame([data])

    # If model needs numeric encoding for categorical columns
    # (Optional: only if your model was trained with numeric encodings)
    for col in ['TransactionType', 'Location', 'CardType', 'MerchantCategory']:
        df[col] = df[col].astype('category').cat.codes

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Make prediction
    prediction = model.predict(df)

    # Convert prediction to label
    output = 'Fraudulent Transaction' if prediction[0] == 1 else 'Legitimate Transaction'

    return render_template('index.html', prediction_text=f"Prediction: {output}")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
