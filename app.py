from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("air_quality_model.pkl")

# Define feature order (same as training)
FEATURES = ['PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
            'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    inputs = {feature: "" for feature in FEATURES}

    if request.method == "POST":
        try:
            # Extract input values and retain them
            inputs = {feature: request.form[feature] for feature in FEATURES}

            # Convert to float
            input_values = np.array([float(inputs[feature]) for feature in FEATURES]).reshape(1, -1)

            # Make prediction
            prediction = round(model.predict(input_values)[0], 2)  # Rounded for readability

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)
