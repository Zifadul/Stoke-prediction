from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load models
male_model = joblib.load("models/male_model.pkl")
female_model = joblib.load("models/female_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])

        data = np.array([[age, hypertension, heart_disease, glucose, bmi, smoking]])

        if gender == "male":
            model = male_model
        else:
            model = female_model

        prediction = model.predict(data)[0]
        result = "High Risk" if prediction == 1 else "Low Risk"

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
