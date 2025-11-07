# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and model
df_1 = pd.read_csv("first_telc.csv")
model = pickle.load(open("model.sav", "rb"))

# ---------------------------------------------
# ROUTE 1: Home Page
# ---------------------------------------------
@app.route("/")
def load_page():
    return render_template("home.html", query="")

# ---------------------------------------------
# ROUTE 2: Prediction
# ---------------------------------------------
@app.route("/", methods=["POST"])
def predict():
    # Get all input values from form
    inputs = [request.form[f"query{i}"] for i in range(1, 20)]

    # Prepare DataFrame for prediction
    new_df = pd.DataFrame([inputs], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Combine with reference data for encoding consistency
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Create tenure groups
    labels = [f"{i} - {i + 11}" for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2.tenure.astype(int), range(1, 80, 12),
        right=False, labels=labels
    )

    # Drop original tenure
    df_2.drop(columns=["tenure"], axis=1, inplace=True)

    # Convert categorical columns to dummies
    df_dummies = pd.get_dummies(df_2[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group'
    ]])

    # Predict churn
    single_pred = model.predict(df_dummies.tail(1))
    probability = model.predict_proba(df_dummies.tail(1))[:, 1]

    if single_pred == 1:
        result_text = "This customer is likely to be churned!!"
    else:
        result_text = "This customer is likely to continue!!"

    confidence_text = f"Confidence: {probability[0] * 100:.2f}%"

    # Render result in template
    return render_template(
        "home.html",
        output1=result_text,
        output2=confidence_text,
        **{f"query{i}": request.form[f"query{i}"] for i in range(1, 20)}
    )

# ---------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------
if __name__ == "__main__":
    # Run Flask in debug mode
    app.run(debug=True)
