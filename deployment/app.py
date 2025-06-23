import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Read model and encoder
model = joblib.load("models/model.pkl")
encoder_bundle = joblib.load("models/ordinal_encoder.pkl")
encoder = encoder_bundle["encoder"]
categorical_cols = encoder_bundle["columns"]

# Prediction processing from input forms
def predict_churn(gender, senior, partner, tenure, internet, contract, charges):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "tenure": float(tenure),
        "InternetService": internet,
        "Contract": contract,
        "MonthlyCharges": float(charges)
    }])

    # Convert only what is being encoded (Match the order)
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])

    # Model prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return ("Churn" if pred == 1 else "No Churn", f"{prob:.2%}")

# Gradio UI
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["0", "1"], label="Senior Citizen (0=No, 1=Yes)"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Number(label="Tenure (months)"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Number(label="Monthly Charges"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Churn Probability")
    ]
)

if __name__ == "__main__":
    demo.launch()
