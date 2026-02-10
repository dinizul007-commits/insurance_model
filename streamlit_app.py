import joblib
import streamlit as st
import pandas as pd

# --- Load model---
# Make sure this matches what you trained/saved!
model = joblib.load("insurance_rf_model.pkl")  # change to insurance_gbr_model.pkl if using GB

if hasattr(model, "feature_names_in_"):
    st.write("Model expects columns:", list(model.feature_names_in_))


st.title("Medical Insurance Cost Predictor")
st.write("Predict estimated insurance charges based on lifestyle and demographics.")

age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=22.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])

if st.button("Predict insurance cost"):
    # Build EXACT feature columns used in training
    X_enc = pd.DataFrame([{
        "age": age,
        "bmi": float(bmi),
        "children": int(children),
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0
    }])

    # If model has feature_names_in_, align just in case (safe)
    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    # Debug display (you should SEE smoker_yes flip 0/1 now)
    st.write("Encoded input sent to model:")
    st.dataframe(X_enc)

    pred = float(model.predict(X_enc)[0])
    st.success(f"Estimated insurance charges: ${pred:,.2f}")

