import joblib
import streamlit as st
import pandas as pd

# --- Load model---
model = joblib.load("insurance_rf_model.pkl")

st.title("Medical Insurance Cost Predictor")
st.write("Predict estimated insurance charges based on lifestyle and demographics.")

# --- User inputs ---
age = st.number_input("Age", min_value=0, max_value=120, value=25)

height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)
weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0)

children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])

# --- BMI calculation ---
height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)

st.caption(f"Calculated BMI: {bmi:.2f}")

if st.button("Predict insurance cost"):
    # 1) Create raw input dataframe
    X_raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }])

    # 2) Apply same encoding as training
    X_enc = pd.get_dummies(X_raw, drop_first=True)

    # 3) Align columns with training data
    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    # 4) Predict
    pred = float(model.predict(X_enc)[0])

    st.success(f"Estimated insurance cost (charges): ${pred:,.2f}")
