import joblib
import streamlit as st
import pandas as pd

# --- Load model---
model = joblib.load("insurance_rf_model.pkl")

st.title("Medical Insurance Cost Predictor")
st.write("Predict estimated insurance charges based on lifestyle and demographics.")

# --- User inputs (region removed because YOU dropped it) ---
age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=22.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])

if st.button("Predict insurance cost"):
    # 1) Create raw input dataframe (same feature names as training BEFORE get_dummies)
    X_raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }])

    # 2) Apply SAME encoding style you used: get_dummies(drop_first=True)
    X_enc = pd.get_dummies(X_raw, drop_first=True)

    # 3) Align columns to match what the model was trained on
    #    (RandomForest stores feature_names_in_ when trained on a DataFrame)
    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    # 4) Predict
    pred = float(model.predict(X_enc)[0])

    st.success(f"Estimated insurance cost (charges): ${pred:,.2f}")
