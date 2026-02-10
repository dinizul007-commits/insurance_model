import joblib
import streamlit as st
import pandas as pd

# --- Load model---
# Make sure this matches what you trained/saved!
model = joblib.load("insurance_rf_model.pkl")  # change to insurance_gbr_model.pkl if using GB

st.title("Medical Insurance Cost Predictor")
st.write("Predict estimated insurance charges based on lifestyle and demographics.")

age = st.number_input("Age", min_value=0, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=22.0)
children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])

if st.button("Predict insurance cost"):
    X_raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }])

    X_enc = pd.get_dummies(X_raw, drop_first=True)

    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    # enforce consistent 0/1 ints for dummy columns
    for c in ["sex_male", "smoker_yes"]:
        if c in X_enc.columns:
            X_enc[c] = X_enc[c].astype(int)

    # debug display so you can SEE if smoker_yes changes
    st.write("Encoded input sent to model:")
    st.dataframe(X_enc)

    pred = float(model.predict(X_enc)[0])
    st.success(f"Estimated insurance charges: ${pred:,.2f}")
