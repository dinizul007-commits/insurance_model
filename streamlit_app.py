import joblib
import streamlit as st
import pandas as pd

# ------------------------------
# Load trained model
# ------------------------------
model = joblib.load("insurance_gbr_model.pkl")  # change to insurance_gbr_model.pkl if using GB

st.set_page_config(page_title="Medical Insurance Charges (USD)", layout="centered")

# ------------------------------
# App Title (USD context)
# ------------------------------
st.title("Medical Insurance Charges Predictor (USD)")
st.write("Estimate annual medical charges based on demographic and lifestyle factors.")

st.subheader("Enter your details")

# ==============================
# INPUT LAYOUT (3 columns × 2 rows)
# ==============================

# -------- ROW 1 --------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])

with col3:
    smoker = st.selectbox("Smoker", ["no", "yes"])

# -------- ROW 2 --------
col4, col5, col6 = st.columns(3)

with col4:
    height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=165.0)

with col5:
    weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0)

with col6:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

# ==============================
# BMI CALCULATION
# ==============================
height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)

st.caption(f"Calculated BMI: {bmi:.2f}")

# ==============================
# PREDICT BUTTON
# ==============================
st.markdown("---")
predict = st.button("Predict Medical Insurance Charges", use_container_width=True)

# ==============================
# PREDICTION OUTPUT (BIG)
# ==============================
if predict:
    # Manual encoding (matches training exactly)
    X_enc = pd.DataFrame([{
        "age": age,
        "bmi": float(bmi),
        "children": int(children),
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0
    }])

    # Align columns just in case
    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = float(model.predict(X_enc)[0])

    # Big, clear USD output
    st.markdown(
        f"""
        <div style="text-align:center; padding:40px;">
            <h2>Estimated Annual Medical Charges</h2>
            <h1 style="font-size:48px; color:#2ECC71;">
                USD ${pred:,.2f}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
