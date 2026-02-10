import joblib
import streamlit as st
import pandas as pd

# ------------------------------
# Load trained model
# ------------------------------
model = joblib.load("insurance_rf_model.pkl")

st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

st.title("Medical Insurance Cost Predictor")
st.write("Predict estimated insurance charges based on lifestyle and demographics.")

st.subheader("Enter your details")

# ==============================
# INPUT GRID (2 rows × 3 columns)
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
    children = st.number_input("Children", min_value=0, max_value=10, value=0)

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
predict = st.button("Predict insurance cost", use_container_width=True)

# ==============================
# PREDICTION OUTPUT
# ==============================
if predict:
    # Create raw input DataFrame (same columns as training BEFORE encoding)
    X_raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker
    }])

    # Apply same encoding as training
    X_enc = pd.get_dummies(X_raw, drop_first=True)

    # Align columns with model
    if hasattr(model, "feature_names_in_"):
        X_enc = X_enc.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict
    pred = float(model.predict(X_enc)[0])

    # Big result display
    st.markdown(
        f"""
        <div style="text-align:center; padding:40px;">
            <h2>Estimated Insurance Cost</h2>
            <h1 style="font-size:48px; color:#2ECC71;">
                ${pred:,.2f}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
