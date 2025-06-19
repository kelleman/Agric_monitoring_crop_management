import streamlit as st
import joblib
import numpy as np

# Load the trained model and label encoder
model = joblib.load('crop_model.pkl')
le = joblib.load('label_encoder.pkl')

st.set_page_config(page_title="Crop Management", layout="centered")
st.title("🌱 Smart Agriculture Monitoring and Crop Management System")

st.markdown("Enter the following soil and weather parameters to get a crop recommendation, fertilizer advice, pest alerts, and irrigation tips.")

# ========== Helper Functions ==========

def get_fertilizer_advice(N, P, K):
    advice = []
    if N < 50:
        advice.append("🧪 Add Urea or Ammonium Sulphate for Nitrogen.")
    if P < 30:
        advice.append("🧪 Add Single Super Phosphate (SSP) for Phosphorus.")
    if K < 40:
        advice.append("🧪 Add Muriate of Potash (MOP) for Potassium.")
    return advice or ["✅ Your soil's nutrient levels look good."]

def get_pest_alert(temperature, humidity):
    if temperature > 28 and humidity > 80:
        return "⚠️ High risk of fungal infections like powdery mildew."
    elif humidity > 85:
        return "⚠️ Moist conditions may attract aphids and whiteflies."
    else:
        return "✅ No major pest threats detected from weather conditions."

def get_irrigation_advice(rainfall):
    if rainfall < 50:
        return "💧 Irrigation recommended. Rainfall is insufficient."
    elif 50 <= rainfall <= 200:
        return "✅ Rainfall is adequate. Monitor soil moisture."
    else:
        return "🌧️ High rainfall — ensure proper drainage to avoid root rot."

# ========== Input Form ==========
with st.form("crop_form"):
    N = st.number_input("Nitrogen (N)", min_value=0, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, step=1)
    K = st.number_input("Potassium (K)", min_value=0, step=1)
    temperature = st.number_input("Temperature (°C)", format="%.2f")
    humidity = st.number_input("Humidity (%)", format="%.2f")
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, format="%.2f")
    rainfall = st.number_input("Rainfall (mm)", format="%.2f")
    
    submitted = st.form_submit_button("🌾 Manage Crop")

# ========== Prediction and Advice ==========
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]
    
    st.success(f"✅ Recommended Crop: **{prediction_label}**")
    
    # Fertilizer Advice
    st.subheader("🧾 Fertilizer Advice")
    for tip in get_fertilizer_advice(N, P, K):
        st.info(tip)
    
    # Pest Alerts
    st.subheader("🛡️ Pest Alert")
    st.warning(get_pest_alert(temperature, humidity))
    
    # Irrigation Tips
    st.subheader("🚿 Irrigation Advice")
    st.success(get_irrigation_advice(rainfall))

        # Visuals for Model Evaluation
    st.subheader("📊 Model Performance Overview")

    # Show accuracy image
    st.image("accuracy.png", caption="Model Accuracy", use_container_width=True)

    # Show confusion matrix image
    st.image("confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)

