import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open("fertilizer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("fertilizer_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fertilizer Recommendation App", page_icon="🧪")
st.title("🧪 Fertilizer Recommendation System")

st.write("Enter the soil and crop parameters:")

# Input fields
temperature = st.number_input("🌡️ Temperature (°C)", value=26.0)
humidity = st.number_input("💧 Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
moisture = st.number_input("🌱 Moisture (0 to 1)", min_value=0.0, max_value=1.0, value=0.6)
soil_type = st.number_input("🧬 Soil Type (0 to 9)", min_value=0, max_value=9, value=2)
crop_type = st.number_input("🌾 Crop Type (0 to 10)", min_value=0, max_value=10, value=3)
nitrogen = st.number_input("🧪 Nitrogen", value=10.0)
potassium = st.number_input("🧪 Potassium", value=15.0)
phosphorous = st.number_input("🧪 Phosphorous", value=6.0)

if st.button("🔍 Recommend Fertilizer"):
    input_data = np.array([[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    st.success(f"✅ Recommended Fertilizer: **{prediction[0]}**")
