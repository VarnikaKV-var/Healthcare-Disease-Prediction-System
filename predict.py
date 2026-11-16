import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Predict", page_icon="🔬")

st.title("🔬 Diabetes Risk Prediction")
st.markdown("---")

# Load model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except:
    st.error("Model files not found! Please make sure models folder exists.")
    st.stop()

st.write("Enter your health details below:")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 3)
    glucose = st.slider("Glucose", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)

with col2:
    insulin = st.slider("Insulin", 0, 846, 80)
    bmi = st.slider("BMI", 0.0, 67.1, 32.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
    age = st.slider("Age", 21, 81, 33)

st.markdown("---")

if st.button("🔍 Predict", use_container_width=True):
    data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)
    probability = model.predict_proba(data_scaled)[0][1]
    percent = round(probability * 100, 1)

    st.markdown("### Result:")
    if probability < 0.3:
        st.success(f"✅ Low Risk of Diabetes — {percent}%")
    elif probability < 0.6:
        st.warning(f"⚠️ Medium Risk of Diabetes — {percent}%")
    else:
        st.error(f"🚨 High Risk of Diabetes — {percent}%")

    st.markdown("### Patient Summary:")
    import pandas as pd
    summary = pd.DataFrame({
        "Feature": ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","DPF","Age"],
        "Value": [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    })
    st.table(summary)
