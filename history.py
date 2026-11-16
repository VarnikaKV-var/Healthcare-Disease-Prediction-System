import streamlit as st
import pandas as pd

st.set_page_config(page_title="History", page_icon="📋")

st.title("📋 Prediction History")
st.markdown("---")

# Sample history data
data = {
    "Date": ["2026-05-01", "2026-05-02", "2026-05-03"],
    "Glucose": [120, 150, 90],
    "BMI": [28.5, 33.2, 24.1],
    "Age": [35, 45, 28],
    "Risk": ["Low", "High", "Low"],
    "Probability": ["22%", "78%", "18%"]
}

df = pd.DataFrame(data)

st.write("Your past predictions:")
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.info("ℹ️ In a full app, this would show your actual saved predictions.")
