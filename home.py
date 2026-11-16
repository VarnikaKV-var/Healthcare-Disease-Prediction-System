import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠")

st.title("🏥 Diabetes Risk Prediction System")
st.markdown("---")

st.markdown("""
## Welcome! 👋

This app helps predict the **risk of diabetes** based on your health data using Machine Learning.

### 🔍 What you can do:
- **Predict** your diabetes risk instantly
- **View your history** of past predictions
- **Consult a doctor** if needed
- **Learn more** about diabetes

### 📊 How it works:
1. Go to the **Predict** page
2. Enter your health details
3. Get your risk result instantly!

---
⚠️ *This app is for informational purposes only. Always consult a doctor for medical advice.*
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", "76.62%")
with col2:
    st.metric("Dataset Size", "768 patients")
with col3:
    st.metric("Features Used", "8")
