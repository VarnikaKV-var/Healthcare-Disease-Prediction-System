import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .risk-low { background-color: #d4edda; padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; color: #155724; }
    .risk-medium { background-color: #fff3cd; padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; color: #856404; }
    .risk-high { background-color: #f8d7da; padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
except:
    model_loaded = False

# ── SESSION STATE FOR HISTORY ─────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = [
        {"Date": "2026-05-01 10:30", "Glucose": 120, "BMI": 28.5, "Age": 35, "Risk": "🟢 Low", "Probability": "22%"},
        {"Date": "2026-05-02 14:15", "Glucose": 150, "BMI": 33.2, "Age": 45, "Risk": "🔴 High", "Probability": "78%"},
        {"Date": "2026-05-03 09:00", "Glucose": 90,  "BMI": 24.1, "Age": 28, "Risk": "🟢 Low",  "Probability": "18%"},
    ]

# ── GAUGE CHART FUNCTION ──────────────────────────────
def create_gauge(probability):
    percent = round(probability * 100, 1)
    if probability < 0.3:
        color = "green"
        level = "LOW RISK"
    elif probability < 0.6:
        color = "orange"
        level = "MEDIUM RISK"
    else:
        color = "red"
        level = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Diabetes Risk — {level}", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30],   'color': '#d4edda'},
                {'range': [30, 60],  'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 6},
                'thickness': 0.85,
                'value': percent
            }
        }
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

# ── SIDEBAR ───────────────────────────────────────────
st.sidebar.title("🏥 HealthPredict AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🔬 Predict",
    "📋 History",
    "👨‍⚕️ Doctor",
    "📊 Dashboard",
    "ℹ️ About",
    "🔐 Login",
    "📝 Signup"
])
st.sidebar.markdown("---")
st.sidebar.info("💡 Go to Predict page and enter your health details to get your diabetes risk score!")

# ══════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("#### *Early detection saves lives — Know your risk today!*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Model Accuracy", "76.62%", "Best: Random Forest")
    with col2:
        st.metric("👥 Dataset Size", "768 Patients", "PIMA Indians Dataset")
    with col3:
        st.metric("🔢 Features Used", "8 Features", "Clinical Parameters")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🔍 What this app does:
        - ✅ Predicts **diabetes risk** using ML models
        - ✅ Shows **Low / Medium / High** risk with gauge chart
        - ✅ Saves your **prediction history**
        - ✅ Suggests **doctors** to consult
        - ✅ Shows **model comparison** dashboard
        """)
    with col2:
        st.markdown("""
        ### 📊 How to use:
        1. Click **🔬 Predict** in the sidebar
        2. Move the **sliders** to enter your health data
        3. Click the **Predict** button
        4. See your **risk level** on the gauge chart
        5. Check **📋 History** to see past predictions
        """)
    st.markdown("---")
    st.warning("⚠️ *This app is for informational purposes only. Always consult a qualified doctor for medical advice.*")

# ══════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════
elif page == "🔬 Predict":
    st.title("🔬 Diabetes Risk Prediction")
    st.markdown("*Move the sliders to enter your health details and click Predict*")
    st.markdown("---")

    if not model_loaded:
        st.error("⚠️ Model files not found! Make sure models/best_model.pkl and models/scaler.pkl exist.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📝 Enter Your Details")
            pregnancies    = st.slider("🤰 Pregnancies",               0,   17,  3)
            glucose        = st.slider("🩸 Glucose Level",             0,  200, 120)
            blood_pressure = st.slider("💓 Blood Pressure (mm Hg)",    0,  122,  70)
            skin_thickness = st.slider("📏 Skin Thickness (mm)",       0,   99,  20)
        with col2:
            st.markdown("### 📝 More Details")
            insulin        = st.slider("💉 Insulin (mu U/ml)",         0,  846,  80)
            bmi            = st.slider("⚖️ BMI",                     0.0, 67.1, 32.0)
            dpf            = st.slider("🧬 Diabetes Pedigree Function",0.0, 2.42, 0.5)
            age            = st.slider("🎂 Age",                       21,   81,  33)

        st.markdown("---")
        if st.button("🔍 PREDICT MY RISK", use_container_width=True, type="primary"):
            data        = np.array([[pregnancies, glucose, blood_pressure,
                                     skin_thickness, insulin, bmi, dpf, age]])
            data_scaled = scaler.transform(data)
            probability = model.predict_proba(data_scaled)[0][1]
            percent     = round(probability * 100, 1)

            st.markdown("### 📊 Your Risk Level:")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = create_gauge(probability)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                if probability < 0.3:
                    st.markdown('<div class="risk-low">✅ LOW RISK<br><br>Great! Keep maintaining a healthy lifestyle.</div>', unsafe_allow_html=True)
                elif probability < 0.6:
                    st.markdown('<div class="risk-medium">⚠️ MEDIUM RISK<br><br>Be careful! Diet control and exercise recommended.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-high">🚨 HIGH RISK<br><br>Please consult a doctor immediately!</div>', unsafe_allow_html=True)

            st.markdown("### 📋 Your Input Summary:")
            summary = pd.DataFrame({
                "Feature":       ["Pregnancies","Glucose","Blood Pressure","Skin Thickness","Insulin","BMI","DPF","Age"],
                "Your Value":    [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
                "Normal Range":  ["0-6", "70-99 mg/dL", "60-80 mmHg", "10-30 mm", "16-166 mu/ml", "18.5-24.9", "0.1-0.5", "Any"]
            })
            st.dataframe(summary, use_container_width=True)

            # Save to history
            risk_label = "🟢 Low" if probability < 0.3 else ("🟡 Medium" if probability < 0.6 else "🔴 High")
            new_entry  = {
                "Date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Glucose":     glucose,
                "BMI":         bmi,
                "Age":         age,
                "Risk":        risk_label,
                "Probability": f"{percent}%"
            }
            st.session_state.history.insert(0, new_entry)
            st.success("✅ This prediction has been saved to your History!")

# ══════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════
elif page == "📋 History":
    st.title("📋 Prediction History")
    st.markdown("*All your predictions are saved here automatically*")
    st.markdown("---")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet! Go to **🔬 Predict** page to make your first prediction.")
    else:
        st.success(f"✅ Total predictions made: **{len(st.session_state.history)}**")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 Risk Summary:")
        col1, col2, col3 = st.columns(3)
        low    = sum(1 for r in df['Risk'] if 'Low'    in r)
        medium = sum(1 for r in df['Risk'] if 'Medium' in r)
        high   = sum(1 for r in df['Risk'] if 'High'   in r)
        with col1:
            st.metric("🟢 Low Risk",    low)
        with col2:
            st.metric("🟡 Medium Risk", medium)
        with col3:
            st.metric("🔴 High Risk",   high)

        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════
# DOCTOR PAGE
# ══════════════════════════════════════════════════════
elif page == "👨‍⚕️ Doctor":
    st.title("👨‍⚕️ Consult a Doctor")
    st.markdown("*If you are at medium or high risk, please consult a specialist*")
    st.markdown("---")
    doctors = [
        {"name": "Dr. Anitha Sharma", "speciality": "Endocrinologist",  "hospital": "Apollo Hospital",  "contact": "+91 98765 43210", "exp": "15 years"},
        {"name": "Dr. Rajesh Kumar",  "speciality": "Diabetologist",    "hospital": "Fortis Hospital",  "contact": "+91 91234 56789", "exp": "12 years"},
        {"name": "Dr. Priya Nair",    "speciality": "General Physician", "hospital": "AIIMS Bangalore", "contact": "+91 87654 32109", "exp": "10 years"},
    ]
    for doc in doctors:
        with st.expander(f"🩺 {doc['name']} — {doc['speciality']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🏥 **Hospital:** {doc['hospital']}")
                st.write(f"📞 **Contact:** {doc['contact']}")
            with col2:
                st.write(f"⏳ **Experience:** {doc['exp']}")
                st.write(f"🔬 **Speciality:** {doc['speciality']}")
    st.markdown("---")
    st.warning("⚠️ Always consult a qualified medical professional for diagnosis and treatment.")

# ══════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.title("📊 Model Performance Dashboard")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Best Model",    "Random Forest")
    with col2:
        st.metric("🎯 Best Accuracy", "76.62%")
    with col3:
        st.metric("📈 Best AUC-ROC", "83.05%")

    st.markdown("### 🤖 Model Comparison:")
    comparison = pd.DataFrame({
        "Model":       ["Logistic Regression", "Random Forest ✅", "XGBoost"],
        "Accuracy %":  [70.78, 76.62, 72.08],
        "F1-Score %":  [63.41, 70.49, 65.60],
        "AUC-ROC %":   [82.53, 83.05, 78.37]
    })
    st.dataframe(comparison, use_container_width=True)

    st.markdown("### 📊 Dataset Information:")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Dataset:** PIMA Indians Diabetes Database\n\n**Total Records:** 768\n\n**Diabetic:** 268 (35%)\n\n**Non-Diabetic:** 500 (65%)")
    with col2:
        st.info("**Features:** 8 clinical parameters\n\n**Train Size:** 614 records\n\n**Test Size:** 154 records\n\n**Preprocessing:** SMOTE + StandardScaler")

# ══════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("---")
    st.markdown("""
    ## 🏥 Diabetes Risk Prediction System

    Built as a **Data Analytics & AI Internship Project** at **Incanto Dynamics Pvt. Ltd.**

    ### 🤖 ML Models Used:
    | Model | Accuracy | F1-Score | AUC-ROC |
    |---|---|---|---|
    | Logistic Regression | 70.78% | 63.41% | 82.53% |
    | **Random Forest ✅** | **76.62%** | **70.49%** | **83.05%** |
    | XGBoost | 72.08% | 65.60% | 78.37% |

    ### 🛠️ Technologies:
    Python • Streamlit • Scikit-learn • XGBoost • SHAP • SMOTE • Pandas • NumPy • Plotly

    ### 👩‍💻 Developer:
    **Varnika KV** — Data Analytics & AI Intern
    **Company:** Incanto Dynamics Private Limited, Bengaluru

    ---
    *Built with ❤️ using Streamlit*
    """)

# ══════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════
elif page == "🔐 Login":
    st.title("🔐 Login")
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            st.markdown("### Welcome Back! 👋")
            username = st.text_input("👤 Username")
            password = st.text_input("🔑 Password", type="password")
            submit   = st.form_submit_button("Login", use_container_width=True)
            if submit:
                if username == "admin" and password == "admin123":
                    st.success(f"✅ Welcome, {username}!")
                    st.balloons()
                else:
                    st.error("❌ Invalid username or password!")
        st.info("**Demo:** Username: `admin` | Password: `admin123`")

# ══════════════════════════════════════════════════════
# SIGNUP PAGE
# ══════════════════════════════════════════════════════
elif page == "📝 Signup":
    st.title("📝 Create Account")
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("signup_form"):
            st.markdown("### Create New Account 🆕")
            name     = st.text_input("👤 Full Name")
            email    = st.text_input("📧 Email")
            username = st.text_input("🔖 Username")
            password = st.text_input("🔑 Password",         type="password")
            confirm  = st.text_input("🔑 Confirm Password", type="password")
            submit   = st.form_submit_button("Sign Up", use_container_width=True)
            if submit:
                if not name or not email or not username or not password:
                    st.error("❌ Please fill in all fields!")
                elif password != confirm:
                    st.error("❌ Passwords do not match!")
                else:
                    st.success(f"✅ Account created for **{name}**! Please go to Login.")
                    st.balloons()
