import streamlit as st

st.set_page_config(page_title="Signup", page_icon="📝")

st.title("📝 Create Account")
st.markdown("---")

with st.form("signup_form"):
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    submit = st.form_submit_button("Sign Up")

    if submit:
        if not name or not email or not username or not password:
            st.error("Please fill in all fields!")
        elif password != confirm:
            st.error("Passwords do not match!")
        else:
            st.success(f"Account created successfully for {name}! ✅ Please go to Login.")
