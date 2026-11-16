import streamlit as st

st.set_page_config(page_title="Login", page_icon="🔐")

st.title("🔐 Login")
st.markdown("---")

with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

    if submit:
        if username == "admin" and password == "admin123":
            st.success(f"Welcome, {username}! ✅")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
        else:
            st.error("Invalid username or password!")

st.markdown("---")
st.write("Don't have an account? Go to **Signup** page.")
