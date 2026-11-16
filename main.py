import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auth import init_session
init_session()
from auth import logged_in, is_doctor, current_user, logout
with st.sidebar:
    st.markdown("## HealthPredict AI")
    st.markdown("---")
    if st.button("Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    if st.button("About", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()
    if logged_in():
        if st.button("Predict", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
        if st.button("History", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()
        if is_doctor():
            if st.button("Doctor Dashboard", use_container_width=True):
                st.session_state.page = "doctor"
                st.rerun()
        st.markdown("---")
        u = current_user()
        st.write("Logged in: " + u['full_name'])
        st.caption(u['role'] + " | " + u['email'])
        if st.button("Logout", use_container_width=True):
            logout()
    else:
        st.markdown("---")
        if st.button("Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()
        if st.button("Sign Up", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()
page = st.session_state.get("page", "home")
if page == "home":
    from page_views.home import show
    show()
elif page == "login":
    from page_views.login import show
    show()
elif page == "signup":
    from page_views.signup import show
    show()
elif page == "predict":
    from page_views.predict import show
    show()
elif page == "history":
    from page_views.history import show
    show()
elif page == "doctor":
    from page_views.doctor import show
    show()
elif page == "about":
    from page_views.about import show
    show()