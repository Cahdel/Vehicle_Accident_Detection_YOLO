import streamlit as st
from modules import detectVideo

st.set_page_config(
    page_title="Vehicle and Accident Detection App",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

detectVideo.main()
