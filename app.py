import streamlit as st
from modules import detectImage, detectVideo, detectRealTime

# Function to clear session state (preserve navigation keys)
def reset_session_state():
    keys_to_keep = {'current_page', 'last_page'}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

st.set_page_config(
    page_title="Vehicle and Accident Detection App",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="auto"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Detection Mode",
    ["Image Detection", "Video Detection", "Real-Time Detection"],
    key="current_page"
)

# Detect changes in page
if "last_page" not in st.session_state:
    st.session_state.last_page = page  # Initialize first page

if st.session_state.last_page != page:
    reset_session_state()  # Clear session state if page changes
    st.session_state.last_page = page  # Update last page

# Logic to display page according to selection
if page == "Image Detection":
    detectImage.main()  # Call main function from detectImage module
elif page == "Video Detection":
    detectVideo.main()  # Call main function from detectVideo module
elif page == "Real-Time Detection":
    detectRealTime.main()  # Call main function from detectRealTime module
