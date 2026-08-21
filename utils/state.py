import streamlit as st
from collections import defaultdict

def reset_state():
    st.session_state.frame_index = 0
    st.session_state.counted_vehicle_ids = set()
    st.session_state.counted_accident_ids = set()
    st.session_state.vehicle_count = defaultdict(int)
    st.session_state.accident_count = defaultdict(int)
    st.session_state.last_frame = None
    st.session_state.vehicle_accident_data = []
    st.session_state.annotated_frames = []
    st.session_state.accident_messages = []
    st.session_state.accident_snapshots = []
    st.session_state.notified_accident_ids = set()
    st.session_state.frame_count = 0
    # Release VideoWriter if active
    writer = st.session_state.get('video_writer')
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    st.session_state.video_writer = None
    st.session_state.video_output_path = None
    st.session_state.total_processing_time = 0.0
    st.session_state.processed_frames_count = 0

def init_state():
    if 'frame_index' not in st.session_state:
        st.session_state.frame_index = 0
    if 'counted_vehicle_ids' not in st.session_state:
        st.session_state.counted_vehicle_ids = set()
    if 'counted_accident_ids' not in st.session_state:
        st.session_state.counted_accident_ids = set()
    if 'vehicle_count' not in st.session_state:
        st.session_state.vehicle_count = defaultdict(int)
    if 'accident_count' not in st.session_state:
        st.session_state.accident_count = defaultdict(int)
    if 'last_frame' not in st.session_state:
        st.session_state.last_frame = None
    if 'vehicle_accident_data' not in st.session_state:
        st.session_state.vehicle_accident_data = []
    if 'annotated_frames' not in st.session_state:
        st.session_state.annotated_frames = []  
    if 'accident_messages' not in st.session_state:
        st.session_state.accident_messages = []
    if 'accident_snapshots' not in st.session_state:
        st.session_state.accident_snapshots = []
    if 'notified_accident_ids' not in st.session_state:
        st.session_state.notified_accident_ids = set()
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'video_writer' not in st.session_state:
        st.session_state.video_writer = None
    if 'video_output_path' not in st.session_state:
        st.session_state.video_output_path = None
    if 'total_processing_time' not in st.session_state:
        st.session_state.total_processing_time = 0.0
    if 'processed_frames_count' not in st.session_state:
        st.session_state.processed_frames_count = 0
