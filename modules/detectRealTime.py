import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import time as t

from utils.polygon_utils import (
    load_or_draw_polygon,
    display_polygon_info,
    draw_polygon_on_frame,
)
from utils.saveOutputs import save_and_provide_download_button
from utils.displayStatistics import update_real_time_statistics
from utils.state import reset_state, init_state
from utils.components import load_model_and_initialize_components
from utils.detect import process_frame, init_video_writer, release_video_writer
from utils.obs_camera import init_obs_camera, send_frame_to_obs, cleanup_obs_camera

def _probe_camera(index, backend_id=None, warmup_reads=3):
    if backend_id is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend_id)

    if not cap.isOpened():
        cap.release()
        return None

    frame = None
    for _ in range(warmup_reads):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            break

    cap.release()

    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]
    return width, height

def _capture_first_frame(camera, max_attempts=30, delay_seconds=0.1, warmup_grabs=10):
    """Read the first valid frame with retries for slow-starting cameras."""
    for _ in range(warmup_grabs):
        camera.grab()

    for _ in range(max_attempts):
        ret, frame = camera.read()
        if ret and frame is not None and frame.size > 0:
            return True, frame
        t.sleep(delay_seconds)
    return False, None

@st.cache_data(ttl=60)
def get_available_cameras(max_cameras=10):
    """Deteksi kamera yang siap digunakan (berhasil baca frame). Cached for 60 seconds."""
    available_cameras = []

    backends = []
    if os.name == "nt":
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("DSHOW", cv2.CAP_DSHOW))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("MSMF", cv2.CAP_MSMF))
    backends.append(("DEFAULT", None))

    for index in range(max_cameras):
        for backend_name, backend_id in backends:
            result = _probe_camera(index, backend_id)
            if result is not None:
                width, height = result
                label = f"Camera {index} - {backend_name} ({width}x{height}) READY"
                available_cameras.append({
                    "label": label,
                    "index": index,
                    "backend": backend_id
                })
                break

    return available_cameras

def main():
    # Streamlit UI
    st.title("Vehicle and Accident Detection - Real-Time")

    # Get current directory for model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models')

    # Input sidebar
    st.sidebar.header("Step 1: Setup Webcam")
    
    # Deteksi dan pilih kamera
    available_cameras = get_available_cameras()
    if not available_cameras:
        st.error("No ready cameras detected")
        st.info("Tips: start OBS Virtual Camera in OBS, lalu refresh halaman")
        return

    camera_options = [camera["label"] for camera in available_cameras]
    selected_label = st.sidebar.selectbox(
        "Pilih Kamera",
        camera_options,
        help="Only cameras that successfully returned a frame are listed"
    )
    selected_camera = next(camera for camera in available_cameras if camera["label"] == selected_label)
    camera_id = selected_camera["index"]
    camera_backend = selected_camera["backend"]

    if st.session_state.get("realtime_selected_camera") != selected_label:
        st.session_state.realtime_selected_camera = selected_label
        st.session_state.realtime_first_frame = None
        st.session_state.realtime_first_frame_captured = False

    frame_width = 640
    frame_height = 360
    resolution_wh = (frame_width, frame_height)

    # Get FPS from webcam
    FPS = 25 / 2.2  # Simpang Pidada

    # Initialize session state
    if 'polygon_defined' not in st.session_state:
        st.session_state.polygon_defined = False
    if 'detection_area' not in st.session_state:
        st.session_state.detection_area = None
    
    init_state()
    
    st.info("Webcam ready")
    
    # STEP 2: Define Detection Area - capture first frame
    st.markdown("---")
    st.header("Step 2: Define Detection Area")
    st.info("Ambil frame pertama dari webcam dan pilih area deteksi")
    
    # Capture first frame untuk polygon setup
    first_frame_captured = st.session_state.get("realtime_first_frame_captured", False)
    first_frame = st.session_state.get("realtime_first_frame", None)
    
    if not first_frame_captured:
        # Only open camera temporarily for first frame capture
        with st.spinner("Capturing first frame from camera..."):
            if camera_backend is None:
                temp_camera = cv2.VideoCapture(camera_id)
            else:
                temp_camera = cv2.VideoCapture(camera_id, camera_backend)
            
            if not temp_camera.isOpened():
                st.error("❌ Cannot open camera for frame capture")
                return
            
            ret, first_frame = _capture_first_frame(temp_camera, max_attempts=60)
            
            # Immediately release temporary camera
            temp_camera.release()
            t.sleep(0.2)  # Give time for OS to release camera
        
        if ret and first_frame is not None:
            first_frame_resized = cv2.resize(first_frame, (frame_width, frame_height))
            st.session_state.realtime_first_frame = first_frame_resized
            st.session_state.realtime_first_frame_captured = True
            first_frame = first_frame_resized
            st.success("✅ First frame captured!")
        else:
            st.error("❌ Tidak dapat membaca frame dari webcam. Silakan coba lagi atau pilih kamera lain.")
            if st.button("🔄 Retry Capture"):
                st.session_state.realtime_first_frame_captured = False
                st.rerun()
            st.stop()
    
    if first_frame is not None:
        # Get all available locations
        try:
            from utils.location_manager import get_all_location_names, delete_location, is_preset_location
            all_locations = get_all_location_names()
        except:
            all_locations = ["Fullscreen 360p"]
        
        # Add custom drawing option
        all_locations.append("Draw Custom Area")
        
        # Location selection with management
        col_select, col_manage = st.columns([3, 1])
        
        with col_select:
            location = st.selectbox(
                "Select Location",
                all_locations,
                index=len(all_locations) - 1,
                key="realtime_location_select"
            )
        
        with col_manage:
            # Show delete button only for saved (non-preset) locations
            if location not in ["Draw Custom Area", "Fullscreen 720p", "Fullscreen 360p"]:
                try:
                    if not is_preset_location(location):
                        if st.button("🗑️ Delete", use_container_width=True, key="realtime_delete_location"):
                            if delete_location(location):
                                st.success(f"Deleted '{location}'")
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                except:
                    pass
            else:
                # Empty placeholder to maintain layout
                st.write("")
        
        # Load atau draw polygon
        SOURCE = load_or_draw_polygon(
            first_frame,
            location,
            mode_type="realtime",
            use_preset=True,
            key_prefix="realtime_polygon"
        )
        
        # Check if polygon is valid
        if SOURCE is not None and len(SOURCE) >= 3:
            st.session_state.polygon_defined = True
            st.session_state.detection_area = SOURCE
            
            # Display polygon info in a cleaner way
            st.markdown("---")
            # Removed duplicate success message - already shown in polygon_utils
            
            # Only show "Save This Area" for custom drawn polygons (not presets)
            if location == "Draw Custom Area":
                # Optional: Save this area for future use
                with st.expander("💾 Save This Area for Future Use", expanded=False):
                    st.write("Save this detection area with a custom name so you can reuse it later.")
                    new_location_name = st.text_input(
                        "Location Name:", 
                        key="realtime_save_location_name", 
                        placeholder="e.g., Office Entrance"
                    )
                    if st.button("Save Location", key="realtime_save_location_btn", type="primary"):
                        if new_location_name and new_location_name.strip():
                            try:
                                from utils.location_manager import add_location
                                if add_location(new_location_name.strip(), SOURCE, "realtime"):
                                    st.success(f"✅ Location '{new_location_name.strip()}' saved successfully!")
                                    st.info("↻ Refresh the page to see it in the location dropdown")
                                else:
                                    st.error("Failed to save location")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Please enter a location name")
            
            with st.expander("📊 Area Details", expanded=False):
                display_polygon_info(SOURCE, first_frame.shape)
            
            with st.expander("👁️ Preview Detection Area", expanded=True):
                preview_frame = draw_polygon_on_frame(first_frame, SOURCE.tolist())
                st.image(preview_frame, channels="BGR", use_container_width=True)
        else:
            st.warning("⏳ Please define a valid detection area (minimum 3 points)")
            st.session_state.polygon_defined = False
            st.stop()
    else:
        st.error("❌ Tidak dapat menangkap frame awal dari webcam")
        camera.release()
        st.stop()
    
    # STEP 3: Detection Settings - hanya bisa akses kalau polygon sudah valid
    if not st.session_state.polygon_defined or SOURCE is None:
        st.warning("⚠️ Silakan definisikan Detection Area terlebih dahulu")
        camera.release()
        st.stop()
    
    st.markdown("---")
    st.header("Detection Settings")
    
    # Get list of models
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Select Model", model_files)
    
    # Control buttons
    if 'status' not in st.session_state:
        st.session_state.status = 'stopped'
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start", use_container_width=True, type="primary"):
            st.session_state.status = 'running'
    with col2:
        if st.button("Pause", use_container_width=True):
            st.session_state.status = 'paused'
    with col3:
        if st.button("Stop", use_container_width=True):
            st.session_state.status = 'stopped'
    
    # Sidebar settings
    st.sidebar.markdown("---")
    st.sidebar.header("Detection Parameters")
    
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)
    
    # USE SAVED DETECTION AREA
    SOURCE = st.session_state.detection_area
    
    # Setup placeholders untuk display
    stframe = st.empty()
    vehicle_stats_placeholder = st.empty()
    accident_stats_placeholder = st.empty()

    # Set start time to current time
    start_time = datetime.now()

    start_processing_time = t.time()

    model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, FPS)

    plot_counter = 0
    obs_started = False
    camera = None

    # Detection loop - only initialize camera and OBS when actually running
    if st.session_state.status == 'running':
        # Open camera for detection
        try:
            if camera_backend is None:
                camera = cv2.VideoCapture(camera_id)
            else:
                camera = cv2.VideoCapture(camera_id, camera_backend)
            
            if not camera.isOpened():
                st.error("❌ Cannot open camera for detection")
                return
            
            # Warm up camera
            for _ in range(5):
                camera.grab()
                t.sleep(0.05)
                
        except Exception as e:
            st.error(f"❌ Error opening camera: {e}")
            return
        
        # Initialize VideoWriter for saving results to disk
        init_video_writer(frame_width, frame_height, int(FPS))
        
        # Initialize OBS Virtual Camera streaming
        obs_camera = init_obs_camera(
            width=frame_width,
            height=frame_height,
            fps=int(FPS)
        )
        
        if obs_camera.is_active() or obs_camera.start():
            obs_started = True

    while st.session_state.status == 'running':
        if camera is None or not camera.isOpened():
            st.error("Camera not available")
            break
        
        ret, frame = camera.read()
        if not ret:
            st.warning("Failed to read frame from webcam")
            t.sleep(0.1)
            continue

        frame = cv2.resize(frame, (frame_width, frame_height))

        elapsed_time = t.time() - start_processing_time
        current_time = start_time + timedelta(seconds=elapsed_time)
        
        process_frame(
            frame, 
            model, 
            byte_track, 
            polygon_zone, 
            vehicle_classes, 
            accident_classes, 
            start_time, 
            elapsed_time, 
            current_time, 
            frame_height, 
            box_annotator, 
            label_annotator, 
            trace_annotator, 
            stframe, 
            thickness, 
            FPS, 
            confidence_threshold, 
            iou_threshold, 
            text_scale
        )
        
        # Send processed frame to OBS Virtual Camera
        if obs_started and st.session_state.last_frame is not None:
            # Convert RGB frame from session_state back to BGR for OBS
            frame_bgr = cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_RGB2BGR)
            send_frame_to_obs(frame_bgr)

        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            plot_counter, 
            vehicle_classes
        )

    # Release the webcam when finished
    if camera is not None:
        camera.release()
    
    # Stop OBS Virtual Camera streaming
    if obs_started:
        cleanup_obs_camera()
    
    # Release VideoWriter
    release_video_writer()

    # Display results and provide download when stopped
    if st.session_state.status == 'stopped' and st.session_state.last_frame is not None:
        # Display the last detected frame
        stframe.image(st.session_state.last_frame, channels="RGB")

        # Display final statistics
        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            plot_counter, 
            vehicle_classes
        )

        # Display all accident messages
        snapshots = st.session_state.get('accident_snapshots', [])
        for i, message in enumerate(st.session_state.accident_messages):
            with st.container(border=True):
                st.error(message)
                if i < len(snapshots) and snapshots[i] is not None:
                    st.image(
                        snapshots[i],
                        caption=f"📸 Snapshot Accident #{i+1}",
                        use_container_width=True
                    )

        # Save video and xlsx files and provide a download button for ZIP files
        video_output_path = st.session_state.get('video_output_path')
        if video_output_path and os.path.exists(video_output_path) and st.session_state.get('frame_count', 0) > 0:
            save_and_provide_download_button(current_dir, FPS, model)

    # Display last frame and chart when status 'paused'
    if st.session_state.status == 'paused' and st.session_state.last_frame is not None:
        stframe.image(st.session_state.last_frame, channels="RGB")

        plot_counter = update_real_time_statistics(
            vehicle_stats_placeholder, 
            accident_stats_placeholder, 
            plot_counter, 
            vehicle_classes
        )
        
        # Display all accident_messages stored in st.session_state
        snapshots = st.session_state.get('accident_snapshots', [])
        for i, message in enumerate(st.session_state.accident_messages):
            with st.container(border=True):
                st.error(message)
                if i < len(snapshots) and snapshots[i] is not None:
                    st.image(
                        snapshots[i],
                        caption=f"📸 Snapshot Accident #{i+1}",
                        use_container_width=True
                    )
