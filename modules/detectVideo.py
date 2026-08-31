import streamlit as st
import numpy as np
import supervision as sv
from tempfile import NamedTemporaryFile
import os
import time
from datetime import datetime, timedelta
import cv2

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

def main():
    # Streamlit UI
    st.title("Vehicle and Accident Detection - Video")
    
    # Get current directory for model path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models')

    # Sidebar input
    st.sidebar.header("Step 1: Upload Video")

    # Upload video
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    SOURCE = None
    
    if uploaded_video is not None:
        # Initialize session state untuk tracking
        if 'polygon_defined' not in st.session_state:
            st.session_state.polygon_defined = False
        if 'detection_area' not in st.session_state:
            st.session_state.detection_area = None
        if 'last_uploaded_video' not in st.session_state:
            st.session_state.last_uploaded_video = None
            
        # Reset if new video uploaded
        if st.session_state.last_uploaded_video != uploaded_video.name:
            st.session_state.polygon_defined = False
            st.session_state.detection_area = None
            st.session_state.last_uploaded_video = uploaded_video.name
            reset_state()
        
        # Save uploaded video to temp file (single temp file used throughout)
        tfile = NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        temp_file_path = tfile.name
        
        try:
            video_info = sv.VideoInfo.from_video_path(video_path=temp_file_path)
            frame_height = video_info.height
            frame_width = video_info.width
            resolution_wh = video_info.resolution_wh
            FPS = video_info.fps
            total_frames = video_info.total_frames
            
            # Capture first frame using cv2 (efficient, doesn't load entire video)
            cap = cv2.VideoCapture(temp_file_path)
            ret, first_frame = cap.read()
            cap.release()
            
            if not ret or first_frame is None:
                st.error("❌ Tidak dapat membaca frame dari video")
                st.stop()
            
            # STEP 2: Define Detection Area
            st.markdown("---")
            st.header("Step 2: Define Detection Area")
            
            # Get all available locations
            try:
                from utils.location_manager import get_all_location_names, delete_location, is_preset_location
                all_locations = get_all_location_names()
            except:
                all_locations = ["Fullscreen 720p"]
            
            # Add custom drawing option
            all_locations.append("Draw Custom Area")
            
            # Location selection with management
            col_select, col_manage = st.columns([3, 1])
            
            with col_select:
                location = st.selectbox(
                    "Select Location",
                    all_locations,
                    index=len(all_locations) - 1,
                    key="video_location_select"
                )
            
            with col_manage:
                # Show delete button only for saved (non-preset) locations
                if location not in ["Draw Custom Area", "Fullscreen 720p", "Fullscreen 360p"]:
                    try:
                        if not is_preset_location(location):
                            if st.button("🗑️ Delete", use_container_width=True, key="video_delete_location"):
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
                mode_type="video",
                use_preset=True,
                key_prefix="video_polygon"
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
                            key="video_save_location_name", 
                            placeholder="e.g., Main Street Intersection"
                        )
                        if st.button("Save Location", key="video_save_location_btn", type="primary"):
                            if new_location_name and new_location_name.strip():
                                try:
                                    from utils.location_manager import add_location
                                    if add_location(new_location_name.strip(), SOURCE, "video"):
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
            
            # STEP 3: Detection Settings
            if not st.session_state.polygon_defined or SOURCE is None:
                st.warning("⚠️ Silakan definisikan Detection Area terlebih dahulu")
                st.stop()
            
            st.markdown("---")
            st.header("Detection Settings")
            
            # Hardcoded model for HF Spaces deployment
            selected_model = "Augmen3x-Yolov11m.pt"

            # Control buttons
            if 'status' not in st.session_state:
                st.session_state.status = 'paused'
            
            init_state()

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Start/Restart", use_container_width=True, type="primary"):
                    st.session_state.status = 'running'
                    reset_state()
            with col2:
                if st.button("Continue", use_container_width=True):
                    st.session_state.status = 'running'
            with col3:
                if st.button("Pause", use_container_width=True):
                    st.session_state.status = 'paused'
            
            # Sidebar settings
            st.sidebar.markdown("---")
            st.sidebar.header("Detection Parameters")
            
            confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
            iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)
            
            # USE SAVED DETECTION AREA
            SOURCE = st.session_state.detection_area
            
            # Placeholder untuk video display
            stframe = st.empty()
            progress_placeholder = st.empty()
            vehicle_stats_placeholder = st.empty()
            accident_stats_placeholder = st.empty()

            # Set start time to current time
            start_time = datetime.now()

            model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, vehicle_classes, accident_classes, thickness, text_scale = load_model_and_initialize_components(
                model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, FPS
            )

            plot_counter = 0

            # Detection loop using generator (memory efficient — one frame at a time)
            if st.session_state.status == 'running':
                # Initialize VideoWriter for saving results to disk
                init_video_writer(frame_width, frame_height, FPS)
                
                frame_gen = sv.get_video_frames_generator(source_path=temp_file_path)
                
                # Skip already processed frames (for Continue functionality)
                for _ in range(st.session_state.frame_index):
                    try:
                        next(frame_gen)
                    except StopIteration:
                        break
                
                total_processing_time = st.session_state.get('total_processing_time', 0.0)
                processed_frames_count = st.session_state.get('processed_frames_count', 0)
                
                for frame in frame_gen:
                    if st.session_state.status != 'running':
                        break
                    
                    st.session_state.frame_index += 1

                    elapsed_time = st.session_state.frame_index / FPS
                    current_time = start_time + timedelta(seconds=elapsed_time)
                    
                    frame_start = time.time()
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
                    total_processing_time += time.time() - frame_start
                    processed_frames_count += 1

                    plot_counter = update_real_time_statistics(
                        vehicle_stats_placeholder, 
                        accident_stats_placeholder, 
                        plot_counter, 
                        vehicle_classes
                    )
                    
                    # Update progress bar
                    if total_frames and total_frames > 0:
                        progress = min(st.session_state.frame_index / total_frames, 1.0)
                        progress_placeholder.progress(progress, text=f"Processing frame {st.session_state.frame_index}/{total_frames}")
                
                st.session_state.status = 'paused'
                st.session_state.total_processing_time = total_processing_time
                st.session_state.processed_frames_count = processed_frames_count
                
                # Release VideoWriter
                release_video_writer()
                
                # Clear progress bar
                progress_placeholder.empty()

            # Save and provide download button
            video_output_path = st.session_state.get('video_output_path')
            if video_output_path and os.path.exists(video_output_path) and st.session_state.get('frame_count', 0) > 0:
                save_and_provide_download_button(current_dir, FPS, model)

            # Display last frame and chart when status 'paused'
            if st.session_state.status == 'paused' and st.session_state.last_frame is not None:
                stframe.image(st.session_state.last_frame, channels="RGB")

                # Show processing time metrics
                total_proc_time = st.session_state.get('total_processing_time', 0.0)
                proc_frames = st.session_state.get('processed_frames_count', 0)
                if total_proc_time > 0:
                    st.markdown("---")
                    st.subheader("⏱️ Detection Processing Time")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.metric(
                            label="Total Processing Time",
                            value=f"{total_proc_time:.2f} s",
                            help="Total cumulative time spent on YOLO inference across all processed frames."
                        )
                    with col_t2:
                        avg_per_frame = (total_proc_time / proc_frames) if proc_frames > 0 else 0.0
                        st.metric(
                            label="Avg. Time per Frame",
                            value=f"{avg_per_frame:.4f} s",
                            help="Average detection processing time per frame."
                        )

                plot_counter = update_real_time_statistics(
                    vehicle_stats_placeholder, 
                    accident_stats_placeholder, 
                    plot_counter, 
                    vehicle_classes
                )

                # Display all accident_messages stored in st.session_state
                if st.session_state.accident_messages:
                    st.markdown("---")
                    st.subheader("🚨 Accident Detections")
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
                else:
                    st.success("✓ No accidents detected in this video")
                
        except Exception as e:
            st.error(f"Error during processing: {e}")
        finally:
            # Cleanup temp file
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file: {e}")
    else:
        st.info("👆 Upload video terlebih dahulu")
        st.stop()
