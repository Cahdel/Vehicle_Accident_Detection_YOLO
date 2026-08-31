import pandas as pd
import zipfile
import os
from io import BytesIO
import streamlit as st
import cv2

def save_video(frames, output_path, fps):
    """Legacy function: save frames from RAM to video file."""
    if frames:
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        video.release()

def create_excel_data(vehicle_accident_data, vehicle_count, accident_count, model_names):
    """Create Excel data from detection results."""
    vehicle_accident_df = pd.DataFrame(vehicle_accident_data)
    all_classes = list(model_names.values())

    vehicle_count_df = pd.DataFrame(list(vehicle_count.items()), columns=["Class", "Count_vehicle"])
    accident_count_df = pd.DataFrame(list(accident_count.items()), columns=["Class", "Count_accident"])

    vehicle_count_df = vehicle_count_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()
    accident_count_df = accident_count_df.set_index("Class").reindex(all_classes, fill_value=0).reset_index()

    count_df = pd.merge(vehicle_count_df, accident_count_df, on="Class", how="outer", suffixes=("_vehicle", "_accident"))
    count_df["Count"] = count_df["Count_vehicle"] + count_df["Count_accident"]
    count_df = count_df.drop(columns=["Count_vehicle", "Count_accident"])
    count_df = count_df.drop_duplicates(subset=["Class"])

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        vehicle_accident_df.to_excel(writer, index=False, sheet_name='Events')
        count_df.to_excel(writer, index=False, sheet_name='Summary')
    return output.getvalue()

def zip_results(video_file_path, excel_data):
    """Create ZIP file containing video and Excel results."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        if os.path.exists(video_file_path):
            zip_file.write(video_file_path, os.path.basename(video_file_path))
        zip_file.writestr('detection_result.xlsx', excel_data)
    return zip_buffer.getvalue()

def save_and_provide_download_button(current_dir, FPS, model):
    """Save detection results and provide download button.
    
    Uses the video file already written by VideoWriter on disk (if available),
    otherwise falls back to saving annotated_frames from RAM.
    """
    video_file_path = st.session_state.get('video_output_path')
    
    # Fallback: if no VideoWriter path, use legacy approach
    if video_file_path is None or not os.path.exists(video_file_path):
        video_file_path = os.path.join('/tmp', 'output_video.mp4')
        # Legacy: save from RAM if annotated_frames exist
        if st.session_state.get('annotated_frames'):
            save_video(st.session_state.annotated_frames, video_file_path, FPS)

    # Create Excel data
    excel_data = create_excel_data(
        st.session_state.vehicle_accident_data, 
        st.session_state.vehicle_count, 
        st.session_state.accident_count, 
        model.names
    )
    
    # Create ZIP
    zip_data = zip_results(video_file_path, excel_data)
    
    # Provide download button for ZIP files
    st.sidebar.download_button(
        label="Download Detection Results",
        data=zip_data,
        file_name='output_files.zip',
        mime='application/zip'
    )

    # Remove video file after zipping
    try:
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
    except Exception as e:
        print(f"Warning: Could not remove temp video file: {e}")