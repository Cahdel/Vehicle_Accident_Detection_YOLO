import streamlit as st
import cv2
import supervision as sv
import os
from datetime import timedelta

from utils.annotation import add_annotations

def get_video_writer():
    """Get or create a VideoWriter for saving annotated frames to disk instead of RAM."""
    if 'video_writer' not in st.session_state or st.session_state.video_writer is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputTest')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'output_video.mp4')
        st.session_state.video_output_path = output_path
        st.session_state.video_writer = None  # Will be initialized on first frame
    return st.session_state.get('video_writer')

def init_video_writer(frame_width, frame_height, fps):
    """Initialize the VideoWriter with frame dimensions."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputTest')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'output_video.mp4')
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    st.session_state.video_writer = writer
    st.session_state.video_output_path = output_path
    return writer

def release_video_writer():
    """Release the VideoWriter if it exists."""
    writer = st.session_state.get('video_writer')
    if writer is not None:
        writer.release()
        st.session_state.video_writer = None

def process_accident_detection(tracker_id, class_name, elapsed_time, timestamp, frame=None):
    # Check if it has been recorded before (avoid duplication)
    already_recorded = any(
        d["ID"] == tracker_id and d["Timestamp"] == timestamp and d["Class"] == class_name
        for d in st.session_state.vehicle_accident_data
    )
    if not already_recorded:
        st.session_state.vehicle_accident_data.append({
            "Sec": elapsed_time, 
            "Timestamp": timestamp, 
            "ID": tracker_id, 
            "Class": class_name, 
        })

        # Accident enumeration only if tracker_id has never been enumerated for an accident.
        if tracker_id not in st.session_state.counted_accident_ids:
            st.session_state.accident_count[class_name] += 1
            st.session_state.counted_accident_ids.add(tracker_id)

    if st.session_state.vehicle_accident_data:
        latest_accident = st.session_state.vehicle_accident_data[-1]  # Retrieve the latest accident data (last element)

        # Check if tracker_id is already in notified_accident_ids
        if tracker_id not in st.session_state.notified_accident_ids:
            # Format notification message
            accident_message = (
                f"🚨 **Accident Detected!**\n\n"
                f"🔹 **Sec:** {latest_accident['Sec']:.0f}\n"
                f"🕒 **Timestamp:** {latest_accident['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🆔 **ID:** {latest_accident['ID']}\n"
                f"🚗 **Class:** {latest_accident['Class']}"
            )

            # Save snapshot frame (convert BGR to RGB for display)
            snapshot = None
            if frame is not None:
                snapshot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Add accident_message and snapshot to session state
            st.session_state.accident_messages.append(accident_message)
            st.session_state.accident_snapshots.append(snapshot)

            # Add tracker_id to notified_accident_ids
            st.session_state.notified_accident_ids.add(tracker_id)

def process_frame(frame, model, byte_track, polygon_zone, vehicle_classes, accident_classes, start_time, elapsed_time, current_time, frame_height, box_annotator, label_annotator, trace_annotator, stframe, thickness, FPS, confidence_threshold, iou_threshold, text_scale):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    result = model(gray_frame_rgb)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > confidence_threshold]
    detections = detections[polygon_zone.trigger(detections)]
    detections = detections.with_nms(threshold=iou_threshold)
    detections = byte_track.update_with_detections(detections=detections)

    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        class_name = model.names[class_id]
        # Vehicle enumeration only if tracker_id has never been enumerated for vehicle
        if class_name in vehicle_classes and tracker_id not in st.session_state.counted_vehicle_ids:
            st.session_state.vehicle_count[class_name] += 1
            st.session_state.counted_vehicle_ids.add(tracker_id)

    labels = []
    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        class_name = model.names[class_id]
        timestamp = start_time + timedelta(seconds=elapsed_time)

        if class_name in vehicle_classes:
            st.session_state.vehicle_accident_data.append({
                "Sec": elapsed_time, 
                "Timestamp": timestamp, 
                "ID": tracker_id, 
                "Class": class_name, 
            })
        labels.append(f"#{tracker_id}")

    # Build annotated frame FIRST so it can be used as snapshot when accident detected
    annotated_frame = frame.copy()

    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    annotated_frame = add_annotations(annotated_frame, vehicle_classes, thickness, text_scale, frame_height, current_time)

    # Process accident detection with the fully annotated frame as snapshot
    for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
        class_name = model.names[class_id]
        timestamp = start_time + timedelta(seconds=elapsed_time)
        if class_name in accident_classes:
            process_accident_detection(tracker_id, class_name, elapsed_time, timestamp, annotated_frame)

    # Write frame to VideoWriter (on disk) instead of accumulating in RAM
    writer = st.session_state.get('video_writer')
    if writer is not None and writer.isOpened():
        writer.write(annotated_frame)  # Write BGR frame to video
    st.session_state.frame_count = st.session_state.get('frame_count', 0) + 1

    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    stframe.image(annotated_frame_rgb, channels="RGB")
    st.session_state.last_frame = annotated_frame_rgb