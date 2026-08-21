import cv2
import streamlit as st

def add_annotations(annotated_frame, vehicle_classes, thickness, text_scale, frame_height, current_time):
    y_offset = int(30 * frame_height / 360)
    
    total_accidents = sum(st.session_state.accident_count.values())
    text = f"Accidents: {total_accidents}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    cv2.rectangle(annotated_frame, (10, y_offset - text_height - baseline), (10 + text_width, y_offset + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return annotated_frame