import streamlit as st
import torch
from collections import deque, defaultdict
from ultralytics import YOLO
import supervision as sv

@st.cache_resource
def load_yolo_model(model_path, selected_model):
    """Load YOLO model with caching to avoid reloading on every Streamlit rerun."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(f'{model_path}/{selected_model}').to(device)
    return model

def load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, FPS):
    model = load_yolo_model(model_path, selected_model)

    vehicle_classes = {"bus", "car", "motorcycle", "truck"}
    accident_classes = {
        "bus_bus_accident",
        "bus_object_accident",
        "bus_truck_accident",
        "car_bus_accident",
        "car_car_accident",
        "car_motorcycle_accident",
        "car_object_accident",
        "car_truck_accident",
        "motorcycle_bus_accident",
        "motorcycle_motorcycle_accident",
        "motorcycle_object_accident",
        "motorcycle_truck_accident",
        "truck_object_accident",
        "truck_truck_accident"
    }

    byte_track = sv.ByteTrack(
        frame_rate=FPS, 
        track_activation_threshold=confidence_threshold
    )

    thickness = int(sv.calculate_optimal_line_thickness(resolution_wh) / 2)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=FPS * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    polygon_zone = sv.PolygonZone(polygon=SOURCE)

    return model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, vehicle_classes, accident_classes, thickness, text_scale