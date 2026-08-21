import streamlit as st
import cv2
import numpy as np
import supervision as sv
import plotly.express as px
import pandas as pd
from io import BytesIO
import os
import time
from PIL import Image
from utils.components import load_yolo_model

def main():
    st.title("Vehicle and Accident Detection - Image")

    # Sidebar - Model Input
    st.sidebar.header("Input Settings")

    model_path = "models"  # Ensure model folder is available
    model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
    selected_model = st.sidebar.selectbox("Select Model", model_files)

    # Sidebar - Confidence and IoU Threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.05)

    # Sidebar - Upload Image
    uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Show uploaded image
        st.subheader("Original Image")
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, channels="BGR")

        # Convert image to grayscale for detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert grayscale to RGB (copy grayscale channel to 3 channels)
        gray_image_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        height, width, _ = image.shape
        resolution_wh = (width, height)

        # Load model (cached — won't reload on every rerun)
        model = load_yolo_model(model_path, selected_model)

        # Detect objects in the grayscale image converted to RGB
        detection_start = time.time()
        results = model(gray_image_rgb)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > confidence_threshold]
        detections = detections.with_nms(threshold=iou_threshold)
        detection_time = time.time() - detection_start

        thickness = sv.calculate_optimal_line_thickness(resolution_wh)
        text_scale = sv.calculate_optimal_text_scale(resolution_wh)

        # Add labels based on class names
        labels = [
            f"{model.names[class_id]} ({confidence:.2f})"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Add annotations to the colored image
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )

        annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Show annotated image
        st.subheader("Annotated Image")
        st.image(annotated_image, channels="RGB")

        # Show processing time
        st.metric(
            label="⏱️ Detection Processing Time",
            value=f"{detection_time:.4f} s",
            help="Time taken to run YOLO inference and apply confidence/NMS filtering on this image."
        )

        # Detection statistics
        class_counts = pd.DataFrame(detections.class_id, columns=["Class"])
        class_counts["Class"] = class_counts["Class"].map(model.names)
        counts_df = class_counts["Class"].value_counts().reset_index()
        counts_df.columns = ["Class", "Count"]

        # Show chart
        st.subheader("Graph of Number of Detected Objects")
        fig = px.bar(counts_df, x="Count", y="Class", orientation='h', title="Count of Detections per Class")
        st.plotly_chart(fig)

        # Save the result image for download
        # Convert to PIL Image (RGB)
        pil_image = Image.fromarray(annotated_image)

        # Save image to buffer in JPEG format
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Download button in sidebar
        st.sidebar.download_button(
            label="Download Result Image",
            data=buffer,
            file_name="annotated_image.jpg",
            mime="image/jpeg"
        )
