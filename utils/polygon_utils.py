import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json
from pathlib import Path


# Preset polygon coordinates untuk berbagai lokasi CCTV
PRESET_POLYGONS = {
    "Fullscreen 720p": {
        "video": np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]]),
        "realtime": None
    },
    "Fullscreen 360p": {
        "realtime": np.array([[0, 0], [640, 0], [640, 360], [0, 360]]),
        "video": None
    }
}


def init_polygon_state(key_prefix=""):
    """Initialize polygon state dalam session state"""
    prefix = f"{key_prefix}_" if key_prefix else ""
    
    if f"{prefix}polygon_points" not in st.session_state:
        st.session_state[f"{prefix}polygon_points"] = []
    if f"{prefix}drawing_mode" not in st.session_state:
        st.session_state[f"{prefix}drawing_mode"] = "polygon"
    if f"{prefix}stroke_width" not in st.session_state:
        st.session_state[f"{prefix}stroke_width"] = 2
    if f"{prefix}reset_counter" not in st.session_state:
        st.session_state[f"{prefix}reset_counter"] = 0
    if f"{prefix}saved_polygon" not in st.session_state:
        st.session_state[f"{prefix}saved_polygon"] = None


def get_preset_polygon(location, mode_type="video"):
    """Get preset polygon untuk lokasi yang dipilih"""
    if location in PRESET_POLYGONS:
        polygon = PRESET_POLYGONS[location].get(mode_type)
        if polygon is not None:
            return polygon.copy()
    return None


def draw_polygon_on_frame(frame, polygon_points, color=(0, 255, 0), thickness=2, alpha=0.3):
    """
    Draw polygon pada frame dengan transparansi
    
    Args:
        frame: Input frame (numpy array BGR)
        polygon_points: List or numpy array of polygon points [(x1,y1), (x2,y2), ...]
        color: Warna polygon (BGR format)
        thickness: Ketebalan garis
        alpha: Transparansi overlay (0-1)
    
    Returns:
        Frame dengan polygon yang ter-draw
    """
    # Handle None or empty
    if polygon_points is None:
        return frame
    
    # Convert to list if numpy array
    if isinstance(polygon_points, np.ndarray):
        polygon_points = polygon_points.tolist()
    
    # Check if has enough points
    if len(polygon_points) < 2:
        return frame
    
    frame_copy = frame.copy()
    overlay = frame.copy()
    
    # Convert polygon points to numpy array
    if len(polygon_points) > 0:
        pts = np.array(polygon_points, dtype=np.int32)
        
        # Draw filled polygon dengan transparansi
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], color)
        
        # Draw polygon outline
        cv2.polylines(frame_copy, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Blend overlay dengan frame
        cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
        
        # Draw points
        for point in polygon_points:
            cv2.circle(frame_copy, tuple(point), 5, color, -1)
    
    return frame_copy


def draw_interactive_canvas(frame, height, width, key_prefix="", mode_type="video"):
    """
    Draw interactive canvas untuk polygon drawing
    
    Args:
        frame: Input frame untuk canvas background
        height: Height frame asli
        width: Width frame asli
        key_prefix: Prefix untuk session state keys
        mode_type: "video" atau "realtime"
    
    Returns:
        Tuple (polygon_points_array, canvas_result)
    """
    init_polygon_state(key_prefix)
    
    prefix = f"{key_prefix}_" if key_prefix else ""
    
    # Resize gambar agar fit di viewport (max 600px width)
    max_canvas_width = 600
    scale_factor = 1.0
    
    if width > max_canvas_width:
        scale_factor = max_canvas_width / width
        canvas_width = max_canvas_width
        canvas_height = int(height * scale_factor)
    else:
        canvas_width = width
        canvas_height = height
    
    # Limit height juga
    max_canvas_height = 500
    if canvas_height > max_canvas_height:
        scale_factor = scale_factor * (max_canvas_height / canvas_height)
        canvas_height = max_canvas_height
        canvas_width = int(width * scale_factor)
    
    # Store scale factor di session state untuk later use
    st.session_state[f"{prefix}canvas_scale_factor"] = scale_factor
    
    # Resize frame untuk display
    if scale_factor < 1.0:
        resized_frame = cv2.resize(frame, (canvas_width, canvas_height))
    else:
        resized_frame = frame
        canvas_height = height
        canvas_width = width
    
    # Convert frame ke PIL Image
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Draw canvas full-width dengan dynamic key berdasarkan reset counter
    canvas_key = f"{prefix}polygon_canvas_{st.session_state[f'{prefix}reset_counter']}"
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.2)",
        stroke_width=st.session_state[f"{prefix}stroke_width"],
        stroke_color="rgba(0, 255, 0, 1)",
        background_image=pil_image,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        key=canvas_key,
        display_toolbar=True
    )
    
    # Extract polygon points dari canvas
    extracted_points = extract_polygon_from_canvas(canvas_result, (canvas_height, canvas_width))
    point_count = len(extracted_points) if extracted_points else 0
    
    # Display info dan buttons di bawah canvas
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Reset Canvas", key=f"{prefix}reset_btn", use_container_width=True):
            st.session_state[f"{prefix}reset_counter"] += 1
            st.session_state[f"{prefix}polygon_points"] = []
            st.rerun()
    
    with col2:
        # Save button - hanya muncul jika valid
        if point_count >= 3:
            saved_points = extracted_points
            if scale_factor < 1.0:
                saved_points = [(int(p[0] / scale_factor), int(p[1] / scale_factor)) for p in extracted_points]
            
            if st.button("Save Polygon", key=f"{prefix}save_btn", use_container_width=True, type="primary"):
                st.session_state[f"{prefix}saved_polygon"] = saved_points
                # Removed duplicate success message - will show in main flow
        else:
            st.button("Save Polygon", key=f"{prefix}save_btn_disabled", use_container_width=True, disabled=True)
    
    # Scale points back to original frame size jika ada scaling
    polygon_points = extracted_points
    if scale_factor < 1.0 and polygon_points:
        polygon_points = [(int(p[0] / scale_factor), int(p[1] / scale_factor)) for p in polygon_points]
    
    return polygon_points, canvas_result


def extract_polygon_from_canvas(canvas_result, frame_shape):
    """
    Extract polygon vertices dari st_canvas dengan mode polygon.
    Handle SVG path format: [["M",x,y],["L",x,y],["L",x,y],["z"]]
    """
    polygon_points = []
    
    if canvas_result is None:
        return polygon_points
    
    if not hasattr(canvas_result, 'json_data') or canvas_result.json_data is None:
        return polygon_points
    
    try:
        json_data = canvas_result.json_data
        objects = json_data.get("objects", [])
        
        for obj in objects:
            obj_type = obj.get("type", "")
            
            # Handle path objects (SVG format)
            if obj_type == "path":
                path_data = obj.get("path", [])
                
                # Parse SVG path commands
                # Format: [["M",x,y], ["L",x,y], ["L",x,y], ..., ["z"]]
                for command in path_data:
                    if isinstance(command, (list, tuple)) and len(command) >= 3:
                        cmd_type = command[0]
                        
                        # M = Move (start point), L = Line (vertex)
                        if cmd_type in ["M", "L"]:
                            try:
                                x = int(round(float(command[1])))
                                y = int(round(float(command[2])))
                                
                                # Validate coordinates
                                frame_height, frame_width = frame_shape
                                if 0 <= x < frame_width and 0 <= y < frame_height:
                                    # Avoid duplicates
                                    if not polygon_points or polygon_points[-1] != (x, y):
                                        polygon_points.append((x, y))
                            except (TypeError, ValueError, IndexError):
                                continue
            
            # Handle polyline/polygon objects (alternative formats)
            elif obj_type in ["polyline", "polygon"]:
                # Try different attribute names
                points_data = obj.get("points") or obj.get("vertices") or obj.get("coordinates", [])
                
                if points_data:
                    for point in points_data:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            try:
                                x = int(round(float(point[0])))
                                y = int(round(float(point[1])))
                                
                                frame_height, frame_width = frame_shape
                                if 0 <= x < frame_width and 0 <= y < frame_height:
                                    if not polygon_points or polygon_points[-1] != (x, y):
                                        polygon_points.append((x, y))
                            except (TypeError, ValueError):
                                continue
                                
    except Exception as e:
        pass
    
    return polygon_points


def polygon_to_numpy_array(polygon_points):
    """
    Convert polygon points to numpy array format (OpenCV compatible)
    
    Args:
        polygon_points: List of (x, y) tuples or numpy array
    
    Returns:
        Numpy array shape (n_points, 2) atau None jika invalid
    """
    if polygon_points is None:
        return None
    
    # If already numpy array, just validate
    if isinstance(polygon_points, np.ndarray):
        if len(polygon_points) < 3:
            return None
        return polygon_points
    
    # If list or other sequence
    if len(polygon_points) < 3:
        return None
    
    try:
        return np.array(polygon_points, dtype=np.int32)
    except Exception as e:
        st.error(f"Error converting polygon: {e}")
        return None


def show_polygon_preview(frame, polygon_points, title="Polygon Preview"):
    """Show preview of polygon on frame"""
    if polygon_points is None or len(polygon_points) < 3:
        st.warning("⚠️ Polygon tidak valid (minimum 3 titik)")
        return
    
    preview_frame = draw_polygon_on_frame(frame, polygon_points)
    st.image(preview_frame, channels="BGR", caption=title, use_container_width=True)


def load_or_draw_polygon(frame, location, mode_type="video", use_preset=True, key_prefix=""):
    """
    Load preset polygon atau allow drawing custom polygon
    
    Args:
        frame: Input frame
        location: Lokasi CCTV ("Simpang Pidada", "Custom", dll)
        mode_type: "video" atau "realtime"
        use_preset: Whether to offer preset loading
        key_prefix: Prefix untuk session state
    
    Returns:
        Numpy array polygon atau None
    """
    init_polygon_state(key_prefix)
    prefix = f"{key_prefix}_" if key_prefix else ""
    
    # Jika location adalah preset atau saved location dan use_preset=True
    if use_preset and location != "Draw Custom Area":
        # Try to get from location_manager first (supports both preset and saved)
        try:
            from utils.location_manager import get_location_polygon
            preset_polygon = get_location_polygon(location, mode_type)
        except:
            preset_polygon = get_preset_polygon(location, mode_type)
        
        if preset_polygon is not None:
            # Save to session state and return immediately
            st.session_state[f"{prefix}saved_polygon"] = preset_polygon
            return preset_polygon
    
    # Jika location adalah Draw Custom Area atau tidak ada preset
    st.info("📌 Click on the image to create polygon points (minimum 3 points required)")
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Draw interactive canvas
    polygon_points, canvas_result = draw_interactive_canvas(
        frame, frame_height, frame_width, key_prefix, mode_type
    )
    
    # Convert to numpy array
    if polygon_points is not None and len(polygon_points) >= 3:
        polygon_array = polygon_to_numpy_array(polygon_points)
        
        if polygon_array is not None:
            # Show preview
            show_polygon_preview(frame, polygon_points, "Your Custom Detection Area")
            
            # Check if user has clicked "Save Polygon" button on canvas
            # Save Polygon button sets the saved_polygon in session state
            if st.session_state.get(f"{prefix}saved_polygon") is not None:
                # Auto-confirm when polygon is saved via canvas button
                st.session_state[f"{prefix}polygon_points"] = polygon_array
                
                # No notification here - will be shown in main module (detectVideo/detectRealTime)
                return polygon_array
            else:
                # Polygon drawn but not yet saved via canvas button
                st.warning("⏳ Click 'Save Polygon' button above to confirm the area")
    
    return None


def display_polygon_info(polygon, frame_shape):
    """Display informasi tentang polygon"""
    if polygon is None or len(polygon) < 3:
        st.warning("⚠️ No valid polygon")
        return
    
    frame_height, frame_width = frame_shape[:2]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Points", len(polygon))
    
    with col2:
        # Calculate polygon area
        area = cv2.contourArea(polygon)
        st.metric("Area (px²)", f"{area:.0f}")
    
    with col3:
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(polygon)
        st.metric("Coverage", f"{(area / (frame_width * frame_height) * 100):.1f}%")
    
    # Display coordinates (removed nested expander)
    st.markdown("**📍 Coordinates (OpenCV Format):**")
    st.code(str(polygon.tolist()), language="python")


def save_polygon_to_file(polygon_array, location, filename=None):
    """Save polygon ke JSON file untuk future use"""
    if polygon_array is None:
        st.error("Cannot save None polygon")
        return None
    
    if filename is None:
        filename = f"polygon_{location.lower().replace(' ', '_')}.json"
    
    polygon_data = {
        "location": location,
        "points": polygon_array.tolist(),
        "timestamp": str(st.session_state.get("current_time", "")),
        "point_count": len(polygon_array)
    }
    
    filepath = Path(filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(polygon_data, f, indent=2)
        return str(filepath)
    except Exception as e:
        st.error(f"Error saving polygon: {e}")
        return None
