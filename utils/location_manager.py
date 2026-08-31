import json
import os
from pathlib import Path
import numpy as np
import streamlit as st

# File untuk menyimpan lokasi yang disave
LOCATIONS_FILE = "saved_locations.json"

def get_locations_file_path():
    """Get full path untuk locations file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, '..', LOCATIONS_FILE)

def load_saved_locations():
    """Load saved locations dari JSON file"""
    filepath = get_locations_file_path()
    
    if not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Convert lists back to numpy arrays
            for location_name, location_data in data.items():
                if 'video' in location_data and location_data['video'] is not None:
                    location_data['video'] = np.array(location_data['video'])
                if 'realtime' in location_data and location_data['realtime'] is not None:
                    location_data['realtime'] = np.array(location_data['realtime'])
            return data
    except Exception as e:
        st.error(f"Error loading saved locations: {e}")
        return {}

def save_locations_to_file(locations_dict):
    """Save locations dictionary ke JSON file"""
    filepath = get_locations_file_path()
    
    try:
        # Convert numpy arrays to lists untuk JSON serialization
        data_to_save = {}
        for location_name, location_data in locations_dict.items():
            data_to_save[location_name] = {}
            if 'video' in location_data:
                if isinstance(location_data['video'], np.ndarray):
                    data_to_save[location_name]['video'] = location_data['video'].tolist()
                else:
                    data_to_save[location_name]['video'] = location_data['video']
            if 'realtime' in location_data:
                if isinstance(location_data['realtime'], np.ndarray):
                    data_to_save[location_name]['realtime'] = location_data['realtime'].tolist()
                else:
                    data_to_save[location_name]['realtime'] = location_data['realtime']
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving locations: {e}")
        return False

def add_location(location_name, polygon_array, mode_type="video"):
    """Add atau update location"""
    locations = load_saved_locations()
    
    if location_name not in locations:
        locations[location_name] = {}
    
    locations[location_name][mode_type] = polygon_array
    
    return save_locations_to_file(locations)

def delete_location(location_name):
    """Delete location dari saved locations"""
    locations = load_saved_locations()
    
    if location_name in locations:
        del locations[location_name]
        return save_locations_to_file(locations)
    return False

def get_all_location_names():
    """Get list semua nama lokasi yang tersimpan"""
    locations = load_saved_locations()
    # Tambahkan preset locations
    preset_names = ["Fullscreen 720p", "Fullscreen 360p"]
    saved_names = list(locations.keys())
    
    # Gabungkan dan remove duplicates
    all_names = preset_names + [name for name in saved_names if name not in preset_names]
    
    return all_names

def get_location_polygon(location_name, mode_type="video"):
    """Get polygon untuk location tertentu"""
    # Cek preset locations dulu
    from utils.polygon_utils import PRESET_POLYGONS
    
    if location_name in PRESET_POLYGONS:
        polygon = PRESET_POLYGONS[location_name].get(mode_type)
        if polygon is not None:
            return polygon.copy()
    
    # Cek saved locations
    locations = load_saved_locations()
    if location_name in locations:
        polygon = locations[location_name].get(mode_type)
        if polygon is not None:
            if isinstance(polygon, list):
                return np.array(polygon)
            return polygon.copy()
    
    return None

def is_preset_location(location_name):
    """Check apakah location adalah preset atau custom"""
    from utils.polygon_utils import PRESET_POLYGONS
    return location_name in PRESET_POLYGONS
