"""
Aspect Ratio and Specifications Detection Module

This module contains functions for validating images against specific 
format requirements including dimensions, DPI, file format, and size.
"""

import os
import shutil
from PIL import Image, ExifTags
import numpy as np
import cv2
from pathlib import Path
import logging

# Configure paths
BASE_PROJECT_PATH = 'C:\\Users\\Mohamed Sameh\\Downloads\\IT_Task'
SOURCE_DIR = os.path.join(BASE_PROJECT_PATH, 'Photos4Testing')
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, 'ValidAspectRatioResults')
INVALID_DIR = os.path.join(OUTPUT_DIR, 'Invalid')
VALID_DIR = os.path.join(OUTPUT_DIR, 'Valid')

# Set up logging
logger = logging.getLogger(__name__)

# Image specifications
SPECS = {
    'website': {
        'slider': {
            'width': 1920,
            'height': 705,
            'dpi': (72, 72),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg'] # should be jpg only
        },
        'gallery': {
            'width': 800,
            'height': 600,
            'dpi': (72, 72),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg'] # should be jpg only
        }
    },
    'publication': {
        'digital_a4': {
            'min_width': 210,
            'min_height': 297,
            'width': 2480,
            'height': 3508,
            'dpi_range': (150, 200),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'PNG', 'png'], # should be jpg or png only
            'color_mode': 'RGB',
            'max_file_size_mb': 1
        },
        'digital_b5': {
            'min_width': 176,
            'min_height': 250,
            'width': 2069,
            'height': 2953,
            'dpi_range': (150, 200),
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'PNG', 'png'], # should be jpg or png only
            'color_mode': 'RGB',
        },
        'print_a4': {
            'min_width': 210,
            'min_height': 297,
            'width': 2480,
            'height': 3508,
            'dpi': (300, 300),
            # should be jpg or tiff or pdf only
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'TIFF', 'tiff', 'tif', 'TIF', 'PDF', 'pdf'],
            'color_mode': 'CMYK'
        },
        'print_b5': {
            'min_width': 176,
            'min_height': 250,
            'width': 2079,
            'height': 2953,
            'dpi': (300, 300),
            # should be jpg or tiff or pdf only
            'formats': ['JPEG', 'JPG', 'jpeg', 'jpg', 'TIFF', 'tiff', 'tif', 'TIF', 'PDF', 'pdf'],
            'color_mode': 'CMYK'
        }
    }
}


def get_image_dpi(image):
    """Get DPI information from image EXIF data."""
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'XResolution':
                    x_dpi = float(value)
                elif tag == 'YResolution':
                    y_dpi = float(value)
            return (x_dpi, y_dpi)
    except:
        pass
    
    # Default DPI if not found
    return image.info.get('dpi', (72, 72))


def get_file_size_mb(file_path):
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def meets_specifications(image_path, target_specs=None):
    """
    Check if an image meets any of the defined specifications.
    
    Parameters:
        image_path (str): Path to the image file
        target_specs (dict): Optional specific specs to check against
    
    Returns:
        tuple(bool, str): (True, spec_name) if image meets specs, (False, reason) otherwise
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            file_format = img.format
            dpi = get_image_dpi(img)
            file_size_mb = get_file_size_mb(image_path)
            
            # If specific specs provided, check only those
            if target_specs:
                specs_to_check = target_specs
            else:
                specs_to_check = SPECS
            
            # Check against all specification categories
            for category, category_specs in specs_to_check.items():
                for spec_name, spec in category_specs.items():
                    # Check format
                    if file_format not in spec.get('formats', []):
                        continue
                    
                    # Check dimensions
                    if 'width' in spec and 'height' in spec:
                        if width != spec['width'] or height != spec['height']:
                            continue
                    
                    # Check minimum dimensions
                    if 'min_width' in spec and 'min_height' in spec:
                        if width < spec['min_width'] or height < spec['min_height']:
                            continue
                    
                    # Check DPI
                    if 'dpi' in spec:
                        required_dpi = spec['dpi']
                        if abs(dpi[0] - required_dpi[0]) > 5 or abs(dpi[1] - required_dpi[1]) > 5:
                            continue
                    
                    # Check DPI range
                    if 'dpi_range' in spec:
                        min_dpi, max_dpi = spec['dpi_range']
                        if not (min_dpi <= dpi[0] <= max_dpi and min_dpi <= dpi[1] <= max_dpi):
                            continue
                    
                    # Check file size
                    if 'max_file_size_mb' in spec:
                        if file_size_mb > spec['max_file_size_mb']:
                            continue
                    
                    # Check color mode (basic check)
                    if 'color_mode' in spec:
                        required_mode = spec['color_mode']
                        if required_mode == 'CMYK' and img.mode != 'CMYK':
                            continue
                        if required_mode == 'RGB' and img.mode not in ['RGB', 'RGBA']:
                            continue
                    
                    # If we get here, all checks passed
                    return True, f"{category}_{spec_name}"
            
            return False, f"No matching specifications (size: {width}x{height}, format: {file_format}, dpi: {dpi})"
    
    except Exception as e:
        logger.error(f"Error checking specifications for {image_path}: {e}")
        return False, f"Error: {str(e)}"


if __name__ == "__main__":
    # Example usage
    try:
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(INVALID_DIR, exist_ok=True)
        os.makedirs(VALID_DIR, exist_ok=True)
        
        print(f"Aspect Ratio and Specifications Detection Module")
        print(f"Source Directory: {SOURCE_DIR}")
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"Ready for specifications validation.")
        
    except Exception as e:
        print(f"Error setting up directories: {e}")