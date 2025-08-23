"""
Aspect Ratio and Specifications Detection Module

This module contains functions for validating images against specific 
format requirements including dimensions, DPI, file format, and size.
"""

import os
from PIL import Image, ExifTags
import logging

# Configure paths  
BASE_PROJECT_PATH = os.getcwd()
SOURCE_DIR = os.path.join(BASE_PROJECT_PATH, 'photos4testing')
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, 'Results')
INVALID_DIR = os.path.join(OUTPUT_DIR, 'invalid')
VALID_DIR = os.path.join(OUTPUT_DIR, 'valid')

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
            'width': 2079,  # Fixed: was 2069, now consistent with print_b5
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
    """Get DPI information from image EXIF data with proper error handling."""
    x_dpi = y_dpi = 72.0  # Default values initialized
    
    try:
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'XResolution':
                    try:
                        x_dpi = float(value) if isinstance(value, (int, float)) else 72.0
                    except (ValueError, TypeError):
                        x_dpi = 72.0
                elif tag == 'YResolution':
                    try:
                        y_dpi = float(value) if isinstance(value, (int, float)) else 72.0
                    except (ValueError, TypeError):
                        y_dpi = 72.0
    except (AttributeError, ValueError, TypeError) as e:
        logger.debug(f"Could not extract DPI from EXIF: {e}")
    
    # Fallback to PIL info if still default values
    if x_dpi == y_dpi == 72.0:
        dpi_info = image.info.get('dpi')
        if dpi_info and len(dpi_info) >= 2:
            try:
                x_dpi, y_dpi = float(dpi_info[0]), float(dpi_info[1])
            except (ValueError, TypeError, IndexError):
                pass  # Keep defaults
    
    return (x_dpi, y_dpi)


def get_file_size_mb(file_path):
    """Get file size in megabytes with error handling."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, IOError) as e:
        logger.warning(f"Could not get file size for {file_path}: {e}")
        return 0.0


def _check_dpi_requirements(actual_dpi, spec):
    """Check if DPI meets specification requirements."""
    x_dpi, y_dpi = actual_dpi
    
    # Check exact DPI requirement
    if 'dpi' in spec:
        required_dpi = spec['dpi']
        tolerance = 5  # Allow small tolerance for DPI matching
        if (abs(x_dpi - required_dpi[0]) > tolerance or 
            abs(y_dpi - required_dpi[1]) > tolerance):
            return False
    
    # Check DPI range requirement
    if 'dpi_range' in spec:
        min_dpi, max_dpi = spec['dpi_range']
        if not (min_dpi <= x_dpi <= max_dpi and min_dpi <= y_dpi <= max_dpi):
            return False
    
    return True


def _check_color_mode(actual_mode, required_mode):
    """Check if color mode meets requirements."""
    if not required_mode:
        return True
    
    if required_mode == 'CMYK':
        return actual_mode == 'CMYK'
    elif required_mode == 'RGB':
        return actual_mode in ['RGB', 'RGBA']
    
    return True


def _matches_single_spec(properties, spec):
    """Check if properties match a single specification."""
    # Format check
    if properties['format'] not in spec.get('formats', []):
        return False
    
    # Dimension checks - prioritize exact over minimum
    if 'width' in spec and 'height' in spec:
        if properties['width'] != spec['width'] or properties['height'] != spec['height']:
            return False
    elif 'min_width' in spec and 'min_height' in spec:
        if properties['width'] < spec['min_width'] or properties['height'] < spec['min_height']:
            return False
    
    # DPI checks
    if not _check_dpi_requirements(properties['dpi'], spec):
        return False
    
    # File size check
    if 'max_file_size_mb' in spec:
        if properties['file_size_mb'] > spec['max_file_size_mb']:
            return False
    
    # Color mode check
    if not _check_color_mode(properties['mode'], spec.get('color_mode')):
        return False
    
    return True


def _generate_failure_reason(properties):
    """Generate detailed failure reason with comprehensive information."""
    return (f"No matching specifications found:\n"
            f"  Size: {properties['width']}×{properties['height']}px\n" 
            f"  Format: {properties['format']}\n"
            f"  DPI: {properties['dpi'][0]:.1f}×{properties['dpi'][1]:.1f}\n"
            f"  Color Mode: {properties['mode']}\n"
            f"  File Size: {properties['file_size_mb']:.2f}MB")


def _check_against_specs(properties, specs_to_check):
    """Check image properties against specifications with priority ordering."""
    matches = []
    
    for category, category_specs in specs_to_check.items():
        for spec_name, spec in category_specs.items():
            if _matches_single_spec(properties, spec):
                # Assign priority: print > digital, exact dimensions > minimum
                priority = 0
                if 'print' in category:
                    priority += 100
                if 'width' in spec and 'height' in spec:
                    priority += 10  # Exact dimensions have higher priority
                
                matches.append((priority, category, spec_name, spec))
    
    if matches:
        # Return highest priority match (highest priority value)
        matches.sort(key=lambda x: x[0], reverse=True)
        _, category, spec_name, _ = matches[0]
        return True, f"{category}_{spec_name}"
    
    return False, _generate_failure_reason(properties)
def meets_specifications(image_path, target_specs=None):
    """
    Check if an image meets any of the defined specifications with optimized loading.
    
    Parameters:
        image_path (str): Path to the image file
        target_specs (dict): Optional specific specs to check against
    
    Returns:
        tuple(bool, str): (True, spec_name) if image meets specs, (False, reason) otherwise
    """
    # Input validation
    if not image_path:
        return False, "Error: No image path provided"
    
    if not os.path.exists(image_path):
        return False, f"Error: Image file not found: {image_path}"
    
    try:
        # Single image load with comprehensive property extraction
        with Image.open(image_path) as img:
            properties = {
                'width': img.size[0],
                'height': img.size[1],
                'format': img.format,
                'mode': img.mode,
                'dpi': get_image_dpi(img),
                'file_size_mb': get_file_size_mb(image_path)
            }
        
        # Use target specs if provided, otherwise use all SPECS
        specs_to_check = target_specs if target_specs else SPECS
        
        return _check_against_specs(properties, specs_to_check)
        
    except (IOError, OSError) as e:
        logger.error(f"Error opening image {image_path}: {e}")
        return False, f"Error: Cannot open image file - {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error checking specifications for {image_path}: {e}")
        return False, f"Error: {str(e)}"


def validate_spec_consistency():
    """Validate that specifications are logically consistent."""
    issues = []
    
    for category, category_specs in SPECS.items():
        for spec_name, spec in category_specs.items():
            # Check for conflicting dimension requirements
            if 'width' in spec and 'min_width' in spec:
                if spec['width'] < spec['min_width']:
                    issues.append(f"{category}.{spec_name}: width < min_width")
            
            if 'height' in spec and 'min_height' in spec:
                if spec['height'] < spec['min_height']:
                    issues.append(f"{category}.{spec_name}: height < min_height")
            
            # Check DPI consistency
            if 'dpi' in spec and 'dpi_range' in spec:
                dpi_val = spec['dpi'][0]  # Assume square DPI
                min_dpi, max_dpi = spec['dpi_range']
                if not (min_dpi <= dpi_val <= max_dpi):
                    issues.append(f"{category}.{spec_name}: DPI outside range")
    
    return issues


if __name__ == "__main__":
    # Example usage
    try:
        print(f"Aspect Ratio and Specifications Detection Module")
        print(f"Source Directory: photos4testing")
        print(f"Output Directory: Results")
        
        # Validate specifications consistency
        consistency_issues = validate_spec_consistency()
        if consistency_issues:
            print("\nSpecification Consistency Issues:")
            for issue in consistency_issues:
                print(f"  - {issue}")
        else:
            print("All specifications are consistent")
        
        print(f"Ready for specifications validation.")
        
        # Example check
        import sys
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                is_compliant, details = meets_specifications(image_path)
                print(f"Image: {os.path.basename(image_path)}")
                print(f"Compliant: {is_compliant}")
                print(f"Details: {details}")
            else:
                print(f"Image file not found: {image_path}")
        else:
            print("Usage: python Spec_detector.py <image_path>")
            print("Or import this module to use meets_specifications() function")
        
    except Exception as e:
        print(f"Error: {e}")