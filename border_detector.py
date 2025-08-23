import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm
import json
import logging
import shutil
import traceback
import tempfile
import argparse
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consolidated and calibrated configuration with consistent thresholds
DEFAULT_CONFIG = {
    'BASE_PROJECT_PATH': os.environ.get('PHOTOVALIDATOR_PATH', os.getcwd()),  # Cross-platform compatible
    'SOURCE_DIR': 'photos4testing',
    'OUTPUT_DIR': 'Results',  # Unified output folder
    'PARALLEL_PROCESSING': True,
    'SAVE_INTERMEDIATE': False,
    'EXPORT_JSON': False,
    
    # === CORE DETECTION THRESHOLDS (calibrated and consistent) ===
    'COLOR_DIFF_THRESHOLD': 30,           # Primary color difference threshold
    'COLOR_DIFF_THRESHOLD_LOW': 15,       # Lower threshold for subtle borders
    'COLOR_DIFF_THRESHOLD_HIGH': 50,      # Higher threshold for strong borders
    
    'UNIFORMITY_THRESHOLD': 0.75,         # Border uniformity requirement (0-1)
    'UNIFORMITY_THRESHOLD_STRICT': 0.85,  # Strict uniformity for high confidence
    'TEXTURE_SENSITIVITY': 1.8,
    'CONFIDENCE_THRESHOLD_LOW': 0.35,     # Below this = no border
    'CONFIDENCE_THRESHOLD_MANUAL': 0.55,  # Above this = manual review needed  
    'CONFIDENCE_THRESHOLD_INVALID': 0.58, # Above this = invalid (border detected)
    'CONFIDENCE_THRESHOLD_OBVIOUS': 0.85, # Above this = very obvious border
    
    # === EDGE DETECTION PARAMETERS ===
    'CANNY_THRESHOLD1': 50,
    'CANNY_THRESHOLD2': 150,
    'EDGE_DILATION_ITERATIONS': 2,
    
    # === BORDER SIZE CONSTRAINTS ===
    'MIN_BORDER_SIZE_PX': 2,              # Minimum border width in pixels
    'MAX_BORDER_SIZE_RATIO': 0.15,        # Maximum border as ratio of image dimension
    'MIN_BORDER_SIZE_RATIO': 0.005,       # Minimum border as ratio of image dimension
    'PERIMETER_FOCUS_WIDTH': 0.15,        # Width of perimeter region as ratio of image dimension
    
    # === FRAME DETECTION PARAMETERS ===
    'FRAME_AREA_MIN': 0.1,                # Minimum frame area as ratio of image
    'FRAME_AREA_MAX': 0.9,                # Maximum frame area as ratio of image
    
    # === QUALITY AND NOISE THRESHOLDS ===
    'NOISE_THRESHOLD': 25,                # Noise level above which penalties apply
    'LOW_CONTRAST_THRESHOLD': 30,         # Standard deviation below which image is low contrast
    'UNIFORM_BG_THRESHOLD': 0.8,          # Uniformity level for uniform background detection
    
    # === FEATURE DETECTION FLAGS ===
    'ENABLE_OBVIOUS_BORDER_CHECK': True,
    'ENABLE_TEXTURE_DETECTION': True,
    'ENABLE_VIGNETTE_DETECTION': True,
    'ENABLE_ARTIFACT_DETECTION': True,
    'ENABLE_SYMMETRY_CHECK': True,
    'ENABLE_GRADIENT_ANALYSIS': True,
    
    # === OBVIOUS BORDER DETECTION PARAMETERS ===
    'OBVIOUS_BORDER_MIN_WIDTH': 3,
    'OBVIOUS_BORDER_COLOR_DIFF': 25,
    'OBVIOUS_BORDER_STD_THRESHOLD': 15,
    
    # === QUALITY PENALTIES ===
    'NOISE_PENALTY_FACTOR': 0.8,          # Penalty factor for noisy images
    'UNIFORM_BG_PENALTY_FACTOR': 0.7,     # Penalty factor for uniform background
    'ARTIFACT_PENALTY_FACTOR': 0.6,       # Penalty factor for compression artifacts
    'SOLID_COLOR_STD_THRESHOLD': 3,       # Standard deviation threshold for solid color images
    
    # === MANUAL REVIEW SYSTEM ===
    'SAVE_MANUAL_REVIEW_SEPARATELY': True,
    
    # === LEGACY COMPATIBILITY (deprecated - use CONFIDENCE_THRESHOLD_* instead) ===
    'MANUAL_REVIEW_THRESHOLD_LOW': 0.35,
    'MANUAL_REVIEW_THRESHOLD_HIGH': 0.58,
    # === ADAPTIVE THRESHOLDS ===
    'ADAPTIVE_THRESHOLDS': False,  # Set to True if you want adaptive thresholds enabled by default
    # === CENTER EXCLUSION (legacy, usually disabled for better detection) ===
    'CENTER_EXCLUSION': False,  # Set to True to focus only on perimeter, False for full image analysis
    'USE_PERIMETER_MASKING': False,  # Set to True to enable perimeter masking for edge detection
    
    # === PREPROCESSING OPTIONS ===
    'APPLY_GAUSSIAN_BLUR': False,  # Disable by default - can smear thin borders
    'BLUR_KERNEL_SIZE': 3,         # Smaller kernel when blur is needed
    'USE_BILATERAL_FILTER': False, # Better for preserving edges while reducing noise
    'BILATERAL_D': 9,              # Diameter for bilateral filter
    'BILATERAL_SIGMA_COLOR': 75,   # Color similarity threshold
    'BILATERAL_SIGMA_SPACE': 75,   # Spatial proximity threshold
    'SKIP_PREPROCESSING': True,    # Skip all preprocessing for cleanest edge detection
}


def validate_preprocessing_config(config):
    """Validate preprocessing configuration for conflicts."""
    preprocess_flags = [
        config.get('APPLY_GAUSSIAN_BLUR', False),
        config.get('USE_BILATERAL_FILTER', False)
    ]
    
    if sum(preprocess_flags) > 1:
        raise ValueError("Multiple preprocessing methods enabled - choose only one")
    
    if config.get('SKIP_PREPROCESSING', False) and any(preprocess_flags):
        logger.warning("SKIP_PREPROCESSING=True overrides other preprocessing flags")


def validate_threshold_consistency(config):
    """Validate that threshold values are logically consistent."""
    # Check confidence thresholds
    low = config.get('CONFIDENCE_THRESHOLD_LOW', 0.4)
    manual = config.get('CONFIDENCE_THRESHOLD_MANUAL', 0.55)
    invalid = config.get('CONFIDENCE_THRESHOLD_INVALID', 0.58)
    obvious = config.get('CONFIDENCE_THRESHOLD_OBVIOUS', 0.85)
    
    if not (low < manual < invalid < obvious):
        raise ValueError(f"Confidence thresholds must be in ascending order: {low} < {manual} < {invalid} < {obvious}")
    
    # Check border size constraints
    min_ratio = config.get('MIN_BORDER_SIZE_RATIO', 0.005)
    max_ratio = config.get('MAX_BORDER_SIZE_RATIO', 0.15)
    
    if min_ratio >= max_ratio:
        raise ValueError(f"MIN_BORDER_SIZE_RATIO ({min_ratio}) must be less than MAX_BORDER_SIZE_RATIO ({max_ratio})")


class ConfigManager:
    """Centralized configuration management with validation."""
    
    def __init__(self, config_path=None, base_config=None):
        self.config = (base_config or DEFAULT_CONFIG).copy()
        
        if config_path and os.path.exists(config_path):
            self._load_external_config(config_path)
        
        self._validate_config()
    
    def _load_external_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                external_config = json.load(f)
            self.config.update(external_config)
            logger.info(f"Loaded external configuration from {config_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load external config {config_path}: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration consistency."""
        validate_preprocessing_config(self.config)
        validate_threshold_consistency(self.config)
        logger.debug("Configuration validation passed")
    
    def get_config(self):
        """Get validated configuration."""
        return self.config.copy()


class ImagePropertiesCache:
    """Cache for expensive image property calculations."""
    
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def get_properties(self, img_path, image=None):
        """Get cached image properties or calculate them."""
        try:
            # Create cache key based on file path and modification time
            cache_key = f"{img_path}_{os.path.getmtime(img_path)}"
            
            if cache_key in self.cache:
                # Move to end of access order (LRU)
                self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                return self.cache[cache_key]
            
            # Calculate new properties
            properties = self._calculate_properties(img_path, image)
            
            # Add to cache with LRU eviction
            if len(self.cache) >= self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            self.cache[cache_key] = properties
            self.access_order.append(cache_key)
            
            return properties
            
        except OSError as e:
            logger.warning(f"Could not cache properties for {img_path}: {e}")
            return self._calculate_properties(img_path, image)
    
    def _calculate_properties(self, img_path, image=None):
        """Calculate image properties."""
        if image is None:
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Could not load image: {img_path}")
        
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        return {
            'height': height,
            'width': width,
            'min_dimension': min(height, width),
            'image_area': height * width,
            'gray': gray,
            'global_mean': np.mean(gray),
            'global_std': np.std(gray),
            'file_size': os.path.getsize(img_path) if os.path.exists(img_path) else 0
        }

def load_images(config):
    """Load all images from the source directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']
    images_path = os.path.join(config['BASE_PROJECT_PATH'], config['SOURCE_DIR'])
    images = []
    
    logger.info(f"Looking for images in: {images_path}")
    
    if not os.path.exists(images_path):
        logger.error(f"Source directory does not exist: {images_path}")
        return []
    
    try:
        for filename in os.listdir(images_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                img_path = os.path.join(images_path, filename)
                try:
                    # Just validate file exists and has reasonable size, don't load the actual image yet
                    if os.path.getsize(img_path) > 0:
                        images.append((filename, img_path))
                        logger.debug(f"Found image: {filename}")
                    else:
                        logger.warning(f"Empty file: {img_path}")
                except Exception as e:
                    logger.error(f"Error checking image {img_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error accessing directory {images_path}: {str(e)}")
    
    logger.info(f"Found {len(images)} valid images")
    return images

def detect_texture_based_frame(gray, height, width, config):
    """Detect frames that have textured patterns, focusing only on the perimeter."""
    # Create ROIs for the border regions based on config perimeter width
    border_width = int(min(width, height) * config['PERIMETER_FOCUS_WIDTH'])
    
    # Extract border regions
    top_border = gray[:border_width, :]
    bottom_border = gray[height-border_width:, :]
    left_border = gray[:, :border_width]
    right_border = gray[:, width-border_width:]
    
    # Calculate local binary pattern (a simple version)
    def calculate_texture_variance(region):
        if region.size == 0:
            return 0
        # Apply Sobel filter to detect texture edges
        sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        # Combine edge responses
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        # Calculate variance as a measure of texture
        return np.var(magnitude)
    
    # Calculate texture variance for each border and center
    center = gray[border_width:height-border_width, border_width:width-border_width]
    
    border_texture = np.mean([
        calculate_texture_variance(top_border),
        calculate_texture_variance(bottom_border),
        calculate_texture_variance(left_border),
        calculate_texture_variance(right_border)
    ])
    
    center_texture = calculate_texture_variance(center)
    
    # If border texture is significantly different from center, it might be a textured frame
    texture_ratio = border_texture / (center_texture + 1e-10)  # Avoid division by zero
    
    return texture_ratio > config['TEXTURE_SENSITIVITY'], texture_ratio

def create_perimeter_mask(height, width, config, min_dimension=None):
    """
    Create a mask that only includes the perimeter region of the image.
    
    IMPORTANT: This mask should be used to FILTER edge detection results,
    NOT to mask the input image before edge detection. Masking the input
    creates artificial edges at the mask boundary that will be detected
    as false borders.
    
    Args:
        height: Image height
        width: Image width
        config: Configuration dictionary
        min_dimension: Pre-computed min(height, width) for efficiency
        
    Returns:
        Binary mask where 255 = perimeter region, 0 = center region
    """
    if min_dimension is None:
        min_dimension = min(height, width)
    
    perimeter_width = int(min_dimension * config['PERIMETER_FOCUS_WIDTH'])
    
    # Create mask with entire image as perimeter initially
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Exclude center region if CENTER_EXCLUSION is enabled
    if config.get('CENTER_EXCLUSION', False):
        # Create inner rectangle (center) and set it to 0 (black)
        # This leaves only the perimeter area as 255 (white)
        inner_x1 = perimeter_width
        inner_y1 = perimeter_width
        inner_x2 = width - perimeter_width
        inner_y2 = height - perimeter_width
        
        # Only exclude center if we have a meaningful perimeter
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            cv2.rectangle(mask, (inner_x1, inner_y1), (inner_x2, inner_y2), 0, -1)
    
    return mask

def check_border_symmetry(gray, height, width, config):
    """Check if the potential border exhibits symmetry, which is common in actual borders."""
    perimeter_width = int(min(height, width) * config['PERIMETER_FOCUS_WIDTH'])
    
    # Get border regions
    top = gray[:perimeter_width, :]
    bottom = gray[height-perimeter_width:, :]
    left = gray[:, :perimeter_width]
    right = gray[:, width-perimeter_width:]
    
    # Calculate histograms for each region
    hist_top = cv2.calcHist([top], [0], None, [64], [0, 256])
    hist_bottom = cv2.calcHist([bottom], [0], None, [64], [0, 256])
    hist_left = cv2.calcHist([left], [0], None, [64], [0, 256])
    hist_right = cv2.calcHist([right], [0], None, [64], [0, 256])
    
    # Normalize histograms
    hist_top = cv2.normalize(hist_top, hist_top).flatten()
    hist_bottom = cv2.normalize(hist_bottom, hist_bottom).flatten()
    hist_left = cv2.normalize(hist_left, hist_left).flatten()
    hist_right = cv2.normalize(hist_right, hist_right).flatten()
    
    # Compare histograms using correlation (higher is more similar)
    tb_correlation = cv2.compareHist(hist_top, hist_bottom, cv2.HISTCMP_CORREL)
    lr_correlation = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
    
    # Average the correlations - higher values indicate more symmetry
    symmetry_score = (tb_correlation + lr_correlation) / 2.0
    
    return symmetry_score > 0.7, symmetry_score

def analyze_gradient_patterns(image, height, width, config):
    """Analyze gradient patterns at image boundaries to detect frames."""
    perimeter_width = int(min(height, width) * config['PERIMETER_FOCUS_WIDTH'])
    
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create perimeter mask
    mask = create_perimeter_mask(height, width, config)
    
    # Calculate average gradient in perimeter vs center
    perimeter_gradient = cv2.mean(magnitude, mask=mask)[0]
    
    # Create inverse mask for center
    center_mask = np.ones((height, width), dtype=np.uint8) * 255
    center_mask = cv2.subtract(center_mask, mask)
    
    center_gradient = cv2.mean(magnitude, mask=center_mask)[0]
    
    # A strong gradient in the perimeter compared to center suggests a frame
    gradient_ratio = perimeter_gradient / (center_gradient + 1e-10)
    
    return gradient_ratio > 1.5, gradient_ratio

def calculate_adaptive_thresholds(image, config, global_mean=None, global_std=None):
    """Calculate adaptive thresholds based on image characteristics."""
    # If adaptive thresholds are disabled, return the default config
    if not config['ADAPTIVE_THRESHOLDS']:
        return config
    
    # Create a copy to avoid modifying the original
    adapted_config = config.copy()
    
    # Use pre-computed statistics if available, otherwise compute them
    if global_mean is not None and global_std is not None:
        mean_val = global_mean
        std_val = global_std
    else:
        # Convert to grayscale for analysis (fallback)
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Get image statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
    
    # Adjust Canny thresholds based on image contrast
    if std_val < 30:  # Low contrast image
        adapted_config['CANNY_THRESHOLD1'] = max(10, int(mean_val * 0.3))
        adapted_config['CANNY_THRESHOLD2'] = max(30, int(mean_val * 0.8))
    elif std_val > 60:  # High contrast image
        adapted_config['CANNY_THRESHOLD1'] = max(30, int(mean_val * 0.4))
        adapted_config['CANNY_THRESHOLD2'] = max(90, int(mean_val * 1.0))
    
    # Adjust color difference threshold based on image characteristics
    if std_val < 20:  # Very uniform image - may need more sensitivity
        adapted_config['COLOR_DIFF_THRESHOLD'] = max(15, config['COLOR_DIFF_THRESHOLD'] * 0.7)
    elif std_val > 50:  # Very varied image - needs more difference to detect borders
        adapted_config['COLOR_DIFF_THRESHOLD'] = min(50, config['COLOR_DIFF_THRESHOLD'] * 1.3)
    
    # Adjust texture sensitivity based on image texture
    texture_mean = np.mean(cv2.Laplacian(gray, cv2.CV_64F))
    if texture_mean < 3:  # Low texture image
        adapted_config['TEXTURE_SENSITIVITY'] = max(1.5, config['TEXTURE_SENSITIVITY'] * 0.8)
    elif texture_mean > 10:  # High texture image
        adapted_config['TEXTURE_SENSITIVITY'] = min(3.0, config['TEXTURE_SENSITIVITY'] * 1.2)
    
    # Adjust perimeter width based on image size
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    
    if min_dim < 500:  # Small image
        adapted_config['PERIMETER_FOCUS_WIDTH'] = min(0.25, config['PERIMETER_FOCUS_WIDTH'] * 1.2)
    elif min_dim > 2000:  # Large image
        adapted_config['PERIMETER_FOCUS_WIDTH'] = max(0.1, config['PERIMETER_FOCUS_WIDTH'] * 0.8)
    
    return adapted_config

def adapt_thresholds_to_resolution(image, config):
    """
    Adapt detection thresholds based on image resolution to maintain consistent detection
    across different image sizes.
    
    Args:
        image: Input image
        config: Configuration dictionary
        
    Returns:
        Adapted configuration with resolution-specific thresholds
    """
    height, width = image.shape[:2]
    
    # Create a copy to avoid modifying the original
    adapted_config = config.copy()
    
    # Calculate resolution factors
    image_area = width * height
    reference_area = 1920 * 1080  # Reference resolution (2MP)
    
    # Calculate scaling factor - how much bigger/smaller this image is compared to reference
    scale_factor = (image_area / reference_area) ** 0.5  # Square root to convert area ratio to linear ratio
    
    # Adapt color difference threshold based on resolution
    if scale_factor < 0.5:  # Very small images
        # For smaller images, reduce threshold as color differences appear smaller
        adapted_config['COLOR_DIFF_THRESHOLD'] = max(15, config['COLOR_DIFF_THRESHOLD'] * 0.7)
    elif scale_factor > 2.0:  # Very large images
        # For larger images, increase threshold as noise can create false differences
        adapted_config['COLOR_DIFF_THRESHOLD'] = min(50, config['COLOR_DIFF_THRESHOLD'] * 1.2)
    
    # Adjust minimum required uniformity based on resolution
    # Smaller images have less pixels for sampling, so we should be more lenient
    if scale_factor < 0.7:
        adapted_config['BORDER_UNIFORMITY_THRESHOLD'] = 0.6  # More lenient
    elif scale_factor > 2.0:
        adapted_config['BORDER_UNIFORMITY_THRESHOLD'] = 0.75  # More strict
    else:
        adapted_config['BORDER_UNIFORMITY_THRESHOLD'] = 0.7  # Default
    
    # Set resolution-specific border detection score threshold
    if scale_factor < 0.5:
        # For very small images, be more lenient with threshold
        adapted_config['BORDER_SCORE_THRESHOLD'] = 1.1
    elif scale_factor > 3.0:
        # For very high res images, be more strict
        adapted_config['BORDER_SCORE_THRESHOLD'] = 1.3
    else:
        adapted_config['BORDER_SCORE_THRESHOLD'] = 1.2  # Default
    
    # Adjust border width sampling based on resolution
    if scale_factor < 0.5:
        adapted_config['BORDER_WIDTH_SAMPLES'] = [0.01, 0.02, 0.04]  # Fewer but wider for small images
    elif scale_factor > 2.0:
        # For high-res images, use more granular width sampling
        adapted_config['BORDER_WIDTH_SAMPLES'] = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
    else:
        adapted_config['BORDER_WIDTH_SAMPLES'] = [0.01, 0.02, 0.03, 0.05]  # Default
    
    # Log resolution adaptations if debug is enabled
    if logger.level <= logging.DEBUG:
        logger.debug(f"Resolution adaptation: {width}x{height}, scale_factor={scale_factor:.2f}")
        logger.debug(f"Adapted COLOR_DIFF_THRESHOLD: {adapted_config['COLOR_DIFF_THRESHOLD']}")
        logger.debug(f"Adapted BORDER_SCORE_THRESHOLD: {adapted_config['BORDER_SCORE_THRESHOLD']}")
    
    return adapted_config

def detect_uniform_background_regions(gray, height, width, config):
    """
    Detect if the image has large uniform background regions that might
    create false border detections.
    
    Args:
        gray: Grayscale image
        height: Image height  
        width: Image width
        config: Configuration dictionary
        
    Returns:
        tuple: (has_uniform_background, uniformity_score)
    """
    # Divide image into a grid to check for uniform regions
    grid_size = 8
    cell_height = height // grid_size
    cell_width = width // grid_size
    
    uniform_cells = 0
    total_cells = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * cell_height
            y2 = min((i + 1) * cell_height, height)
            x1 = j * cell_width
            x2 = min((j + 1) * cell_width, width)
            
            cell = gray[y1:y2, x1:x2]
            if cell.size > 0:
                cell_std = np.std(cell)
                if cell_std < 8:  # Very uniform cell
                    uniform_cells += 1
                total_cells += 1
    
    uniformity_ratio = uniform_cells / max(1, total_cells)
    
    # Check if edges are predominantly uniform
    border_width = int(min(width, height) * 0.05)
    
    # Extract edge regions
    top_edge = gray[:border_width, :]
    bottom_edge = gray[height-border_width:, :]
    left_edge = gray[:, :border_width]
    right_edge = gray[:, width-border_width:]
    
    edge_uniformity_scores = []
    for edge in [top_edge, bottom_edge, left_edge, right_edge]:
        if edge.size > 0:
            edge_std = np.std(edge)
            edge_uniformity = 1.0 - min(1.0, edge_std / 20.0)
            edge_uniformity_scores.append(edge_uniformity)
    
    avg_edge_uniformity = np.mean(edge_uniformity_scores) if edge_uniformity_scores else 0
    
    # Combine metrics
    overall_uniformity = 0.6 * uniformity_ratio + 0.4 * avg_edge_uniformity
    
    # High uniformity suggests potential false positive risk
    has_uniform_background = overall_uniformity > 0.7
    
    return has_uniform_background, overall_uniformity

def detect_compression_artifacts(gray, height, width):
    """
    Detect compression artifacts that might be mistaken for borders.
    JPEG compression can create block boundaries that look like borders.
    
    Args:
        gray: Grayscale image (pre-computed)
        height: Image height
        width: Image width
        
    Returns:
        tuple: (has_artifacts, artifact_score)
    """
    # Check for 8x8 block patterns typical of JPEG compression
    block_size = 8
    
    # Look for horizontal and vertical lines at block boundaries
    h_lines = []
    v_lines = []
    
    # Check horizontal block boundaries
    for y in range(block_size, height - block_size, block_size):
        row_above = gray[y-1, :]
        row_below = gray[y, :]
        diff = np.mean(np.abs(row_above.astype(float) - row_below.astype(float)))
        h_lines.append(diff)
    
    # Check vertical block boundaries  
    for x in range(block_size, width - block_size, block_size):
        col_left = gray[:, x-1]
        col_right = gray[:, x]
        diff = np.mean(np.abs(col_left.astype(float) - col_right.astype(float)))
        v_lines.append(diff)
    
    # If there are regular patterns of differences at 8-pixel intervals,
    # it suggests compression artifacts
    avg_h_diff = np.mean(h_lines) if h_lines else 0
    avg_v_diff = np.mean(v_lines) if v_lines else 0
    
    # Check for periodicity in the differences
    h_std = np.std(h_lines) if h_lines else 0
    v_std = np.std(v_lines) if v_lines else 0
    
    # Low standard deviation with moderate mean differences suggests regular pattern
    h_regularity = avg_h_diff / (h_std + 1e-5) if h_std > 0 else 0
    v_regularity = avg_v_diff / (v_std + 1e-5) if v_std > 0 else 0
    
    # Calculate artifact score
    artifact_score = min(1.0, (h_regularity + v_regularity) / 10.0)
    
    # Threshold for detecting compression artifacts
    has_artifacts = artifact_score > 0.3 and (avg_h_diff > 2 or avg_v_diff > 2)
    
    return has_artifacts, artifact_score

def detect_actual_border_regions(image, detection_result, config):
    """
    Detect the actual border/frame regions based on the detection results.
    
    Args:
        image: Input image
        detection_result: Dictionary with detection results
        config: Configuration dictionary
        
    Returns:
        Dictionary containing the actual border coordinates and parameters
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    
    border_info = {
        'simple_border_regions': [],
        'frame_regions': [],
        'textured_regions': [],
        'actual_border_width': 0
    }
    
    # If simple border detected, find the actual border width
    if detection_result.get('simple_border_detected', False):
        # Test different border widths to find the best match
        min_dimension = min(width, height)  # Calculate once
        border_widths = [
            int(min_dimension * ratio) for ratio in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]
        ]
        
        best_border_width = 0
        best_difference = 0
        
        for border_width in border_widths:
            if border_width < 2:
                continue
                
            # Create border mask
            border_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(border_mask, (0, 0), (width, height), 255, border_width)
            
            # Create center mask
            center_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(center_mask, (border_width, border_width), 
                         (width - border_width, height - border_width), 255, -1)
            
            # Calculate color difference
            border_mean = cv2.mean(gray, mask=border_mask)[0]
            center_mean = cv2.mean(gray, mask=center_mask)[0]
            
            color_diff = abs(border_mean - center_mean)
            
            # Check border uniformity
            _, border_std = cv2.meanStdDev(gray, mask=border_mask)
            border_uniformity = 1.0 - min(1.0, border_std[0][0] / 30.0)
            
            # Score this border width
            score = color_diff * border_uniformity
            
            if score > best_difference and color_diff > config.get('COLOR_DIFF_THRESHOLD', 30) * 0.5:
                best_difference = score
                best_border_width = border_width
        
        if best_border_width > 0:
            border_info['actual_border_width'] = best_border_width
            border_info['simple_border_regions'] = [
                {'type': 'top', 'coords': (0, 0, width, best_border_width)},
                {'type': 'bottom', 'coords': (0, height - best_border_width, width, height)},
                {'type': 'left', 'coords': (0, 0, best_border_width, height)},
                {'type': 'right', 'coords': (width - best_border_width, 0, width, height)}
            ]
    
    # If frame detected, find contours and analyze them
    if detection_result.get('frame_detected', False):
        # Apply minimal preprocessing for contour detection
        if config.get('SKIP_PREPROCESSING', True):
            # No preprocessing for cleanest edges
            processed = gray.copy()
        elif config.get('USE_BILATERAL_FILTER', False):
            # Use bilateral filter to preserve edges while reducing noise
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
        elif config.get('APPLY_GAUSSIAN_BLUR', False):
            # Only use Gaussian blur if explicitly enabled
            blur_kernel_size = config.get('BLUR_KERNEL_SIZE', 3)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1
            blur_kernel_size = max(3, blur_kernel_size)
            processed = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
        else:
            processed = gray.copy()
        edges = cv2.Canny(processed, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > height * width * 0.1:  # Significant contour
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's near the image perimeter (likely a frame)
                margin = min(width, height) * 0.05
                if (x < margin or y < margin or 
                    x + w > width - margin or y + h > height - margin):
                    
                    border_info['frame_regions'].append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area
                    })
    
    # If textured frame detected, identify texture regions
    if detection_result.get('textured_frame_detected', False):
        # Use perimeter focus to identify textured regions
        perimeter_width = int(min(width, height) * config.get('PERIMETER_FOCUS_WIDTH', 0.1))
        
        border_info['textured_regions'] = [
            {'type': 'top', 'coords': (0, 0, width, perimeter_width)},
            {'type': 'bottom', 'coords': (0, height - perimeter_width, width, height)},
            {'type': 'left', 'coords': (0, 0, perimeter_width, height)},
            {'type': 'right', 'coords': (width - perimeter_width, 0, width, height)}
        ]
    
    return border_info

def detect_simple_border_patterns(image, gray, height, width, config, global_mean, global_std, min_dimension):
    """
    Revised simple border detection with focus on reducing false positives.
    Uses a cleaner, more robust approach to identify deliberate borders.
    """
    # Use consistent border size constraints from config
    min_border_px = max(config.get('MIN_BORDER_SIZE_PX', 2), 1)
    max_border_ratio = config.get('MAX_BORDER_SIZE_RATIO', 0.15)
    min_border_ratio = config.get('MIN_BORDER_SIZE_RATIO', 0.005)
    
    # Calculate border width samples based on constraints
    border_widths = []
    for ratio in [0.008, 0.015, 0.025, 0.040, 0.060]:
        if min_border_ratio <= ratio <= max_border_ratio:
            width_px = max(min_border_px, int(min_dimension * ratio))
            border_widths.append(width_px)
    
    # Remove duplicates and sort
    border_widths = sorted(list(set(border_widths)))
    
    if not border_widths:
        return False, 0, 0
    
    # Use pre-computed global statistics (no redundant calculation)
    # global_mean and global_std are passed as parameters
    
    # Use consistent low contrast threshold
    is_low_contrast = global_std < config.get('LOW_CONTRAST_THRESHOLD', 30)
    
    best_score = 0
    best_difference = 0
    best_width = 0
    
    for border_width in border_widths:
        # === CORE BORDER ANALYSIS ===
        
        # Create precise border and center masks
        border_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(border_mask, (0, 0), (width, height), 255, border_width)
        
        center_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(center_mask, (border_width, border_width), 
                     (width - border_width, height - border_width), 255, -1)
        
        # Basic statistics
        border_mean = cv2.mean(gray, mask=border_mask)[0]
        center_mean = cv2.mean(gray, mask=center_mask)[0]
        _, border_std = cv2.meanStdDev(gray, mask=border_mask)
        
        # === CORE METRICS ===
        
        # 1. Color difference (fundamental requirement)
        color_difference = abs(border_mean - center_mean)
        if color_difference < config['COLOR_DIFF_THRESHOLD'] * 0.5:  # Minimum threshold
            continue  # Skip if difference is too small
            
        # 2. Border uniformity (critical for real borders)
        border_uniformity = calculate_border_uniformity(gray, border_mask, global_std)
        if border_uniformity < 0.75:  # Strict uniformity requirement
            continue  # Skip non-uniform borders
            
        # 3. Side consistency (all four sides should be similar)
        side_consistency = calculate_side_consistency(gray, border_width, height, width)
        if side_consistency < 0.70:  # All sides must be reasonably consistent
            continue  # Skip inconsistent borders
            
        # 4. Transition quality (clean edge between border and content)
        transition_score = calculate_transition_quality(gray, border_width, height, width)
        
        # === ADVANCED VALIDATION ===
        
        # 5. Corner consistency (corners should match border properties)
        corner_score = validate_corner_consistency(gray, border_width, height, width)
        
        # 6. Color channel consistency (for color images)
        color_consistency = 1.0
        if len(image.shape) > 2:
            color_consistency = validate_color_consistency(image, border_mask, center_mask)
        
        # === SCORING ===
        
        # Calculate normalized scores using consistent thresholds
        color_score = min(1.0, color_difference / config.get('COLOR_DIFF_THRESHOLD', 30))
        uniformity_score = max(0.0, (border_uniformity - 0.5) * 2.0)  # Scale 0.5-1.0 to 0.0-1.0
        
        # Composite score with balanced weights
        composite_score = (
            0.40 * color_score +           # Color difference (increased weight)
            0.30 * uniformity_score +      # Border uniformity (increased weight)
            0.15 * side_consistency +      # Side consistency
            0.10 * transition_score +      # Transition quality
            0.05 * corner_score           # Corner consistency
        )
        
        # Apply size-based penalty using consistent ratio threshold
        size_ratio = border_width / min_dimension
        max_size_ratio = config.get('MAX_BORDER_SIZE_RATIO', 0.15)
        if size_ratio > max_size_ratio * 0.5:  # Penalty starts at 50% of max allowed
            size_penalty = max(0.7, 1.0 - (size_ratio - max_size_ratio * 0.5) * 3.0)
            composite_score *= size_penalty
        
        # Apply low contrast penalty
        if is_low_contrast:
            composite_score *= 0.9  # Moderate penalty for low contrast images
        
        # Track best result
        if composite_score > best_score:
            best_score = composite_score
            best_difference = color_difference
            best_width = border_width
    
    # === FINAL VALIDATION ===
    
    # Use consistent confidence threshold
    detection_threshold = config.get('CONFIDENCE_THRESHOLD_LOW', 0.35)
    
    # Adjust threshold for image characteristics
    if is_low_contrast:
        detection_threshold *= 1.2  # Slightly higher threshold for low contrast
    
    # Final decision with vignette check
    simple_border_detected = best_score > detection_threshold
    
    if simple_border_detected:
        # Check for vignetting effects that might indicate false positive
        has_vignette, vignette_strength = detect_vignetting(image, gray, height, width, min_dimension)
        if has_vignette and vignette_strength > 0.6 and best_score < detection_threshold * 1.5:
            # Strong vignette with borderline score suggests false positive
            simple_border_detected = False
            best_score *= 0.8
    
    if config.get('CONFIDENCE_DEBUG_LOGGING', False):
        logger.debug(f"Simple border detection: score={best_score:.3f}, "
                    f"threshold={detection_threshold:.3f}, "
                    f"detected={simple_border_detected}, "
                    f"width={best_width}, diff={best_difference:.1f}")
    
    return simple_border_detected, best_difference, best_score


def calculate_border_uniformity(gray, border_mask, global_std):
    """Calculate how uniform the border region is."""
    _, border_std = cv2.meanStdDev(gray, mask=border_mask)
    
    # Normalize by global standard deviation
    norm_factor = max(5.0, global_std)  # Prevent division by very small numbers
    normalized_std = border_std[0][0] / norm_factor
    
    # Convert to uniformity score (lower std = higher uniformity)
    uniformity = 1.0 - min(1.0, normalized_std / 2.0)
    
    return uniformity


def calculate_side_consistency(gray, border_width, height, width):
    """Calculate consistency between all four sides of the border."""
    # Create individual side masks
    sides = {
        'top': gray[:border_width, :],
        'bottom': gray[height-border_width:, :],
        'left': gray[:, :border_width], 
        'right': gray[:, width-border_width:]
    }
    
    # Calculate mean for each side
    side_means = [np.mean(side) for side in sides.values()]
    
    # Calculate consistency (lower variance = higher consistency)
    if len(side_means) > 1:
        mean_variance = np.var(side_means)
        consistency = 1.0 - min(1.0, mean_variance / 400.0)  # Scale factor
    else:
        consistency = 1.0
        
    return consistency


def calculate_transition_quality(gray, border_width, height, width):
    """Calculate quality of transition from border to center."""
    # Apply gradient detection at the transition boundary
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create mask for transition area (just inside the border)
    transition_mask = np.zeros_like(gray, dtype=np.uint8)
    if border_width > 2:
        cv2.rectangle(transition_mask, (border_width-1, border_width-1),
                     (width-border_width+1, height-border_width+1), 255, 2)
        
        # Calculate average gradient strength at transition
        avg_gradient = cv2.mean(gradient_mag, mask=transition_mask)[0]
        transition_score = min(1.0, avg_gradient / 30.0)  # Normalize
    else:
        transition_score = 0.5  # Default for very thin borders
        
    return transition_score


def validate_corner_consistency(gray, border_width, height, width):
    """Validate that corners are consistent with border properties."""
    if border_width < 4:  # Too small to analyze corners meaningfully
        return 0.8  # Neutral score
    
    corner_size = min(border_width * 2, 15)  # Reasonable corner analysis size
    
    # Extract corner regions
    corners = [
        gray[:corner_size, :corner_size],                           # Top-left
        gray[:corner_size, width-corner_size:],                     # Top-right
        gray[height-corner_size:, :corner_size],                    # Bottom-left
        gray[height-corner_size:, width-corner_size:]               # Bottom-right
    ]
    
    # Calculate statistics for each corner
    corner_means = [np.mean(corner) for corner in corners]
    
    # Check consistency between corners
    if len(corner_means) > 1:
        corner_variance = np.var(corner_means)
        consistency = 1.0 - min(1.0, corner_variance / 200.0)
    else:
        consistency = 1.0
        
    return consistency


def validate_color_consistency(image, border_mask, center_mask):
    """Validate consistency across color channels."""
    # Split into color channels
    channels = cv2.split(image)
    
    channel_diffs = []
    for channel in channels:
        border_mean = cv2.mean(channel, mask=border_mask)[0]
        center_mean = cv2.mean(channel, mask=center_mask)[0]
        channel_diffs.append(abs(border_mean - center_mean))
    
    # Check if differences are consistent across channels
    if len(channel_diffs) > 1:
        diff_variance = np.var(channel_diffs)
        diff_mean = np.mean(channel_diffs)
        
        # Normalize variance by mean to get coefficient of variation
        if diff_mean > 0:
            cv_score = 1.0 - min(1.0, diff_variance / (diff_mean ** 2))
        else:
            cv_score = 0.0
    else:
        cv_score = 1.0
        
    return cv_score

def contour_detection(contours, height, width, adapted_config, perimeter_mask):
    """
    Optimized contour detection with early filtering and improved performance.
    """
    if not contours:
        return {
            'frame_detected': False,
            'best_contour': None,
            'confidence': 0.0
        }
    
    # === EARLY FILTERING FOR PERFORMANCE ===
    # Pre-filter by area before expensive operations
    image_area = height * width
    min_area = image_area * adapted_config.get('FRAME_AREA_MIN', 0.05)
    max_area = image_area * adapted_config.get('FRAME_AREA_MAX', 0.95)
    
    # Quick area-based filtering
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            valid_contours.append((contour, area))
    
    if not valid_contours:
        return {
            'frame_detected': False,
            'best_contour': None,
            'confidence': 0.0
        }
    
    # Sort by area (largest first) and limit to top 5 for efficiency
    valid_contours.sort(key=lambda x: x[1], reverse=True)
    top_contours = [contour for contour, area in valid_contours[:5]]
    
    candidates = []
    
    for contour in top_contours:
        # Quick complexity check - skip overly complex contours
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) > 20:  # Skip overly complex shapes
            continue
        
        # Check if contour is at the perimeter (critical for frame detection)
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # Check overlap with perimeter mask
        perimeter_overlap = cv2.bitwise_and(contour_mask, perimeter_mask)
        overlap_ratio = np.sum(perimeter_overlap > 0) / max(1, np.sum(contour_mask > 0))
        
        if overlap_ratio < 0.6:  # At least 60% should be at perimeter
            continue
        
        # Calculate essential properties efficiently
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        perimeter = cv2.arcLength(contour, True)
        
        # Rectangularity (lower values indicate more rectangular shapes)
        rectangularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio analysis
        rect = cv2.minAreaRect(contour)
        box_width, box_height = rect[1]
        aspect_ratio = box_width / max(box_height, 1e-6)
        aspect_score = 1.0 - min(1.0, abs(np.log(max(aspect_ratio, 1e-6))) / 2.0)
        
        # Centeredness calculation
        moments = cv2.moments(contour)
        centeredness = 0.5  # Default moderate score
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            center_distance = np.sqrt((cx - width/2)**2 + (cy - height/2)**2)
            max_distance = np.sqrt((width/2)**2 + (height/2)**2)
            centeredness = 1.0 - (center_distance / max_distance)
        
        # Optimized scoring for frame-like properties
        if 4 <= len(approx) <= 12:  # Reasonable polygon complexity
            frame_score = (
                0.25 * (1.0 - rectangularity) +  # Prefer rectangular shapes (low rectangularity)
                0.25 * overlap_ratio +            # Good perimeter overlap
                0.20 * centeredness +             # Centered in image
                0.15 * aspect_score +             # Reasonable aspect ratio
                0.15 * min(1.0, area_ratio * 4)   # Substantial but not overwhelming size
            )
            
            # Bonus for optimal area range
            if 0.15 <= area_ratio <= 0.85:
                frame_score *= 1.1
            
            candidates.append({
                'contour': contour,
                'score': frame_score,
                'approx': approx,
                'area_ratio': area_ratio,
                'rectangularity': rectangularity
            })
    
    # Return best candidate if it meets minimum threshold
    if candidates:
        best = max(candidates, key=lambda x: x['score'])
        min_score_threshold = adapted_config.get('FRAME_DETECTION_THRESHOLD', 0.5)
        frame_detected = best['score'] > min_score_threshold
        
        return {
            'frame_detected': frame_detected,
            'best_contour': best['contour'],
            'confidence': best['score']
        }
    else:
        return {
            'frame_detected': False,
            'best_contour': None,
            'confidence': 0.0
        }

def compute_per_side_uniformity(gray, border_width, height, width):
    """
    Compute uniformity for each side of the border separately.
    Returns the minimum uniformity, side consistency, and individual scores.
    """
    # Create masks for each side
    top_mask = np.zeros((height, width), dtype=np.uint8)
    bottom_mask = np.zeros((height, width), dtype=np.uint8)
    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill each side mask
    top_mask[:border_width, :] = 255
    bottom_mask[height-border_width:, :] = 255
    left_mask[:, :border_width] = 255
    right_mask[:, width-border_width:] = 255
    
    # Calculate standard deviation for each side
    sides = [top_mask, bottom_mask, left_mask, right_mask]
    uniformity_scores = []
    
    for mask in sides:
        _, std = cv2.meanStdDev(gray, mask=mask)
        uniformity = 1.0 - min(1.0, std[0][0] / 30.0)
        uniformity_scores.append(uniformity)
    
    min_uniformity = min(uniformity_scores)
    side_consistency = 1.0 - (np.std(uniformity_scores) / np.mean(uniformity_scores) if np.mean(uniformity_scores) > 0 else 0)
    
    return min_uniformity, side_consistency, uniformity_scores

def compute_border_symmetry(gray, border_width, height, width):
    """
    Compute symmetry between opposite sides of the border.
    """
    # Create masks for opposite sides
    top_mask = np.zeros((height, width), dtype=np.uint8)
    bottom_mask = np.zeros((height, width), dtype=np.uint8)
    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)
    
    top_mask[:border_width, :] = 255
    bottom_mask[height-border_width:, :] = 255
    left_mask[:, :border_width] = 255
    right_mask[:, width-border_width:] = 255
    
    # Calculate mean for each side
    top_mean = cv2.mean(gray, mask=top_mask)[0]
    bottom_mean = cv2.mean(gray, mask=bottom_mask)[0]
    left_mean = cv2.mean(gray, mask=left_mask)[0]
    right_mean = cv2.mean(gray, mask=right_mask)[0]
    
    # Calculate symmetry scores
    tb_symmetry = 1.0 - min(1.0, abs(top_mean - bottom_mean) / 50.0)
    lr_symmetry = 1.0 - min(1.0, abs(left_mean - right_mean) / 50.0)
    
    return (tb_symmetry + lr_symmetry) / 2.0

def compute_transition_sharpness(gray, border_width, height, width):
    """
    Compute the sharpness of transition from border to center.
    """
    # Create border and center masks
    border_mask = np.zeros((height, width), dtype=np.uint8)
    center_mask = np.zeros((height, width), dtype=np.uint8)
    
    cv2.rectangle(border_mask, (0, 0), (width, height), 255, border_width)
    cv2.rectangle(center_mask, (border_width, border_width), 
                 (width - border_width, height - border_width), 255, -1)
    
    # Calculate gradients at the transition
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create transition mask (area between border and center)
    transition_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(transition_mask, (border_width-1, border_width-1), 
                 (width-border_width+1, height-border_width+1), 255, 2)
    
    # Calculate average gradient at transition
    transition_gradient = cv2.mean(gradient_magnitude, mask=transition_mask)[0]
    
    # Normalize to 0-1 range
    sharpness_score = min(1.0, transition_gradient / 50.0)
    
    return sharpness_score

def detect_vignetting(image, gray, height, width, min_dimension):
    """
    Detect vignetting effect that might be mistaken for a border.
    Vignetting causes darker corners and edges compared to center.
    """
    # Use pre-computed grayscale image (no redundant conversion)
    # gray parameter is already provided
    
    # Create masks for center and corners
    center_size = min_dimension // 4  # Use pre-computed min_dimension
    center_x, center_y = width // 2, height // 2
    
    # Center region mask
    center_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(center_mask, 
                 (center_x - center_size, center_y - center_size),
                 (center_x + center_size, center_y + center_size), 255, -1)
    
    # Corner regions
    corner_size = min(height, width) // 8
    corner_masks = []
    
    # Four corners
    corners = [
        (0, 0),  # Top-left
        (width - corner_size, 0),  # Top-right
        (0, height - corner_size),  # Bottom-left
        (width - corner_size, height - corner_size)  # Bottom-right
    ]
    
    for corner_x, corner_y in corners:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (corner_x, corner_y), 
                     (corner_x + corner_size, corner_y + corner_size), 255, -1)
        corner_masks.append(mask)
    
    # Calculate brightness for center and corners
    center_brightness = cv2.mean(gray, mask=center_mask)[0]
    corner_brightnesses = [cv2.mean(gray, mask=mask)[0] for mask in corner_masks]
    
    # Calculate vignetting score
    avg_corner_brightness = np.mean(corner_brightnesses)
    brightness_ratio = center_brightness / (avg_corner_brightness + 1e-5)
    
    # Check consistency across corners (vignetting affects all corners similarly)
    corner_consistency = 1.0 - (np.std(corner_brightnesses) / (np.mean(corner_brightnesses) + 1e-5))
    
    # Vignetting detected if center is significantly brighter and corners are consistent
    vignette_strength = max(0, brightness_ratio - 1.0)  # How much brighter center is
    vignette_score = min(1.0, vignette_strength * corner_consistency)
    
    has_vignette = vignette_score > 0.3 and brightness_ratio > 1.2
    
    return has_vignette, vignette_score

def color_based_frame_detection(image, gray, height, width, config, min_dimension):
    """
    Specialized frame detection using color analysis in multiple color spaces.
    This function focuses on detecting subtle color differences that define frames.
    
    Returns:
        tuple: (frame_detected, color_difference, confidence_score)
    """
    # Convert to multiple color spaces for better analysis
    # Use pre-computed grayscale image (no redundant conversion)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) > 2 else None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) if len(image.shape) > 2 else None
    
    # Define regions of interest with varying sizes
    # Test multiple border widths to handle different border sizes
    widths = [
        int(min_dimension * 0.01),
        int(min_dimension * 0.02),
        int(min_dimension * 0.03),
        int(min_dimension * 0.05),
    ]
    
    best_score = 0.0
    best_diff = 0.0
    
    for border_width in widths:
        if border_width < 1:
            continue
        
        # Create masks for different regions
        border_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(border_mask, (0, 0), (width, height), 255, border_width)
        
        # Create a slightly inset mask to check for frame edges
        inner_border_mask = np.zeros((height, width), dtype=np.uint8)
        inset = max(1, border_width // 2)
        cv2.rectangle(inner_border_mask, (border_width, border_width), 
                     (width - border_width, height - border_width), 255, inset)
        
        # Center mask for comparison
        center_mask = np.zeros((height, width), dtype=np.uint8)
        center_offset = border_width + inset
        cv2.rectangle(center_mask, (center_offset, center_offset), 
                    (width - center_offset, height - center_offset), 255, -1)
                    
        # Calculate scores for each color space
        scores = []
        
        # Analyze grayscale
        border_mean = cv2.mean(gray, mask=border_mask)[0]
        inner_mean = cv2.mean(gray, mask=inner_border_mask)[0]
        center_mean = cv2.mean(gray, mask=center_mask)[0]
        
        # Check if border is uniform (important for detecting deliberate borders)
        _, border_std = cv2.meanStdDev(gray, mask=border_mask)
        border_uniformity = 1.0 - min(1.0, border_std[0][0] / 40.0)  # Lower std means more uniform
        
        # Calculate color differences between regions
        outer_inner_diff = abs(border_mean - inner_mean)
        outer_center_diff = abs(border_mean - center_mean)
        gray_diff = max(outer_inner_diff, outer_center_diff)
        
        # Weight the difference based on border uniformity
        gray_score = (gray_diff / config['COLOR_DIFF_THRESHOLD']) * (1.0 + 0.5 * border_uniformity)
        scores.append(gray_score)
        
        # Process HSV color space if available
        if hsv is not None:
            # Process each channel separately
            for ch in range(3):
                h_border = cv2.mean(hsv[:,:,ch], mask=border_mask)[0]
                h_center = cv2.mean(hsv[:,:,ch], mask=center_mask)[0]
                
                # Special handling for hue channel
                if ch == 0:  # Hue channel
                    # Account for circular nature of hue
                    hue_diff = min(abs(h_border - h_center), 180 - abs(h_border - h_center)) / 90.0
                    scores.append(min(1.5, hue_diff))
                else:  # Saturation and Value
                    diff = abs(h_border - h_center) / 128.0  # Normalize
                    scores.append(min(1.5, diff))
        
        # Process LAB color space if available
        if lab is not None:
            # Process each channel separately (L, a, b)
            for ch in range(3):
                l_border = cv2.mean(lab[:,:,ch], mask=border_mask)[0]
                l_center = cv2.mean(lab[:,:,ch], mask=center_mask)[0]
                
                # Normalize differences based on typical channel ranges
                if ch == 0:  # L channel (0-100)
                    diff = abs(l_border - l_center) / 100.0
                else:  # a,b channels (-127 to 127)
                    diff = abs(l_border - l_center) / 127.0
                
                scores.append(min(1.5, diff))
        
        # Calculate combined score with higher weights to more discriminative channels
        # Weight LAB more heavily as it better represents human color perception
        if len(scores) > 1:
            combined_score = max(scores)  # Use max to catch any strong signal
        else:
            combined_score = scores[0]
            
        # Keep track of best score across different border widths
        if combined_score > best_score:
            best_score = combined_score
            best_diff = gray_diff
    
    # Determine if frame is detected
    frame_detected = best_score > 1.0
    confidence = min(1.0, best_score / 2.0)  # Scale to 0-1
    
    return frame_detected, best_diff, confidence

def analyze_quadrilateral_corners(contour, height, width):
    """
    Specialized function to analyze corners of potential quadrilateral frames.
    Helps detect frames even when contour quality is poor.
    
    Args:
        contour: Contour to analyze
        height: Image height
        width: Image width
        
    Returns:
        tuple: (is_frame, confidence_score)
    """
    # Get minimum area rectangle that fits the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # Use astype(int) for compatibility with recent NumPy versions
    box = box.astype(int)
    
    # Calculate attributes of the rectangle
    rect_center = rect[0]
    rect_size = rect[1]
    rect_angle = rect[2]
    
    # Ensure width is always larger than height
    rect_width = max(rect_size)
    rect_height = min(rect_size)
    
    # Score for centrality of rectangle
    image_center_x = width / 2
    image_center_y = height / 2
    
    # Calculate normalized distance from center of image (0 = perfect center, 1 = at corner)
    center_offset_x = abs(rect_center[0] - image_center_x) / (width / 2)
    center_offset_y = abs(rect_center[1] - image_center_y) / (height / 2)
    
    # A more centered rectangle is more likely a deliberate frame
    center_score = 1.0 - min(1.0, (center_offset_x + center_offset_y) / 2)
    
    # Score for rectangle proportions - frames usually have width and height close to image dimensions
    rect_ratio = rect_height / rect_width if rect_width > 0 else 0
    image_ratio = min(height, width) / max(height, width)
    
    # Similar aspect ratios between rectangle and image suggest a deliberate frame
    ratio_diff = abs(rect_ratio - image_ratio)
    ratio_score = 1.0 - min(1.0, ratio_diff * 5)  # Scale difference for better sensitivity
    
    # Score for parallelism with image edges
    # Frames are usually parallel to image edges, so angle should be close to 0 or 90 degrees
    angle_mod_90 = abs(rect_angle % 90)
    if angle_mod_90 > 45:
        angle_mod_90 = 90 - angle_mod_90  # Convert to range 0-45
        
    # Perfect alignment with image edges scores 1.0
    alignment_score = 1.0 - min(1.0, angle_mod_90 / 45.0) 
    
    # Calculate area ratio between contour and its minimum area rectangle
    # Frames usually fill most of their bounding rectangle
    contour_area = cv2.contourArea(contour)
    rect_area = rect_width * rect_height
    area_ratio = contour_area / rect_area if rect_area > 0 else 0
    fill_score = min(1.0, area_ratio * 1.5)  # Scale to encourage higher fill ratios
    
    # Combine scores with appropriate weights
    frame_score = (
        0.35 * center_score +     # Being centered is important
        0.25 * alignment_score +  # Alignment with edges is important
        0.20 * ratio_score +      # Similar aspect ratio helps
        0.20 * fill_score         # Filling rectangle helps
    )
    
    # Determine if it's likely a frame
    is_frame = frame_score > 0.7
    
    return is_frame, frame_score

def enhance_edges_for_frame_detection(gray, height, width, already_filtered=False):
    """
    Apply advanced edge enhancement techniques to better detect frames,
    especially in low contrast or noisy images.
    
    Args:
        gray: Grayscale image
        height: Image height
        width: Image width
        already_filtered: True if image was already preprocessed with bilateral filter
        
    Returns:
        Enhanced edges image
    """
    # Apply bilateral filter only if not already done
    if not already_filtered:
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    else:
        bilateral = gray
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(bilateral)
    
    # Compute morphological gradient for edge detection
    kernel = np.ones((3, 3), np.uint8)
    morph_gradient = cv2.morphologyEx(clahe_img, cv2.MORPH_GRADIENT, kernel)
    
    # Apply additional filters to enhance edges
    edges1 = cv2.Sobel(clahe_img, cv2.CV_64F, 1, 0, ksize=3)
    edges2 = cv2.Sobel(clahe_img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(np.square(edges1) + np.square(edges2))
    edges = np.uint8(edges / edges.max() * 255)
    
    # Combine edge detection results
    combined_edges = cv2.addWeighted(np.uint8(morph_gradient), 0.7, edges, 0.3, 0)
    
    # Apply adaptive thresholding to better separate edges
    adaptive_thresh = cv2.adaptiveThreshold(combined_edges, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    
    # Clean up with morphological operations
    closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    return closing

class BorderDetector:
    """
    Modular border detection class that breaks down the monolithic detection process.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager.get_config() if config_manager else DEFAULT_CONFIG.copy()
        self.cache = ImagePropertiesCache()
        self.stats = {'processed': 0, 'detected': 0, 'errors': 0}
    
    def detect(self, img_path):
        """
        Main detection orchestrator with comprehensive error handling.
        
        Returns:
            dict: Detection result with all relevant information
        """
        start_time = time.time()
        
        try:
            # Stage 1: Load and validate image
            image = self._load_and_validate_image(img_path)
            if image is None:
                return self._create_error_result(img_path, "Failed to load image")
            
            # Stage 2: Extract image properties (cached)
            properties = self._extract_image_properties(img_path, image)
            
            # Stage 3: Run detection pipeline
            detection_results = self._run_detection_pipeline(image, properties)
            
            # Stage 4: Generate final result
            final_result = self._generate_final_result(img_path, detection_results, properties)
            final_result['processing_time'] = time.time() - start_time
            
            self.stats['processed'] += 1
            if final_result.get('border_detected', False):
                self.stats['detected'] += 1
            
            return final_result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Detection failed for {img_path}: {e}")
            return self._create_error_result(img_path, str(e))
    
    def _load_and_validate_image(self, img_path):
        """Load and validate image with proper error handling."""
        try:
            if not os.path.exists(img_path):
                logger.error(f"Image file not found: {img_path}")
                return None
            
            image = cv2.imread(img_path)
            if image is None:
                logger.error(f"Could not decode image: {img_path}")
                return None
            
            # Basic validation
            if image.size == 0:
                logger.error(f"Empty image: {img_path}")
                return None
            
            return image
            
        except (IOError, OSError) as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return None
    
    def _extract_image_properties(self, img_path, image):
        """Extract and cache image properties to avoid redundant calculations."""
        return self.cache.get_properties(img_path, image)
    
    def _create_error_result(self, img_path, error_message):
        """Create standardized error result."""
        return {
            'image_path': img_path,
            'border_detected': False,
            'confidence': 0.0,
            'reason': f"Error: {error_message}",
            'error': True,
            'processing_time': 0.0
        }
    
    def _run_detection_pipeline(self, image, properties):
        """
        Run the complete detection pipeline using existing optimized functions.
        
        Args:
            image: Loaded image
            properties: Pre-computed image properties
            
        Returns:
            dict: Detection results from all methods
        """
        height, width = properties['height'], properties['width']
        gray = properties['gray']
        min_dimension = properties['min_dimension']
        global_mean = properties['global_mean']
        global_std = properties['global_std']
        
        # Adapt configuration for this image
        adapted_config = calculate_adaptive_thresholds(image, self.config, global_mean, global_std)
        adapted_config = adapt_thresholds_to_resolution(image, adapted_config)
        
        results = {
            'detection_methods': [],
            'confidence_scores': [],
            'adapted_config': adapted_config
        }
        
        # Check for obvious borders first (fastest)
        obvious_detected = False
        if adapted_config.get('ENABLE_OBVIOUS_BORDER_CHECK', True):
            obvious_detected, border_info, obvious_confidence = detect_obvious_borders_fast(
                image, gray, adapted_config, min_dimension)
            if obvious_detected:
                results['detection_methods'].append('obvious')
                results['confidence_scores'].append(('obvious', obvious_confidence))
                results['obvious_border_info'] = border_info
        
        # Simple border detection
        simple_detected, color_diff, simple_confidence = detect_simple_border_patterns(
            image, gray, height, width, adapted_config, global_mean, global_std, min_dimension)
        
        if simple_detected:
            results['detection_methods'].append('simple')
            results['confidence_scores'].append(('simple', simple_confidence))
            results['color_difference'] = color_diff
        
        # Frame contour detection
        frame_detected = False
        if adapted_config.get('ENABLE_FRAME_DETECTION', True):
            # Enhanced edge detection
            enhanced_edges = enhance_edges_for_frame_detection(gray, height, width)
            contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                perimeter_mask = create_perimeter_mask(height, width, adapted_config, min_dimension)
                frame_result = contour_detection(contours, height, width, adapted_config, perimeter_mask)
                if frame_result and frame_result.get('frame_detected', False):
                    frame_detected = True
                    results['detection_methods'].append('frame')
                    results['confidence_scores'].append(('frame', frame_result.get('confidence', 0.0)))
                    results['frame_contour'] = frame_result.get('best_contour')
        
        # Textured frame detection
        textured_detected = False
        if adapted_config.get('ENABLE_TEXTURE_DETECTION', True):
            textured_detected, texture_ratio = detect_texture_based_frame(gray, height, width, adapted_config)
            if textured_detected:
                results['detection_methods'].append('texture')
                results['confidence_scores'].append(('texture', min(1.0, texture_ratio / 3.0)))
                results['texture_ratio'] = texture_ratio
        
        # Quality assessments
        has_uniform_bg, uniformity_score = detect_uniform_background_regions(gray, height, width, adapted_config)
        has_artifacts, artifact_score = detect_compression_artifacts(gray, height, width)
        has_vignette, vignette_score = detect_vignetting(image, gray, height, width, min_dimension)
        
        results.update({
            'obvious_detected': obvious_detected,
            'simple_detected': simple_detected,
            'frame_detected': frame_detected,
            'textured_detected': textured_detected,
            'has_uniform_bg': has_uniform_bg,
            'uniformity_score': uniformity_score,
            'has_artifacts': has_artifacts,
            'artifact_score': artifact_score,
            'has_vignette': has_vignette,
            'vignette_score': vignette_score,
            'noise_std': np.std(gray - cv2.medianBlur(gray, 3))
        })
        
        return results
    
    def _generate_final_result(self, img_path, detection_results, properties):
        """
        Generate final detection result with confidence scoring.
        
        Args:
            img_path: Path to image file
            detection_results: Results from detection pipeline
            properties: Image properties
            
        Returns:
            dict: Final detection result
        """
        # Calculate robust confidence score
        confidence = calculate_robust_confidence(
            detection_results, 
            None,  # We don't need the full image here
            properties['gray'],
            detection_results['has_uniform_bg'],
            detection_results['has_artifacts'],
            detection_results['noise_std'],
            detection_results['has_vignette'],
            detection_results['vignette_score'],
            detection_results['adapted_config']
        )
        
        # Determine if border is detected
        border_detected = (
            detection_results.get('obvious_detected', False) or
            detection_results['simple_detected'] or 
            detection_results['frame_detected'] or 
            detection_results['textured_detected']
        ) and confidence > detection_results['adapted_config'].get('CONFIDENCE_THRESHOLD_LOW', 0.35)
        
        # Generate detection reason
        reason = generate_detection_reason(
            detection_results['simple_detected'],
            detection_results['frame_detected'],
            detection_results['textured_detected'],
            detection_results,
            confidence,
            detection_results['has_uniform_bg'],
            detection_results['has_vignette'],
            detection_results['vignette_score'],
            detection_results['has_artifacts']
        )
        
        # Build result in legacy format for compatibility
        result = {
            'image_path': img_path,
            'img_path': img_path,  # Legacy field
            'border_detected': border_detected,
            'obvious_border_detected': detection_results.get('obvious_detected', False),
            'simple_border_detected': detection_results['simple_detected'],
            'frame_detected': detection_results['frame_detected'],
            'textured_frame_detected': detection_results['textured_detected'],
            'confidence': confidence,
            'reason': reason,
            'color_difference': detection_results.get('color_difference', 0.0),
            'frame_contour': detection_results.get('frame_contour'),
            'adapted_thresholds': True,
            'error': False
        }
        
        return result

    def get_statistics(self):
        """Get processing statistics."""
        return self.stats.copy()


def print_ascii_table(detections, title="Border/Frame Detected Images"):
    """
    Print an ASCII table of images.
    Each detection should be a dict with keys depending on the table content.
    """
    if not detections:
        print(f"No data to display for {title}.")
        return
    headers = list(detections[0].keys())
    col_widths = [
        max(len(header), max((len(str(d[header])) for d in detections), default=0))
        for header in headers
    ]
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(f"\n{title}")
    print(separator)
    print(header_row)
    print(separator)
    for d in detections:
        row = "| " + " | ".join([
            str(d[header]).ljust(col_widths[i])
            for i, header in enumerate(headers)
        ]) + " |"
        print(row)
    print(separator)

def get_border_type(result):
    if result.get('manual_review', False):
        return "Manual Review"
    if result.get('simple_border', False):
        return "Simple Border"
    if result.get('textured_frame', False):
        return "Textured Frame"
    if result.get('frame', False):
        return "Frame Structure"
    return "No Border"

def print_main_style_border_summary(detections_list: list, config: dict) -> None:
    """
    Print summary in the same style as main_optimized.py for consistency.
    
    Args:
        detections_list: List of detection results
        config: Configuration dictionary
    """
    print("=" * 120)
    print("BORDER DETECTION RESULTS")
    print("=" * 120)
    
    print(f"\nTESTS PERFORMED:")
    print(f"  Enabled: borders")
    print(f"  Disabled: editing, specifications, text, watermarks")
    
    # Calculate statistics with new three-tier system
    total = len(detections_list)
    
    # Categorize images based on confidence thresholds
    invalid_images = []      # High confidence borders (≥0.58)
    manual_review_images = []  # Medium confidence borders (0.35-0.58)
    valid_images = []        # Low confidence borders (<0.35)
    
    for d in detections_list:
        confidence = float(d.get('Confidence', 0))
        if confidence >= config.get('CONFIDENCE_THRESHOLD_INVALID', 0.58):
            invalid_images.append(d)
        elif confidence >= config.get('CONFIDENCE_THRESHOLD_LOW', 0.35):
            manual_review_images.append(d)
        else:
            valid_images.append(d)
    
    invalid_count = len(invalid_images)
    manual_review_count = len(manual_review_images) 
    valid_count = len(valid_images)
    
    print(f"\nSUMMARY:")
    print(f"  Total Images Processed: {total}")
    print(f"  Valid Images: {valid_count}")
    print(f"  Invalid Images: {invalid_count}")
    print(f"  Manual Review Needed (moved): {manual_review_count}")
    print(f"  Success Rate: {(valid_count/total*100):.1f}%" if total > 0 else "  Success Rate: 0%")
    
    # Display each category with updated logic
    if valid_images:
        print(f"\nVALID IMAGES ({len(valid_images)} images):")
        print("-" * 120)
        print(f"All validation checks passed - images copied to 'Results\\valid' folder")
    
    if invalid_images:
        print(f"\nINVALID IMAGES ({len(invalid_images)} images):")
        print("-" * 120)
        
        for i, detection in enumerate(invalid_images, 1):
            filename = os.path.basename(detection.get('Image Path', ''))
            confidence = float(detection.get('Confidence', 0))
            border_type = detection.get('Border/Frame Type', 'Unknown')
            reason = f"Border detected - {border_type} (confidence: {confidence:.2f})"
            
            print(f"\n{i:2d}. {filename}")
            print(f"    Failures:")
            print(f"      • Borders: {reason}")
    
    if manual_review_images:
        print(f"\nMANUAL REVIEW NEEDED ({len(manual_review_images)} images):")
        print("-" * 120)
        
        for i, detection in enumerate(manual_review_images, 1):
            filename = os.path.basename(detection.get('Image Path', ''))
            confidence = float(detection.get('Confidence', 0))
            border_type = detection.get('Border/Frame Type', 'Unknown')
            reason = f"Border detected - {border_type} (confidence: {confidence:.2f})"
            
            print(f"\n{i:2d}. {filename}")
            print(f"    High-confidence border detected: {reason}")
    
    print(f"\nOUTPUT STRUCTURE:")
    print(f"  Valid images: Results\\valid")
    print(f"  Invalid images: Results\\invalid")
    print(f"  Manual review needed: Results\\manualreview")
    print(f"  Processing logs: Results\\logs")
    
    print("\n" + "=" * 120)

def main(custom_config=None):
    config = custom_config if custom_config is not None else DEFAULT_CONFIG.copy()
    output_dir = os.path.join(config['BASE_PROJECT_PATH'], config['OUTPUT_DIR'])
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {str(e)}")
        return
        
    image_list = load_images(config)
    if not image_list:
        logger.error("No valid images found to process")
        return
    logger.info(f"Processing {len(image_list)} images...")
    results = []
    if config['PARALLEL_PROCESSING']:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_image, (filename, img_path, config)): (filename, img_path) for (filename, img_path) in image_list}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        for img_info in image_list:
            result = process_image((*img_info, config))
            if result is not None:
                results.append(result)
    logger.info(f"Processed {len(results)} images with detections")

    # --- Post-process: treat all border/frame detections with confidence <= 0.15 as no-border ---
    for r in results:
        if (r.get('simple_border', True) or r.get('frame', True) or r.get('textured_frame', True)) and r.get('confidence', 0.0) <= 0.15:
            r['simple_border'] = False 
            r['frame'] = False
            r['textured_frame'] = False
            r['reason'] = "No border detected (confidence below threshold of 0.15)"

    # --- Transfer files based on detection results after post-processing ---
    for r in results:
        # Determine destination based on detection results and confidence thresholds
        confidence = r.get('confidence', 0.0)
        
        # Three-tier classification system:
        # - Invalid: High border confidence (≥0.58) - images with clear borders
        # - Manual Review: Medium border confidence (0.35-0.58) - needs human verification
        # - Valid: Low border confidence (<0.35) - clean images
        
        if confidence >= config.get('CONFIDENCE_THRESHOLD_INVALID', 0.58):
            # High confidence border detected - mark as invalid
            destination_dir = os.path.join(r['output_dir'], 'invalid')
            log_message = f"Invalid - border detected: {r['filename']} to {destination_dir} (confidence: {confidence:.3f})"
        elif confidence >= config.get('CONFIDENCE_THRESHOLD_LOW', 0.35):
            # Medium confidence - manual review needed
            destination_dir = os.path.join(r['output_dir'], 'manualreview')
            log_message = f"Manual review needed: {r['filename']} to {destination_dir} (confidence: {confidence:.3f})"
        else:
            # Low confidence - valid (clean) image
            destination_dir = os.path.join(r['output_dir'], 'valid')
            log_message = f"Valid - no border detected: {r['filename']} to {destination_dir} (confidence: {confidence:.3f})"

        # Copy the original image to the determined destination
        try:
            os.makedirs(destination_dir, exist_ok=True)
            shutil.copy2(r['img_path'], os.path.join(destination_dir, r['filename']))
            logger.info(log_message)
        except Exception as e:
            logger.error(f"Failed to copy {r['filename']}: {str(e)}")
    if config['EXPORT_JSON']:
        try:
            json_path = os.path.join(output_dir, 'detection_results.json')
            with open(json_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)
            logger.info(f"Results exported to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving results to JSON: {str(e)}")
    logger.info("Processing complete")

    # --- ASCII table output for border/frame-detected images ---
    border_detections = []
    manual_review_detections = []
    
    for r in results:
        if r.get('manual_review', False):
            manual_review_detections.append({
                'Filename': r['filename'],
                'Confidence': f"{r.get('confidence', 0.0):.3f}",
                'Border/Frame Type': get_border_type(r),
                'Detection Reason': r.get('detection_reason', 'Unknown')[:50] + ('...' if len(r.get('detection_reason', 'Unknown')) > 50 else '')
            })
        elif r.get('simple_border') or r.get('frame') or r.get('textured_frame'):
            border_detections.append({
                'Filename': r['filename'],
                'Confidence': f"{r.get('confidence', 0.0):.3f}",
                'Border/Frame Type': get_border_type(r),
                'Color Difference': f"{r.get('color_difference', 0.0):.1f}",
                'Detection Reason': r.get('detection_reason', 'Unknown')[:50] + ('...' if len(r.get('detection_reason', 'Unknown')) > 50 else '')
            })

    # Sort manual review detections by confidence (descending)
    manual_review_detections.sort(key=lambda x: float(x['Confidence']), reverse=True)

    # Simple summary output (no verbose tables)
    total_images = len(results)
    border_count = len(border_detections)
    manual_review_count = len(manual_review_detections)
    no_border_count = total_images - border_count - manual_review_count
    
    # Calculate border type statistics
    border_types = {}
    high_confidence_count = 0
    for detection in border_detections:
        border_type = detection['Border/Frame Type']
        border_types[border_type] = border_types.get(border_type, 0) + 1
        
        # Count high confidence detections
        confidence = float(detection['Confidence'])
        if confidence >= config.get('CONFIDENCE_THRESHOLD_INVALID', 0.58):
            high_confidence_count += 1
    
    # Convert results to unified format for main pipeline style summary
    unified_detections = []
    for r in results:
        unified_detections.append({
            'Image Path': r.get('filename', ''),
            'Has Border/Frame': 'Yes' if r.get('confidence', 0.0) >= config.get('CONFIDENCE_THRESHOLD_LOW', 0.35) else 'No',
            'Confidence': r.get('confidence', 0.0),
            'Border/Frame Type': get_border_type(r),
            'Color Difference': r.get('color_difference', 0.0),
            'Detection Reason': r.get('detection_reason', 'Unknown')
        })
    
    # Use main pipeline style summary
    print_main_style_border_summary(unified_detections, config)
    
def has_border_or_frame(pil_img, show_debug=False, debug_output_dir="debug"):
    """
    Interface function for photo filter integration.
    Uses the new BorderDetector class with minimal wrapper.
    
    Parameters:
        pil_img (PIL.Image): Input image in PIL format
        show_debug (bool): Save debug information
        debug_output_dir (str): Debug output directory
    
    Returns:
        tuple(bool, str): (False, reason) if border/frame detected, (True, "valid") otherwise
    """
    try:
        import tempfile
        
        # Save PIL image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            pil_img.save(temp_file.name, 'JPEG')
            temp_path = temp_file.name
        
        try:
            # Use the new BorderDetector class
            detector = BorderDetector()
            result = detector.detect(temp_path)
            
            if result is None or result.get('error', False):
                return True, "error"
            
            # Check if any border/frame was detected
            simple_border = result.get('simple_border_detected', False)
            frame_detected = result.get('frame_detected', False)
            textured_frame = result.get('textured_frame_detected', False)
            confidence = result.get('confidence', 0.0)
            color_diff = result.get('color_difference', 0.0)
            
            has_border = simple_border or frame_detected or textured_frame
            
            if has_border:
                if simple_border:
                    reason = f"simple border detected (confidence: {confidence:.2f}, color_diff: {color_diff:.1f})"
                elif frame_detected:
                    reason = f"frame detected (confidence: {confidence:.2f})"
                elif textured_frame:
                    reason = f"textured frame detected (confidence: {confidence:.2f})"
                else:
                    reason = f"border/frame detected (confidence: {confidence:.2f})"
                return False, reason
            else:
                return True, "valid"
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        logger.warning(f"Error in border detection: {e}")
        return True, "error"

def process_image(args):
    """
    Process a single image for border/frame detection.
    
    Args:
        args: Tuple containing (filename, img_path, config)
        
    Returns:
        Dictionary with detection results and file information, or None if processing failed
    """
    try:
        filename, img_path, config = args
        
        logger.info(f"Processing image: {filename}")
        
        # Perform border detection using the new BorderDetector class
        config_manager = ConfigManager(base_config=config)
        detector = BorderDetector(config_manager)
        detection_result = detector.detect(img_path)
        
        if detection_result is None or detection_result.get('error', False):
            logger.error(f"Border detection failed for: {filename}")
            return None
        
        # Add filename and output directory information to the result
        output_dir = os.path.join(config['BASE_PROJECT_PATH'], config['OUTPUT_DIR'])
        
        result = {
            'filename': filename,
            'img_path': img_path,
            'output_dir': output_dir,
            'simple_border': detection_result.get('simple_border_detected', False),
            'frame': detection_result.get('frame_detected', False),
            'textured_frame': detection_result.get('textured_frame_detected', False),
            'confidence': detection_result.get('confidence', 0.0),
            'color_difference': detection_result.get('color_difference', 0.0),
            'manual_review': detection_result.get('manual_review', False),
            'adapted_thresholds': detection_result.get('adapted_thresholds', False),
            'has_uniform_background': detection_result.get('has_uniform_background', False),
            'has_compression_artifacts': detection_result.get('has_compression_artifacts', False),
            'noise_penalty_applied': detection_result.get('noise_penalty_applied', False),
            'reason': detection_result.get('detection_reason', 'No detection reason provided'),
            'detection_reason': detection_result.get('detection_reason', 'No detection reason provided')  # Add both for compatibility
        }
        
        # Removed visualization - simplified processing
        
        logger.debug(f"Completed processing: {filename} - Border: {result['simple_border']}, "
                    f"Frame: {result['frame']}, Textured: {result['textured_frame']}, "
                    f"Confidence: {result['confidence']:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def detect_obvious_borders_fast(image, gray, config, min_dimension):
    """
    Fast pre-screening to detect very obvious borders without complex analysis.
    Checks for solid color borders on all four sides with clear color differences.
    
    Args:
        image: Input image (BGR or grayscale)
        gray: Pre-computed grayscale image
        config: Configuration dictionary
        min_dimension: Pre-computed min(height, width)
        
    Returns:
        tuple: (has_obvious_border, border_info, confidence)
            border_info contains: {'top': bool, 'bottom': bool, 'left': bool, 'right': bool, 'width': int}
    """
    height, width = image.shape[:2]
    # Use pre-computed grayscale image (no redundant conversion)
    
    min_width = config.get('OBVIOUS_BORDER_MIN_WIDTH', 3)
    min_color_diff = config.get('OBVIOUS_BORDER_COLOR_DIFF', 25)
    max_std_threshold = config.get('OBVIOUS_BORDER_STD_THRESHOLD', 15)
    
    # Test different border widths quickly (only small ones for obvious detection)
    max_test_width = min(min_dimension // 10, 20)  # Test up to 10% of image or 20px max
    border_widths = range(min_width, max_test_width + 1)
    
    best_border_info = {'top': False, 'bottom': False, 'left': False, 'right': False, 'width': 0}
    best_confidence = 0.0
    best_total_score = 0.0
    
    for border_width in border_widths:
        # Extract border regions
        top_border = gray[:border_width, :]
        bottom_border = gray[height-border_width:, :]
        left_border = gray[:, :border_width]
        right_border = gray[:, width-border_width:]
        
        # Extract center region (avoiding border areas)
        center = gray[border_width:height-border_width, border_width:width-border_width]
        
        if center.size == 0:  # Skip if border too large
            continue
            
        # Calculate means
        center_mean = np.mean(center)
        top_mean = np.mean(top_border)
        bottom_mean = np.mean(bottom_border) 
        left_mean = np.mean(left_border)
        right_mean = np.mean(right_border)
        
        # Calculate color differences
        top_diff = abs(top_mean - center_mean)
        bottom_diff = abs(bottom_mean - center_mean)
        left_diff = abs(left_mean - center_mean)
        right_diff = abs(right_mean - center_mean)
        
        # Check uniformity of each border (low std = uniform)
        top_std = np.std(top_border)
        bottom_std = np.std(bottom_border)
        left_std = np.std(left_border)
        right_std = np.std(right_border)
        
        # Determine which sides have obvious borders
        current_borders = {
            'top': top_diff > min_color_diff and top_std < max_std_threshold,
            'bottom': bottom_diff > min_color_diff and bottom_std < max_std_threshold,
            'left': left_diff > min_color_diff and left_std < max_std_threshold,
            'right': right_diff > min_color_diff and right_std < max_std_threshold,
            'width': border_width
        }
        
        # Count detected border sides
        border_count = sum([current_borders['top'], current_borders['bottom'], 
                          current_borders['left'], current_borders['right']])
        
        if border_count == 0:
            continue
            
        # Calculate confidence based on:
        # 1. Number of sides with borders
        # 2. Strength of color differences  
        # 3. Uniformity of borders
        # 4. Symmetry (opposite sides similar)
        
        avg_color_diff = np.mean([top_diff, bottom_diff, left_diff, right_diff])
        avg_uniformity = 1.0 - np.mean([top_std, bottom_std, left_std, right_std]) / 30.0
        avg_uniformity = max(0, min(1, avg_uniformity))
        
        # Check symmetry: top-bottom and left-right should be similar if both detected
        symmetry_score = 1.0
        if current_borders['top'] and current_borders['bottom']:
            tb_symmetry = 1.0 - min(1.0, abs(top_mean - bottom_mean) / (avg_color_diff + 1e-5))
            symmetry_score *= tb_symmetry
        if current_borders['left'] and current_borders['right']:
            lr_symmetry = 1.0 - min(1.0, abs(left_mean - right_mean) / (avg_color_diff + 1e-5))
            symmetry_score *= lr_symmetry
            
        # Bonus for complete borders (all 4 sides)
        completeness_bonus = 1.0
        if border_count == 4:
            completeness_bonus = 1.5
        elif border_count == 3:
            completeness_bonus = 1.2
        elif border_count == 2:
            # Check if it's opposite sides (stronger than adjacent sides)
            if (current_borders['top'] and current_borders['bottom']) or \
               (current_borders['left'] and current_borders['right']):
                completeness_bonus = 1.1
        
        # Calculate total score
        total_score = (avg_color_diff / min_color_diff) * avg_uniformity * symmetry_score * completeness_bonus
        
        # Update best if this is better
        if total_score > best_total_score:
            best_total_score = total_score
            best_border_info = current_borders.copy()
            # Confidence between 0-1 based on score strength
            best_confidence = min(1.0, total_score / 3.0)
    
    # Determine if we found obvious borders
    border_count = sum([best_border_info['top'], best_border_info['bottom'], 
                       best_border_info['left'], best_border_info['right']])
    has_obvious_border = border_count >= 2 and best_confidence > 0.6  # At least 2 sides with good confidence
    
    return has_obvious_border, best_border_info, best_confidence

def calculate_robust_confidence(detection_results, image, gray, has_uniform_bg, 
                               has_artifacts, noise_std, has_vignette, vignette_score, config):
    """
    Robust confidence calculation that eliminates redundant checks and overlapping logic.
    
    Uses a clear, hierarchical approach:
    1. Evidence aggregation from detection methods
    2. Quality-based adjustments 
    3. Final calibration
    
    Args:
        detection_results: Dictionary with detection results and scores
        image: Original image
        gray: Grayscale image
        has_uniform_bg: Boolean for uniform background
        has_artifacts: Boolean for compression artifacts
        noise_std: Noise level in image
        has_vignette: Boolean for vignette effect
        vignette_score: Vignette strength score
        config: Configuration dictionary
        
    Returns:
        float: Robust confidence score between 0.0 and 1.0
    """
    
    if not detection_results['confidence_scores']:
        return 0.0
    
    # === STEP 1: Evidence Aggregation ===
    # Use weighted maximum instead of average to avoid dilution
    method_weights = {
        'obvious': 1.0,    # Highest weight - most reliable
        'simple': 0.8,     # High weight - good reliability  
        'frame': 0.7,      # Medium weight - contour-based
        'texture': 0.6     # Lower weight - more ambiguous
    }
    
    weighted_scores = []
    for method, score in detection_results['confidence_scores']:
        weight = method_weights.get(method, 0.5)
        weighted_scores.append(score * weight)
    
    # Use the maximum weighted score as base evidence
    base_evidence = max(weighted_scores) if weighted_scores else 0.0
    
    # === STEP 2: Quality Adjustments (Non-overlapping) ===
    quality_factor = 1.0
    
    # Noise adjustment (single, consistent threshold)
    if noise_std > config['NOISE_THRESHOLD']:
        noise_penalty = max(0.7, 1.0 - (noise_std - config['NOISE_THRESHOLD']) / 50.0)
        quality_factor *= noise_penalty
    
    # Uniform background adjustment (context-dependent)
    if has_uniform_bg:
        if base_evidence > config['CONFIDENCE_THRESHOLD_INVALID']:
            # Strong evidence + uniform background = likely real border
            quality_factor *= 1.1
        else:
            # Weak evidence + uniform background = likely false positive
            quality_factor *= 0.8
    
    # Compression artifacts adjustment
    if has_artifacts:
        quality_factor *= 0.85  # Moderate, consistent penalty
    
    # Vignette adjustment (only if significant)
    if has_vignette and vignette_score > 0.5:
        if base_evidence < config['CONFIDENCE_THRESHOLD_INVALID']:
            # Vignette without strong border evidence suggests natural effect
            quality_factor *= 0.9
    
    # === STEP 3: Final Calibration ===
    adjusted_confidence = base_evidence * quality_factor
    
    # Apply consistent calibration curve
    if adjusted_confidence < 0.2:
        final_confidence = adjusted_confidence * 0.8  # Compress very low scores
    elif adjusted_confidence > 0.8:
        final_confidence = 0.8 + (adjusted_confidence - 0.8) * 0.5  # Compress very high scores
    else:
        final_confidence = adjusted_confidence
    
    return max(0.0, min(1.0, final_confidence))

def generate_detection_reason(simple_detected, frame_detected, textured_detected,
                             detection_results, confidence, has_uniform_bg, has_vignette, 
                             vignette_score, has_artifacts):
    """
    Generate clear, non-redundant detection reason.
    
    Args:
        simple_detected: Boolean for simple border detection
        frame_detected: Boolean for frame detection  
        textured_detected: Boolean for textured frame detection
        detection_results: Dictionary with detection results
        confidence: Overall confidence score
        has_uniform_bg: Boolean for uniform background
        has_vignette: Boolean for vignette effect
        vignette_score: Vignette strength score
        has_artifacts: Boolean for compression artifacts
        
    Returns:
        str: Clear description of detection reasoning
    """
    
    if not any([simple_detected, frame_detected, textured_detected]):
        if has_vignette and vignette_score > 0.5:
            return f"Vignette effect detected (strength: {vignette_score:.2f}) but no deliberate border"
        elif has_artifacts:
            return "Compression artifacts detected but no deliberate border"
        elif has_uniform_bg:
            return "Uniform background but insufficient border evidence"
        else:
            return "No border or frame detected"
    
    # Determine primary detection method
    primary_method = None
    primary_confidence = 0.0
    
    for method, score in detection_results['confidence_scores']:
        if score > primary_confidence:
            primary_confidence = score
            primary_method = method
    
    # Generate reason based on primary method
    if primary_method == 'obvious':
        return f"Obvious border detected with high confidence ({confidence:.2f})"
    elif primary_method == 'simple':
        if has_uniform_bg:
            return f"Simple uniform border detected (confidence: {confidence:.2f})"
        else:
            return f"Simple border detected via color/texture analysis (confidence: {confidence:.2f})"
    elif primary_method == 'frame':
        return f"Frame structure detected via contour analysis (confidence: {confidence:.2f})"
    elif primary_method == 'texture':
        return f"Textured border detected via texture analysis (confidence: {confidence:.2f})"
    else:
        return f"Border detected via multiple methods (confidence: {confidence:.2f})"

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Border and Frame Detection for PhotoValidator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_CONFIG['SOURCE_DIR'],
                        help='Input directory containing images to analyze (default: photos4testing)')
    
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_CONFIG['OUTPUT_DIR'],
                        help='Output directory for results (default: Results)')
    
    parser.add_argument('--parallel', action='store_true', default=DEFAULT_CONFIG['PARALLEL_PROCESSING'],
                        help='Enable parallel processing (default: enabled)')
    
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                        help='Disable parallel processing')
    
    parser.add_argument('--save-intermediate', action='store_true', default=DEFAULT_CONFIG['SAVE_INTERMEDIATE'],
                        help='Save intermediate processing images (default: disabled)')
    
    parser.add_argument('--export-json', action='store_true', default=DEFAULT_CONFIG['EXPORT_JSON'],
                        help='Export results to JSON file (default: disabled)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config['SOURCE_DIR'] = args.input
    config['OUTPUT_DIR'] = args.output
    config['PARALLEL_PROCESSING'] = args.parallel
    config['SAVE_INTERMEDIATE'] = args.save_intermediate
    config['EXPORT_JSON'] = args.export_json
    
    # Run main detection
    main(config)