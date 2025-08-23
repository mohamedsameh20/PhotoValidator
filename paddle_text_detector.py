"""
PaddleOCR-Based Text Detection System - PRODUCTION READY & COMPLETELY INDEPENDENT

STANDALONE SYSTEM - Does NOT use or depend on text_detector.py

State-of-the-art text detection using PaddlePaddle's DB (Differentiable Binarization) model
PERFORMANCE TARGET: <100ms per image with superior accuracy

COMPLETE INDEPENDENCE:
- No imports from text_detector.py or any legacy systems
- Uses PaddleOCR DB (Differentiable Binarization) model directly
- Completely separate folder structure: PADDLE_OCR_RESULTS/
- Independent configuration and processing pipeline
- Self-contained with its own optimization and organization

Features:
- DB (Differentiable Binarization) for robust text detection
- CRNN for accurate text recognition
- Handles complex layouts, orientations, and challenging scenarios
- Production-optimized with GPU acceleration support
- Confidence-based organization and validation
- Comprehensive progress tracking and statistics
- User-friendly batch processing interface
- THREAD-SAFE: Can be called from multiple threads safely
"""

import cv2
import numpy as np
import os
import re
import json
import time
import logging
import threading
import warnings
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass, field
import shutil
from PIL import Image
import traceback
from functools import wraps

# More targeted warning suppression (keep important warnings visible)
warnings.filterwarnings("ignore", message=".*ccache.*")
warnings.filterwarnings("ignore", message=".*recompiling.*")
warnings.filterwarnings("ignore", message=".*Could not find files for the given pattern.*")
warnings.filterwarnings("ignore", message=".*deprecated.*", category=DeprecationWarning)
# Keep other UserWarnings visible for debugging

# Set environment variables to reduce PaddlePaddle logging noise
os.environ['GLOG_minloglevel'] = '2'
os.environ['FLAGS_print_model_stats'] = '0'
os.environ['FLAGS_enable_parallel_graph'] = '0'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'

# Additional imports for output suppression
import contextlib
import sys

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# PaddleOCR imports with warnings suppressed
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
    # print("PaddleOCR successfully imported")  # Suppressed for clean output
except ImportError as e:
    PADDLE_AVAILABLE = False
    print(f"PaddleOCR not available: {e}")
    print("Install with: pip install paddlepaddle paddleocr")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions for better type safety
class BatchResult(TypedDict):
    """Typed structure for batch processing results."""
    batch_processing_success: bool
    total_images: int
    successfully_processed: int
    processing_time_seconds: float
    average_time_per_image_ms: float
    average_time_per_image_seconds: float
    statistics: Dict
    results: List[Dict]
    organization_folders: Dict[str, str]
    performance_summary: Dict

class ProcessingResult(TypedDict):
    """Typed structure for single image processing results."""
    image_path: str
    processing_success: bool
    processing_time_ms: float
    ocr_time_ms: float
    image_dimensions: Tuple[int, int]
    text_detections: int
    paddle_results: List[Dict]
    confidence_metrics: Dict
    organization_result: Dict
    method: str

@dataclass
class PaddleDetectorConfig:
    """Configuration for PaddleTextDetector with cross-platform support."""
    # Core detection parameters
    confidence_threshold: float = 0.5
    recognition_threshold: float = 0.6
    min_text_length: int = 1
    max_text_length: int = 500
    angle_threshold: float = 15
    min_bbox_area: int = 50
    
    # Performance parameters
    use_gpu: bool = False
    lang: str = 'en'
    use_textline_orientation: bool = True
    det_model_dir: Optional[str] = None
    rec_model_dir: Optional[str] = None
    
    # Processing options
    use_dilation: bool = True
    save_crop_res: bool = False
    crop_res_save_dir: str = './crop_results'
    
    # Cross-platform paths (auto-detected if None)
    base_directory: Optional[Path] = None
    input_folder: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize platform-specific paths if not provided."""
        if self.base_directory is None:
            # Use current working directory or detected project root
            self.base_directory = self._detect_project_root()
        
        if self.input_folder is None:
            self.input_folder = self.base_directory
    
    def _detect_project_root(self) -> Path:
        """Detect project root directory in a cross-platform way."""
        # Try to find project root by looking for common project files
        current = Path.cwd()
        
        # Look for project indicators
        project_indicators = [
            'requirements.txt', 'setup.py', 'pyproject.toml', 
            '.git', 'README.md', 'main.py'
        ]
        
        # Traverse up the directory tree
        for parent in [current] + list(current.parents):
            if any((parent / indicator).exists() for indicator in project_indicators):
                logger.info(f"Detected project root: {parent}")
                return parent
        
        # Fallback to current working directory
        logger.info(f"Using current working directory as project root: {current}")
        return current
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'PaddleDetectorConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert path strings to Path objects
        if 'base_directory' in data and data['base_directory']:
            data['base_directory'] = Path(data['base_directory'])
        if 'input_folder' in data and data['input_folder']:
            data['input_folder'] = Path(data['input_folder'])
        
        return cls(**data)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        
        # Convert to serializable format
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            else:
                data[key] = value
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")

def monitor_performance(func):
    """Decorator to monitor function performance and memory usage."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        
        # Try to get memory info if psutil is available
        start_memory = None
        try:
            import psutil
            start_memory = psutil.Process().memory_info().rss
        except ImportError:
            pass
        
        result = func(self, *args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if start_memory is not None:
            try:
                end_memory = psutil.Process().memory_info().rss
                memory_delta = (end_memory - start_memory) / 1024 / 1024  # MB
                logger.debug(f"{func.__name__}: {execution_time:.2f}s, Memory: {memory_delta:+.1f}MB")
            except:
                logger.debug(f"{func.__name__}: {execution_time:.2f}s")
        else:
            logger.debug(f"{func.__name__}: {execution_time:.2f}s")
        
        return result
    return wrapper

@dataclass
class PaddleTextResult:
    """Enhanced data class for PaddleOCR text detection results"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    polygon: List[Tuple[int, int]]   # Original polygon points
    detection_confidence: float      # Separate detection confidence
    recognition_confidence: float    # Separate recognition confidence
    angle: float = 0.0              # Text rotation angle
    
@dataclass
class PaddleValidationResult:
    """Enhanced validation result with PaddleOCR specifics"""
    is_valid: bool
    matched_text: List[str]
    confidence_scores: List[float]
    total_confidence: float
    detection_confidence: float
    recognition_confidence: float
    structured_output: Dict

class PaddleTextDetector:
    """
    STANDALONE Production-ready PaddleOCR text detection system
    
    COMPLETELY INDEPENDENT - No dependencies on text_detector.py or legacy systems
    THREAD-SAFE - Uses threading locks to prevent PaddleOCR conflicts
    CROSS-PLATFORM - Works on Windows, Linux, and macOS
    
    Superior accuracy with state-of-the-art DB model:
    - Uses PaddleOCR's Differentiable Binarization (DB) model
    - CRNN for text recognition
    - Independent processing pipeline
    - Separate result organization in PADDLE_OCR_RESULTS/
    - No shared code with old text detection systems
    - GPU memory management and OOM handling
    """
    
    # Class-level threading lock for PaddleOCR instances
    _paddle_lock = threading.Lock()
    
    def __init__(self, 
                 config: Optional[PaddleDetectorConfig] = None,
                 input_folder: Optional[str] = None,
                 use_gpu: Optional[bool] = None,
                 use_textline_orientation: Optional[bool] = None,
                 lang: Optional[str] = None,
                 det_model_dir: Optional[str] = None,
                 rec_model_dir: Optional[str] = None):
        """
        Initialize PaddleTextDetector with flexible configuration.
        
        Args:
            config: Configuration object (recommended approach)
            input_folder: Override input folder (for backward compatibility)
            use_gpu: Override GPU setting (for backward compatibility)
            use_textline_orientation: Override orientation setting
            lang: Override language setting
            det_model_dir: Override detection model directory
            rec_model_dir: Override recognition model directory
        """
        
        if not PADDLE_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Install with:\n"
                "pip install paddlepaddle paddleocr\n"
                "For GPU support: pip install paddlepaddle-gpu"
            )
        
        # Initialize configuration
        if config is None:
            config = PaddleDetectorConfig()
        
        # Apply parameter overrides for backward compatibility
        if input_folder is not None:
            config.input_folder = Path(input_folder)
        if use_gpu is not None:
            config.use_gpu = use_gpu
        if use_textline_orientation is not None:
            config.use_textline_orientation = use_textline_orientation
        if lang is not None:
            config.lang = lang
        if det_model_dir is not None:
            config.det_model_dir = det_model_dir
        if rec_model_dir is not None:
            config.rec_model_dir = rec_model_dir
        
        self.config = config
        self.input_folder = config.input_folder
        self.use_gpu = config.use_gpu
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Initialize PaddleOCR models
        self._initialize_paddle_models()
        
        # Setup cross-platform organization folders
        self.setup_organization_folders()
        
        logger.info(f"PaddleTextDetector initialized successfully")
        logger.info(f"Platform: {platform.system()}")
        logger.info(f"Base directory: {config.base_directory}")
        logger.info(f"Input folder: {self.input_folder}")
        logger.info(f"GPU enabled: {config.use_gpu}")
        logger.info(f"Textline orientation: {config.use_textline_orientation}")
        logger.info(f"Language: {config.lang}")
    
    def _initialize_paddle_models(self):
        """Initialize PaddleOCR models with thread-safe initialization and GPU memory management"""
        
        # Thread-safe initialization using class-level lock
        with self._paddle_lock:
            try:
                logger.info(f"Thread-safe initialization of PaddleOCR v3.1.0...")
                
                # Minimal parameters to prevent "Unknown exception" errors
                base_params = {
                    'lang': self.config.lang,
                    'use_angle_cls': self.config.use_textline_orientation,
                }
                
                # Add custom model paths if provided
                if self.config.det_model_dir:
                    base_params['det_model_dir'] = self.config.det_model_dir
                if self.config.rec_model_dir:
                    base_params['rec_model_dir'] = self.config.rec_model_dir
                
                # Add GPU configuration if enabled
                if self.config.use_gpu:
                    base_params['use_gpu'] = True
                    logger.info("GPU support enabled for PaddleOCR")
                    
                    # Clear GPU memory before initialization
                    self.clear_gpu_memory()
                
                logger.info(f"Initializing PaddleOCR with parameters: {base_params}")
                
                # Initialize full OCR model with complete output suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with suppress_stdout_stderr():
                        self.ocr_full = PaddleOCR(**base_params)
                
                # For detection-only, we'll use the same model but only extract bounding boxes
                # This avoids unsupported 'rec' parameter issues
                self.ocr_detection_only = self.ocr_full
                
                logger.info("Thread-safe PaddleOCR models initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR models: {e}")
                logger.error(f"Error details: {traceback.format_exc()}")
                raise
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache if using GPU."""
        if self.config.use_gpu:
            try:
                # Try PaddlePaddle GPU memory clearing
                try:
                    import paddle
                    if hasattr(paddle, 'device') and hasattr(paddle.device, 'cuda'):
                        paddle.device.cuda.empty_cache()
                        logger.debug("PaddlePaddle GPU memory cleared")
                except:
                    pass
                
                # Also try PyTorch if available (in case of mixed GPU usage)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.debug("PyTorch GPU memory cleared")
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")
    
    def validate_installation(self) -> bool:
        """Validate PaddleOCR installation and functionality."""
        try:
            logger.info("Validating PaddleOCR installation...")
            
            # Create test image with text
            test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Test detection
            results = self.detect_and_recognize_text(test_img)
            validation_success = len(results) > 0 and any("TEST" in r.text.upper() for r in results)
            
            if validation_success:
                logger.info("PaddleOCR installation validation PASSED")
            else:
                logger.warning("PaddleOCR installation validation FAILED - no text detected in test image")
            
            return validation_success
            
        except Exception as e:
            logger.error(f"PaddleOCR validation failed: {e}")
            return False
    
    def setup_organization_folders(self):
        """Set up cross-platform folder structure for confidence-based organization."""
        # Use configuration-based base directory (cross-platform)
        base_directory = self.config.base_directory
        self.organized_images_dir = base_directory / "PADDLE_OCR_RESULTS"
        
        # Create organization subdirectories
        self.invalid_folder = self.organized_images_dir / "INVALID_WATERMARKED"        # High confidence text detected
        self.manual_review_folder = self.organized_images_dir / "MANUAL_REVIEW"       # Medium confidence
        self.valid_folder = self.organized_images_dir / "VALID_CLEAN"                 # Low/no text confidence
        self.debug_folder = self.organized_images_dir / "DEBUG_VISUALIZATIONS"        # Debug outputs
        
        # Create directories with proper error handling
        folders_to_create = [
            self.organized_images_dir, 
            self.invalid_folder,
            self.manual_review_folder, 
            self.valid_folder, 
            self.debug_folder
        ]
        
        for folder in folders_to_create:
            try:
                folder.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {folder}")
            except Exception as e:
                logger.error(f"Failed to create directory {folder}: {e}")
                raise
        
        logger.info(f"Cross-platform PaddleOCR organization folders created:")
        logger.info(f"  Platform: {platform.system()}")
        logger.info(f"  Base directory: {base_directory}")
        logger.info(f"  Organization root: {self.organized_images_dir}")
        logger.info(f"  - INVALID_WATERMARKED: {self.invalid_folder}")
        logger.info(f"  - MANUAL_REVIEW: {self.manual_review_folder}")
        logger.info(f"  - VALID_CLEAN: {self.valid_folder}")
        logger.info(f"  - DEBUG_VISUALIZATIONS: {self.debug_folder}")
    
    def detect_text_regions_only(self, image: Union[str, np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Fast text detection only - returns bounding boxes
        Ideal for integration with existing OCR systems
        """
        try:
            # Handle both file path and numpy array input
            if isinstance(image, str):
                img_array = cv2.imread(image)
                if img_array is None:
                    logger.error(f"Could not load image: {image}")
                    return []
            else:
                img_array = image
            
            # Run detection-only model with thread safety and compatibility
            start_time = time.time()
            
            with self._paddle_lock:
                try:
                    # Try modern predict method first
                    results = self.ocr_detection_only.predict(img_array)
                except AttributeError:
                    try:
                        # Fall back to deprecated ocr method
                        results = self.ocr_detection_only.ocr(img_array, det=True, rec=False)
                    except Exception:
                        # If parameters not supported, try simple call
                        results = self.ocr_detection_only.ocr(img_array)
                except Exception as e:
                    logger.error(f"Thread-safe detection failed: {e}")
                    return []
            
            detection_time = (time.time() - start_time) * 1000
            
            rectangles = []
            if results and results[0]:
                for detection in results[0]:
                    if isinstance(detection, list) and len(detection) >= 4:
                        # Convert polygon to bounding rectangle
                        points = np.array(detection, dtype=np.int32)
                        x, y, w, h = cv2.boundingRect(points)
                        
                        # Filter by minimum area
                        if w * h >= self.config['min_bbox_area']:
                            rectangles.append((x, y, w, h))
            
            logger.info(f"Detection completed in {detection_time:.2f}ms, found {len(rectangles)} regions")
            return rectangles
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []
    
    @monitor_performance
    def detect_and_recognize_text(self, image: Union[str, np.ndarray], 
                                detection_only: bool = False) -> List[PaddleTextResult]:
        """
        Complete text detection and recognition pipeline using PaddleOCR v3.x API
        with enhanced error handling for robustness and GPU memory management.
        
        Args:
            image: Image file path or numpy array
            detection_only: If True, only detect regions without recognition
            
        Returns:
            List of PaddleTextResult objects
        """
        try:
            # Step 1: Preprocess image
            processed_img, image_path = self._preprocess_input_image(image)
            if processed_img is None:
                return []
            
            # Step 2: Execute OCR with fallback strategies
            ocr_results = self._execute_paddle_ocr_with_fallback(processed_img, image_path)
            if ocr_results is None:
                return []
            
            # Step 3: Process results based on mode
            if detection_only:
                results = self._process_detection_only_results_v3(ocr_results)
            else:
                results = self._process_full_ocr_results_v3(ocr_results)
            
            logger.info(f"Processed {len(results)} text elements from {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"OCR processing failed for {image}: {e}")
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM detected, clearing memory")
                self.clear_gpu_memory()
            return []
    
    def _preprocess_input_image(self, image: Union[str, np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
        """
        Preprocess input image and extract image path.
        
        Returns:
            Tuple of (processed_image_array, image_path)
        """
        # Handle input format
        if isinstance(image, str):
            img_array = cv2.imread(image)
            if img_array is None:
                logger.error(f"Could not load image: {image}")
                return None, image
            image_path = image
        else:
            img_array = image
            image_path = "numpy_array"
        
        logger.info(f"Starting OCR processing for: {image_path}")
        logger.debug(f"Image shape: {img_array.shape}")
        
        # Validate and preprocess image
        processed_img = self._validate_and_preprocess_image(img_array, image_path)
        if processed_img is None:
            logger.error(f"Image preprocessing failed for: {image_path}")
            return None, image_path
        
        return processed_img, image_path
    
    def _execute_paddle_ocr_with_fallback(self, processed_img: np.ndarray, image_path: str) -> Optional[List]:
        """
        Execute PaddleOCR with multiple fallback strategies for maximum compatibility.
        
        Returns:
            OCR results or None if all strategies fail
        """
        ocr_results = None
        last_error = None
        
        # Thread-safe OCR execution using class-level lock
        with self._paddle_lock:
            # Strategy 1: Try with RGB conversion (recommended approach)
            try:
                logger.debug("Strategy 1: Thread-safe OCR with RGB conversion")
                img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                ocr_results = self.ocr_full.ocr(img_rgb)
                logger.debug("Strategy 1 succeeded")
                return ocr_results
            except Exception as e1:
                last_error = e1
                logger.debug(f"Strategy 1 failed: {e1}")
                
                # Check for GPU OOM and clear memory
                if "out of memory" in str(e1).lower():
                    logger.warning("GPU OOM detected in strategy 1, clearing memory")
                    self.clear_gpu_memory()
            
            # Strategy 2: Try with original BGR image
            try:
                logger.debug("Strategy 2: Thread-safe OCR with BGR image")
                ocr_results = self.ocr_full.ocr(processed_img)
                logger.debug("Strategy 2 succeeded")
                return ocr_results
            except Exception as e2:
                last_error = e2
                logger.debug(f"Strategy 2 failed: {e2}")
                
                # Check for GPU OOM and clear memory
                if "out of memory" in str(e2).lower():
                    logger.warning("GPU OOM detected in strategy 2, clearing memory")
                    self.clear_gpu_memory()
            
            # Strategy 3: Try with further preprocessing
            try:
                logger.debug("Strategy 3: Thread-safe OCR with additional preprocessing")
                further_processed = self._preprocess_problematic_image(processed_img)
                img_rgb = cv2.cvtColor(further_processed, cv2.COLOR_BGR2RGB)
                ocr_results = self.ocr_full.ocr(img_rgb)
                logger.debug("Strategy 3 succeeded")
                return ocr_results
            except Exception as e3:
                last_error = e3
                logger.debug(f"Strategy 3 failed: {e3}")
        
        # All strategies failed
        error_msg = str(last_error).lower()
        if "unknown exception" in error_msg:
            logger.warning(f"PaddleOCR encountered unknown exception for {image_path} - this is often related to image format/orientation issues and is recoverable")
        else:
            logger.error(f"All OCR strategies failed for {image_path}. Last error: {last_error}")
        
        return None
    
    def _preprocess_problematic_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Preprocess images that cause PaddleOCR unknown exceptions
        Common issues: unusual aspect ratios, extreme orientations, color spaces
        """
        try:
            # Ensure image is in BGR format
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # Convert BGRA to BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
            
            # Ensure reasonable dimensions
            h, w = img_array.shape[:2]
            max_size = 2000
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            
            # Ensure minimum dimensions
            min_size = 32
            if min(h, w) < min_size:
                scale = min_size / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                logger.debug(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")
            
            return img_array
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return img_array
    
    def _validate_and_preprocess_image(self, img_array: np.ndarray, image_path: str) -> Optional[np.ndarray]:
        """
        Comprehensive image validation and preprocessing to prevent PaddleOCR "Unknown exception" errors
        This addresses the root causes of PaddleOCR v3.1.0 compatibility issues
        """
        try:
            if img_array is None:
                logger.error(f"Image array is None for {image_path}")
                return None
            
            # Check image shape validity
            if len(img_array.shape) not in [2, 3]:
                logger.error(f"Invalid image shape {img_array.shape} for {image_path}")
                return None
            
            # Handle different image formats that cause PaddleOCR issues
            
            # 1. Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                logger.debug(f"Converted grayscale to BGR for {image_path}")
            
            # 2. Handle RGBA images (alpha channel causes issues)
            elif img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                logger.debug(f"Converted BGRA to BGR for {image_path}")
            
            # 3. Handle single channel color images
            elif img_array.shape[2] == 1:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                logger.debug(f"Converted single channel to BGR for {image_path}")
            
            # Validate dimensions
            h, w = img_array.shape[:2]
            if h <= 0 or w <= 0:
                logger.error(f"Invalid image dimensions {w}x{h} for {image_path}")
                return None
            
            # Handle extreme dimensions that cause PaddleOCR v3.1.0 "Unknown exception"
            max_dimension = 3000  # Conservative limit for v3.1.0
            min_dimension = 20    # Minimum reasonable size for OCR
            
            # Calculate if resizing is needed
            needs_resize = False
            scale = 1.0
            
            # Check maximum dimension
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                needs_resize = True
                logger.debug(f"Image too large ({max(h, w)} > {max_dimension}), will scale by {scale:.3f}")
            
            # Check minimum dimension
            if min(h, w) < min_dimension:
                min_scale = min_dimension / min(h, w)
                if min_scale > scale:  # Use the larger scale factor
                    scale = min_scale
                    needs_resize = True
                logger.debug(f"Image too small ({min(h, w)} < {min_dimension}), will scale by {scale:.3f}")
            
            # Check extreme aspect ratios that cause issues
            aspect_ratio = max(h, w) / min(h, w)
            if aspect_ratio > 20:  # Very extreme aspect ratio
                # Limit aspect ratio to prevent OCR failures
                if h > w:  # Tall image
                    new_h = min(h, int(w * 20))
                    crop_y = (h - new_h) // 2
                    img_array = img_array[crop_y:crop_y + new_h, :, :]
                    logger.debug(f"Cropped tall image to reduce aspect ratio for {image_path}")
                else:  # Wide image
                    new_w = min(w, int(h * 20))
                    crop_x = (w - new_w) // 2
                    img_array = img_array[:, crop_x:crop_x + new_w, :]
                    logger.debug(f"Cropped wide image to reduce aspect ratio for {image_path}")
                
                # Recalculate dimensions after cropping
                h, w = img_array.shape[:2]
            
            # Apply resizing if needed
            if needs_resize:
                new_h, new_w = int(h * scale), int(w * scale)
                # Use appropriate interpolation based on scaling direction
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
                logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} for {image_path}")
            
            # Ensure image is contiguous in memory (prevents OpenCV/PaddleOCR errors)
            if not img_array.flags['C_CONTIGUOUS']:
                img_array = np.ascontiguousarray(img_array)
                logger.debug(f"Made image contiguous in memory for {image_path}")
            
            # Ensure correct data type
            if img_array.dtype != np.uint8:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                logger.debug(f"Converted image to uint8 for {image_path}")
            
            # Final validation
            final_h, final_w = img_array.shape[:2]
            if final_h < min_dimension or final_w < min_dimension:
                logger.error(f"Final image too small ({final_w}x{final_h}) for {image_path}")
                return None
            
            if final_h > max_dimension or final_w > max_dimension:
                logger.error(f"Final image too large ({final_w}x{final_h}) for {image_path}")
                return None
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image validation/preprocessing failed for {image_path}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _process_detection_only_results_v3(self, ocr_results) -> List[PaddleTextResult]:
        """Process detection-only results for PaddleOCR v3.x"""
        results = []
        
        if not ocr_results or len(ocr_results) == 0:
            logger.info("No detection results found")
            return results
        
        # PaddleOCR v3.x returns a list with a single dict containing all results
        result_dict = ocr_results[0] if isinstance(ocr_results, list) else ocr_results
        
        if 'dt_polys' not in result_dict:
            logger.info("No detection polygons found in results")
            return results
        
        dt_polys = result_dict['dt_polys']
        logger.info(f"Processing {len(dt_polys)} detection results")
        
        for i, poly in enumerate(dt_polys):
            try:
                # Convert polygon points
                points = np.array(poly, dtype=np.float32)
                polygon = [(int(p[0]), int(p[1])) for p in points]
                
                # Convert to bounding rectangle
                x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
                
                # Filter by area
                if w * h >= self.config['min_bbox_area']:
                    result = PaddleTextResult(
                        text=f"detected_region_{i}",
                        confidence=0.8,  # Default for detection-only
                        bbox=(x, y, w, h),
                        polygon=polygon,
                        detection_confidence=0.8,
                        recognition_confidence=0.0,
                        angle=self._calculate_text_angle(points)
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Failed to process detection {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} detection results")
        return results
    
    def _process_full_ocr_results_v3(self, ocr_results) -> List[PaddleTextResult]:
        """Process full OCR results for PaddleOCR v3.x"""
        results = []
        
        if not ocr_results or len(ocr_results) == 0:
            logger.info("No OCR results found")
            return results
        
        # PaddleOCR v3.x returns a list with a single dict containing all results
        result_dict = ocr_results[0] if isinstance(ocr_results, list) else ocr_results
        
        # Extract the components
        rec_texts = result_dict.get('rec_texts', [])
        rec_scores = result_dict.get('rec_scores', [])
        rec_polys = result_dict.get('rec_polys', [])
        
        logger.info(f"Processing {len(rec_texts)} OCR results")
        logger.info(f"Texts: {rec_texts}")
        logger.info(f"Scores: {rec_scores}")
        
        # Make sure all arrays have the same length
        min_length = min(len(rec_texts), len(rec_scores), len(rec_polys))
        
        for i in range(min_length):
            try:
                text = rec_texts[i].strip()
                confidence = float(rec_scores[i])
                poly = rec_polys[i]
                
                logger.info(f"Processing text: '{text}' with confidence: {confidence:.3f}")
                
                # Convert polygon points
                points = np.array(poly, dtype=np.float32)
                polygon = [(int(p[0]), int(p[1])) for p in points]
                x, y, w, h = cv2.boundingRect(np.array(points, dtype=np.int32))
                
                # Apply validation filters
                if not self._is_valid_paddle_result(text, confidence, w, h):
                    logger.info(f"Text '{text}' filtered out (confidence: {confidence:.3f}, size: {w}x{h})")
                    continue
                
                # Calculate detection confidence (estimate)
                detection_confidence = min(confidence + 0.1, 1.0)
                
                # Calculate text angle
                angle = self._calculate_text_angle(points)
                
                result = PaddleTextResult(
                    text=text,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    polygon=polygon,
                    detection_confidence=detection_confidence,
                    recognition_confidence=confidence,
                    angle=angle
                )
                results.append(result)
                logger.info(f"Added valid text result: '{text}' ({confidence:.3f})")
                
            except Exception as e:
                logger.warning(f"Failed to process OCR result {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(results)} valid OCR results")
        return results
    
    def _is_valid_paddle_result(self, text: str, confidence: float, width: int, height: int) -> bool:
        """Validate PaddleOCR result using configuration-based criteria"""
        # Confidence check using config
        if confidence < self.config.recognition_threshold:
            logger.debug(f"Text '{text}' rejected: confidence {confidence:.3f} below threshold {self.config.recognition_threshold}")
            return False
        
        # Text length check using config
        if not (self.config.min_text_length <= len(text) <= self.config.max_text_length):
            logger.debug(f"Text '{text}' rejected: length {len(text)} outside range [{self.config.min_text_length}, {self.config.max_text_length}]")
            return False
        
        # Area check using config
        if width * height < self.config.min_bbox_area:
            logger.debug(f"Text '{text}' rejected: area {width*height} below minimum {self.config.min_bbox_area}")
            return False
        
        # Content check - more permissive, allow any printable characters
        if not any(c.isprintable() and not c.isspace() for c in text):
            logger.debug(f"Text '{text}' rejected: no printable characters")
            return False
        
        # Reject only extreme noise patterns
        if self._is_noise_pattern(text):
            logger.debug(f"Text '{text}' rejected: noise pattern")
            return False
        
        return True
    
    def _is_noise_pattern(self, text: str) -> bool:
        """Detect noise patterns in text"""
        cleaned = text.strip().lower()
        
        # Very short repetitive patterns
        if len(cleaned) <= 3:
            if len(set(cleaned)) == 1:  # All same character
                return True
        
        # Repetitive patterns
        if len(cleaned) >= 4:
            unique_chars = len(set(cleaned))
            char_diversity = unique_chars / len(cleaned)
            if char_diversity < 0.3:  # Less than 30% diversity
                return True
        
        # Only special characters
        if re.match(r'^[^\w\s]{2,}$', cleaned):
            return True
        
        return False
    
    def _calculate_text_angle(self, points: np.ndarray) -> float:
        """Calculate text rotation angle from polygon points"""
        try:
            # Use the first two points to calculate angle
            p1, p2 = points[0], points[1]
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            return angle
        except:
            return 0.0
    
    def process_single_image(self, image_path: Path, 
                           save_debug: bool = False,
                           detection_only: bool = False) -> Dict:
        """
        Process a single image with comprehensive analysis
        
        Args:
            image_path: Path to image file
            save_debug: Save debug visualization
            detection_only: Only perform detection without recognition
            
        Returns:
            Comprehensive processing results
        """
        start_time = time.time()
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Load and validate image
            img = cv2.imread(str(image_path))
            if img is None:
                return self._create_error_result(image_path, "Could not load image")
            
            height, width = img.shape[:2]
            
            # Run PaddleOCR
            ocr_start = time.time()
            paddle_results = self.detect_and_recognize_text(img, detection_only=detection_only)
            ocr_time = (time.time() - ocr_start) * 1000
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_paddle_confidence(paddle_results)
            
            # Create debug visualization if requested
            if save_debug and paddle_results:
                self._save_debug_visualization(img, paddle_results, image_path)
            
            # Organize image based on confidence
            organization_result = self.organize_image_by_confidence(image_path, confidence_metrics)
            
            total_time = (time.time() - start_time) * 1000
            
            # Compile results
            result = {
                'image_path': str(image_path),
                'processing_success': True,
                'processing_time_ms': total_time,
                'ocr_time_ms': ocr_time,
                'image_dimensions': (width, height),
                'text_detections': len(paddle_results),
                'paddle_results': [
                    {
                        'text': r.text,
                        'confidence': r.confidence,
                        'detection_confidence': r.detection_confidence,
                        'recognition_confidence': r.recognition_confidence,
                        'bbox': r.bbox,
                        'polygon': r.polygon,
                        'angle': r.angle
                    }
                    for r in paddle_results
                ],
                'confidence_metrics': confidence_metrics,
                'organization_result': organization_result,
                'method': 'PaddleOCR_DB_Model'
            }
            
            logger.info(f"Successfully processed {image_path.name} in {total_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return self._create_error_result(image_path, str(e))
    
    def test_and_visualize(self, image_path: str, show_result: bool = True) -> Dict:
        """
        Test text detection on a single image and create visualization
        
        Args:
            image_path: Path to the image to test
            show_result: Whether to display results in console
            
        Returns:
            Detection results with visualization paths
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image not found: {image_path}")
                return {'error': f'Image not found: {image_path}'}
            
            logger.info(f"Testing text detection on: {image_path}")
            
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # Detect text
            results = self.detect_and_recognize_text(img, detection_only=False)
            
            # Always create visualization for testing
            self._save_debug_visualization(img, results, image_path)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_paddle_confidence(results)
            
            test_result = {
                'image_path': str(image_path),
                'image_dimensions': img.shape[:2][::-1],  # (width, height)
                'detection_success': True,
                'text_regions_found': len(results),
                'results': [
                    {
                        'text': r.text,
                        'confidence': r.confidence,
                        'bbox': r.bbox,
                        'polygon': r.polygon
                    }
                    for r in results
                ],
                'confidence_metrics': confidence_metrics,
                'visualization_paths': {
                    'detailed': str(self.debug_folder / f"paddle_debug_{image_path.stem}.jpg"),
                    'simple': str(self.debug_folder / f"paddle_simple_{image_path.stem}.jpg")
                }
            }
            
            if show_result:
                print(f"\n{'='*60}")
                print(f"TEXT DETECTION RESULTS for {image_path.name}")
                print(f"{'='*60}")
                print(f"Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
                print(f"Text regions found: {len(results)}")
                print(f"Overall confidence: {confidence_metrics['overall_confidence']:.1f}%")
                
                if results:
                    print(f"\nDETECTED TEXT:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. '{result.text}' (confidence: {result.confidence:.3f})")
                        x, y, w, h = result.bbox
                        print(f"     Location: x={x}, y={y}, width={w}, height={h}")
                
                print(f"\nVISUALIZATIONS SAVED:")
                print(f"  • Detailed: {test_result['visualization_paths']['detailed']}")
                print(f"  • Simple: {test_result['visualization_paths']['simple']}")
                
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            logger.error(traceback.format_exc())
            return {'error': str(e)}

    def _calculate_paddle_confidence(self, results: List[PaddleTextResult]) -> Dict:
        """Calculate comprehensive confidence metrics for PaddleOCR results"""
        if not results:
            return {
                'overall_confidence': 0.0,
                'detection_confidence': 0.0,
                'recognition_confidence': 0.0,
                'text_count': 0,
                'average_confidence': 0.0,
                'high_confidence_count': 0,
                'medium_confidence_count': 0,
                'low_confidence_count': 0
            }
        
        # Extract confidence values
        detection_confidences = [r.detection_confidence for r in results]
        recognition_confidences = [r.recognition_confidence for r in results]
        overall_confidences = [r.confidence for r in results]
        
        # Calculate averages
        avg_detection = np.mean(detection_confidences)
        avg_recognition = np.mean(recognition_confidences)
        avg_overall = np.mean(overall_confidences)
        
        # Count confidence levels
        high_confidence = sum(1 for c in overall_confidences if c >= 0.8)
        medium_confidence = sum(1 for c in overall_confidences if 0.5 <= c < 0.8)
        low_confidence = sum(1 for c in overall_confidences if c < 0.5)
        
        # Calculate overall confidence (weighted by text length and confidence)
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = len(result.text) * result.confidence
            weighted_confidence += weight * result.confidence
            total_weight += weight
        
        overall_confidence = (weighted_confidence / total_weight) if total_weight > 0 else 0.0
        overall_confidence *= 100  # Convert to percentage
        
        return {
            'overall_confidence': overall_confidence,
            'detection_confidence': avg_detection * 100,
            'recognition_confidence': avg_recognition * 100,
            'text_count': len(results),
            'average_confidence': avg_overall * 100,
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            'total_text_length': sum(len(r.text) for r in results),
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            }
        }
    
    def organize_image_by_confidence(self, image_path: Path, confidence_metrics: Dict) -> Dict:
        """Organize image based on PaddleOCR confidence metrics"""
        try:
            overall_confidence = confidence_metrics.get('overall_confidence', 0.0)
            text_count = confidence_metrics.get('text_count', 0)
            
            # Determine destination based on confidence and text count
            if overall_confidence > 80 or text_count > 5:
                destination_folder = self.invalid_folder
                category = "INVALID_WATERMARKED"
                reason = f"HIGH TEXT CONFIDENCE: {overall_confidence:.1f}% ({text_count} text regions detected)"
            elif overall_confidence > 60 or text_count > 2:
                destination_folder = self.manual_review_folder
                category = "MANUAL_REVIEW"
                reason = f"MEDIUM TEXT CONFIDENCE: {overall_confidence:.1f}% ({text_count} text regions detected)"
            else:
                destination_folder = self.valid_folder
                category = "VALID_CLEAN"
                reason = f"LOW TEXT CONFIDENCE: {overall_confidence:.1f}% ({text_count} text regions detected)"
            
            # Create destination path with conflict resolution
            destination_path = destination_folder / image_path.name
            counter = 1
            original_destination = destination_path
            
            while destination_path.exists():
                stem = original_destination.stem
                suffix = original_destination.suffix
                destination_path = destination_folder / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file
            shutil.copy2(str(image_path), str(destination_path))
            
            organization_result = {
                'source_path': str(image_path),
                'destination_path': str(destination_path),
                'category': category,
                'confidence': overall_confidence,
                'text_count': text_count,
                'reason': reason,
                'organized_successfully': True
            }
            
            logger.info(f"Organized {image_path.name} -> {category} ({overall_confidence:.1f}%, {text_count} texts)")
            return organization_result
            
        except Exception as e:
            error_msg = f"Failed to organize {image_path.name}: {str(e)}"
            logger.error(error_msg)
            return {
                'source_path': str(image_path),
                'destination_path': None,
                'category': None,
                'confidence': confidence_metrics.get('overall_confidence', 0.0),
                'reason': error_msg,
                'organized_successfully': False
            }

    def _save_debug_visualization(self, img: np.ndarray, results: List[PaddleTextResult], 
                                image_path: Path):
        """Save enhanced debug visualization with detected text regions"""
        try:
            # Create visualization image
            debug_img = img.copy()
            height, width = debug_img.shape[:2]
            
            # Create a larger canvas for additional info
            info_height = 200
            canvas = np.ones((height + info_height, width, 3), dtype=np.uint8) * 240
            canvas[:height, :width] = debug_img
            
            # Color scheme for different confidence levels
            colors = {
                'high': (0, 255, 0),      # Green for high confidence
                'medium': (0, 165, 255),  # Orange for medium confidence
                'low': (0, 0, 255),       # Red for low confidence
                'bbox': (255, 0, 255),    # Magenta for bounding box
                'polygon': (255, 255, 0)  # Cyan for polygon
            }
            
            info_lines = []
            info_lines.append(f"Image: {image_path.name}")
            info_lines.append(f"Dimensions: {width}x{height}")
            info_lines.append(f"Text Regions Found: {len(results)}")
            info_lines.append("=" * 50)
            
            for i, result in enumerate(results):
                # Determine confidence level and color
                conf = result.confidence
                if conf >= 0.8:
                    conf_level = 'high'
                    conf_color = colors['high']
                elif conf >= 0.5:
                    conf_level = 'medium'
                    conf_color = colors['medium']
                else:
                    conf_level = 'low'
                    conf_color = colors['low']
                
                # Draw polygon outline (more accurate)
                polygon_points = np.array(result.polygon, dtype=np.int32)
                cv2.polylines(canvas, [polygon_points], True, colors['polygon'], 2)
                
                # Draw bounding box
                x, y, w, h = result.bbox
                cv2.rectangle(canvas, (x, y), (x + w, y + h), colors['bbox'], 1)
                
                # Fill polygon with semi-transparent color
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [polygon_points], conf_color)
                cv2.addWeighted(canvas, 0.8, overlay, 0.2, 0, canvas)
                
                # Add text label with background
                label = f"{i+1}: {result.text[:30]}"
                label_conf = f"({conf:.2f})"
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                (conf_width, conf_height), _ = cv2.getTextSize(
                    label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                # Draw background rectangle for text
                bg_x = max(0, x)
                bg_y = max(text_height + 5, y - 5)
                bg_width = max(text_width, conf_width) + 10
                bg_height = text_height + conf_height + 15
                
                cv2.rectangle(canvas, 
                            (bg_x, bg_y - bg_height), 
                            (bg_x + bg_width, bg_y), 
                            (0, 0, 0), -1)
                
                # Draw text labels
                cv2.putText(canvas, label, (bg_x + 5, bg_y - conf_height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(canvas, label_conf, (bg_x + 5, bg_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
                
                # Add to info
                info_lines.append(f"{i+1}. '{result.text}' ({conf:.3f}) - {conf_level}")
                if len(info_lines) > 15:  # Limit info display
                    info_lines.append("... (more results)")
                    break
            
            # Add information panel at bottom
            y_offset = height + 20
            for line in info_lines:
                cv2.putText(canvas, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 20
                if y_offset > height + info_height - 20:
                    break
            
            # Add legend
            legend_x = width - 200
            legend_y = height + 20
            cv2.putText(canvas, "Legend:", (legend_x, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            legend_y += 25
            
            legend_items = [
                ("High Conf (≥0.8)", colors['high']),
                ("Med Conf (≥0.5)", colors['medium']),
                ("Low Conf (<0.5)", colors['low']),
                ("Polygon", colors['polygon']),
                ("BBox", colors['bbox'])
            ]
            
            for item, color in legend_items:
                cv2.rectangle(canvas, (legend_x, legend_y - 10), (legend_x + 15, legend_y), color, -1)
                cv2.putText(canvas, item, (legend_x + 20, legend_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                legend_y += 18
            
            # Save debug image
            debug_filename = f"paddle_debug_{image_path.stem}.jpg"
            debug_path = self.debug_folder / debug_filename
            cv2.imwrite(str(debug_path), canvas)
            
            logger.info(f"Enhanced debug visualization saved: {debug_path}")
            
            # Also save a simple overlay version
            simple_debug = img.copy()
            for i, result in enumerate(results):
                x, y, w, h = result.bbox
                conf = result.confidence
                color = (0, 255, 0) if conf >= 0.7 else (0, 165, 255) if conf >= 0.4 else (0, 0, 255)
                
                cv2.rectangle(simple_debug, (x, y), (x + w, y + h), color, 2)
                cv2.putText(simple_debug, f"{result.text[:20]} ({conf:.2f})", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            simple_debug_filename = f"paddle_simple_{image_path.stem}.jpg"
            simple_debug_path = self.debug_folder / simple_debug_filename
            cv2.imwrite(str(simple_debug_path), simple_debug)
            
            logger.info(f"Simple debug visualization saved: {simple_debug_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug visualization: {e}")
            logger.warning(traceback.format_exc())

    def _create_error_result(self, image_path: Path, error_message: str) -> Dict:
        """Create error result dictionary with enhanced error categorization"""
        error_msg_lower = error_message.lower()
        
        # Categorize common PaddleOCR errors
        error_category = 'general_error'
        user_friendly_message = error_message
        
        if 'unknown exception' in error_msg_lower:
            error_category = 'paddle_unknown_exception'
            user_friendly_message = 'PaddleOCR encountered an unknown exception (likely image format or orientation issue)'
        elif 'orientation' in error_msg_lower or 'textline' in error_msg_lower:
            error_category = 'paddle_orientation_issue'
            user_friendly_message = 'PaddleOCR text orientation processing issue'
        elif 'memory' in error_msg_lower or 'cuda' in error_msg_lower or 'gpu' in error_msg_lower:
            error_category = 'paddle_resource_issue'
            user_friendly_message = 'PaddleOCR memory or GPU resource issue'
        elif 'could not load' in error_msg_lower:
            error_category = 'image_load_error'
            user_friendly_message = 'Could not load or read the image file'
        
        return {
            'image_path': str(image_path),
            'processing_success': False,
            'error': user_friendly_message,
            'error_category': error_category,
            'original_error': error_message,
            'processing_time_ms': 0,
            'text_detections': 0,
            'paddle_results': [],
            'confidence_metrics': {
                'overall_confidence': 0.0,
                'text_count': 0
            },
            'method': 'PaddleOCR_DB_Model',
            'graceful_failure': error_category in ['paddle_unknown_exception', 'paddle_orientation_issue', 'paddle_resource_issue']
        }

    def get_image_files(self) -> List[Path]:
        """Get all supported image files from input folder"""
        if not self.input_folder.exists():
            logger.error(f"Input folder does not exist: {self.input_folder}")
            return []
        
        image_files = []
        for file_path in self.input_folder.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        logger.info(f"Found {len(image_files)} image files")
        return image_files
    
    @monitor_performance
    def process_batch(self, detection_only: bool = False, save_debug: bool = False, 
                     max_images: Optional[int] = None, start_from: int = 0) -> BatchResult:
        """
        Process all images in the input folder with progress tracking
        
        Args:
            detection_only: Only perform detection without recognition
            save_debug: Save debug visualizations
            max_images: Maximum number of images to process (None for all)
            start_from: Index to start processing from (for resuming)
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        image_files = self.get_image_files()
        
        if not image_files:
            logger.warning("No image files found to process")
            return {'error': 'No image files found'}
        
        # Apply start_from and max_images filters
        if start_from > 0:
            image_files = image_files[start_from:]
        if max_images:
            image_files = image_files[:max_images]
        
        total_images = len(image_files)
        logger.info(f"Starting batch processing of {total_images} images")
        logger.info(f"Detection only: {detection_only}")
        logger.info(f"Save debug: {save_debug}")
        
        results = []
        successful_processing = 0
        
        # Progress tracking
        batch_start_time = time.time()
        processing_times = []
        
        for i, image_path in enumerate(image_files, 1):
            image_start_time = time.time()
            
            # Enhanced progress information with real-time feedback
            progress_pct = (i / total_images) * 100
            elapsed_total = time.time() - batch_start_time
            
            # Print to both logger and console for immediate visibility
            progress_msg = f"\n[{i:3d}/{total_images}] ({progress_pct:5.1f}%) Processing: {image_path.name}"
            print(progress_msg)  # Immediate console output
            logger.info(progress_msg)
            
            # Real-time time estimation
            if processing_times:
                avg_time = np.mean(processing_times)
                remaining_images = total_images - i
                estimated_remaining_sec = remaining_images * avg_time
                estimated_remaining_min = estimated_remaining_sec / 60
                
                time_msg = f"Elapsed: {elapsed_total/60:.1f}min | ETA: {estimated_remaining_min:.1f}min | Avg: {avg_time:.1f}s/img"
                print(time_msg)  # Immediate console output
                logger.info(time_msg)
            else:
                time_msg = f"Starting batch processing..."
                print(time_msg)
                logger.info(time_msg)
            
            try:
                result = self.process_single_image(image_path, save_debug, detection_only)
                image_time = time.time() - image_start_time
                processing_times.append(image_time)
                
                # Immediate result feedback
                if result.get('processing_success', False):
                    successful_processing += 1
                    text_count = result.get('text_detections', 0)
                    confidence = result.get('confidence_metrics', {}).get('overall_confidence', 0)
                    category = result.get('organization_result', {}).get('category', 'unknown')
                    
                    success_msg = f"Success in {image_time:.1f}s | {text_count} texts | {confidence:.1f}% conf | → {category}"
                    print(success_msg)  # Immediate console output
                    logger.info(success_msg)
                else:
                    error_msg = result.get('error', 'Unknown error')[:60]
                    fail_msg = f"Failed in {image_time:.1f}s: {error_msg}"
                    print(fail_msg)  # Immediate console output
                    logger.info(fail_msg)
                
                results.append(result)
                
                # Periodic summary for longer batches
                if i % 10 == 0 or i == total_images:
                    success_rate = (successful_processing / i) * 100
                    current_avg = sum(processing_times) / len(processing_times) if processing_times else 0
                    summary_msg = f"Progress Summary: {successful_processing}/{i} success ({success_rate:.1f}%) | Current avg: {current_avg:.1f}s/img"
                    print(summary_msg)  # Immediate console output
                    logger.info(summary_msg)
                
                if result.get('processing_success', False):
                    successful_processing += 1
                    
                # Track processing time
                image_time = time.time() - image_start_time
                processing_times.append(image_time)
                
                # Show current stats
                logger.info(f"Processed in {image_time:.2f}s")
                if result.get('confidence_metrics'):
                    confidence = result['confidence_metrics'].get('overall_confidence', 0)
                    text_count = result['confidence_metrics'].get('text_count', 0)
                    category = result.get('organization_result', {}).get('category', 'unknown')
                    logger.info(f"  Result: {category} ({confidence:.1f}% confidence, {text_count} text regions)")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path.name}: {e}")
                results.append(self._create_error_result(image_path, str(e)))
                processing_times.append(0.1)  # Minimal time for failed processing
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_batch_statistics(results)
        
        batch_result = {
            'batch_processing_success': True,
            'total_images': total_images,
            'successfully_processed': successful_processing,
            'processing_time_seconds': total_time,
            'average_time_per_image_ms': (total_time * 1000) / total_images if total_images > 0 else 0,
            'average_time_per_image_seconds': total_time / total_images if total_images > 0 else 0,
            'statistics': stats,
            'results': results,
            'organization_folders': {
                'invalid': str(self.invalid_folder),
                'manual_review': str(self.manual_review_folder),
                'valid': str(self.valid_folder),
                'debug': str(self.debug_folder)
            },
            'performance_summary': {
                'total_time_minutes': total_time / 60,
                'images_per_minute': (total_images / total_time) * 60 if total_time > 0 else 0,
                'success_rate_percent': (successful_processing / total_images * 100) if total_images > 0 else 0
            }
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Successfully processed: {successful_processing}/{total_images} images")
        logger.info(f"Average time per image: {total_time/total_images:.2f} seconds")
        logger.info(f"Processing rate: {(total_images/total_time)*60:.1f} images/minute")
        
        return batch_result
    
    def _calculate_batch_statistics(self, results: List[Dict]) -> Dict:
        """Calculate comprehensive batch statistics"""
        total_images = len(results)
        successful_results = [r for r in results if r.get('processing_success', False)]
        
        if not successful_results:
            return {'error': 'No successful processing results'}
        
        # Category distribution
        categories = {'INVALID_WATERMARKED': 0, 'MANUAL_REVIEW': 0, 'VALID_CLEAN': 0}
        confidence_values = []
        text_counts = []
        processing_times = []
        
        for result in successful_results:
            org_result = result.get('organization_result', {})
            category = org_result.get('category')
            if category in categories:
                categories[category] += 1
            
            confidence_metrics = result.get('confidence_metrics', {})
            confidence_values.append(confidence_metrics.get('overall_confidence', 0.0))
            text_counts.append(confidence_metrics.get('text_count', 0))
            processing_times.append(result.get('processing_time_ms', 0))
        
        return {
            'total_images': total_images,
            'successful_processing': len(successful_results),
            'success_rate': (len(successful_results) / total_images * 100) if total_images > 0 else 0,
            'category_distribution': categories,
            'confidence_statistics': {
                'average': np.mean(confidence_values) if confidence_values else 0,
                'median': np.median(confidence_values) if confidence_values else 0,
                'min': np.min(confidence_values) if confidence_values else 0,
                'max': np.max(confidence_values) if confidence_values else 0,
                'std': np.std(confidence_values) if confidence_values else 0
            },
            'text_count_statistics': {
                'average': np.mean(text_counts) if text_counts else 0,
                'median': np.median(text_counts) if text_counts else 0,
                'min': np.min(text_counts) if text_counts else 0,
                'max': np.max(text_counts) if text_counts else 0,
                'total': sum(text_counts)
            },
            'performance_statistics': {
                'average_processing_time_ms': np.mean(processing_times) if processing_times else 0,
                'median_processing_time_ms': np.median(processing_times) if processing_times else 0,
                'min_processing_time_ms': np.min(processing_times) if processing_times else 0,
                'max_processing_time_ms': np.max(processing_times) if processing_times else 0
            }
        }
    
    def process_batch_optimized(self, images: List[Path], batch_size: int = 4) -> List[ProcessingResult]:
        """
        Process images in optimized batches with memory management.
        
        Args:
            images: List of image paths to process
            batch_size: Number of images to process before clearing memory
            
        Returns:
            List of processing results
        """
        results = []
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(images)} images in {total_batches} batches of {batch_size}")
        
        for batch_idx in range(0, len(images), batch_size):
            batch = images[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)")
            
            # Preload batch images for efficiency
            loaded_images = []
            for img_path in batch:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        loaded_images.append((img_path, img))
                    else:
                        logger.warning(f"Could not load image: {img_path}")
                except Exception as e:
                    logger.error(f"Error loading {img_path}: {e}")
            
            # Process batch
            for img_path, img in loaded_images:
                try:
                    result = self.process_single_image(img_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    # Add error result
                    results.append(self._create_error_result(img_path, str(e)))
            
            # Clear memory between batches
            if self.config.use_gpu:
                self.clear_gpu_memory()
            
            logger.info(f"Completed batch {batch_num}/{total_batches}")
        
        return results

def main():
    """Example usage of PaddleTextDetector with new configuration system"""
    
    print("Starting Enhanced PaddleOCR Text Detection System")
    print("=" * 60)
    
    # Create configuration
    print("Creating cross-platform configuration...")
    config = PaddleDetectorConfig(
        use_gpu=False,  # Set to True if you have GPU
        use_textline_orientation=True,
        lang='en',
        confidence_threshold=0.5,
        recognition_threshold=0.6
    )
    
    # For backward compatibility, you can also specify a custom input folder
    photos_folder = config.base_directory / "photos4testing"
    if photos_folder.exists():
        config.input_folder = photos_folder
        print(f"Using photos4testing folder: {photos_folder}")
    else:
        print(f"Using base directory: {config.base_directory}")
    
    # Optional: Save configuration for future use
    config_path = config.base_directory / "paddle_detector_config.json"
    config.save_to_file(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Initialize detector with configuration
    print("Initializing PaddleOCR detector...")
    detector = PaddleTextDetector(config=config)
    print("PaddleOCR detector initialized successfully!")
    
    # Validate installation
    print("Validating PaddleOCR installation...")
    if detector.validate_installation():
        print("Installation validation PASSED!")
    else:
        print("Installation validation FAILED!")
        return
    
    # Get total image count first
    all_images = detector.get_image_files()
    total_count = len(all_images)
    print(f"Found {total_count} images in {detector.input_folder}")

    if total_count == 0:
        print("No images found to process. Please check the input folder.")
        return

    # Always process all images in the folder by default
    max_images = None
    print(f"Processing ALL {total_count} images...")

    # Process images
    print(f"\nStarting batch processing...")
    print("Tip: This will show progress updates every few seconds")

    results = detector.process_batch(
        detection_only=False,
        save_debug=True,
        max_images=max_images
    )
    
    # Print results summary
    if 'statistics' in results:
        stats = results['statistics']
        perf = results.get('performance_summary', {})
        
        print(f"\nPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"SUMMARY:")
        print(f"  • Platform: {platform.system()}")
        print(f"  • Total images processed: {stats['total_images']}")
        print(f"  • Success rate: {stats['success_rate']:.1f}%")
        print(f"  • Total time: {perf.get('total_time_minutes', 0):.1f} minutes")
        print(f"  • Average per image: {perf.get('images_per_minute', 0):.1f} images/minute")
        
        print(f"\nORGANIZATION RESULTS:")
        for category, count in stats['category_distribution'].items():
            print(f"  • {category.title()}: {count} images")
        
        print(f"\nTEXT DETECTION STATS:")
        print(f"  • Total text regions detected: {stats['text_count_statistics']['total']}")
        print(f"  • Average per image: {stats['text_count_statistics']['average']:.1f}")
        
        print(f"\nCONFIDENCE STATS:")
        print(f"  • Average confidence: {stats['confidence_statistics']['average']:.1f}%")
        print(f"  • Range: {stats['confidence_statistics']['min']:.1f}% - {stats['confidence_statistics']['max']:.1f}%")
        
        print(f"\nCROSS-PLATFORM ORGANIZED FOLDERS:")
        org_folders = results.get('organization_folders', {})
        for folder_type, path in org_folders.items():
            print(f"  • {folder_type.title()}: {path}")
        
        print(f"\nCONFIGURATION USED:")
        print(f"  • Confidence threshold: {config.confidence_threshold}")
        print(f"  • Recognition threshold: {config.recognition_threshold}")
        print(f"  • Min bbox area: {config.min_bbox_area}")
        print(f"  • GPU enabled: {config.use_gpu}")
        print(f"  • Language: {config.lang}")
            
    else:
        print("Processing failed. Check the logs above for details.")

if __name__ == "__main__":
    main()
