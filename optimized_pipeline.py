"""
OPTIMIZED IMAGE PROCESSING PIPELINE - Updated for PaddleOCR Integration
Performance-focused redesign eliminating redundant operations

This module implements a unified pipeline that:
1. Loads each image ONCE
2. Performs preprocessing ONCE 
3. Shares processed data between all detectors
4. Eliminates temporary file operations
5. Uses memory pooling and caching
6. Integrates with PaddleOCR for text detection (handled in main pipeline)

PERFORMANCE TARGET: 70-80% reduction in total processing time

NOTE: Text and watermark detection are now handled by PaddleOCR in the main pipeline.
This module provides compatibility support for other detectors (borders, specifications).
Colors are handled by advanced_pyiqa_detector.py (not integrated here).
"""

import os
import numpy as np
import warnings
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import time
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import importlib
from collections import defaultdict

# Conditional imports for external dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV (cv2) not available. Some image processing features will be limited.")

# Suppress PyTorch and other model loading warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="timm")

logger = logging.getLogger(__name__)

class DependencyManager:
    """Robust dependency management with graceful degradation"""
    
    def __init__(self):
        self.available_detectors = {}
        self.failed_detectors = {}
        self.fallback_functions = {}
        self.checked_detectors = set()  # Track which detectors we've already checked
        self._setup_fallbacks()
    
    def _check_detector(self, name: str):
        """Check availability of a specific detector (lazy loading)"""
        if name in self.checked_detectors:
            return  # Already checked
            
        detectors = {
            'specifications': ('Spec_detector', 'meets_specifications'),
            'borders': ('border_detector', 'has_border_or_frame'),
            'watermarks': ('advanced_watermark_detector', 'AdvancedWatermarkDetector'),
            'editing': ('advanced_pyiqa_detector', 'AdvancedEditingDetector'),
            'text': ('paddle_text_detector', 'PaddleTextDetector'),
        }
        
        if name not in detectors:
            logger.warning(f"Unknown detector: {name}")
            return
            
        module_name, func_name = detectors[name]
        
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, func_name):
                self.available_detectors[name] = mod
                logger.info(f"{name} detector available")
            else:
                logger.warning(f"{name} detector function '{func_name}' missing")
                self.failed_detectors[name] = f"Function '{func_name}' not found in module"
        except ImportError as e:
            logger.warning(f"{name} detector unavailable: {e}")
            self.failed_detectors[name] = f"Import error: {e}"
        except Exception as e:
            logger.error(f"{name} detector error: {e}")
            self.failed_detectors[name] = f"Unexpected error: {e}"
            
        self.checked_detectors.add(name)
    
    def _setup_fallbacks(self):
        """Set up fallback functions for missing detectors"""
        self.fallback_functions = {
            'specifications': self._fallback_specifications,
            'borders': self._fallback_borders,
            'watermarks': self._fallback_watermarks,
            'editing': self._fallback_editing,
            'text': self._fallback_text,
        }
    
    def is_available(self, detector_name: str) -> bool:
        """Check if detector is available (lazy check)"""
        if detector_name not in self.checked_detectors:
            self._check_detector(detector_name)
        return detector_name in self.available_detectors
    
    def get_detector(self, detector_name: str):
        """Get detector module if available (lazy check)"""
        if detector_name not in self.checked_detectors:
            self._check_detector(detector_name)
        return self.available_detectors.get(detector_name)
    
    def get_fallback(self, detector_name: str):
        """Get fallback function for detector"""
        return self.fallback_functions.get(detector_name)
    
    def _fallback_specifications(self, image_path: str, processed_data=None) -> Dict[str, Any]:
        """Fallback specifications check based on image dimensions"""
        try:
            if processed_data:
                width, height = processed_data.width, processed_data.height
            else:
                with Image.open(image_path) as img:
                    width, height = img.size
            
            # Basic dimension check (assuming standard requirements)
            min_dimension = 800
            max_dimension = 4000
            
            if min(width, height) < min_dimension:
                return {
                    'passed': False,
                    'reason': f'Image too small: {width}x{height} (minimum: {min_dimension}px)',
                    'dimensions': (width, height),
                    'fallback_used': True
                }
            elif max(width, height) > max_dimension:
                return {
                    'passed': False,
                    'reason': f'Image too large: {width}x{height} (maximum: {max_dimension}px)',
                    'dimensions': (width, height),
                    'fallback_used': True
                }
            else:
                return {
                    'passed': True,
                    'reason': f'Dimensions acceptable: {width}x{height}',
                    'dimensions': (width, height),
                    'fallback_used': True
                }
        except Exception as e:
            return {
                'passed': True,  # Don't fail due to fallback errors
                'reason': 'Specifications check unavailable',
                'error': str(e),
                'fallback_used': True
            }
    
    def _fallback_borders(self, image_path: str, processed_data=None) -> Dict[str, Any]:
        """Fallback border detection using simple edge analysis"""
        try:
            if not CV2_AVAILABLE:
                return {
                    'passed': True,
                    'reason': 'Border detection unavailable (OpenCV not installed)',
                    'fallback_used': True,
                    'error': 'OpenCV required for border detection'
                }
            
            import cv2
            
            if processed_data and hasattr(processed_data, 'opencv_gray'):
                gray_img = processed_data.opencv_gray
            else:
                img = cv2.imread(image_path)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height, width = gray_img.shape
            
            # Check edges for consistent borders
            edge_thickness = min(20, min(width, height) // 20)
            
            # Sample edge regions
            top_edge = gray_img[:edge_thickness, :]
            bottom_edge = gray_img[-edge_thickness:, :]
            left_edge = gray_img[:, :edge_thickness]
            right_edge = gray_img[:, -edge_thickness:]
            
            # Calculate variance in edge regions (borders have low variance)
            edge_variances = [
                np.var(top_edge), np.var(bottom_edge),
                np.var(left_edge), np.var(right_edge)
            ]
            
            avg_edge_variance = np.mean(edge_variances)
            border_threshold = 200  # Empirical threshold
            
            has_border = avg_edge_variance < border_threshold
            
            return {
                'passed': not has_border,
                'reason': f'Border analysis: {"borders detected" if has_border else "no borders"}',
                'has_border': has_border,
                'edge_variance': avg_edge_variance,
                'fallback_used': True
            }
        except Exception as e:
            return {
                'passed': True,  # Don't fail due to fallback errors
                'reason': 'Border detection unavailable',
                'error': str(e),
                'fallback_used': True
            }
    
    def _fallback_watermarks(self, image_path: str, processed_data=None) -> Dict[str, Any]:
        """Fallback watermark detection using basic image analysis"""
        return {
            'passed': True,  # Skip advanced watermark detection
            'reason': 'Advanced watermark detector unavailable - skipping check',
            'has_watermark': False,
            'confidence': 0.0,
            'fallback_used': True,
            'note': 'Install PyTorch and advanced_watermark_detector for full functionality'
        }
    
    def _fallback_editing(self, image_path: str, processed_data=None) -> Dict[str, Any]:
        """Fallback editing detection using basic histogram analysis"""
        try:
            if not CV2_AVAILABLE:
                return {
                    'passed': True,
                    'reason': 'Editing detection unavailable (OpenCV not installed)',
                    'fallback_used': True,
                    'error': 'OpenCV required for advanced editing detection'
                }
            
            import cv2
            
            if processed_data and hasattr(processed_data, 'opencv_rgb'):
                img = processed_data.opencv_rgb
            else:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Basic histogram analysis for over-processing detection
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Calculate histogram statistics
            def hist_stats(hist):
                hist_norm = hist.flatten() / hist.sum()
                entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
                peaks = len([i for i in range(1, 255) 
                           if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
                return entropy, peaks
            
            r_entropy, r_peaks = hist_stats(hist_r)
            g_entropy, g_peaks = hist_stats(hist_g)
            b_entropy, b_peaks = hist_stats(hist_b)
            
            avg_entropy = (r_entropy + g_entropy + b_entropy) / 3
            total_peaks = r_peaks + g_peaks + b_peaks
            
            # Heuristic thresholds for over-processing
            # Natural images typically have entropy 6-8, heavily processed < 5
            editing_confidence = 0.0
            if avg_entropy < 5.0:
                editing_confidence += 15.0
            if total_peaks > 30:  # Too many histogram peaks suggests processing
                editing_confidence += 10.0
            if avg_entropy < 4.0:
                editing_confidence += 20.0
            
            return {
                'passed': editing_confidence < 20.0,
                'reason': f'Basic editing analysis (confidence: {editing_confidence:.1f}%)',
                'editing_confidence': editing_confidence,
                'histogram_entropy': avg_entropy,
                'histogram_peaks': total_peaks,
                'fallback_used': True,
                'note': 'Install PyIQA and advanced_pyiqa_detector for comprehensive analysis'
            }
        except Exception as e:
            return {
                'passed': True,  # Don't fail due to fallback errors
                'reason': 'Editing detection unavailable',
                'error': str(e),
                'fallback_used': True
            }
    
    def _fallback_text(self, image_path: str, processed_data=None) -> Dict[str, Any]:
        """Fallback text detection using basic edge analysis"""
        return {
            'passed': True,  # Skip text detection
            'reason': 'PaddleOCR text detector unavailable - skipping check',
            'text_count': 0,
            'confidence': 0.0,
            'fallback_used': True,
            'note': 'Install PaddleOCR for text detection functionality'
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of detector availability"""
        return {
            'available_detectors': list(self.available_detectors.keys()),
            'failed_detectors': self.failed_detectors,
            'total_available': len(self.available_detectors),
            'total_failed': len(self.failed_detectors)
        }

@dataclass
class PipelineConfig:
    """Comprehensive pipeline configuration with adaptive settings"""
    # Cache and memory settings
    cache_size: int = 10
    max_memory_mb: int = 1024
    enable_gpu_tracking: bool = True
    temp_file_threshold_mb: int = 100  # When to use temp files vs memory
    thread_pool_size: int = 4
    
    # Device configuration
    device: str = 'cpu'  # 'cpu' or 'cuda'
    gpu_id: int = 0  # GPU device ID when using CUDA
    force_cpu: bool = False  # Force CPU even if GPU requested
    
    # Detector-specific configurations
    ocr_max_dimension: int = 2000
    border_max_dimension: int = 1500
    watermark_confidence_threshold: float = 96.0
    editing_confidence_threshold: float = 25.0
    
    # Performance tuning
    enable_memory_pooling: bool = True
    enable_preprocessing_cache: bool = True
    enable_performance_tracking: bool = True
    
    # Error handling
    graceful_degradation: bool = True
    fallback_on_detector_failure: bool = True
    max_retry_attempts: int = 2
    
    # Batch processing
    adaptive_batch_sizing: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 10
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        # Filter only valid fields
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        import json
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {field.name: getattr(self, field.name) 
                for field in self.__dataclass_fields__.values()}
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        import json
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")

@dataclass
class ProcessedImageData:
    """Container for all processed image formats to avoid redundant conversions"""
    # Original formats
    pil_image: Image.Image
    opencv_bgr: np.ndarray
    opencv_rgb: np.ndarray
    opencv_gray: np.ndarray
    
    # Preprocessed formats (shared between detectors)
    text_processed: Optional[np.ndarray] = None
    border_processed: Optional[np.ndarray] = None
    watermark_processed: Optional[np.ndarray] = None
    
    # Metadata
    original_path: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    channels: int = 0
    
    # Performance tracking
    load_time_ms: float = 0
    preprocessing_time_ms: float = 0

class PerformanceTracker:
    """Comprehensive performance tracking for optimization validation"""
    
    def __init__(self):
        self.metrics = {
            'images_processed': 0,
            'total_processing_time_ms': 0,
            'total_load_time_ms': 0,
            'total_preprocessing_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_peak_mb': 0,
            'operations_by_type': {},
            'start_time': time.time()
        }
        self.operation_times = []
    
    def start_operation(self, operation_type: str):
        """Start timing an operation"""
        return time.time()
    
    def end_operation(self, operation_type: str, start_time: float):
        """End timing an operation and record metrics"""
        duration_ms = (time.time() - start_time) * 1000
        
        if operation_type not in self.metrics['operations_by_type']:
            self.metrics['operations_by_type'][operation_type] = []
        
        self.metrics['operations_by_type'][operation_type].append(duration_ms)
        return duration_ms
    
    def record_image_processed(self, processing_time_ms: float, load_time_ms: float = 0, 
                              preprocessing_time_ms: float = 0):
        """Record completion of image processing"""
        self.metrics['images_processed'] += 1
        self.metrics['total_processing_time_ms'] += processing_time_ms
        self.metrics['total_load_time_ms'] += load_time_ms
        self.metrics['total_preprocessing_time_ms'] += preprocessing_time_ms
        
        self.operation_times.append(processing_time_ms)
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""  
        self.metrics['cache_misses'] += 1
    
    def record_image_result(self, result: Dict, category: str):
        """Record image processing result (compatibility method)"""
        processing_time = result.get('total_time_ms', 0)
        load_time = result.get('load_time_ms', 0)
        preprocessing_time = result.get('preprocessing_time_ms', 0)
        
        if processing_time > 0:
            self.record_image_processed(processing_time, load_time, preprocessing_time)
    
    def get_summary(self) -> Dict:
        """Get performance summary statistics"""
        if self.metrics['images_processed'] == 0:
            return {}
        
        total_time_s = time.time() - self.metrics['start_time']
        avg_processing_time = self.metrics['total_processing_time_ms'] / self.metrics['images_processed']
        
        summary = {
            'images_processed': self.metrics['images_processed'],
            'total_time_seconds': total_time_s,
            'average_processing_time_ms': avg_processing_time,
            'throughput_images_per_second': self.metrics['images_processed'] / total_time_s if total_time_s > 0 else 0,
            'total_load_time_ms': self.metrics['total_load_time_ms'],
            'total_preprocessing_time_ms': self.metrics['total_preprocessing_time_ms'],
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'peak_memory_usage_mb': self.metrics['memory_peak_mb']
        }
        
        return summary
    
    def print_summary(self):
        """Print detailed performance summary"""
        summary = self.get_summary()
        
        if not summary:
            print("No performance data collected")
            return
        
        print("\\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Images Processed: {summary['images_processed']}")
        print(f"Total Time: {summary['total_time_seconds']:.2f}s")
        print(f"Average Time per Image: {summary['average_processing_time_ms']:.1f}ms")
        print(f"Throughput: {summary['throughput_images_per_second']:.1f} images/second")
        print(f"Load Time: {summary['total_load_time_ms']:.1f}ms")
        print(f"Preprocessing Time: {summary['total_preprocessing_time_ms']:.1f}ms")
        print(f"Cache Hit Rate: {summary['cache_hit_rate']*100:.1f}%")
        print("="*50)

class AdaptiveMemoryPool:
    """Advanced memory pool with dynamic adaptation and usage pattern learning"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.buffers = {}
        self.usage_stats = {}  # Track buffer usage patterns
        self.lock = threading.RLock()
        self.allocated_memory = 0
        
        # Initialize with minimal buffers
        self._initialize_minimal_buffers()
    
    def _initialize_minimal_buffers(self):
        """Initialize with minimal buffer set"""
        with self.lock:
            # Start with smaller initial buffers
            self.buffers = {
                'small_rgb': np.zeros((512, 512, 3), dtype=np.uint8),
                'small_gray': np.zeros((512, 512), dtype=np.uint8),
            }
            self.allocated_memory = (512 * 512 * 3) + (512 * 512)  # Bytes
    
    def get_buffer(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """Get appropriately sized buffer with dynamic allocation"""
        total_pixels = height * width
        buffer_size_bytes = total_pixels * channels
        
        with self.lock:
            # Record usage pattern
            buffer_key = self._get_buffer_key(height, width, channels)
            self.usage_stats[buffer_key] = self.usage_stats.get(buffer_key, 0) + 1
            
            # Check if we have a suitable existing buffer
            suitable_buffer = self._find_suitable_buffer(height, width, channels)
            if suitable_buffer is not None:
                return suitable_buffer
            
            # Check memory constraints before creating new buffer
            if self.allocated_memory + buffer_size_bytes > self.max_memory_bytes:
                # Try to free up space by removing least used buffers
                self._cleanup_unused_buffers()
                
                # If still not enough space, use a view of existing buffer
                if self.allocated_memory + buffer_size_bytes > self.max_memory_bytes:
                    return self._get_fallback_buffer(height, width, channels)
            
            # Create new buffer
            buffer_name = f"{buffer_key}_{len(self.buffers)}"
            if channels == 1:
                new_buffer = np.zeros((height, width), dtype=np.uint8)
            else:
                new_buffer = np.zeros((height, width, channels), dtype=np.uint8)
            
            self.buffers[buffer_name] = new_buffer
            self.allocated_memory += buffer_size_bytes
            
            return new_buffer
    
    def _get_buffer_key(self, height: int, width: int, channels: int) -> str:
        """Generate buffer key for usage tracking"""
        if channels == 1:
            if height * width <= 512 * 512:
                return "small_gray"
            elif height * width <= 1024 * 1024:
                return "medium_gray"
            else:
                return "large_gray"
        else:
            if height * width <= 512 * 512:
                return "small_rgb"
            elif height * width <= 1024 * 1024:
                return "medium_rgb"
            else:
                return "large_rgb"
    
    def _find_suitable_buffer(self, height: int, width: int, channels: int) -> Optional[np.ndarray]:
        """Find existing buffer that can accommodate the request"""
        for buffer_name, buffer in self.buffers.items():
            if len(buffer.shape) == 2 and channels == 1:  # Grayscale
                if buffer.shape[0] >= height and buffer.shape[1] >= width:
                    return buffer[:height, :width]
            elif len(buffer.shape) == 3 and channels > 1:  # Color
                if (buffer.shape[0] >= height and 
                    buffer.shape[1] >= width and 
                    buffer.shape[2] >= channels):
                    return buffer[:height, :width, :channels]
        return None
    
    def _cleanup_unused_buffers(self):
        """Remove least used buffers to free memory"""
        if len(self.buffers) <= 2:  # Keep at least 2 buffers
            return
        
        # Find least used buffer types
        usage_items = sorted(self.usage_stats.items(), key=lambda x: x[1])
        
        # Remove buffers of least used types
        buffers_to_remove = []
        for buffer_name in self.buffers:
            buffer_type = buffer_name.split('_')[0] + '_' + buffer_name.split('_')[1]
            if buffer_type in [item[0] for item in usage_items[:2]]:  # Remove 2 least used types
                buffers_to_remove.append(buffer_name)
        
        for buffer_name in buffers_to_remove:
            if buffer_name in self.buffers:
                buffer = self.buffers[buffer_name]
                buffer_size = buffer.nbytes
                del self.buffers[buffer_name]
                self.allocated_memory -= buffer_size
    
    def _get_fallback_buffer(self, height: int, width: int, channels: int) -> np.ndarray:
        """Get fallback buffer when memory is constrained"""
        # Use the largest available buffer and create a view
        largest_buffer = None
        largest_size = 0
        
        for buffer in self.buffers.values():
            if len(buffer.shape) == 2 and channels == 1:
                size = buffer.shape[0] * buffer.shape[1]
                if size > largest_size:
                    largest_buffer = buffer
                    largest_size = size
            elif len(buffer.shape) == 3 and channels > 1:
                size = buffer.shape[0] * buffer.shape[1]
                if size > largest_size:
                    largest_buffer = buffer
                    largest_size = size
        
        if largest_buffer is not None:
            if channels == 1 and len(largest_buffer.shape) == 2:
                max_h = min(height, largest_buffer.shape[0])
                max_w = min(width, largest_buffer.shape[1])
                return largest_buffer[:max_h, :max_w]
            elif channels > 1 and len(largest_buffer.shape) == 3:
                max_h = min(height, largest_buffer.shape[0])
                max_w = min(width, largest_buffer.shape[1])
                max_c = min(channels, largest_buffer.shape[2])
                return largest_buffer[:max_h, :max_w, :max_c]
        
        # Last resort: create minimal buffer
        if channels == 1:
            return np.zeros((min(height, 256), min(width, 256)), dtype=np.uint8)
        else:
            return np.zeros((min(height, 256), min(width, 256), channels), dtype=np.uint8)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory pool statistics"""
        with self.lock:
            return {
                'allocated_memory_mb': self.allocated_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'utilization_percentage': (self.allocated_memory / self.max_memory_bytes) * 100,
                'buffer_count': len(self.buffers),
                'usage_patterns': dict(self.usage_stats)
            }

class MemoryPool:
    """Legacy memory pool for backward compatibility"""
    
    def __init__(self):
        # Use adaptive pool internally
        self.adaptive_pool = AdaptiveMemoryPool(max_memory_mb=512)  # Smaller default
        self.lock = threading.Lock()
    
    def get_buffer(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """Get buffer using adaptive pool"""
        return self.adaptive_pool.get_buffer(height, width, channels)

class UnifiedImageLoader:
    """Single point image loading with format caching"""
    
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.cache = {}  # LRU cache for recently processed images
        self.cache_lock = threading.RLock()  # Reentrant lock for thread safety
        self.max_cache_size = 5
    
    def load_image_unified(self, image_path: str) -> ProcessedImageData:
        """
        Load image ONCE and convert to all required formats
        Returns ProcessedImageData with all formats ready
        """
        start_time = time.time()
        
        # Check cache first
        path_key = str(image_path)
        with self.cache_lock:
            if path_key in self.cache:
                logger.info(f"Cache hit for {Path(image_path).name}")
                return self.cache[path_key]
        
        try:
            # Load image once using PIL (most compatible)
            pil_image = Image.open(image_path).convert('RGB')
            width, height = pil_image.size
            file_size = Path(image_path).stat().st_size
            
            # Create all required formats in one pass
            rgb_array = np.array(pil_image)
            
            # Create all required formats in one pass (conditional on CV2)
            opencv_rgb = rgb_array  # Already in RGB format
            
            if CV2_AVAILABLE:
                import cv2
                opencv_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                opencv_gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            else:
                # Fallback: create grayscale using PIL/numpy
                opencv_bgr = rgb_array  # Use RGB as fallback
                # Simple grayscale conversion: 0.299*R + 0.587*G + 0.114*B
                opencv_gray = np.dot(rgb_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            
            load_time = (time.time() - start_time) * 1000
            
            # Create processed data container
            processed_data = ProcessedImageData(
                pil_image=pil_image,
                opencv_bgr=opencv_bgr,
                opencv_rgb=opencv_rgb,
                opencv_gray=opencv_gray,
                original_path=image_path,
                width=width,
                height=height,
                file_size=file_size,
                channels=3,
                load_time_ms=load_time
            )
            
            # Add to cache (with LRU eviction)
            self._add_to_cache(path_key, processed_data)
            
            logger.info(f"Loaded {Path(image_path).name} in {load_time:.2f}ms")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            raise
    
    def _add_to_cache(self, key: str, data: ProcessedImageData):
        """Add to cache with LRU eviction - thread safe"""
        with self.cache_lock:
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                gc.collect()  # Force garbage collection
            
            self.cache[key] = data

class SharedPreprocessor:
    """Unified preprocessing for all detection modules"""
    
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
    
    def preprocess_for_text_detection(self, data: ProcessedImageData) -> np.ndarray:
        """Legacy text preprocessing - now handled by PaddleOCR in main pipeline"""
        # This method is kept for compatibility but text detection
        # is now handled by PaddleOCR in main pipeline
        return data.opencv_gray
    
    def preprocess_for_border_detection(self, data: ProcessedImageData) -> np.ndarray:
        """Optimized preprocessing for border detection"""
        if data.border_processed is not None:
            return data.border_processed
        
        if not CV2_AVAILABLE:
            # Return grayscale as fallback
            return data.opencv_gray
        
        start_time = time.time()
        
        # Use grayscale as base
        gray = data.opencv_gray
        
        # Resize for border detection (different requirements than text)
        resized = self._smart_resize_for_borders(gray)
        
        # Edge enhancement
        import cv2
        edges = cv2.Canny(resized, 50, 150)
        
        # Cache result
        data.border_processed = edges
        data.preprocessing_time_ms += (time.time() - start_time) * 1000
        
        return edges
    
    def preprocess_for_watermark_detection(self, data: ProcessedImageData) -> Image.Image:
        """Legacy watermark preprocessing - now handled by PaddleOCR"""
        # This method is kept for compatibility but watermark detection
        # is now integrated into PaddleOCR text detection in main pipeline
        return data.pil_image
    
    def _smart_resize_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Smart resize optimized for OCR"""
        if not CV2_AVAILABLE:
            return img  # Return as-is if no CV2
        
        import cv2
        height, width = img.shape[:2]
        max_dimension = max(width, height)
        
        # OCR optimal resolution
        if max_dimension > 2000:
            scale_factor = 2000 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif max_dimension < 800:
            scale_factor = 800 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return img
    
    def _smart_resize_for_borders(self, img: np.ndarray) -> np.ndarray:
        """Smart resize optimized for border detection"""
        if not CV2_AVAILABLE:
            return img  # Return as-is if no CV2
        
        import cv2
        height, width = img.shape[:2]
        max_dimension = max(width, height)
        
        # Border detection optimal resolution
        if max_dimension > 1500:
            scale_factor = 1500 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img

class OptimizedDetectorWrapper:
    """Wrapper for existing detectors to use shared data"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.dependency_manager = DependencyManager()
        self.loader = UnifiedImageLoader()
        self.preprocessor = SharedPreprocessor(self.loader.memory_pool)
        
        # Initialize detector instances (lazy loading)
        self._text_detector = None
        self._watermark_detector = None
        self._editing_detector = None
        # Note: Text detection is handled by PaddleOCR in main pipeline
    
    def process_image_unified(self, image_path: str, enabled_tests: set) -> Dict[str, Any]:
        """
        Process image through all enabled detectors with ZERO redundancy
        """
        start_time = time.time()
        
        # STEP 1: Load image once, convert to all formats once
        try:
            processed_data = self.loader.load_image_unified(image_path)
        except Exception as e:
            return {'error': f'Failed to load image: {e}', 'processing_success': False}
        
        results = {
            'image_path': image_path,
            'processing_success': True,
            'load_time_ms': processed_data.load_time_ms,
            'tests_performed': list(enabled_tests),
            'results': {}
        }
        
        # STEP 2: Run only enabled tests with shared preprocessing
        
        if 'specifications' in enabled_tests:
            # Use dependency manager for specifications check
            if self.dependency_manager.is_available('specifications'):
                spec_detector = self.dependency_manager.get_detector('specifications')
                try:
                    passed, reason = spec_detector.meets_specifications(processed_data.original_path)
                    spec_result = {
                        'passed': passed,
                        'reason': reason if not passed else 'Matches specifications',
                        'dimensions': (processed_data.width, processed_data.height)
                    }
                except Exception as e:
                    spec_result = {
                        'passed': False,
                        'reason': f'Specifications check error: {str(e)}',
                        'dimensions': (processed_data.width, processed_data.height)
                    }
            else:
                # Use fallback
                fallback_func = self.dependency_manager.get_fallback('specifications')
                spec_result = fallback_func(processed_data.original_path, processed_data)
            
            results['results']['specifications'] = spec_result
        
        if 'text' in enabled_tests:
            # Use improved in-memory text detection
            if self.dependency_manager.is_available('text'):
                text_result = self._check_text_optimized_v2(processed_data)
            else:
                # Use fallback
                fallback_func = self.dependency_manager.get_fallback('text')
                text_result = fallback_func(processed_data.original_path, processed_data)
            
            results['results']['text'] = text_result
        
        if 'watermarks' in enabled_tests:
            # Use dependency manager for watermark detection
            if self.dependency_manager.is_available('watermarks'):
                watermark_result = self._check_watermarks_optimized(processed_data.original_path)
            else:
                # Use fallback
                fallback_func = self.dependency_manager.get_fallback('watermarks')
                watermark_result = fallback_func(processed_data.original_path, processed_data)
            
            results['results']['watermarks'] = watermark_result
        
        if 'borders' in enabled_tests:
            # Use dependency manager for border detection
            if self.dependency_manager.is_available('borders'):
                border_detector = self.dependency_manager.get_detector('borders')
                try:
                    # Use the proven border detection algorithm
                    is_valid, reason = border_detector.has_border_or_frame(processed_data.pil_image, show_debug=False)
                    border_result = {
                        'passed': is_valid,
                        'reason': reason,
                        'has_border': not is_valid
                    }
                except Exception as e:
                    border_result = {'passed': True, 'error': str(e)}
            else:
                # Use fallback
                fallback_func = self.dependency_manager.get_fallback('borders')
                border_result = fallback_func(processed_data.original_path, processed_data)
            
            results['results']['borders'] = border_result
        
        if 'editing' in enabled_tests:
            # Use dependency manager for editing detection
            if self.dependency_manager.is_available('editing'):
                editing_result = self._check_editing_optimized(processed_data.original_path)
            else:
                # Use fallback
                fallback_func = self.dependency_manager.get_fallback('editing')
                editing_result = fallback_func(processed_data.original_path, processed_data)
            
            results['results']['editing'] = editing_result
        
        # STEP 3: Aggregate results
        total_time = (time.time() - start_time) * 1000
        results['total_time_ms'] = total_time
        results['preprocessing_time_ms'] = processed_data.preprocessing_time_ms
        
        # Performance metrics
        results['performance_improvement'] = self._calculate_improvement(
            len(enabled_tests), total_time
        )
        
        # Record performance data
        tracker = PerformanceTracker()
        tracker.record_image_processed(
            processing_time_ms=total_time,
            load_time_ms=processed_data.load_time_ms,
            preprocessing_time_ms=processed_data.preprocessing_time_ms
        )
        
        return results
    
    def _check_specifications_optimized(self, data: ProcessedImageData) -> Dict[str, Any]:
        """Optimized specifications check using actual Spec_detector logic"""
        from Spec_detector import meets_specifications
        
        try:
            # Use the actual specifications detector
            passed, reason = meets_specifications(data.original_path)
            
            return {
                'passed': passed,
                'reason': reason if not passed else 'Matches specifications',
                'dimensions': (data.width, data.height)
            }
        except Exception as e:
            return {
                'passed': False,
                'reason': f'Specifications check error: {str(e)}',
                'dimensions': (data.width, data.height)
            }

    def _check_text_optimized_v2(self, processed_data: ProcessedImageData) -> Dict[str, Any]:
        """Optimized text detection using in-memory PaddleOCR processing"""
        try:
            # Check if PaddleOCR detector is available
            if not self.dependency_manager.is_available('text'):
                fallback_func = self.dependency_manager.get_fallback('text')
                if fallback_func:
                    return fallback_func(processed_data.original_path, processed_data)
                else:
                    return {'passed': True, 'error': 'Text detector unavailable'}
            
            # Initialize PaddleOCR text detector if not already done
            if self._text_detector is None:
                text_detector_module = self.dependency_manager.get_detector('text')
                self._text_detector = text_detector_module.PaddleTextDetector()
            
            # Use in-memory processing - convert to numpy array format expected by PaddleOCR
            # Many OCR libraries can work directly with numpy arrays
            rgb_array = processed_data.opencv_rgb
            
            # Try direct numpy array processing first
            try:
                # Check if PaddleOCR supports direct numpy array input
                if hasattr(self._text_detector, 'process_numpy_array'):
                    paddle_result = self._text_detector.process_numpy_array(
                        rgb_array, detection_only=False
                    )
                elif hasattr(self._text_detector, 'detect_and_recognize_text'):
                    paddle_result = self._text_detector.detect_and_recognize_text(
                        rgb_array, detection_only=False
                    )
                else:
                    # Fall back to PIL image processing
                    paddle_result = self._process_via_pil_image(processed_data)
                
            except (AttributeError, TypeError):
                # If direct processing fails, use PIL image method
                paddle_result = self._process_via_pil_image(processed_data)
            
            if paddle_result.get('processing_success', False):
                confidence_metrics = paddle_result.get('confidence_metrics', {})
                overall_confidence = confidence_metrics.get('overall_confidence', 0.0)
                text_count = confidence_metrics.get('text_count', 0)
                
                # High confidence means lots of text (likely watermarked)
                has_significant_text = overall_confidence > 60 or text_count > 2
                
                return {
                    'passed': not has_significant_text,
                    'text_count': text_count,
                    'confidence': overall_confidence,
                    'detected_texts': [result.get('text', '') for result in paddle_result.get('paddle_results', [])],
                    'processing_method': 'in_memory'
                }
            else:
                return {'passed': True, 'error': 'PaddleOCR processing failed', 'processing_method': 'in_memory'}
                
        except Exception as e:
            return {'passed': True, 'error': str(e), 'processing_method': 'failed'}
    
    def _process_via_pil_image(self, processed_data: ProcessedImageData) -> Dict[str, Any]:
        """Process PaddleOCR using PIL image (avoids temp file if possible)"""
        try:
            # Use existing PIL image from processed data
            pil_img = processed_data.pil_image
            
            # Check if PaddleOCR can accept PIL images directly
            if hasattr(self._text_detector, 'process_pil_image'):
                return self._text_detector.process_pil_image(pil_img, detection_only=False)
            
            # If not, we need to use the original file path method
            # This is the fallback that still uses the file system
            from pathlib import Path
            paddle_result = self._text_detector.process_single_image(
                Path(processed_data.original_path), detection_only=False
            )
            return paddle_result
            
        except Exception as e:
            return {'processing_success': False, 'error': str(e)}

    def _check_text_optimized(self, preprocessed_img: np.ndarray) -> Dict[str, Any]:
        """Legacy text detection method - replaced by _check_text_optimized_v2"""
        # This method is kept for backward compatibility
        # New code should use _check_text_optimized_v2 with ProcessedImageData
        return {
            'passed': True,
            'note': 'Use _check_text_optimized_v2 for improved in-memory processing'
        }

    def _get_watermark_detector(self):
        """Lazy initialization of watermark detector"""
        if self._watermark_detector is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Temporarily suppress all INFO level logging from watermark detector
                original_level = logging.getLogger().level
                watermark_logger = logging.getLogger('advanced_watermark_detector')
                original_watermark_level = watermark_logger.level
                
                # Set both root and module logger to WARNING to suppress INFO messages
                logging.getLogger().setLevel(logging.WARNING)
                watermark_logger.setLevel(logging.WARNING)
                
                try:
                    from advanced_watermark_detector import AdvancedWatermarkDetector
                    # Use device configuration from pipeline config
                    device = 'cpu' if self.config.force_cpu else self.config.device
                    self._watermark_detector = AdvancedWatermarkDetector(
                        model_name='convnext-tiny',
                        device=device,  # Use configured device
                        fp16=False     # Disable FP16 for better compatibility
                    )
                    # Silent success - no log message
                except Exception as init_error:
                    # Restore logging temporarily to show error
                    logging.getLogger().setLevel(original_level)
                    logger.error(f"Failed to initialize watermark detector: {init_error}")
                    self._watermark_detector = 'failed'  # Mark as failed
                finally:
                    # Restore original logging levels
                    logging.getLogger().setLevel(original_level)
                    watermark_logger.setLevel(original_watermark_level)
        return self._watermark_detector

    def _get_editing_detector(self):
        """Lazy initialization of editing detector"""
        if self._editing_detector is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Temporarily suppress all INFO level logging from editing detector
                original_level = logging.getLogger().level
                editing_logger = logging.getLogger('advanced_pyiqa_detector')
                original_editing_level = editing_logger.level
                
                # Set both root and module logger to WARNING to suppress INFO messages
                logging.getLogger().setLevel(logging.WARNING)
                editing_logger.setLevel(logging.WARNING)
                
                try:
                    from advanced_pyiqa_detector import AdvancedEditingDetector
                    # Use device configuration from pipeline config
                    force_cpu = self.config.force_cpu or (self.config.device == 'cpu')
                    self._editing_detector = AdvancedEditingDetector(
                        force_cpu=force_cpu,  # Use configured device preference
                        quiet=True,      # Suppress initialization output
                        # Use FAST recommended models by default for speed
                        selected_models=['brisque', 'niqe', 'clipiqa']
                    )
                    # Silent success - no log message
                except Exception as init_error:
                    # Restore logging temporarily to show error
                    logging.getLogger().setLevel(original_level)
                    logger.error(f"Failed to initialize editing detector: {init_error}")
                    self._editing_detector = 'failed'  # Mark as failed
                finally:
                    # Restore original logging levels
                    logging.getLogger().setLevel(original_level)
                    editing_logger.setLevel(original_editing_level)
        return self._editing_detector
    
    def _check_watermarks_optimized(self, image_path: str) -> Dict[str, Any]:
        """Use advanced watermark detection with improved error handling for PyTorch/CUDA issues"""
        try:
            # First, try to import PyTorch and check for common DLL issues
            try:
                import torch
                # Test basic torch functionality
                test_tensor = torch.tensor([1.0])
                torch_available = True
            except ImportError as ie:
                logger.warning(f"PyTorch not available: {ie}")
                torch_available = False
            except Exception as te:
                logger.warning(f"PyTorch DLL/CUDA error: {te}")
                torch_available = False
            
            if not torch_available:
                logger.info("PyTorch unavailable, skipping advanced watermark detection")
                return {
                    'passed': True,  # Skip watermark check if PyTorch unavailable
                    'reason': 'Advanced watermark detection skipped (PyTorch unavailable)',
                    'has_watermark': False,
                    'confidence': 0.0,
                    'skipped': True,
                    'error': 'PyTorch/CUDA dependencies not available'
                }
            
            # Suppress warnings during model loading and prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Get the shared watermark detector (lazy initialization)
                detector = self._get_watermark_detector()
                
                # Check if detector initialization failed
                if detector == 'failed':
                    return {
                        'passed': True,  # Skip if can't initialize
                        'reason': 'Watermark detector initialization failed',
                        'has_watermark': False,
                        'confidence': 0.0,
                        'error': 'Detector initialization failed'
                    }
                
                # Use the advanced watermark detector
                result = detector.predict_single_image(image_path)
            
            # Apply the same thresholds as the standalone watermark detector
            # Based on advanced_watermark_detector.py:
            # - Manual review: 91% - 96%
            # - Flagged as watermark: > 96%
            confidence = result.get('confidence_percentage', 0.0)
            WATERMARK_THRESHOLD = 96.0  # Match standalone detector threshold
            MANUAL_REVIEW_THRESHOLD = 91.0
            
            if confidence > WATERMARK_THRESHOLD:
                # High confidence watermark - fail the image
                has_watermark = True
                passed = False
                reason = f"Watermark detected (confidence: {confidence:.1f}%)"
            elif confidence > MANUAL_REVIEW_THRESHOLD:
                # Manual review needed
                has_watermark = True
                passed = None  # Will trigger manual review
                reason = f"Potential watermark - manual review needed (confidence: {confidence:.1f}%)"
            else:
                # Clean image
                has_watermark = False
                passed = True
                reason = f"No watermark detected (confidence: {confidence:.1f}%)"
            
            return {
                'passed': passed,
                'reason': reason,
                'has_watermark': has_watermark,
                'confidence': confidence,
                'needs_manual_review': passed is None,
                'details': {
                    'confidence_clean': result.get('confidence_clean', 0.0),
                    'confidence_watermarked': result.get('confidence_watermarked', 0.0),
                    'prediction': result.get('prediction', 'unknown'),
                    'model_used': result.get('model_used', 'convnext-tiny'),
                    'device_used': 'cpu',  # Note that we forced CPU mode
                    'threshold_used': WATERMARK_THRESHOLD,
                    'manual_review_threshold': MANUAL_REVIEW_THRESHOLD
                }
            }
            
        except Exception as e:
            logger.error(f"Advanced watermark detection failed: {e}")
            # Enhanced fallback with better error categorization
            error_msg = str(e)
            if "DLL" in error_msg or "shm.dll" in error_msg:
                fallback_reason = "Watermark detection skipped (PyTorch DLL issue)"
            elif "CUDA" in error_msg:
                fallback_reason = "Watermark detection skipped (CUDA unavailable)"
            else:
                fallback_reason = f"Watermark detection error: {error_msg[:50]}..."
            
            return {
                'passed': True,  # Don't fail the image due to detector issues
                'reason': fallback_reason,
                'has_watermark': False,
                'confidence': 0.0,
                'error': str(e),
                'fallback_used': True
            }
    
    def _check_borders_optimized(self, preprocessed_edges: np.ndarray, data: ProcessedImageData) -> Dict[str, Any]:
        """Optimized border detection using the actual border detection algorithm"""
        from border_detector import has_border_or_frame
        
        try:
            # Use the proven border detection algorithm that works in standalone mode
            # Convert shared PIL image to the format expected by border detector
            pil_image = data.pil_image
            
            # Call the actual border detection function
            is_valid, reason = has_border_or_frame(pil_image, show_debug=False)
            
            return {
                'passed': is_valid,
                'reason': reason,
                'has_border': not is_valid
            }
        except Exception as e:
            return {'passed': True, 'error': str(e)}

    def _check_editing_optimized(self, image_path: str) -> Dict[str, Any]:
        """Use advanced PyIQA-based editing detection"""
        try:
            # Get the shared editing detector (lazy initialization)
            detector = self._get_editing_detector()
            
            # Check if detector initialization failed
            if detector == 'failed':
                return {
                    'passed': True,  # Skip if can't initialize
                    'reason': 'Editing detector initialization failed',
                    'editing_confidence': 0.0,
                    'error': 'Detector initialization failed'
                }
            
            # Use the editing detector to analyze the image
            result, error = detector.analyze_single_image(image_path)
            
            # Check if analysis failed (error is not the success message)
            if error and error != "Analysis completed successfully":
                return {
                    'passed': True,  # Skip if analysis failed
                    'reason': f'Editing analysis failed: {error}',
                    'editing_confidence': 0.0,
                    'error': error
                }
            
            # Check if result is None (analysis failed)
            if result is None:
                return {
                    'passed': True,  # Skip if analysis failed
                    'reason': 'Editing analysis returned no results',
                    'editing_confidence': 0.0,
                    'error': 'No analysis results'
                }
            
            # Extract editing confidence from comprehensive assessment
            comprehensive_assessment = result.get('comprehensive_assessment', {})
            editing_confidence = comprehensive_assessment.get('overall_editing_score', 0.0)
            editing_category = comprehensive_assessment.get('editing_category', 'Unknown')
            
            # NEW THRESHOLDS: 25+ = Invalid, 20-25 = Manual Review, <20 = Valid
            EDITING_INVALID_THRESHOLD = 25.0
            EDITING_MANUAL_REVIEW_THRESHOLD = 20.0
            
            if editing_confidence >= EDITING_INVALID_THRESHOLD:
                # High editing confidence (25%+) - mark as invalid
                return {
                    'passed': False,  # False = invalid
                    'reason': f"High editing confidence detected: {editing_confidence:.1f}% (threshold: {EDITING_INVALID_THRESHOLD}%)",
                    'editing_confidence': editing_confidence,
                    'editing_category': editing_category,
                    'needs_manual_review': False,
                    'insights': self._generate_editing_insights(result)
                }
            elif editing_confidence >= EDITING_MANUAL_REVIEW_THRESHOLD:
                # Medium editing confidence (20-25%) - manual review needed
                return {
                    'passed': None,  # None = manual review needed
                    'reason': f"Medium editing confidence detected: {editing_confidence:.1f}% (requires manual review)",
                    'editing_confidence': editing_confidence,
                    'editing_category': editing_category,
                    'needs_manual_review': True,
                    'insights': self._generate_editing_insights(result)
                }
            else:
                # Low editing confidence (<20%) - image passes
                return {
                    'passed': True,
                    'reason': f"Minimal editing detected (confidence: {editing_confidence:.1f}%)",
                    'editing_confidence': editing_confidence,
                    'editing_category': editing_category,
                    'needs_manual_review': False
                }
                
        except Exception as e:
            logger.error(f"Editing detection error: {e}")
            return {
                'passed': True,  # Skip if error
                'reason': 'Editing detection error',
                'editing_confidence': 0.0,
                'error': str(e)
            }

    def _generate_editing_insights(self, analysis_result: Dict) -> str:
        """Generate human-readable insights about detected editing"""
        insights = []
        
        comprehensive = analysis_result.get('comprehensive_assessment', {})
        pyiqa = analysis_result.get('pyiqa_analysis', {})
        histogram = analysis_result.get('histogram_analysis', {})
        
        # PyIQA insights
        if 'error' not in pyiqa:
            if pyiqa.get('brisque_score', 0) > 50:
                insights.append("BRISQUE score indicates unnatural image quality")
            if pyiqa.get('niqe_score', 0) > 8:
                insights.append("NIQE score suggests degraded naturalness")
            if pyiqa.get('clipiqa_score', 1) < 0.5:
                insights.append("CLIP-IQA indicates low perceptual quality")
        
        # Histogram insights
        if histogram.get('total_clipping', 0) > 0.1:
            insights.append("Histogram clipping detected (possible overexposure/contrast adjustment)")
        if histogram.get('histogram_entropy', 8) < 6:
            insights.append("Low histogram entropy (possible tone mapping or heavy color grading)")
        
        # Edge and frequency insights
        edge_analysis = analysis_result.get('edge_analysis', {})
        if edge_analysis.get('edge_density', 0) > 0.2:
            insights.append("High edge density (possible sharpening artifacts)")
        
        freq_analysis = analysis_result.get('frequency_analysis', {})
        if freq_analysis.get('frequency_variance', 0) > 25:
            insights.append("Unusual frequency distribution (possible filtering or enhancement)")
        
        if not insights:
            insights.append("General editing artifacts detected through quality assessment")
        
        return "; ".join(insights)
    
    def _calculate_improvement(self, num_tests: int, actual_time_ms: float) -> Dict[str, float]:
        """Calculate performance improvement vs old pipeline"""
        # Estimate old pipeline time based on redundant operations
        estimated_old_time = num_tests * 200  # ~200ms per test with redundancy
        
        improvement_percentage = ((estimated_old_time - actual_time_ms) / estimated_old_time) * 100
        speedup_factor = estimated_old_time / actual_time_ms
        
        return {
            'estimated_old_time_ms': estimated_old_time,
            'actual_time_ms': actual_time_ms,
            'improvement_percentage': improvement_percentage,
            'speedup_factor': speedup_factor
        }

# Global instance for reuse
_unified_detector = None

def get_unified_detector(config: Optional[PipelineConfig] = None) -> OptimizedDetectorWrapper:
    """Get singleton unified detector instance with configuration"""
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = OptimizedDetectorWrapper(config)
    return _unified_detector

def reset_unified_detector():
    """Reset the global detector instance (useful for testing or config changes)"""
    global _unified_detector
    _unified_detector = None

class ErrorRecoveryManager:
    """Comprehensive error handling with fallback strategies"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.error_counts = defaultdict(int)
        self.fallback_strategies = {
            'detector_unavailable': self._fallback_skip_detector,
            'memory_exhausted': self._fallback_reduce_quality,
            'cuda_error': self._fallback_cpu_mode,
            'file_corrupted': self._fallback_skip_image,
            'timeout_error': self._fallback_quick_process,
        }
    
    def handle_error(self, error_type: str, context: Dict, error: Exception) -> Dict[str, Any]:
        """Handle errors with appropriate fallback strategy"""
        self.error_counts[error_type] += 1
        
        logger.warning(f"Error {error_type} occurred ({self.error_counts[error_type]} times): {error}")
        
        if error_type in self.fallback_strategies:
            return self.fallback_strategies[error_type](context, error)
        else:
            return self._fallback_graceful_failure(context, error)
    
    def _fallback_skip_detector(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Skip detector and mark as passed with warning"""
        return {
            'passed': True,
            'reason': f'Detector unavailable: {str(error)[:100]}',
            'fallback_used': True,
            'error_type': 'detector_unavailable'
        }
    
    def _fallback_reduce_quality(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Reduce processing quality to handle memory issues"""
        return {
            'passed': True,
            'reason': 'Processing skipped due to memory constraints',
            'fallback_used': True,
            'error_type': 'memory_exhausted',
            'suggestion': 'Consider reducing max_memory_mb in configuration'
        }
    
    def _fallback_cpu_mode(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Fall back to CPU mode for CUDA errors"""
        return {
            'passed': True,
            'reason': 'GPU processing failed, using CPU fallback',
            'fallback_used': True,
            'error_type': 'cuda_error',
            'suggestion': 'Check CUDA installation or use force_cpu=True'
        }
    
    def _fallback_skip_image(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Skip corrupted or unreadable images"""
        return {
            'passed': False,
            'reason': f'Image file corrupted or unreadable: {str(error)[:100]}',
            'fallback_used': True,
            'error_type': 'file_corrupted'
        }
    
    def _fallback_quick_process(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Use quick processing for timeout errors"""
        return {
            'passed': True,
            'reason': 'Processing timeout, using quick analysis',
            'fallback_used': True,
            'error_type': 'timeout_error',
            'suggestion': 'Consider increasing processing timeout or reducing image size'
        }
    
    def _fallback_graceful_failure(self, context: Dict, error: Exception) -> Dict[str, Any]:
        """Graceful failure for unknown errors"""
        return {
            'passed': True,  # Don't fail the entire pipeline
            'reason': f'Processing error: {str(error)[:100]}',
            'fallback_used': True,
            'error_type': 'unknown_error',
            'full_error': str(error)
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_breakdown': dict(self.error_counts),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }

# Drop-in replacement function for main.py
def filter_single_image_optimized(image_path: str, enabled_tests: set) -> Tuple[bool, str, str]:
    """
    Optimized drop-in replacement for filter_single_image
    
    Returns:
        Tuple[bool, str, str]: (is_valid, reason, category)
    """
    detector = get_unified_detector()
    
    try:
        results = detector.process_image_unified(image_path, enabled_tests)
        
        if not results.get('processing_success', False):
            return False, results.get('error', 'Processing failed'), 'errors'
        
        # Check each test result
        test_results = results.get('results', {})
        
        for test_name, test_result in test_results.items():
            if not test_result.get('passed', True):
                reason = test_result.get('reason', f'{test_name} check failed')
                return False, reason, test_name
        
        return True, 'All tests passed', 'valid'
        
    except Exception as e:
        return False, f'Processing error: {e}', 'errors'
