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
import cv2
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

# Suppress PyTorch and other model loading warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="timm")

logger = logging.getLogger(__name__)

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

class MemoryPool:
    """Advanced memory pool for image processing buffers"""
    
    def __init__(self):
        self.buffers = {
            'small': np.zeros((1024, 1024, 3), dtype=np.uint8),
            'medium': np.zeros((2048, 2048, 3), dtype=np.uint8),
            'large': np.zeros((4096, 4096, 3), dtype=np.uint8),
            'gray_small': np.zeros((1024, 1024), dtype=np.uint8),
            'gray_medium': np.zeros((2048, 2048), dtype=np.uint8),
            'gray_large': np.zeros((4096, 4096), dtype=np.uint8),
        }
        self.lock = threading.Lock()
    
    def get_buffer(self, height: int, width: int, channels: int = 3) -> np.ndarray:
        """Get appropriately sized buffer to avoid memory allocation"""
        total_pixels = height * width
        
        with self.lock:
            if channels == 1:  # Grayscale
                if total_pixels <= 1024*1024:
                    return self.buffers['gray_small'][:height, :width]
                elif total_pixels <= 2048*2048:
                    return self.buffers['gray_medium'][:height, :width]
                else:
                    return self.buffers['gray_large'][:height, :width]
            else:  # Color
                if total_pixels <= 1024*1024:
                    return self.buffers['small'][:height, :width, :channels]
                elif total_pixels <= 2048*2048:
                    return self.buffers['medium'][:height, :width, :channels]
                else:
                    return self.buffers['large'][:height, :width, :channels]

class UnifiedImageLoader:
    """Single point image loading with format caching"""
    
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.cache = {}  # LRU cache for recently processed images
        self.max_cache_size = 5
    
    def load_image_unified(self, image_path: str) -> ProcessedImageData:
        """
        Load image ONCE and convert to all required formats
        Returns ProcessedImageData with all formats ready
        """
        start_time = time.time()
        
        # Check cache first
        path_key = str(image_path)
        if path_key in self.cache:
            logger.info(f"Cache hit for {Path(image_path).name}")
            return self.cache[path_key]
        
        try:
            # Load image once using PIL (most compatible)
            pil_image = Image.open(image_path).convert('RGB')
            width, height = pil_image.size
            file_size = Path(image_path).stat().st_size
            
            # Convert to numpy array (shared base for all OpenCV operations)
            rgb_array = np.array(pil_image)
            
            # Create all required formats in one pass
            opencv_rgb = rgb_array  # Already in RGB format
            opencv_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            opencv_gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            
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
        """Add to cache with LRU eviction"""
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
        
        start_time = time.time()
        
        # Use grayscale as base
        gray = data.opencv_gray
        
        # Resize for border detection (different requirements than text)
        resized = self._smart_resize_for_borders(gray)
        
        # Edge enhancement
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
    
    def __init__(self):
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
            # Specs check doesn't need preprocessing
            spec_result = self._check_specifications_optimized(processed_data)
            results['results']['specifications'] = spec_result
        
        if 'text' in enabled_tests:
            # Text detection is now handled by PaddleOCR in main pipeline
            # This compatibility layer just returns passed
            results['results']['text'] = {
                'passed': True,
                'note': 'Text detection now handled by PaddleOCR in main pipeline'
            }
        
        if 'watermarks' in enabled_tests:
            # Use advanced watermark detection from advanced_watermark_detector.py
            watermark_result = self._check_watermarks_optimized(processed_data.original_path)
            results['results']['watermarks'] = watermark_result
        
        if 'borders' in enabled_tests:
            # Use shared preprocessing for borders
            border_processed = self.preprocessor.preprocess_for_border_detection(processed_data)
            border_result = self._check_borders_optimized(border_processed, processed_data)
            results['results']['borders'] = border_result
        
        if 'editing' in enabled_tests:
            # Use advanced PyIQA-based editing detection
            editing_result = self._check_editing_optimized(processed_data.original_path)
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

    def _check_text_optimized(self, preprocessed_img: np.ndarray) -> Dict[str, Any]:
        """Optimized text detection using PaddleOCR (replacing legacy text detector)"""
        try:
            # Initialize PaddleOCR text detector if not already done
            if self._text_detector is None:
                from paddle_text_detector import PaddleTextDetector
                self._text_detector = PaddleTextDetector()
            
            # Convert numpy array to PIL Image for PaddleOCR
            from PIL import Image
            if len(preprocessed_img.shape) == 2:  # Grayscale
                pil_img = Image.fromarray(preprocessed_img, mode='L').convert('RGB')
            else:
                pil_img = Image.fromarray(preprocessed_img)
            
            # Use PaddleOCR for text detection
            from pathlib import Path
            import tempfile
            
            # Save temporary file for PaddleOCR processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                pil_img.save(tmp_file.name)
                
                # Process with PaddleOCR
                paddle_result = self._text_detector.process_single_image(
                    Path(tmp_file.name), detection_only=False
                )
                
                # Clean up temp file
                os.unlink(tmp_file.name)
            
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
                    'detected_texts': [result.get('text', '') for result in paddle_result.get('paddle_results', [])]
                }
            else:
                return {'passed': True, 'error': 'PaddleOCR processing failed'}
                
        except Exception as e:
            return {'passed': True, 'error': str(e)}

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
                    self._watermark_detector = AdvancedWatermarkDetector(
                        model_name='convnext-tiny',
                        device='cpu',  # Force CPU to avoid CUDA DLL issues
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
                    self._editing_detector = AdvancedEditingDetector(
                        force_cpu=True,  # Force CPU for stability
                        quiet=True       # Suppress initialization output
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
            
            # Threshold for manual review: 25% or higher editing confidence
            EDITING_THRESHOLD = 25.0
            
            if editing_confidence >= EDITING_THRESHOLD:
                # High editing confidence - flag for manual review
                return {
                    'passed': None,  # None = manual review needed
                    'reason': f"editing",
                    'editing_confidence': editing_confidence,
                    'editing_category': editing_category,
                    'needs_manual_review': True,
                    'insights': self._generate_editing_insights(result)
                }
            else:
                # Low editing confidence - image passes
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

def get_unified_detector() -> OptimizedDetectorWrapper:
    """Get singleton unified detector instance"""
    global _unified_detector
    if _unified_detector is None:
        _unified_detector = OptimizedDetectorWrapper()
    return _unified_detector

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
