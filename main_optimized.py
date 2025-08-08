"""
OPTIMIZED MAIN CONTROLLER - NEW ARCHITECTURE WITH PADDLEOCR

This is the completely restructured main photo filtering system.
Key features:
1. Integrated PaddleOCR text detection (replaces watermark detection)
2. New organized output structure: valid, invalid (with subcategories), manual review needed
3. Smart routing based on text detection confidence and validation results
4. Unified image loading and processing
5. Sequential processing with complete validation
6. High-performance text detection using PaddleOCR's DB model
7. EDITING DETECTION BEHAVIOR: Images with confidence ≥25% are moved to invalid folder,
   20-25% confidence moved to manual review folder, <20% considered valid.
"""

# Set critical environment variables BEFORE any imports
import os
os.environ['GLOG_minloglevel'] = '3'  # Only FATAL messages
os.environ['FLAGS_print_model_stats'] = '0'
os.environ['FLAGS_enable_parallel_graph'] = '0'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0'
os.environ['FLAGS_allocator_strategy'] = 'naive_best_fit'

# Set logging level to suppress all INFO messages
import logging
logging.getLogger().setLevel(logging.WARNING)

import shutil
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Set, Optional

# Suppress PyTorch warnings about torch.load and other verbose warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="timm")

# Suppress PaddleOCR warnings
warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
warnings.filterwarnings("ignore", category=UserWarning, message=".*ccache.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*recompiling.*")
warnings.filterwarnings("ignore", message=".*Could not find files for the given pattern.*")

# Custom logging filter to block specific messages
class PaddleLogFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these patterns
        blocked_patterns = [
            "Could not find files for the given pattern",
            "INFO: Could not find files",
            "No ccache found"
        ]
        
        message = record.getMessage()
        for pattern in blocked_patterns:
            if pattern in message:
                return False
        return True

# Apply custom filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(PaddleLogFilter())

# Set PaddleOCR logging to reduce noise
import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress PaddlePaddle INFO logs
os.environ['FLAGS_print_model_stats'] = '0'
os.environ['FLAGS_enable_parallel_graph'] = '0'

# Additional PaddlePaddle environment variables to suppress logging
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0'

# Redirect stdout temporarily during PaddleOCR operations
import contextlib
import sys
from io import StringIO

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

# Import optimized pipeline
from optimized_pipeline import (
    get_unified_detector, 
    filter_single_image_optimized,
    OptimizedDetectorWrapper
)

# Lazy import for heavy dependencies
def get_paddle_text_detector():
    """Lazy import PaddleOCR text detector to speed up startup"""
    try:
        from paddle_text_detector import PaddleTextDetector
        return PaddleTextDetector
    except ImportError:
        return None

# ===== CENTRALIZED UTILITIES (Integrated to eliminate redundancy) =====

class ImageProcessor:
    """Centralized image processing to eliminate redundancy across detectors"""
    
    def __init__(self):
        self.cache = {}  # Simple cache for recently processed images
        self.max_cache_size = 10  # Increased cache size for better performance
        self._pil_image = None  # Cached PIL Image reference
    
    def load_and_validate_image(self, image_path: str) -> Tuple[bool, str, Optional[Dict]]:
        """
        Load and validate image, return validation status and metadata.
        OPTIMIZED: Uses caching and early validation to avoid redundant operations.
        
        Returns:
            Tuple[bool, str, Optional[Dict]]: (is_valid, reason, metadata)
        """
        # Check cache first
        if image_path in self.cache:
            cached = self.cache[image_path]
            return cached['is_valid'], cached['reason'], cached['metadata']
        
        try:
            # Fast pre-checks before expensive PIL operation
            if not os.path.exists(image_path):
                result = (False, "File does not exist", None)
                self._cache_result(image_path, result)
                return result
            
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext not in SUPPORTED_FORMATS:
                result = (False, f"Unsupported format: {file_ext}", None)
                self._cache_result(image_path, result)
                return result
            
            # Quick file size check (avoid loading huge corrupt files)
            file_size = os.path.getsize(image_path)
            if file_size < 100:  # Less than 100 bytes is likely corrupt
                result = (False, "File too small (likely corrupt)", None)
                self._cache_result(image_path, result)
                return result
            
            # Load image using PIL (most compatible) - OPTIMIZED: Load once, cache reference
            from PIL import Image
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_info = img.format or file_ext.upper().lstrip('.')
                    
                    # Basic validation - OPTIMIZED: Early exit on invalid dimensions
                    if width == 0 or height == 0:
                        result = (False, "Invalid dimensions", None)
                        self._cache_result(image_path, result)
                        return result
                    
                    # OPTIMIZED: Skip minimum size check here, do it in main processing
                    metadata = {
                        'width': width,
                        'height': height,
                        'format': format_info,
                        'file_size': file_size,
                        'path': image_path,
                        'filename': os.path.basename(image_path)
                    }
                    
                    result = (True, "Valid image", metadata)
                    self._cache_result(image_path, result)
                    return result
            except Exception as pil_error:
                result = (False, f"PIL cannot read image: {str(pil_error)}", None)
                self._cache_result(image_path, result)
                return result
                
        except Exception as e:
            result = (False, f"Image validation error: {str(e)}", None)
            self._cache_result(image_path, result)
            return result
    
    def _cache_result(self, image_path: str, result: Tuple):
        """Cache validation result to avoid redundant processing"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[image_path] = {
            'is_valid': result[0],
            'reason': result[1], 
            'metadata': result[2]
        }

class OutputManager:
    """Centralized output management to eliminate redundant file operations"""
    
    def __init__(self, base_output_dir: str = "Results"):
        self.base_output_dir = base_output_dir
        self.structure = {
            'valid': os.path.join(base_output_dir, 'valid'),
            'invalid': os.path.join(base_output_dir, 'invalid'),
            'manualreview': os.path.join(base_output_dir, 'manualreview'),
            'logs': os.path.join(base_output_dir, 'logs')
        }
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create output directory structure"""
        for dir_path in self.structure.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def copy_to_category(self, source_path: str, category: str, reason: str = "") -> str:
        """
        Copy file to appropriate category directory.
        
        Returns:
            str: Target path where file was copied
        """
        if category not in self.structure:
            category = 'invalid'  # Fallback
            
        target_dir = self.structure[category]
        filename = os.path.basename(source_path)
        target_path = os.path.join(target_dir, filename)
        
        # Handle filename conflicts
        counter = 1
        original_target = target_path
        while os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(source_path, target_path)
        return target_path

# Global instances to eliminate redundancy
_global_processor = None
_global_output_manager = None

def get_processor() -> ImageProcessor:
    """Get global image processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = ImageProcessor()
    return _global_processor

def get_output_manager() -> OutputManager:
    """Get global output manager instance"""
    global _global_output_manager
    if _global_output_manager is None:
        _global_output_manager = OutputManager(OUTPUT_DIR)
    return _global_output_manager

# ===== END CENTRALIZED UTILITIES =====

# Import configuration - Simplified (no external config file needed)
# Only supported formats are defined inline
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}

# Configure paths - Updated for new structure
PHOTOS_DIR = "photos4testing"  # Input folder
OUTPUT_DIR = "Results"  # Main output folder

# Define the simplified directory structure
OUTPUT_STRUCTURE = {
    'valid': 'valid',
    'invalid': 'invalid',
    'manual_review': 'manualreview',
    'logs': 'logs'
}

def check_pytorch_environment():
    """Check PyTorch environment and return compatibility info."""
    pytorch_info = {
        'available': False,
        'cuda_available': False,
        'version': None,
        'error': None,
        'recommendation': 'cpu'
    }
    
    try:
        import torch
        pytorch_info['available'] = True
        pytorch_info['version'] = torch.__version__
        
        # Test basic tensor operations to catch DLL issues early
        test_tensor = torch.tensor([1.0, 2.0])
        _ = test_tensor.sum()
        
        # Check CUDA availability (but don't test it to avoid DLL issues)
        pytorch_info['cuda_available'] = torch.cuda.is_available()
        
        if pytorch_info['cuda_available']:
            pytorch_info['recommendation'] = 'cuda'
        else:
            pytorch_info['recommendation'] = 'cpu'
            
    except ImportError as e:
        pytorch_info['error'] = f"PyTorch not installed: {e}"
    except Exception as e:
        pytorch_info['error'] = f"PyTorch DLL/environment error: {e}"
        if "shm.dll" in str(e) or "DLL" in str(e):
            pytorch_info['error'] += " (Try reinstalling PyTorch with: pip install torch --force-reinstall)"
    
    return pytorch_info

def setup_logging():
    """Set up minimal logging - suppress all INFO level messages for clean output."""
    # Set root logger to WARNING level to suppress INFO messages
    logging.getLogger().setLevel(logging.WARNING)
    
    # We only need a logger object for the summary report, no file logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Add a null handler to prevent any unwanted logging output
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger

def create_output_directories():
    """Create the simplified output directory structure."""
    directories = [
        OUTPUT_DIR,
        os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['valid']),
        os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['invalid']),
        os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['manual_review']),
        os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['logs'])
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def get_image_files(source_dir: str) -> List[str]:
    """Get all image files from the source directory."""
    if not os.path.exists(source_dir):
        return []
    
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
                image_files.append(os.path.join(root, file))
    
    return sorted(image_files)

def process_single_image_sequential(
    image_path: str, 
    text_detector,
    enabled_tests: Set[str], 
    logger: logging.Logger,
    dry_run: bool = False, 
    show_progress: bool = True
) -> Tuple[str, bool, str, str, Dict]:
    """
    Process single image in SEQUENTIAL mode - stops on first failure.
    Test order: specifications → borders → watermarks → editing → text
    
    Returns:
        Tuple[filename, is_valid, reason, destination_category, detailed_results]
    """
    filename = os.path.basename(image_path)
    all_results = {}
    
    # Define sequential test order (specifications first, text last)
    test_order = ['specifications', 'borders', 'watermarks', 'editing', 'text']
    ordered_tests = [test for test in test_order if test in enabled_tests]
    
    try:
        # Phase 1: Basic validation using integrated processor (always first)
        try:
            processor = get_processor()
            is_valid, validation_reason, img_metadata = processor.load_and_validate_image(image_path)
            
            if not is_valid:
                reason = f'Basic validation failed: {validation_reason}'
                if show_progress:
                    print(f"  X FAIL Basic validation: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
            
            # Size check
            MIN_WIDTH, MIN_HEIGHT = 100, 100
            if img_metadata['width'] < MIN_WIDTH or img_metadata['height'] < MIN_HEIGHT:
                reason = f'Image too small: {img_metadata["width"]}x{img_metadata["height"]} (minimum: {MIN_WIDTH}x{MIN_HEIGHT})'
                if show_progress:
                    print(f"  X FAIL Size check: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
            
            if show_progress:
                print(f"  ✓ PASS Basic validation")
            
            all_results['basic_validation'] = {'passed': True, 'format': img_metadata['format'], 'size': f'{img_metadata["width"]}x{img_metadata["height"]}'}
                
        except Exception as e:
            reason = f'Cannot process image: {str(e)}'
            if show_progress:
                print(f"  X FAIL Basic validation: {reason}")
            
            if not dry_run:
                copy_file_to_new_structure(image_path, 'invalid', reason, filename)
            
            return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
        
        # Phase 2: Sequential test execution (STOP ON FIRST FAILURE)
        for test_name in ordered_tests:
            if show_progress:
                print(f"  → Running {test_name} test...")
            
            if test_name == 'text' and text_detector:
                # Text detection using PaddleOCR
                try:
                    from pathlib import Path
                    paddle_result = text_detector.process_single_image(Path(image_path), detection_only=False)
                    
                    if paddle_result.get('processing_success', False):
                        confidence_metrics = paddle_result.get('confidence_metrics', {})
                        overall_confidence = confidence_metrics.get('overall_confidence', 0.0)
                        text_count = confidence_metrics.get('text_count', 0)
                        
                        # Apply text confidence thresholds
                        if overall_confidence > 80 or text_count > 5:  # High text confidence = watermarked
                            reason = f'High text confidence: {overall_confidence:.1f}% ({text_count} text regions) - SEQUENTIAL STOP'
                            if show_progress:
                                print(f"  X FAIL Text check: {reason}")
                            
                            if not dry_run:
                                copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                            
                            all_results['text_detection'] = {'passed': False, 'confidence': overall_confidence, 'text_count': text_count}
                            return filename, False, reason, 'invalid', all_results
                            
                        elif overall_confidence > 60 or text_count > 2:  # Medium confidence needs manual review
                            reason = f'Text needs manual review: {overall_confidence:.1f}% ({text_count} text regions) - SEQUENTIAL STOP'
                            if show_progress:
                                print(f"  ? MANUAL Text check: {reason}")
                            
                            if not dry_run:
                                copy_file_to_new_structure(image_path, 'manualreview', reason, filename)
                            
                            all_results['text_detection'] = {'passed': False, 'needs_manual_review': True, 'confidence': overall_confidence, 'text_count': text_count}
                            return filename, False, reason, 'manualreview', all_results
                            
                        else:
                            if show_progress:
                                print(f"  ✓ PASS Text check")
                            all_results['text_detection'] = {'passed': True, 'confidence': overall_confidence, 'text_count': text_count}
                    else:
                        # Handle processing errors - continue in sequential mode
                        error_msg = paddle_result.get('error', 'Text detection failed')
                        if show_progress:
                            print(f"  ! WARNING Text detection failed: {error_msg}")
                        all_results['text_detection'] = {'passed': None, 'error': error_msg}
                        
                except Exception as e:
                    # Graceful degradation on text detection failure - continue
                    if show_progress:
                        print(f"  ! WARNING Text detection failed: {str(e)}")
                    all_results['text_detection'] = {'passed': None, 'error': str(e)}
                    
            else:
                # Other tests using unified detector
                try:
                    detector = get_unified_detector()
                    result = detector.process_image_unified(image_path, {test_name})
                    
                    if not result.get('processing_success', False):
                        error_msg = result.get('error', f'{test_name} test failed')
                        reason = f'{test_name.title()} test failed: {error_msg} - SEQUENTIAL STOP'
                        if show_progress:
                            print(f"  X FAIL {test_name.title()} check: {error_msg}")
                        
                        if not dry_run:
                            copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                        
                        all_results[f'{test_name}_test'] = {'passed': False, 'error': error_msg}
                        return filename, False, reason, 'invalid', all_results
                    else:
                        # Check test result
                        test_results = result.get('results', {})
                        test_result = test_results.get(test_name, {})
                        passed = test_result.get('passed', True)
                        needs_manual_review = test_result.get('needs_manual_review', False)
                        
                        if needs_manual_review:
                            reason = f'{test_name.title()} needs manual review: {test_result.get("reason", "Unknown")} - SEQUENTIAL STOP'
                            if show_progress:
                                print(f"  ? MANUAL {test_name.title()} check: {test_result.get('reason', 'Unknown')}")
                            
                            if not dry_run:
                                copy_file_to_new_structure(image_path, 'manualreview', reason, filename)
                            
                            all_results[f'{test_name}_test'] = {'passed': False, 'needs_manual_review': True, 'reason': test_result.get('reason')}
                            return filename, False, reason, 'manualreview', all_results
                            
                        elif not passed:
                            reason = f'{test_name.title()} test failed: {test_result.get("reason", "Unknown")} - SEQUENTIAL STOP'
                            if show_progress:
                                print(f"  X FAIL {test_name.title()} check: {test_result.get('reason', 'Unknown')}")
                            
                            if not dry_run:
                                copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                            
                            all_results[f'{test_name}_test'] = {'passed': False, 'reason': test_result.get('reason')}
                            return filename, False, reason, 'invalid', all_results
                        else:
                            if show_progress:
                                print(f"  ✓ PASS {test_name.title()} check")
                            all_results[f'{test_name}_test'] = {'passed': True, 'reason': test_result.get('reason', 'Passed')}
                            
                except Exception as e:
                    reason = f'{test_name.title()} test error: {str(e)} - SEQUENTIAL STOP'
                    if show_progress:
                        print(f"  X ERROR {test_name.title()} check: {str(e)}")
                    
                    if not dry_run:
                        copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                    
                    all_results[f'{test_name}_test'] = {'passed': False, 'error': str(e)}
                    return filename, False, reason, 'invalid', all_results
        
        # If we get here, ALL tests passed in sequential mode
        reason = f'All sequential tests passed ({len(ordered_tests)} tests)'
        if show_progress:
            print(f"  ✓ PASS All sequential tests completed successfully")
        
        if not dry_run:
            copy_file_to_new_structure(image_path, 'valid', reason, filename)
        
        return filename, True, reason, 'valid', all_results
        
    except Exception as e:
        reason = f'Critical processing error: {str(e)}'
        if show_progress:
            print(f"  X CRITICAL ERROR: {reason}")
        
        if not dry_run:
            copy_file_to_new_structure(image_path, 'invalid', reason, filename)
        
        return filename, False, reason, 'invalid', {'critical_error': {'passed': False, 'error': str(e)}}

def process_single_image_with_text_detection(
    image_path: str, 
    text_detector,  # OPTIMIZED: Remove type hint for lazy-loaded class
    enabled_tests: Set[str], 
    logger: logging.Logger,
    dry_run: bool = False, 
    show_progress: bool = True
) -> Tuple[str, bool, str, str, Dict]:
    """
    Process single image with integrated PaddleOCR text detection and smart routing.
    Now runs ALL enabled tests and collects ALL results before making final decision.
    
    Returns:
        Tuple[filename, is_valid, reason, destination_category, detailed_results]
    """
    filename = os.path.basename(image_path)
    all_results = {}
    test_failures = []
    manual_review_reasons = []
    overall_valid = True
    
    try:
        # Phase 1: Basic validation using integrated processor (OPTIMIZED: eliminates redundancy)
        try:
            # Use centralized image validation and loading
            processor = get_processor()
            is_valid, validation_reason, img_metadata = processor.load_and_validate_image(image_path)
            
            if not is_valid:
                reason = f'Basic validation failed: {validation_reason}'
                if show_progress:
                    print(f"  X FAIL Basic validation: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
            
            # OPTIMIZED: Only check size if basic validation passes
            MIN_WIDTH, MIN_HEIGHT = 100, 100
            if img_metadata['width'] < MIN_WIDTH or img_metadata['height'] < MIN_HEIGHT:
                reason = f'Image too small: {img_metadata["width"]}x{img_metadata["height"]} (minimum: {MIN_WIDTH}x{MIN_HEIGHT})'
                if show_progress:
                    print(f"  X FAIL Size check: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
            
            if show_progress:
                print(f"  ✓ PASS Basic validation (format: {img_metadata['format']}, size: {img_metadata['width']}x{img_metadata['height']})")
            
            all_results['basic_validation'] = {'passed': True, 'format': img_metadata['format'], 'size': f'{img_metadata["width"]}x{img_metadata["height"]}'}
                
        except Exception as e:
            reason = f'Cannot process image: {str(e)}'
            if show_progress:
                print(f"  X FAIL Basic validation: {reason}")
            
            if not dry_run:
                copy_file_to_new_structure(image_path, 'invalid', reason, filename)
            
            return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
        
        # Phase 2: Text Detection using PaddleOCR (OPTIMIZED: if enabled and available)
        if text_detector and 'text' in enabled_tests:
            try:
                # OPTIMIZED: Process image with PaddleOCR text detector (reuse existing instance)
                from pathlib import Path
                paddle_result = text_detector.process_single_image(Path(image_path), detection_only=False)
                
                if paddle_result.get('processing_success', False):
                    confidence_metrics = paddle_result.get('confidence_metrics', {})
                    overall_confidence = confidence_metrics.get('overall_confidence', 0.0)
                    text_count = confidence_metrics.get('text_count', 0)
                    
                    # OPTIMIZED: Apply text confidence thresholds (early exit on high confidence)
                    if overall_confidence > 80 or text_count > 5:  # High text confidence = watermarked
                        reason = f'High text confidence: {overall_confidence:.1f}% ({text_count} text regions detected)'
                        if show_progress:
                            print(f"  X FAIL Text check: {reason}")
                        test_failures.append(('text', reason, 'invalid'))
                        overall_valid = False
                        
                    elif overall_confidence > 60 or text_count > 2:  # Medium confidence needs manual review
                        reason = f'Text needs manual review: {overall_confidence:.1f}% ({text_count} text regions detected)'
                        if show_progress:
                            print(f"  ? MANUAL Text check: {reason}")
                        manual_review_reasons.append(('text', reason))
                        overall_valid = False
                        
                    else:
                        if show_progress:
                            print(f"  ✓ PASS Text check")
                    
                    all_results['text_detection'] = {
                        'passed': overall_confidence <= 60 and text_count <= 2,
                        'confidence': overall_confidence,
                        'text_count': text_count,
                        'paddle_result': paddle_result
                    }
                else:
                    # Handle processing errors - OPTIMIZED: Don't fail entire pipeline
                    error_msg = paddle_result.get('error', 'Text detection failed')
                    if show_progress:
                        print(f"  ! WARNING Text detection failed: {error_msg}")
                    all_results['text_detection'] = {'passed': None, 'error': error_msg}
                
            except Exception as e:
                # OPTIMIZED: Graceful degradation on text detection failure
                if show_progress:
                    print(f"  ! WARNING Text detection failed: {str(e)}")
                all_results['text_detection'] = {'passed': None, 'error': str(e)}
        
        # Phase 3: Additional validation tests (OPTIMIZED: if enabled)
        other_tests = enabled_tests - {'text'}  # Remove text from tests since we handle it with PaddleOCR
        
        if other_tests:
            # OPTIMIZED: Get unified detector for other tests (lazy loading)
            detector = get_unified_detector()
            
            try:
                result = detector.process_image_unified(image_path, other_tests)
                
                if not result.get('processing_success', False):
                    error_msg = result.get('error', 'Unknown processing error')
                    if show_progress:
                        print(f"  X Processing failed: {error_msg}")
                    
                    all_results['unified_processing'] = {'passed': False, 'error': error_msg}
                    test_failures.append(('processing', error_msg, 'invalid'))
                    overall_valid = False
                else:
                    # Analyze test results
                    test_results = result.get('results', {})
                    
                    # Define test execution order and display names
                    test_names_display = {
                        'specifications': 'Specifications',
                        'colors': 'Colors',
                        'borders': 'Borders',
                        'editing': 'Editing',
                        'effects': 'Effects',
                        'watermarks': 'Watermarks'
                    }
                    
                    test_order = ['specifications', 'borders', 'editing', 'colors', 'effects', 'watermarks']
                    ordered_tests = [test for test in test_order if test in other_tests]
                    ordered_tests.extend([test for test in other_tests if test not in test_order])
                    
                    # Check each test in order - RUN ALL, DON'T STOP ON FIRST FAILURE
                    for test_name in ordered_tests:
                        if test_name in test_results:
                            test_result = test_results[test_name]
                            passed = test_result.get('passed', True)
                            needs_manual_review = test_result.get('needs_manual_review', False)
                            
                            if show_progress:
                                display_name = test_names_display.get(test_name, test_name.title())
                                
                                # Show all test results (pass, fail, manual review)
                                if needs_manual_review:
                                    status = '? MANUAL'
                                    print(f"  {status} {display_name} check", end="")
                                    reason = test_result.get('reason', f'{test_name} needs manual review')
                                    print(f": {reason}")
                                elif not passed:
                                    status = 'X FAIL'
                                    print(f"  {status} {display_name} check", end="")
                                    reason = test_result.get('reason', f'{test_name} check failed')
                                    print(f": {reason}")
                                else:
                                    status = '✓ PASS'
                                    print(f"  {status} {display_name} check")
                            
                            # Store test result
                            all_results[test_name] = test_result
                            
                            # Handle test failures and manual review cases
                            if needs_manual_review:
                                reason = test_result.get('reason', f'{test_name} needs manual review')
                                manual_review_reasons.append((test_name, reason))
                                overall_valid = False
                                
                            elif not passed:
                                reason = test_result.get('reason', f'{test_name} check failed')
                                
                                # All test failures go to invalid category now
                                test_failures.append((test_name, reason, 'invalid'))
                                overall_valid = False
                    
                    all_results['unified_processing'] = {'passed': True, 'results': test_results}
                    
            except Exception as e:
                error_msg = f"Unified processing error: {str(e)}"
                if show_progress:
                    print(f"  X Unified processing failed: {error_msg}")
                all_results['unified_processing'] = {'passed': False, 'error': error_msg}
                test_failures.append(('processing', error_msg, 'invalid'))
                overall_valid = False
        
        # Phase 4: Determine final outcome based on ALL test results
        # Decision logic: prioritize actual failures over manual review flags
        if test_failures:
            # At least one test failed - this takes priority over everything else
            reason = f"Failed checks: {'; '.join([f'{f[0]}: {f[1]}' for f in test_failures])}"
            category = 'invalid'
            if show_progress:
                print(f"  >> Final Result: INVALID ({reason})")
        elif manual_review_reasons:
            # Tests need manual review - move file to manual review folder
            reasons_text = '; '.join([r[1] for r in manual_review_reasons])
            reason = f"Manual review needed: {reasons_text}"
            category = 'manual_review'
            if show_progress:
                print(f"  >> Final Result: MANUAL REVIEW ({reason})")
        elif overall_valid:
            # All tests passed - image is valid
            reason = 'All validation checks passed'
            category = 'valid'
            if show_progress:
                print(f"  >> Final Result: VALID ({reason})")
        else:
            # This shouldn't happen, but fallback to invalid
            reason = "Unknown processing issue"
            category = 'invalid'
            if show_progress:
                print(f"  >> Final Result: INVALID (fallback - {reason})")
        
        # Copy file to appropriate location 
        if not dry_run:
            copy_file_to_new_structure(image_path, category, reason, filename)
        
        # Update overall_valid for return value
        final_overall_valid = (category == 'valid')
        
        return filename, final_overall_valid, reason, category, all_results
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        if show_progress:
            print(f"  X Critical error: {error_msg}")
        
        if not dry_run:
            copy_file_to_new_structure(image_path, 'invalid', error_msg, filename)
        
        return filename, False, error_msg, 'invalid', {'critical_error': error_msg}

def copy_file_to_new_structure(source_path: str, category: str, reason: str, filename: str):
    """Copy file to appropriate category directory using integrated output manager."""
    try:
        # Use integrated output manager (eliminates redundancy)
        output_manager = get_output_manager()
        target_path = output_manager.copy_to_category(source_path, category, reason)
        return target_path
        
    except Exception as e:
        raise Exception(f"Failed to copy {filename} to {category}: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimized Photo Filtering System'
    )
    
    parser.add_argument('--source', '-s', type=str, default=PHOTOS_DIR,
                       help='Source directory containing images')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Process images but don\'t copy files')
    parser.add_argument('--tests', '-t', type=str, nargs='+',
                       choices=['specifications', 'text', 'borders', 'editing', 'watermarks'],
                       default=['specifications', 'text', 'borders', 'editing', 'watermarks'],
                       help='Tests to run (text detection now uses PaddleOCR)')
    parser.add_argument('--no-specs', action='store_true',
                       help='Skip specifications check (ignore size/format requirements)')
    parser.add_argument('--no-text', action='store_true',
                       help='Skip text detection (disable PaddleOCR watermark detection)')
    parser.add_argument('--no-borders', action='store_true',
                       help='Skip border detection check')
    parser.add_argument('--no-editing', action='store_true',
                       help='Skip editing detection check (PyIQA-based analysis)')
    parser.add_argument('--no-watermarks', action='store_true',
                       help='Skip advanced watermark detection check')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed progress output')
    parser.add_argument('--sequential-mode', action='store_true',
                       help='Run tests sequentially, stop on first failure (specs->borders->watermarks->editing->text)')
    
    return parser.parse_args()

def print_results_table(results: List[Tuple], enabled_tests: set = None):
    """Print a formatted table of processing results with simplified categories."""
    
    # Separate valid and invalid results
    valid_results = [r for r in results if r[1]]
    invalid_results = [r for r in results if not r[1]]
    
    print("=" * 120)
    print("PHOTO FILTERING RESULTS")
    print("=" * 120)
    
    # Show enabled/disabled tests
    if enabled_tests:
        all_tests = {'specifications', 'text', 'watermarks', 'borders', 'editing'}
        enabled = sorted(enabled_tests)
        disabled = sorted(all_tests - enabled_tests)
        
        print(f"\nTESTS PERFORMED:")
        if enabled:
            print(f"  Enabled: {', '.join(enabled)}")
        if disabled:
            print(f"  Disabled: {', '.join(disabled)}")
    
    # Show editing confidence table if editing tests were performed
    if 'editing' in enabled_tests and results:
        print(f"\nEDITING CONFIDENCE ANALYSIS:")
        print("-" * 120)
        print(f"{'Filename':<50} {'Editing Confidence':<20} {'Assessment':<30}")
        print("-" * 120)
        
        # Extract editing confidence from results
        editing_results = []
        for filename, is_valid, reason, category, detailed_results in results:
            editing_confidence = 0.0
            assessment = "No editing data"
            
            # Look for editing results in detailed_results
            if detailed_results and 'editing' in detailed_results:
                editing_data = detailed_results['editing']
                editing_confidence = editing_data.get('editing_confidence', 0.0)
                
                if editing_confidence >= 25.0:
                    assessment = "INVALID - High editing"
                elif editing_confidence >= 20.0:
                    assessment = "MANUAL REVIEW - Medium editing"
                else:
                    assessment = "VALID - Minimal editing"
            elif detailed_results:
                # Check if editing results are in unified_processing results
                unified = detailed_results.get('unified_processing', {})
                if unified.get('results') and 'editing' in unified['results']:
                    editing_data = unified['results']['editing']
                    editing_confidence = editing_data.get('editing_confidence', 0.0)
                    
                    if editing_confidence >= 25.0:
                        assessment = "INVALID - High editing"
                    elif editing_confidence >= 20.0:
                        assessment = "MANUAL REVIEW - Medium editing"
                    else:
                        assessment = "VALID - Minimal editing"
            
            editing_results.append((filename, editing_confidence, assessment))
        
        # Sort by confidence (highest first)
        editing_results.sort(key=lambda x: x[1], reverse=True)
        
        # Display the table
        for filename, confidence, assessment in editing_results:
            filename_short = filename[:47] + "..." if len(filename) > 50 else filename
            print(f"{filename_short:<50} {confidence:>8.1f}%{'':<11} {assessment:<30}")
        
        print("-" * 120)
        print("Note: Images with confidence ≥25% are moved to INVALID folder, 20-25% to MANUAL REVIEW folder.")
        
    # Summary statistics 
    total = len(results)
    valid_count = len(valid_results)
    invalid_count = len(invalid_results)
    manual_review_count = len([r for r in results if r[3] == 'manual_review'])
    
    print(f"\nSUMMARY:")
    print(f"  Total Images Processed: {total}")
    print(f"  Valid Images: {valid_count}")
    print(f"  Invalid Images: {invalid_count}")
    print(f"  Manual Review Needed: {manual_review_count}")
    print(f"  Success Rate: {(valid_count/total*100):.1f}%" if total > 0 else "  Success Rate: 0%")
    
    # Group results by simplified categories
    if results:
        category_groups = {'valid': [], 'invalid': [], 'manual_review': []}
        
        for filename, is_valid, reason, category, _ in results:
            if category in category_groups:
                category_groups[category].append((filename, reason))
            else:
                # Map any unknown categories to invalid
                category_groups['invalid'].append((filename, reason))
        
        # Display each category
        category_display = {
            'valid': 'VALID IMAGES',
            'invalid': 'INVALID IMAGES', 
            'manual_review': 'MANUAL REVIEW NEEDED'
        }
        
        for category, items in category_groups.items():
            if items:
                display_name = category_display.get(category, category.upper())
                print(f"\n{display_name} ({len(items)} images):")
                print("-" * 120)
                
                if category == 'valid':
                    print(f"All validation checks passed - images copied to '{os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['valid'])}' folder")
                else:
                    # Display detailed failure information
                    for i, (filename, reason) in enumerate(items, 1):
                        print(f"\n{i:2d}. {filename}")
                        
                        # Parse and format the failure reasons more clearly
                        if reason.startswith("Failed checks: "):
                            # Remove "Failed checks: " prefix
                            failures = reason[15:]
                            
                            # Split by "; " to get individual failures
                            failure_list = failures.split("; ")
                            
                            print("    Failures:")
                            for j, failure in enumerate(failure_list, 1):
                                # Split each failure by ": " to separate test name from reason
                                if ": " in failure:
                                    test_name, test_reason = failure.split(": ", 1)
                                    print(f"      • {test_name.title()}: {test_reason}")
                                else:
                                    print(f"      • {failure}")
                        else:
                            # If it doesn't match expected format, write as-is
                            print(f"    Issue: {reason}")
    
    print(f"\nOUTPUT STRUCTURE:")
    print(f"  Valid images: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['valid'])}")
    print(f"  Invalid images: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['invalid'])}")
    print(f"  Manual review needed: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['manual_review'])}")
    print(f"  Processing logs: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['logs'])}")
    
    print(f"\nNOTE: Images flagged only for editing review are kept in their original location.")
    print(f"      Check the 'EDITING CONFIDENCE ANALYSIS' table above to see which images need editing review.")
    
    print("\n" + "=" * 120)

def generate_summary_report(results: List[Tuple], elapsed_time: float, enabled_tests: Set[str], logger: logging.Logger, sequential_mode: bool = False):
    """Generate a detailed summary report and save it to the logs folder."""
    try:
        # Calculate statistics 
        total = len(results)
        valid_count = len([r for r in results if r[1]])
        invalid_count = len([r for r in results if not r[1] and r[3] != 'manual_review'])
        manual_review_count = len([r for r in results if r[3] == 'manual_review'])
        
        # Group results by category for detailed reporting
        valid_files = [r[0] for r in results if r[1]]
        invalid_files = [(r[0], r[2]) for r in results if not r[1] and r[3] != 'manual_review']
        manual_review_files = [(r[0], r[2]) for r in results if r[3] == 'manual_review']
        
        # Create summary report
        report_path = os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['logs'], 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PHOTO FILTERING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PROCESSING CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Processing Mode: {'Sequential (stop on first failure)' if sequential_mode else 'Parallel (run all tests)'}\n")
            if sequential_mode:
                f.write(f"Test Order: specifications -> borders -> watermarks -> editing -> text\n")
            f.write(f"Enabled Tests: {', '.join(sorted(enabled_tests)) if enabled_tests else 'None'}\n")
            f.write(f"Processing Time: {elapsed_time:.2f} seconds\n")
            f.write(f"Average Time per Image: {elapsed_time/total:.2f} seconds\n\n" if total > 0 else "Average Time: N/A\n\n")
            
            f.write("PROCESSING STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Images Processed: {total}\n")
            f.write(f"Valid Images: {valid_count}\n")
            f.write(f"Invalid Images: {invalid_count}\n")
            f.write(f"Manual Review Needed: {manual_review_count}\n")
            f.write(f"Success Rate: {(valid_count/total*100):.1f}%\n\n" if total > 0 else "Success Rate: 0%\n\n")
            
            # Valid files section
            if valid_files:
                f.write(f"VALID IMAGES ({valid_count}):\n")
                f.write("-" * 40 + "\n")
                for filename in valid_files:
                    f.write(f"✓ {filename}\n")
                f.write("\n")
            
            # Invalid files section
            if invalid_files:
                f.write(f"INVALID IMAGES ({invalid_count}):\n")
                f.write("-" * 40 + "\n")
                for filename, reason in invalid_files:
                    f.write(f"✗ {filename}: Failed checks:\n")
                    
                    # Parse and format the failure reasons
                    if reason.startswith("Failed checks: "):
                        # Remove "Failed checks: " prefix
                        failures = reason[15:]
                        
                        # Split by "; " to get individual failures
                        failure_list = failures.split("; ")
                        
                        for i, failure in enumerate(failure_list, 1):
                            # Split each failure by ": " to separate test name from reason
                            if ": " in failure:
                                test_name, test_reason = failure.split(": ", 1)
                                f.write(f"   {i}] {test_name}: {test_reason}\n")
                            else:
                                f.write(f"   {i}] {failure}\n")
                    else:
                        # If it doesn't match expected format, write as-is
                        f.write(f"   - {reason}\n")
                    
                    f.write("\n")  # Add blank line between entries
                f.write("\n")
            
            # Manual review section
            if manual_review_files:
                f.write(f"MANUAL REVIEW NEEDED - MOVED ({manual_review_count}):\n")
                f.write("-" * 40 + "\n")
                for filename, reason in manual_review_files:
                    f.write(f"? {filename}: {reason}\n")
                f.write("\n")
            
            f.write("OUTPUT DIRECTORIES:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Valid: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['valid'])}\n")
            f.write(f"Invalid: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['invalid'])}\n")
            f.write(f"Manual Review: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['manual_review'])}\n")
            f.write(f"Logs: {os.path.join(OUTPUT_DIR, OUTPUT_STRUCTURE['logs'])}\n")
            f.write(f"\nNOTE: Editing detection thresholds: ≥25% → Invalid, 20-25% → Manual Review, <20% → Valid\n")
            f.write("=" * 80 + "\n")
        
        # logger.info(f"Summary report saved to: {report_path}")  # Suppress for clean output
        print(f"Summary report: {report_path}")
        
    except Exception as e:
        print(f"Failed to generate summary report: {e}")

def main():
    """Main entry point for optimized photo filtering with PaddleOCR text detection."""
    args = parse_arguments()
    
    # Use the output directory from arguments
    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    
    # Setup
    create_output_directories()
    logger = setup_logging()
    
    # Check PyTorch environment early to warn about potential issues (silently)
    pytorch_info = check_pytorch_environment()
    
    # Convert tests to set and remove disabled tests
    enabled_tests = set(args.tests)
    
    # Remove tests based on --no-* arguments (silently)
    if args.no_specs and 'specifications' in enabled_tests:
        enabled_tests.remove('specifications')
    
    if args.no_text and 'text' in enabled_tests:
        enabled_tests.remove('text')
    
    if args.no_borders and 'borders' in enabled_tests:
        enabled_tests.remove('borders')
    
    if args.no_editing and 'editing' in enabled_tests:
        enabled_tests.remove('editing')
    
    if args.no_watermarks and 'watermarks' in enabled_tests:
        enabled_tests.remove('watermarks')
    
    # Check if any tests are enabled
    if not enabled_tests:
        print("Error: No tests enabled. Use --tests to specify which tests to run.")
        return
    
    # OPTIMIZED: Only initialize PaddleOCR text detector if text detection is enabled
    text_detector = None
    if 'text' in enabled_tests:
        # OPTIMIZED: Initialize PaddleOCR text detector (lazy loading, reduced overhead)
        try:
            PaddleTextDetector = get_paddle_text_detector()
            if PaddleTextDetector is None:
                print("Warning: PaddleOCR not available, skipping text detection")
                enabled_tests.discard('text')  # Remove from enabled tests
            else:
                # Suppress PaddleOCR warnings and stdout noise during initialization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Also suppress stdout for the "Could not find files" message
                    with suppress_stdout_stderr():
                        pass  # PaddleTextDetector import happens in lazy function
                
                # Save original methods
                original_setup_folders = PaddleTextDetector.setup_organization_folders
                original_organize = PaddleTextDetector.organize_image_by_confidence
                
                # Create disabled methods (OPTIMIZED: reduce I/O overhead)
                def disabled_setup_folders(self):
                    """Disabled folder setup - no folders created"""
                    # Set dummy folder paths to prevent None errors
                    from pathlib import Path
                    dummy_path = Path("dummy")
                    self.organized_images_dir = dummy_path
                    self.invalid_folder = dummy_path
                    self.manual_review_folder = dummy_path
                    self.valid_folder = dummy_path
                    self.debug_folder = dummy_path
                
                def disabled_organize(self, image_path, confidence_metrics):
                    """Disabled organization - return success without copying files"""
                    try:
                        # Convert to string if it's a Path object to avoid any path operations
                        if hasattr(image_path, 'name'):
                            filename = image_path.name
                            path_str = str(image_path)
                        else:
                            import os
                            filename = os.path.basename(str(image_path))
                            path_str = str(image_path)
                        
                        return {
                            'source_path': path_str,
                            'destination_path': None,
                            'category': None,
                            'confidence': confidence_metrics.get('overall_confidence', 0.0),
                            'text_count': confidence_metrics.get('text_count', 0),
                            'reason': 'Organization disabled - using main controller',
                            'organized_successfully': True
                        }
                    except Exception as e:
                        # Fallback in case of any error
                        return {
                            'source_path': str(image_path),
                            'destination_path': None,
                            'category': None,
                            'confidence': confidence_metrics.get('overall_confidence', 0.0),
                            'text_count': confidence_metrics.get('text_count', 0),
                            'reason': f'Organization disabled (error: {str(e)})',
                            'organized_successfully': True
                        }
                
                # Monkey patch the class methods
                PaddleTextDetector.setup_organization_folders = disabled_setup_folders
                PaddleTextDetector.organize_image_by_confidence = disabled_organize
                
                # OPTIMIZED: Now initialize the detector with minimal parameters
                text_detector = PaddleTextDetector(
                    input_folder=args.source, 
                    use_gpu=False,  # CPU-only for better compatibility
                    use_textline_orientation=False,  # OPTIMIZED: Disable orientation detection for speed
                    lang='en'
                )
                
                # Also disable any logging that might cause issues
                import logging
                paddle_logger = logging.getLogger('paddle_text_detector')
                paddle_logger.setLevel(logging.CRITICAL)  # Only critical errors
                
                # Restore original methods for any future instances (optional)
                PaddleTextDetector.setup_organization_folders = original_setup_folders
                PaddleTextDetector.organize_image_by_confidence = original_organize
                
                # Silent success - print nothing
        except Exception as e:
            # Silent failure - continue without text detection
            text_detector = None
            enabled_tests.discard('text')  # Remove from enabled tests
    else:
        # Text detection disabled - silent
        pass
    
    # Suppress all startup messages - only show final results
    
    # Get image files
    image_files = get_image_files(args.source)
    if not image_files:
        print(f"Error: No image files found in {args.source}")
        return
    
    # OPTIMIZED: Process images sequentially with integrated watermark detection
    start_time = time.time()
    
    results = []
    total_images = len(image_files)
    
    # OPTIMIZED: Pre-initialize detector to avoid repeated initialization
    unified_detector = None
    other_tests = enabled_tests - {'text'}
    if other_tests:
        unified_detector = get_unified_detector()
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        
        # OPTIMIZED: Show minimal progress for speed
        if not args.quiet:
            mode_indicator = "[SEQ]" if args.sequential_mode else "[PAR]"
            print(f"{mode_indicator} [{i}/{total_images}] Processing: {filename}")
        
        try:
            # Choose processing mode: Sequential or Parallel
            if args.sequential_mode:
                # SEQUENTIAL MODE: Stop on first failure
                if text_detector:
                    result = process_single_image_sequential(
                        image_path, text_detector, enabled_tests, logger, 
                        args.dry_run, show_progress=not args.quiet
                    )
                else:
                    # Sequential mode without text detector
                    result = process_single_image_sequential(
                        image_path, None, enabled_tests, logger, 
                        args.dry_run, show_progress=not args.quiet
                    )
            else:
                # PARALLEL MODE: Run all tests (original behavior)
                if text_detector:
                    result = process_single_image_with_text_detection(
                        image_path, text_detector, enabled_tests, logger, 
                        args.dry_run, show_progress=False  # Always suppress progress for speed
                    )
                else:
                    # OPTIMIZED: Fallback to basic processing without watermark detection
                    result = process_basic_validation_only(
                        image_path, enabled_tests, logger, 
                        args.dry_run, show_progress=False  # Always suppress progress for speed
                    )
            
            results.append(result)
            
            # OPTIMIZED: Show result status only
            if not args.quiet:
                status = "✓ PASS" if result[1] else "✗ FAIL"
                category = result[3].upper()
                if args.sequential_mode and not result[1]:
                    print(f"  {status} -> {category} (stopped early)")
                else:
                    print(f"  {status} -> {category}")
            
        except Exception as e:
            if not args.quiet:
                print(f"  X Critical error: {e}")
            results.append((filename, False, f"Processing error: {e}", 'invalid', {}))
        
        # OPTIMIZED: Remove spacing between images for speed
    
    elapsed_time = time.time() - start_time
    
    # Print results table with mode information
    mode_str = "Sequential (stop on first failure)" if args.sequential_mode else "Parallel (run all tests)"
    print(f"\nProcessing Mode: {mode_str}")
    if args.sequential_mode:
        print(f"Test Order: specifications -> borders -> watermarks -> editing -> text")
    print_results_table(results, enabled_tests)
    
    # Generate detailed summary report (only this, no processing.log)
    generate_summary_report(results, elapsed_time, enabled_tests, logger, args.sequential_mode)
    
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")

def process_basic_validation_only(
    image_path: str, 
    enabled_tests: Set[str], 
    logger: logging.Logger,
    dry_run: bool = False, 
    show_progress: bool = True
) -> Tuple[str, bool, str, str, Dict]:
    """
    Fallback processing function for when PaddleOCR text detector is not available.
    Still runs all other enabled tests (specifications, borders, effects).
    """
    filename = os.path.basename(image_path)
    all_results = {}
    test_failures = []
    manual_review_reasons = []
    overall_valid = True
    
    try:
        # Phase 1: Basic format and size validation (OPTIMIZED: use cached validation)
        try:
            # OPTIMIZED: Use centralized image validation and loading (with caching)
            processor = get_processor()
            is_valid, validation_reason, img_metadata = processor.load_and_validate_image(image_path)
            
            if not is_valid:
                reason = f'Basic validation failed: {validation_reason}'
                if show_progress:
                    print(f"  X FAIL Basic validation: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
            
            # OPTIMIZED: Size check using cached metadata
            MIN_WIDTH, MIN_HEIGHT = 100, 100
            if img_metadata['width'] < MIN_WIDTH or img_metadata['height'] < MIN_HEIGHT:
                reason = f'Image too small: {img_metadata["width"]}x{img_metadata["height"]}'
                if show_progress:
                    print(f"  X FAIL Size check: {reason}")
                
                if not dry_run:
                    copy_file_to_new_structure(image_path, 'invalid', reason, filename)
                
                return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
                
            if show_progress:
                print(f"  ✓ PASS Basic validation (format: {img_metadata['format']}, size: {img_metadata['width']}x{img_metadata['height']})")
            
            all_results['basic_validation'] = {'passed': True, 'format': img_metadata['format'], 'size': f'{img_metadata["width"]}x{img_metadata["height"]}'}
                
        except Exception as e:
            reason = f'Cannot read image: {str(e)}'
            if show_progress:
                print(f"  X FAIL Basic validation: {reason}")
            
            if not dry_run:
                copy_file_to_new_structure(image_path, 'invalid', reason, filename)
            
            return filename, False, reason, 'invalid', {'basic_validation': {'passed': False, 'reason': reason}}
        
        # Phase 2: Run enabled tests (OPTIMIZED: excluding text since PaddleOCR unavailable)
        other_tests = enabled_tests - {'text'}  # Remove text from tests since PaddleOCR unavailable
        
        if other_tests:
            # OPTIMIZED: Get unified detector for other tests (reuse existing instance)
            detector = get_unified_detector()
            
            try:
                result = detector.process_image_unified(image_path, other_tests)
                
                if not result.get('processing_success', False):
                    error_msg = result.get('error', 'Unknown processing error')
                    if show_progress:
                        print(f"  X Processing failed: {error_msg}")
                    
                    all_results['unified_processing'] = {'passed': False, 'error': error_msg}
                    test_failures.append(('processing', error_msg, 'invalid'))
                    overall_valid = False
                else:
                    # OPTIMIZED: Analyze test results with early exit on failures
                    test_results = result.get('results', {})
                    
                    # Define test execution order and display names
                    test_names_display = {
                        'specifications': 'Specifications',
                        'colors': 'Colors',
                        'borders': 'Borders',
                        'editing': 'Editing',
                        'effects': 'Effects',
                        'watermarks': 'Watermarks'
                    }
                    
                    test_order = ['specifications', 'borders', 'editing', 'colors', 'effects', 'watermarks']
                    ordered_tests = [test for test in test_order if test in other_tests]
                    ordered_tests.extend([test for test in other_tests if test not in test_order])
                    
                    # OPTIMIZED: Check each test in order - early exit on severe failures
                    for test_name in ordered_tests:
                        if test_name in test_results:
                            test_result = test_results[test_name]
                            passed = test_result.get('passed', True)
                            needs_manual_review = test_result.get('needs_manual_review', False)
                            
                            if show_progress:
                                display_name = test_names_display.get(test_name, test_name.title())
                                
                                # Show all test results (pass, fail, manual review)
                                if needs_manual_review:
                                    status = '? MANUAL'
                                    print(f"  {status} {display_name} check", end="")
                                    reason = test_result.get('reason', f'{test_name} needs manual review')
                                    print(f": {reason}")
                                elif not passed:
                                    status = 'X FAIL'
                                    print(f"  {status} {display_name} check", end="")
                                    reason = test_result.get('reason', f'{test_name} check failed')
                                    print(f": {reason}")
                                else:
                                    status = '✓ PASS'
                                    print(f"  {status} {display_name} check")
                            
                            # Store test result
                            all_results[test_name] = test_result
                            
                            # Handle test failures and manual review cases
                            if needs_manual_review:
                                reason = test_result.get('reason', f'{test_name} needs manual review')
                                
                                # Special handling for editing detection - don't affect overall_valid
                                if test_name == 'editing':
                                    # For editing, just report but don't set overall_valid = False
                                    manual_review_reasons.append((test_name, reason))
                                else:
                                    # For other tests that need manual review, affect overall_valid
                                    manual_review_reasons.append((test_name, reason))
                                    overall_valid = False
                                
                            elif not passed:
                                reason = test_result.get('reason', f'{test_name} check failed')
                                test_failures.append((test_name, reason, 'invalid'))
                                overall_valid = False
                                
            except Exception as e:
                if show_progress:
                    print(f"  ! WARNING Additional tests failed: {str(e)}")
                all_results['unified_processing'] = {'passed': False, 'error': str(e)}
                test_failures.append(('processing', str(e), 'invalid'))
                overall_valid = False
        
        else:
            # No tests to run (text detection unavailable, no other tests enabled)
            pass
        
        # Final decision and routing
        # Filter out editing from manual review reasons for file movement decisions
        non_editing_manual_review = [r for r in manual_review_reasons if r[0] != 'editing']
        editing_manual_review = [r for r in manual_review_reasons if r[0] == 'editing']
        
        # Decision logic: prioritize actual failures over manual review flags
        if test_failures:
            # If any tests failed, route to invalid
            failure_details = [f"{test}: {reason}" for test, reason, _ in test_failures]
            reason = f"Failed checks: {'; '.join(failure_details)}"
            category = 'invalid'
            overall_valid = False
            
        elif non_editing_manual_review:
            # If non-editing tests need manual review, route to manual review
            reasons_list = [f"{test}: {reason}" for test, reason in non_editing_manual_review]
            reasons_text = '; '.join(reasons_list)
            reason = f"Manual review needed: {reasons_text}"
            category = 'manual_review'
            
        elif overall_valid:
            # All tests passed
            if editing_manual_review:
                reason = 'All validation checks passed (editing flagged for review)'
            else:
                reason = 'All validation checks passed'
            category = 'valid'
            
        else:
            # Fallback to invalid
            reason = "Unknown processing issue"
            category = 'invalid'
            overall_valid = False
        
        # Copy file to appropriate folder
        if not dry_run:
            # Special handling for editing detection:
            # - If image is invalid or needs manual review for non-editing reasons, move it
            # - If image is valid but has editing flagged, keep in place
            if category == 'valid' and editing_manual_review:
                # File stays in original location - just report editing needs review
                pass  
            else:
                # Move file for invalid or non-editing manual review
                copy_file_to_new_structure(image_path, category, reason, filename)
        
        # Update overall_valid for return value
        final_overall_valid = (category == 'valid')
        
        return filename, final_overall_valid, reason, category, all_results
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        if show_progress:
            print(f"  X Critical error: {error_msg}")
        
        if not dry_run:
            copy_file_to_new_structure(image_path, 'invalid', error_msg, filename)
        
        return filename, False, error_msg, 'invalid', {}

if __name__ == "__main__":
    main()
