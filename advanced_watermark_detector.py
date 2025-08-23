"""
Advanced Watermark Detector using pretrained models from boomb0om/watermark-detection
This module provides comprehensive watermark detection capabilities using state-of-the-art CNN models.
"""

import os
import sys
import json
import time
import warnings
import argparse
from typing import List, Union, Dict, Tuple, Optional
from pathlib import Path
import logging

# More targeted warning suppression - only suppress specific known issues
def suppress_known_warnings():
    """Suppress only specific known warnings that are safe to ignore."""
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    warnings.filterwarnings("ignore", message=".*torch.load.*")
    warnings.filterwarnings("ignore", message=".*FutureWarning.*timm.*")

suppress_known_warnings()

# Import torch normally - only suppress during model loading if needed
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set up configurable logging early
def setup_logging(level='INFO'):
    """Setup logging with configurable level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Better import management with proper fallback
try:
    from wmdetection.models import get_watermarks_detection_model
    from wmdetection.pipelines.predictor import WatermarksPredictor
    from wmdetection.utils.files import list_images, read_image_rgb
except ImportError:
    # Only modify path as fallback if imports fail
    watermark_detection_path = os.path.join(os.path.dirname(__file__), 'watermark-detection')
    if watermark_detection_path not in sys.path:
        sys.path.insert(0, watermark_detection_path)
    
    try:
        from wmdetection.models import get_watermarks_detection_model
        from wmdetection.pipelines.predictor import WatermarksPredictor
        from wmdetection.utils.files import list_images, read_image_rgb
    except ImportError as e:
        logger.error(f"Failed to import watermark detection modules: {e}")
        logger.error("Please ensure the watermark-detection package is properly installed")
        raise


class WatermarkDetectorConfig:
    """Configuration management for watermark detector."""
    
    def __init__(self, config_file=None):
        # Default configuration
        self.model_name = 'convnext-tiny'
        self.device = 'auto'  # auto, cpu, cuda:0, etc.
        self.high_confidence_threshold = 96.0
        self.manual_review_threshold = 91.0
        self.batch_size = 8
        self.num_workers = 4
        self.fp16 = False
        self.cache_dir = './models/watermark_cache'
        self.log_level = 'INFO'
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
    
    def save_to_file(self, config_file):
        """Save configuration to JSON file."""
        config_data = {
            'model_name': self.model_name,
            'device': self.device,
            'high_confidence_threshold': self.high_confidence_threshold,
            'manual_review_threshold': self.manual_review_threshold,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'fp16': self.fp16,
            'cache_dir': self.cache_dir,
            'log_level': self.log_level
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")


class AdvancedWatermarkDetector:
    """
    Advanced watermark detector using pretrained CNN models.
    
    Supports multiple model architectures:
    - convnext-tiny: Best accuracy (93.44%)
    - resnext101_32x8d-large: Large model (84.42%)
    - resnext50_32x4d-small: Compact model (76.22%)
    """
    
    def __init__(
        self, 
        model_name: str = 'convnext-tiny',
        device: Optional[str] = None,
        cache_dir: str = './models/watermark_cache',
        fp16: bool = False,
        config: Optional[WatermarkDetectorConfig] = None
    ):
        """
        Initialize the watermark detector with enhanced error handling.
        
        Args:
            model_name: Model to use ('convnext-tiny', 'resnext101_32x8d-large', 'resnext50_32x4d-small')
            device: Device to run on ('cuda:0', 'cpu', etc.). Auto-detected if None.
            cache_dir: Directory to cache model weights
            fp16: Use half precision (not supported with ConvNeXt models)
            config: Optional WatermarkDetectorConfig object to override other parameters
        """
        # Use config object if provided, otherwise use individual parameters
        if config is not None:
            self.model_name = config.model_name
            self.cache_dir = os.path.abspath(config.cache_dir)
            self.fp16 = config.fp16
            device = config.device if device is None else device
        else:
            self.model_name = model_name
            self.cache_dir = os.path.abspath(cache_dir)
            self.fp16 = fp16
        self.initialization_error = None
        
        # Test PyTorch availability first
        try:
            import torch
            # Test basic tensor operations to catch DLL issues early
            test_tensor = torch.tensor([1.0])
            _ = test_tensor.sum()
        except Exception as torch_error:
            self.initialization_error = f"PyTorch DLL/environment error: {torch_error}"
            logger.error(self.initialization_error)
            raise RuntimeError(self.initialization_error)
        
        # Smart device selection with proper GPU testing
        if device is None or device == 'auto':
            try:
                if torch.cuda.is_available():
                    # Test GPU functionality with a small operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor.sum()
                    self.device = 'cuda:0'
                    logger.info("GPU test successful - using CUDA for processing")
                else:
                    self.device = 'cpu'
                    logger.info("CUDA not available - using CPU for processing")
            except Exception as e:
                logger.warning(f"GPU test failed ({e}) - falling back to CPU for stability")
                self.device = 'cpu'
        else:
            # Validate the provided device
            if device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning(f"CUDA device '{device}' requested but CUDA not available, falling back to CPU")
                self.device = 'cpu'
            else:
                self.device = device
            
        # Validate fp16 settings
        if self.fp16 and model_name.startswith('convnext'):
            logger.warning("FP16 not supported with ConvNeXt models, disabling fp16")
            self.fp16 = False
            
        # Additional safety for CPU mode
        if self.device == 'cpu':
            self.fp16 = False  # Disable FP16 for CPU
            
        logger.info(f"Initializing watermark detector with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Cache directory: {self.cache_dir}")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load model and transforms with enhanced error handling
        self._load_model_safe()
        
    def _load_model_safe(self):
        """Load the watermark detection model with enhanced error handling."""
        try:
            logger.info("Loading model weights...")
            
            # Set environment variables to potentially help with DLL issues
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            
            # Suppress all warnings during model loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model, self.transforms = get_watermarks_detection_model(
                    self.model_name,
                    device=self.device,
                    fp16=self.fp16,
                    pretrained=True,
                    cache_dir=self.cache_dir
                )
            
            # Initialize predictor
            self.predictor = WatermarksPredictor(
                self.model, 
                self.transforms, 
                self.device
            )
            
            logger.info(f"Model {self.model_name} loaded successfully on {self.device}!")
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            
            # Provide helpful suggestions based on error type
            error_str = str(e)
            if "DLL" in error_str or "shm.dll" in error_str:
                suggestion = "Try reinstalling PyTorch: pip install torch --force-reinstall"
            elif "CUDA" in error_str:
                suggestion = "CUDA issue detected, try CPU-only mode"
            elif "memory" in error_str.lower():
                suggestion = "Insufficient memory, try a smaller model"
            elif "UntypedStorage" in error_str and "tagged with auto" in error_str:
                suggestion = "PyTorch device mapping issue. Try: pip install torch --upgrade or use explicit device (cpu/cuda:0)"
            elif "weights_only" in error_str:
                suggestion = "Model compatibility issue with weights_only=True. Update PyTorch or use older model format"
            elif "restore data location" in error_str:
                suggestion = "Model file may be corrupted or incompatible. Try re-downloading the model weights"
            else:
                suggestion = "Check PyTorch installation and dependencies"
                
            logger.error(f"Suggestion: {suggestion}")
            raise RuntimeError(f"{error_msg}. {suggestion}")
    
    def _load_model(self):
        """Legacy method - redirects to safe loading."""
        return self._load_model_safe()
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict watermark presence for a single image with optimized loading.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with prediction results including confidence scores
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        try:
            # Optimized image loading (load once)
            start_time = time.time()
            pil_image = Image.open(image_path).convert('RGB')
            
            # Suppress warnings during prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self.predictor.predict_image(pil_image)
                
            prediction_time = time.time() - start_time
            
            # Extract results from the new format
            has_watermark = bool(result['prediction'])
            confidence_clean = result['confidence_clean']
            confidence_watermarked = result['confidence_watermarked']
            
            return {
                'image_path': image_path,
                'has_watermark': has_watermark,
                'prediction': 'watermarked' if has_watermark else 'clean',
                'confidence': result['prediction'],  # Keep for backward compatibility
                'confidence_clean': confidence_clean,
                'confidence_watermarked': confidence_watermarked,
                'confidence_percentage': confidence_watermarked * 100 if has_watermark else confidence_clean * 100,
                'raw_outputs': result['raw_outputs'],
                'prediction_time': prediction_time,
                'model_used': self.model_name
            }
            
        except FileNotFoundError:
            error_msg = f"File not found: {image_path}"
            logger.error(error_msg)
        except Exception as e:
            # Check for specific error types
            error_str = str(e).lower()
            if "unidentified image" in error_str or "cannot identify image" in error_str:
                error_msg = f"Corrupted or unsupported image format: {image_path}"
            elif "cuda" in error_str and "memory" in error_str:
                error_msg = f"GPU memory exhausted processing: {image_path}"
            elif "dll" in error_str:
                error_msg = f"DLL/library error processing: {image_path}"
            else:
                error_msg = f"Unexpected error processing {image_path}: {e}"
            
            logger.error(error_msg)
        
        # Return error result for any exception
        return {
            'image_path': image_path,
            'has_watermark': None,
            'prediction': 'error',
            'error': error_msg,
            'model_used': self.model_name
        }
    
    def predict_batch(
        self, 
        image_paths: List[str], 
        batch_size: int = 8, 
        num_workers: int = 4,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict watermark presence for multiple images in batch.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            num_workers: Number of worker threads
            show_progress: Show progress bar
            
        Returns:
            List of prediction dictionaries
        """
        logger.info(f"Processing {len(image_paths)} images in batch mode")
        
        # Filter existing files
        valid_paths = []
        invalid_paths = []
        
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                invalid_paths.append(path)
                logger.warning(f"File not found: {path}")
        
        if not valid_paths:
            logger.error("No valid image paths found")
            return []
        
        try:
            start_time = time.time()
            
            # Run batch prediction
            results = self.predictor.run(
                valid_paths,
                num_workers=num_workers,
                bs=batch_size,
                pbar=show_progress
            )
            
            total_time = time.time() - start_time
            avg_time_per_image = total_time / len(valid_paths)
            
            # Format results
            formatted_results = []
            for i, (path, result) in enumerate(zip(valid_paths, results)):
                # Handle new result format
                if isinstance(result, dict):
                    has_watermark = bool(result['prediction'])
                    confidence_clean = result['confidence_clean']
                    confidence_watermarked = result['confidence_watermarked']
                else:
                    # Backward compatibility
                    has_watermark = bool(result)
                    confidence_clean = 1.0 - result if result <= 1 else 0.0
                    confidence_watermarked = result if result <= 1 else 1.0
                
                formatted_results.append({
                    'image_path': path,
                    'has_watermark': has_watermark,
                    'prediction': 'watermarked' if has_watermark else 'clean',
                    'confidence': result['prediction'] if isinstance(result, dict) else result,
                    'confidence_clean': confidence_clean,
                    'confidence_watermarked': confidence_watermarked,
                    'confidence_percentage': confidence_watermarked * 100 if has_watermark else confidence_clean * 100,
                    'raw_outputs': result.get('raw_outputs', []) if isinstance(result, dict) else [],
                    'model_used': self.model_name
                })
            
            # Add invalid paths as errors
            for path in invalid_paths:
                formatted_results.append({
                    'image_path': path,
                    'has_watermark': None,
                    'prediction': 'error',
                    'error': 'File not found',
                    'model_used': self.model_name
                })
            
            logger.info(f"Batch processing completed in {total_time:.2f}s")
            logger.info(f"Average time per image: {avg_time_per_image:.3f}s")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache to prevent memory buildup."""
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def predict_single_image_with_retry(self, image_path: str, max_retries: int = 3) -> Dict:
        """
        Predict with retry mechanism for transient failures.
        
        Args:
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with prediction results
        """
        for attempt in range(max_retries):
            try:
                result = self.predict_single_image(image_path)
                if result.get('prediction') != 'error':
                    return result
                    
                # If we got an error result, retry on certain error types
                error_msg = result.get('error', '').lower()
                if any(keyword in error_msg for keyword in ['memory', 'cuda', 'dll']):
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {image_path}, retrying...")
                        time.sleep(1)  # Brief pause before retry
                        self.clear_gpu_cache()  # Clear cache before retry
                        continue
                
                return result  # Return error result if not retryable
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {image_path}: {e}")
                    time.sleep(1)
                    self.clear_gpu_cache()
                    continue
                # Final attempt failed
                return {
                    'image_path': image_path,
                    'has_watermark': None,
                    'prediction': 'error',
                    'error': f"Failed after {max_retries} attempts: {e}",
                    'model_used': self.model_name
                }
    
    def scan_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        supported_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
        batch_size: int = 8,
        num_workers: int = 4,
        process_individually: bool = False
    ) -> List[Dict]:
        """
        Scan a directory for images and detect watermarks.
        
        Args:
            directory: Directory to scan
            recursive: Scan subdirectories recursively
            supported_extensions: Supported image extensions
            batch_size: Batch size for processing (ignored if process_individually=True)
            num_workers: Number of worker threads (ignored if process_individually=True)
            process_individually: Process images one by one instead of in batches
            
        Returns:
            List of prediction results
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        logger.info(f"Scanning directory: {directory}")
        
        # Find all images
        image_paths = []
        if recursive:
            image_paths = list_images(directory)
        else:
            for file in os.listdir(directory):
                if file.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(directory, file))
        
        logger.info(f"Found {len(image_paths)} images")
        
        if not image_paths:
            logger.warning("No images found in directory")
            return []
        
        # Process individually or in batch
        if process_individually:
            return self.predict_individually(image_paths)
        else:
            return self.predict_batch(
                image_paths, 
                batch_size=batch_size, 
                num_workers=num_workers
            )
    
    def predict_individually(self, image_paths: List[str]) -> List[Dict]:
        """
        Process images one by one with detailed progress tracking.
        
        Args:
            image_paths: List of image paths to process
            
        Returns:
            List of prediction results
        """
        logger.info(f"Processing {len(image_paths)} images individually...")
        
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Process single image
                result = self.predict_single_image(image_path)
                results.append(result)
                
                # Log result
                if result.get('has_watermark') is True:
                    confidence = result.get('confidence_watermarked', 0) * 100
                    logger.info(f"  → WATERMARKED ({confidence:.1f}% confidence)")
                elif result.get('has_watermark') is False:
                    confidence = result.get('confidence_clean', 0) * 100
                    logger.info(f"  → CLEAN ({confidence:.1f}% confidence)")
                else:
                    logger.warning(f"  → ERROR: {result.get('error', 'Unknown error')}")
                
                # Show progress
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining_time = avg_time * (len(image_paths) - i)
                
                logger.info(f"  Progress: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%) | "
                          f"Time per image: {avg_time:.2f}s | "
                          f"ETA: {remaining_time:.1f}s")
                
            except Exception as e:
                logger.error(f"  → ERROR processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'has_watermark': None,
                    'prediction': 'error',
                    'error': str(e),
                    'model_used': self.model_name
                })
            
            print("-" * 80)  # Visual separator
        
        total_time = time.time() - start_time
        logger.info(f"Individual processing completed in {total_time:.2f}s")
        logger.info(f"Average time per image: {total_time/len(image_paths):.3f}s")
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> Dict:
        """
        Generate a comprehensive report from prediction results.
        
        Args:
            results: List of prediction results
            output_file: Optional file to save the report
            
        Returns:
            Report dictionary
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        # Analyze results
        total_images = len(results)
        watermarked_count = sum(1 for r in results if r.get('has_watermark') is True)
        clean_count = sum(1 for r in results if r.get('has_watermark') is False)
        error_count = sum(1 for r in results if r.get('has_watermark') is None)
        
        # Calculate percentages
        watermarked_pct = (watermarked_count / total_images) * 100 if total_images > 0 else 0
        clean_pct = (clean_count / total_images) * 100 if total_images > 0 else 0
        error_pct = (error_count / total_images) * 100 if total_images > 0 else 0
        
        # Get prediction times if available
        times = [r.get('prediction_time', 0) for r in results if 'prediction_time' in r]
        avg_time = np.mean(times) if times else 0
        
        report = {
            'summary': {
                'total_images': total_images,
                'watermarked_images': watermarked_count,
                'clean_images': clean_count,
                'error_images': error_count,
                'watermarked_percentage': round(watermarked_pct, 2),
                'clean_percentage': round(clean_pct, 2),
                'error_percentage': round(error_pct, 2),
                'average_processing_time': round(avg_time, 3),
                'model_used': self.model_name
            },
            'detailed_results': results,
            'watermarked_files': [r['image_path'] for r in results if r.get('has_watermark') is True],
            'clean_files': [r['image_path'] for r in results if r.get('has_watermark') is False],
            'error_files': [r['image_path'] for r in results if r.get('has_watermark') is None]
        }
        
        # Save report if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def print_results_table(self, results: List[Dict]) -> None:
        """
        Print detection results in a formatted ASCII table.
        
        Args:
            results: List of prediction results
        """
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "=" * 120)
        print("WATERMARK DETECTION RESULTS")
        print("=" * 120)
        
        # Table header
        header = f"{'#':<3} {'File Name':<35} {'Status':<12} {'Confidence':<12} {'Clean %':<10} {'Watermark %':<12} {'Time (s)':<8}"
        print(header)
        print("-" * 120)
        
        # Table rows
        for i, result in enumerate(results, 1):
            filename = os.path.basename(result['image_path'])
            # Truncate long filenames
            if len(filename) > 32:
                filename = filename[:29] + "..."
            
            status = result.get('prediction', 'unknown')
            
            if result.get('has_watermark') is None:
                # Error case
                confidence = "ERROR"
                clean_conf = "N/A"
                watermark_conf = "N/A"
                time_taken = "N/A"
            else:
                confidence = f"{result.get('confidence_percentage', 0):.1f}%"
                clean_conf = f"{result.get('confidence_clean', 0) * 100:.1f}%"
                watermark_conf = f"{result.get('confidence_watermarked', 0) * 100:.1f}%"
                time_taken = f"{result.get('prediction_time', 0):.3f}"
            
            # Color coding with text indicators and high-confidence flagging
            if status == 'watermarked':
                # Flag high-confidence watermarks (>95%)
                watermark_confidence = result.get('confidence_watermarked', 0) * 100
                if watermark_confidence > 95:
                    status_display = "FLAGGED"  # High confidence watermark
                else:
                    status_display = "WATERMARK"
            elif status == 'clean':
                status_display = "CLEAN"
            else:
                status_display = "ERROR"
            
            row = f"{i:<3} {filename:<35} {status_display:<12} {confidence:<12} {clean_conf:<10} {watermark_conf:<12} {time_taken:<8}"
            print(row)
        
        print("-" * 120)
        
        # Summary statistics
        total = len(results)
        watermarked = sum(1 for r in results if r.get('has_watermark') is True)
        clean = sum(1 for r in results if r.get('has_watermark') is False)
        errors = sum(1 for r in results if r.get('has_watermark') is None)
        flagged = sum(1 for r in results if r.get('has_watermark') is True and r.get('confidence_watermarked', 0) * 100 > 95)
        
        print(f"SUMMARY: Total: {total} | Clean: {clean} ({clean/total*100:.1f}%) | Watermarked: {watermarked} ({watermarked/total*100:.1f}%) | Flagged (>95%): {flagged} | Errors: {errors} ({errors/total*100:.1f}%)")
        print("=" * 120)

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'fp16_enabled': self.fp16,
            'cache_directory': self.cache_dir,
            'model_loaded': hasattr(self, 'model') and self.model is not None
        }
    
    def get_flagged_images(self, results: List[Dict], confidence_threshold: float = 95.0) -> List[Dict]:
        """
        Get images flagged as high-confidence watermarks.
        
        Args:
            results: List of prediction results
            confidence_threshold: Minimum confidence percentage to flag (default: 95%)
            
        Returns:
            List of flagged image results
        """
        flagged_images = []
        for result in results:
            if (result.get('has_watermark') is True and 
                result.get('confidence_watermarked', 0) * 100 > confidence_threshold):
                flagged_images.append(result)
        
        return flagged_images

    def organize_images_by_confidence(
        self, 
        results: List[Dict], 
        base_output_dir: str = "Results",
        high_confidence_threshold: float = 96.0,
        manual_review_threshold: float = 91.0
    ) -> Dict[str, List[str]]:
        """
        Organize images into folders based on watermark confidence levels.
        
        Args:
            results: List of prediction results
            base_output_dir: Base directory for organized images (default: "Results")
            high_confidence_threshold: Threshold for flagged watermarks (default: 96%)
            manual_review_threshold: Threshold for manual review (default: 91%)
            
        Returns:
            Dictionary with organization statistics
        """
        import shutil
        
        # Create output directories using unified structure
        flagged_dir = os.path.join(base_output_dir, "invalid")  # Watermarked images are invalid
        manual_review_dir = os.path.join(base_output_dir, "manualreview")
        valid_dir = os.path.join(base_output_dir, "valid")
        
        for directory in [flagged_dir, manual_review_dir, valid_dir]:
            os.makedirs(directory, exist_ok=True)
        
        organization_stats = {
            'invalid': [],  # Watermarked images (flagged)
            'manualreview': [],  # Manual review needed
            'valid': [],  # Clean images
            'errors': []
        }
        
        logger.info(f"Organizing {len(results)} images into confidence-based folders...")
        
        for result in results:
            source_path = result['image_path']
            filename = os.path.basename(source_path)
            
            try:
                if result.get('has_watermark') is None:
                    # Error case - skip
                    organization_stats['errors'].append(filename)
                    continue
                
                watermark_confidence = result.get('confidence_watermarked', 0) * 100
                
                if result.get('has_watermark') and watermark_confidence > high_confidence_threshold:
                    # High confidence watermark - flagged as invalid
                    dest_path = os.path.join(flagged_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    organization_stats['invalid'].append(filename)
                    logger.debug(f"Invalid (watermarked): {filename} ({watermark_confidence:.1f}%)")
                    
                elif result.get('has_watermark') and watermark_confidence > manual_review_threshold:
                    # Medium confidence watermark - manual review
                    dest_path = os.path.join(manual_review_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    organization_stats['manualreview'].append(filename)
                    logger.debug(f"Manual review: {filename} ({watermark_confidence:.1f}%)")
                    
                else:
                    # Clean or low confidence - valid
                    dest_path = os.path.join(valid_dir, filename)
                    shutil.copy2(source_path, dest_path)
                    organization_stats['valid'].append(filename)
                    logger.debug(f"Valid: {filename}")
                    
            except Exception as e:
                logger.error(f"Error organizing {filename}: {e}")
                organization_stats['errors'].append(filename)
        
        # Log summary
        logger.info(f"Organization complete:")
        logger.info(f"  - Invalid (watermarked) (>{high_confidence_threshold}%): {len(organization_stats['invalid'])}")
        logger.info(f"  - Manual review ({manual_review_threshold}-{high_confidence_threshold}%): {len(organization_stats['manualreview'])}")
        logger.info(f"  - Valid images: {len(organization_stats['valid'])}")
        logger.info(f"  - Errors: {len(organization_stats['errors'])}")
        
        return organization_stats

    def print_organization_summary(self, organization_stats: Dict[str, List[str]]) -> None:
        """
        Print a summary of the image organization results.
        
        Args:
            organization_stats: Dictionary with organization statistics
        """
        print("\n" + "=" * 80)
        print("IMAGE ORGANIZATION SUMMARY")
        print("=" * 80)
        
        total_processed = sum(len(files) for files in organization_stats.values())
        
        categories = [
            ("INVALID (WATERMARKED) (>96%)", organization_stats['invalid'], "invalid/"),
            ("MANUAL REVIEW (91-96%)", organization_stats['manualreview'], "manualreview/"),
            ("VALID IMAGES", organization_stats['valid'], "valid/"),
            ("ERRORS", organization_stats['errors'], "errors/")
        ]
        
        for category_name, files, folder_name in categories:
            count = len(files)
            percentage = (count / total_processed * 100) if total_processed > 0 else 0
            
            print(f"\n{category_name}")
            print(f"  Count: {count} ({percentage:.1f}%)")
            print(f"  Folder: Results/{folder_name}")
            
            if files and count <= 10:  # Show files if 10 or fewer
                print("  Files:")
                for file in files:
                    print(f"    - {file}")
            elif files and count > 10:  # Show first 5 and last 5 if more than 10
                print("  Files (showing first 5 and last 5):")
                for file in files[:5]:
                    print(f"    - {file}")
                print(f"    ... ({count - 10} more files) ...")
                for file in files[-5:]:
                    print(f"    - {file}")
        
        print("\n" + "=" * 80)

    def print_main_style_summary(self, results: List[Dict]) -> None:
        """
        Print summary in the same style as main_optimized.py for consistency.
        
        Args:
            results: List of prediction results
        """
        print("=" * 120)
        print("WATERMARK DETECTION RESULTS")
        print("=" * 120)
        
        print(f"\nTESTS PERFORMED:")
        print(f"  Enabled: watermarks")
        print(f"  Disabled: editing, specifications, text, borders")
        
        # Calculate statistics
        total = len(results)
        clean_count = sum(1 for r in results if r.get('has_watermark') is False)
        error_count = sum(1 for r in results if r.get('has_watermark') is None)
        
        # Count watermarks by confidence levels
        high_confidence_watermarks = sum(1 for r in results if r.get('has_watermark') is True and 
                                        r.get('confidence_watermarked', 0) >= 0.96)
        manual_review_watermarks = sum(1 for r in results if r.get('has_watermark') is True and 
                                      0.91 <= r.get('confidence_watermarked', 0) < 0.96)
        low_confidence_watermarks = sum(1 for r in results if r.get('has_watermark') is True and 
                                       r.get('confidence_watermarked', 0) < 0.91)
        
        # Calculate totals for display
        invalid_count = high_confidence_watermarks + error_count  # Only high-confidence watermarks + errors are invalid
        valid_count = clean_count + low_confidence_watermarks  # Clean images + low-confidence watermarks are valid
        manual_review_count = manual_review_watermarks
        
        print(f"\nSUMMARY:")
        print(f"  Total Images Processed: {total}")
        print(f"  Valid Images: {valid_count}")
        print(f"  Invalid Images: {invalid_count}")
        print(f"  Manual Review Needed (moved): {manual_review_count}")
        print(f"  Success Rate: {(valid_count/total*100):.1f}%" if total > 0 else "  Success Rate: 0%")
        
        # Group results by categories
        valid_images = []
        invalid_images = []
        manual_review_images = []
        
        for result in results:
            filename = os.path.basename(result.get('image_path', ''))
            
            if result.get('has_watermark') is True:
                confidence = result.get('confidence_watermarked', 0) * 100
                reason = f"Watermark detected (confidence: {confidence:.1f}%)"
                
                # Match the file organization logic:
                # High confidence (96%+) → Invalid
                # Medium confidence (91-96%) → Manual Review  
                # Low confidence (<91%) → Valid (treated as clean/acceptable)
                if confidence >= 96.0:
                    invalid_images.append((filename, reason))
                elif confidence >= 91.0:
                    manual_review_images.append((filename, reason))
                else:
                    # Low confidence watermarks are treated as valid/clean
                    valid_images.append((filename, f"Low confidence watermark ({confidence:.1f}%) - treated as clean"))
            elif result.get('has_watermark') is False:
                confidence = result.get('confidence_clean', 0) * 100
                valid_images.append((filename, f"Clean image (confidence: {confidence:.1f}%)"))
            else:
                error_reason = result.get('error', 'Processing error')
                invalid_images.append((filename, error_reason))
        
        # Display each category
        if valid_images:
            print(f"\nVALID IMAGES ({len(valid_images)} images):")
            print("-" * 120)
            print(f"All validation checks passed - images copied to 'Results\\valid' folder")
        
        if invalid_images:
            print(f"\nINVALID IMAGES ({len(invalid_images)} images):")
            print("-" * 120)
            
            for i, (filename, reason) in enumerate(invalid_images, 1):
                print(f"\n{i:2d}. {filename}")
                print(f"    Failures:")
                print(f"      • Watermarks: {reason}")
        
        if manual_review_images:
            print(f"\nMANUAL REVIEW NEEDED ({len(manual_review_images)} images):")
            print("-" * 120)
            
            for i, (filename, reason) in enumerate(manual_review_images, 1):
                print(f"\n{i:2d}. {filename}")
                print(f"    High-confidence watermark detected: {reason}")
        
        print(f"\nOUTPUT STRUCTURE:")
        print(f"  Valid images: Results\\valid")
        print(f"  Invalid images: Results\\invalid")
        print(f"  Manual review needed: Results\\manualreview")
        print(f"  Processing logs: Results\\logs")
        
        print("\n" + "=" * 120)

    def run_self_test(self) -> bool:
        """
        Run self-test to verify model functionality.
        
        Returns:
            bool: True if self-test passes, False otherwise
        """
        import tempfile
        
        try:
            logger.info("Running self-test...")
            
            # Create small test image
            test_image = Image.new('RGB', (224, 224), color='white')
            
            # Use a proper temp file that gets closed before use
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            
            try:
                # Save image to the closed temp file
                test_image.save(temp_path, 'JPEG')
                
                # Test prediction
                result = self.predict_single_image(temp_path)
                
                # Check if prediction succeeded
                success = (
                    result.get('prediction') is not None and 
                    result.get('prediction') != 'error' and
                    'confidence_clean' in result and
                    'confidence_watermarked' in result and
                    'has_watermark' in result and
                    result.get('has_watermark') is not None  # Should be True or False, not None
                )
                
                if success:
                    logger.info("Self-test passed successfully")
                    return True
                else:
                    logger.error(f"Self-test failed: Invalid result format: {result}")
                    return False
                    
            finally:
                # Cleanup temp file
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
                    
        except Exception as e:
            logger.error(f"Self-test failed with exception: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get comprehensive information about the current model and system."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'fp16_enabled': self.fp16,
            'cache_directory': self.cache_dir,
            'model_loaded': hasattr(self, 'model') and self.model is not None,
            'predictor_ready': hasattr(self, 'predictor') and self.predictor is not None,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'initialization_error': getattr(self, 'initialization_error', None)
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Watermark Detection for PhotoValidator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, default='photos4testing',
                        help='Input directory containing images to analyze (default: photos4testing)')
    
    parser.add_argument('--output', '-o', type=str, default='Results',
                        help='Output directory for results (default: Results)')
    
    parser.add_argument('--model', '-m', type=str, default='convnext-tiny',
                        choices=['convnext-tiny', 'convnext-small', 'convnext-base'],
                        help='Model to use for detection (default: convnext-tiny)')
    
    parser.add_argument('--high-threshold', type=float, default=96.0,
                        help='High confidence threshold for automatic classification (default: 96.0)')
    
    parser.add_argument('--manual-threshold', type=float, default=91.0,
                        help='Manual review threshold (default: 91.0)')
    
    return parser.parse_args()

def main():
    """Example usage and testing."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("Advanced Watermark Detector - Testing")
    print("=" * 50)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"High confidence threshold: {args.high_threshold}%")
    print(f"Manual review threshold: {args.manual_threshold}%")
    print("=" * 50)
    
    try:
        # Create configuration from command line arguments
        config = WatermarkDetectorConfig()
        config.model_name = args.model
        config.high_confidence_threshold = args.high_threshold
        config.manual_review_threshold = args.manual_threshold
        
        # Initialize detector with configuration
        detector = AdvancedWatermarkDetector(config=config)

        print(f"Model info: {detector.get_model_info()}")
        
        # Run self-test
        if detector.run_self_test():
            print("Self-test passed - detector ready")
        else:
            print("! Self-test failed - there may be issues")

        # Use provided input directory or check standard locations
        test_directories = [
            args.input,
            f"./{args.input}",
            f"C:\\Users\\Public\\Python\\ittask\\{args.input}"
        ]
        
        test_images_dir = None
        for test_dir in test_directories:
            if os.path.exists(test_dir):
                test_images_dir = test_dir
                break
        
        if test_images_dir:
            print(f"\nFound test images directory: {test_images_dir}")
            
            # List images first to show what we're about to process
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
            test_images = []
            for root, dirs, files in os.walk(test_images_dir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        test_images.append(os.path.join(root, file))
            
            print(f"Found {len(test_images)} images to process")
            if test_images:
                print("Sample images:")
                for i, img in enumerate(test_images[:5]):
                    print(f"  {i+1}. {os.path.basename(img)}")
                if len(test_images) > 5:
                    print(f"  ... and {len(test_images) - 5} more")
            
            print(f"\nStarting individual processing of all images...")
            print("=" * 80)
            
            # Process images individually (one by one)
            results = detector.scan_directory(
                test_images_dir, 
                recursive=True,
                process_individually=True  # This ensures one-by-one processing
            )

            # Generate and display report in main pipeline style
            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)

            # Use the main pipeline style summary
            detector.print_main_style_summary(results)
            
            # Organize images by confidence levels (but don't show the old summary)
            organization_stats = detector.organize_images_by_confidence(
                results, 
                base_output_dir=args.output,
                high_confidence_threshold=args.high_threshold,
                manual_review_threshold=args.manual_threshold
            )
            
        else:
            print("No test images directory found. Checked locations:")
            for test_dir in test_directories:
                print(f"  - {test_dir}")
            print("\nPlease ensure test images are available in one of these locations.")
            
            # Create a sample directory structure
            print("\nCreating sample directory structure...")
            sample_dir = f"./{args.input}"
            os.makedirs(sample_dir, exist_ok=True)
            print(f"Created directory: {sample_dir}")
            print("Please add test images to this directory and run again.")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
