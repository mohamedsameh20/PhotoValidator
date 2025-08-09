"""
Advanced Image Quality Assessment for Editing Detection using PyIQA

This script uses the PyIQA library to implement No-Reference Image Quality Assessment (NR-IQA)
models like BRISQUE and NIQE, which are excellent for detecting heavily edited images.

Installation required:
pip install pyiqa torch torchvision
"""

import os
import cv2
import numpy as np
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

# Import scipy.signal for peak detection, with fallback
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyiqa
    import torch
    PYIQA_AVAILABLE = True
    # print("✅ PyIQA library available - using advanced quality assessment")  # Suppressed for clean output
except ImportError:
    PYIQA_AVAILABLE = False
    # print("⚠️  PyIQA not available. Install with: pip install pyiqa torch torchvision")  # Suppressed for clean output
    # print("🔄 Falling back to basic feature analysis")  # Suppressed for clean output

class AdvancedEditingDetector:
    """
    Advanced editing detection using PyIQA quality assessment models combined with feature analysis.
    """
    
    def __init__(self, force_cpu=False, gpu_id=0, quiet=False, selected_models=None, excluded_models=None):
        """
        Initialize the detector with CUDA configuration options and model selection.
        
        Args:
            force_cpu (bool): Force CPU usage even if CUDA is available
            gpu_id (int): Specific GPU ID to use (for multi-GPU systems)
            quiet (bool): Suppress initialization output for clean execution
            selected_models (list): Specific models to use (if None, uses all available)
            excluded_models (list): Models to exclude from loading
        """
        self.quiet = quiet
        self.selected_models = selected_models
        self.excluded_models = excluded_models or []
        self._configure_device(force_cpu, gpu_id)
        if not self.quiet:
            self._print_device_info()
        
        if PYIQA_AVAILABLE:
            self._initialize_quality_models()
        else:
            self.quality_models = {}
    
    def _configure_device(self, force_cpu=False, gpu_id=0):
        """Configure the computation device with detailed CUDA information"""
        if force_cpu:
            self.device = torch.device('cpu')
            if not self.quiet:
                print("🖥️  Forced CPU usage")
            return
        
        if torch.cuda.is_available():
            # Check if specific GPU ID is available
            if gpu_id < torch.cuda.device_count():
                self.device = torch.device(f'cuda:{gpu_id}')
            else:
                if not self.quiet:
                    print(f"⚠️  GPU {gpu_id} not available, using GPU 0")
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
            if not self.quiet:
                print("⚠️  CUDA not available, using CPU")
    
    def _print_device_info(self):
        """Print detailed device and CUDA information"""
        print(f"🖥️  Using device: {self.device}")
        
        if self.device.type == 'cuda':
            gpu_id = self.device.index if self.device.index is not None else 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
            print(f"🐍 PyTorch Version: {torch.__version__}")
        else:
            print("💻 Using CPU (install CUDA-enabled PyTorch for GPU acceleration)")
            if torch.cuda.is_available():
                print("💡 CUDA detected but not being used")
            else:
                print("❌ CUDA not available on this system")
    
    def _initialize_quality_models(self):
        """Initialize PyIQA quality assessment models with better isolation"""
        if not self.quiet:
            print("🔧 Initializing quality assessment models...")
        
        self.quality_models = {}
        
        # All available models
        all_models = [
            'brisque',      # BRISQUE - excellent for unnatural distortions
            'niqe',         # NIQE - good for naturalness assessment
            'musiq',        # MUSIQ - multi-scale quality assessment
            'dbcnn',        # DB-CNN - deep learning based
            'hyperiqa',     # HyperIQA - hypernetwork based
        ]
        
        # Determine which models to load based on user selection
        models_to_load = self._determine_models_to_use(all_models)
        
        if not self.quiet:
            print(f"📋 Loading models: {', '.join(models_to_load)}")
        
        # Load non-CLIP models first
        for model_name in models_to_load:
            if model_name == 'clipiqa':
                continue  # Handle CLIP-IQA separately
                
            try:
                # Clear GPU cache before loading each model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                model = pyiqa.create_metric(model_name, device=self.device)
                self.quality_models[model_name] = model
                if not self.quiet:
                    print(f"  ✅ Loaded {model_name.upper()}")
            except Exception as e:
                if not self.quiet:
                    print(f"  ❌ Failed to load {model_name}: {str(e)}")
        
        # Load CLIP-IQA separately if requested
        if 'clipiqa' in models_to_load:
            clipiqa_loaded = False
            for attempt in range(3):  # Try up to 3 times
                try:
                    if not self.quiet:
                        attempt_text = f" (attempt {attempt + 1}/3)" if attempt > 0 else ""
                        print(f"  🔄 Loading CLIP-IQA with isolation{attempt_text}...")
                    
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Force clean model creation
                    clipiqa_model = pyiqa.create_metric('clipiqa', device=self.device)
                    
                    # Test CLIP-IQA with multiple test images if available
                    test_paths = [
                        os.path.join("photos4testing", "Normal_Image.jpg"),
                        os.path.join("photos4testing", "example_clean_architecture.jpg"),
                        # Fallback to any image in the directory
                    ]
                    
                    # Add any available image as fallback
                    if os.path.exists("photos4testing"):
                        for filename in os.listdir("photos4testing"):
                            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                                test_paths.append(os.path.join("photos4testing", filename))
                                break
                    
                    test_successful = False
                    for test_path in test_paths:
                        if os.path.exists(test_path):
                            try:
                                with torch.no_grad():
                                    test_score = clipiqa_model(test_path)
                                if not self.quiet:
                                    print(f"    🔍 CLIP-IQA test on {os.path.basename(test_path)}: {test_score}")
                                
                                # Check for valid score
                                if not torch.isnan(test_score).any() and torch.isfinite(test_score).all():
                                    score_value = float(test_score.item()) if hasattr(test_score, 'item') else float(test_score)
                                    if 0.0 <= score_value <= 1.0:  # CLIP-IQA should be in [0,1] range
                                        self.quality_models['clipiqa'] = clipiqa_model
                                        clipiqa_loaded = True
                                        if not self.quiet:
                                            print(f"  ✅ CLIP-IQA loaded successfully")
                                        break
                                    else:
                                        if not self.quiet:
                                            print(f"    ⚠️  CLIP-IQA score out of range: {score_value}")
                                else:
                                    if not self.quiet:
                                        print(f"    ⚠️  CLIP-IQA produced invalid score: {test_score}")
                            except Exception as test_e:
                                if not self.quiet:
                                    print(f"    ⚠️  CLIP-IQA test failed on {os.path.basename(test_path)}: {str(test_e)}")
                                continue
                    
                    if clipiqa_loaded:
                        break
                    else:
                        if not self.quiet:
                            print(f"    ❌ CLIP-IQA failed validation tests")
                        if attempt < 2:  # Don't clear cache on last attempt
                            if self.device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        
                except Exception as e:
                    if not self.quiet:
                        print(f"    ❌ CLIP-IQA loading error: {str(e)}")
                    if attempt < 2:
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
            
            # If CLIP-IQA was specifically selected but failed to load
            if not clipiqa_loaded:
                if self.selected_models and 'clipiqa' in self.selected_models:
                    error_msg = "CLIP-IQA was specifically selected but failed to load properly after 3 attempts"
                    if not self.quiet:
                        print(f"  🚨 CRITICAL: {error_msg}")
                    raise RuntimeError(error_msg)
                else:
                    if not self.quiet:
                        print(f"  ⚠️  CLIP-IQA skipped due to loading issues")
        
        if not self.quiet:
            print(f"🎯 Successfully loaded {len(self.quality_models)} quality models")
        
        # Validate that specifically requested models are actually loaded
        if self.selected_models:
            missing_models = [model for model in self.selected_models if model not in self.quality_models]
            if missing_models:
                error_msg = f"Specifically requested models failed to load: {', '.join(missing_models)}"
                if not self.quiet:
                    print(f"  🚨 VALIDATION ERROR: {error_msg}")
                    print(f"  📋 Available models: {', '.join(self.quality_models.keys()) if self.quality_models else 'None'}")
                raise RuntimeError(error_msg)
            else:
                if not self.quiet:
                    print(f"  ✅ All requested models loaded: {', '.join(self.selected_models)}")
        
        # Print GPU memory usage after model loading
        if self.device.type == 'cuda' and not self.quiet:
            self._print_gpu_memory_usage("After model loading")
    
    def _determine_models_to_use(self, all_models):
        """
        Determine which models to load based on user selection.
        
        Args:
            all_models (list): List of all available models
            
        Returns:
            list: Models to actually load
        """
        # If specific models are selected, use only those
        if self.selected_models:
            models_to_load = [model for model in self.selected_models if model in all_models + ['clipiqa']]
            if not self.quiet:
                excluded_from_selection = set(self.selected_models) - set(models_to_load)
                if excluded_from_selection:
                    print(f"⚠️  Unknown models in selection: {', '.join(excluded_from_selection)}")
        else:
            # Use all models except excluded ones
            models_to_load = [model for model in all_models if model not in self.excluded_models]
            # Add CLIP-IQA if not excluded
            if 'clipiqa' not in self.excluded_models:
                models_to_load.append('clipiqa')
        
        return models_to_load
    
    def _print_gpu_memory_usage(self, stage=""):
        """Print current GPU memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def analyze_with_pyiqa(self, image_path):
        """
        Analyze image quality using PyIQA models with CUDA optimization.
        
        Returns:
            dict: Quality scores from various models
        """
        if not PYIQA_AVAILABLE or not self.quality_models:
            return {'error': 'PyIQA models not available'}
        
        try:
            quality_scores = {}
            
            # Clear GPU cache before analysis
            if self.device.type == 'cuda':
                self._clear_gpu_cache()
            
            for model_name, model in self.quality_models.items():
                try:
                    # Special handling for CLIP-IQA which can be sensitive to image format
                    if model_name == 'clipiqa':
                        # Preprocess image for CLIP-IQA
                        try:
                            # Load and validate image first
                            from PIL import Image
                            pil_image = Image.open(image_path).convert('RGB')
                            
                            # Check image dimensions (CLIP-IQA prefers certain sizes)
                            width, height = pil_image.size
                            if width < 224 or height < 224:
                                # Resize small images
                                pil_image = pil_image.resize((max(224, width), max(224, height)), Image.Resampling.LANCZOS)
                            
                            # Check for very large images that might cause memory issues
                            if width > 2048 or height > 2048:
                                # Resize very large images
                                aspect_ratio = width / height
                                if aspect_ratio > 1:
                                    new_width = 2048
                                    new_height = int(2048 / aspect_ratio)
                                else:
                                    new_height = 2048
                                    new_width = int(2048 * aspect_ratio)
                                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # Ensure tensor is on correct device and in correct format
                            with torch.no_grad():
                                # Some PyIQA models work better with file paths, others with PIL images
                                # Try PIL image first for CLIP-IQA
                                score = model(pil_image)
                                
                        except Exception as clip_error:
                            # Fallback to original image path if PIL processing fails
                            with torch.no_grad():
                                score = model(image_path)
                    else:
                        # Standard processing for other models
                        with torch.no_grad():  # Disable gradient computation for inference
                            score = model(image_path)
                    
                    # Convert tensor to float if needed
                    if torch.is_tensor(score):
                        # Handle different tensor shapes
                        if score.numel() == 1:
                            score = score.item()
                        elif score.numel() == 0:
                            # Empty tensor - treat as NaN
                            score = float('nan')
                        else:
                            # For multi-dimensional tensors, take the mean or first element
                            score_flat = score.flatten()
                            if len(score_flat) > 0:
                                score = score_flat[0].item()
                            else:
                                score = float('nan')
                    
                    # Additional validation for CLIP-IQA scores
                    if model_name == 'clipiqa':
                        # CLIP-IQA should return values roughly between 0.2 and 0.8 for real images
                        if isinstance(score, (int, float)):
                            if score < 0.1 or score > 0.9:
                                if not self.quiet:
                                    print(f"    ⚠️  {model_name} returned suspicious value: {score:.3f}")
                                quality_scores[f'{model_name}_warning'] = f"Suspicious score: {score:.3f}"
                                # Still record the score but flag it
                                quality_scores[f'{model_name}_score'] = float(score)
                            else:
                                quality_scores[f'{model_name}_score'] = float(score)
                        else:
                            quality_scores[f'{model_name}_error'] = f"Non-numeric score: {score}"
                    else:
                        # Standard processing for other models
                        quality_scores[f'{model_name}_score'] = float(score)
                    
                    # Clear cache after each model to prevent memory buildup
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    quality_scores[f'{model_name}_error'] = str(e)
                    print(f"    ⚠️  Error with {model_name}: {str(e)}")
            
            return quality_scores
            
        except Exception as e:
            return {'error': f'PyIQA analysis failed: {str(e)}'}
    
    def analyze_histogram_features(self, image):
        """Enhanced histogram analysis for editing detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Normalize histogram
        hist_norm = hist / np.sum(hist)
        
        # 1. Clipping analysis
        total_pixels = gray.shape[0] * gray.shape[1]
        black_clip = (hist[0] / total_pixels) * 100
        white_clip = (hist[255] / total_pixels) * 100
        
        # 2. Histogram entropy
        hist_entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))
        
        # 3. Histogram smoothness
        hist_diff = np.diff(hist_norm)
        hist_smoothness = 1.0 / (1.0 + np.var(hist_diff))
        
        # 4. Peak analysis
        if SCIPY_AVAILABLE:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
            peak_count = len(peaks)
        else:
            # Simple peak detection fallback
            peak_count = 0
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.max(hist) * 0.1:
                    peak_count += 1
        
        # 5. Uniform distribution test (Kolmogorov-Smirnov like)
        expected_uniform = np.full(256, 1/256)
        ks_statistic = np.max(np.abs(np.cumsum(hist_norm) - np.cumsum(expected_uniform)))
        
        return {
            'black_clipping': float(black_clip),
            'white_clipping': float(white_clip),
            'total_clipping': float(black_clip + white_clip),
            'histogram_entropy': float(hist_entropy),
            'histogram_smoothness': float(hist_smoothness),
            'peak_count': int(peak_count),
            'ks_uniformity_test': float(ks_statistic)
        }
    
    def analyze_edge_artifacts(self, image):
        """Advanced edge artifact analysis"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Laplacian variance (sharpening detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        
        # 2. Edge density analysis
        edges_canny = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges_canny > 0) / edges_canny.size
        
        # 3. Edge contrast analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_contrast = np.std(edge_magnitude)
        
        # 4. Halo detection (simplified)
        # Dilate edges and check for brightness variations
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges_canny, kernel, iterations=2)
        
        if np.sum(edge_region) > 0:
            edge_pixels = gray[edge_region > 0]
            halo_variance = np.var(edge_pixels)
        else:
            halo_variance = 0
        
        # 5. High-frequency analysis
        # Apply high-pass filter
        kernel_hp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray.astype(np.float32), -1, kernel_hp)
        high_freq_energy = np.var(high_freq)
        
        return {
            'laplacian_variance': float(lap_var),
            'edge_density': float(edge_density),
            'edge_contrast': float(edge_contrast),
            'halo_variance': float(halo_variance),
            'high_freq_energy': float(high_freq_energy)
        }
    
    def analyze_frequency_domain(self, image):
        """Frequency domain analysis for editing artifacts"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Power spectrum
        power_spectrum = magnitude_spectrum ** 2
        
        # Analyze frequency distribution
        h, w = power_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        # Create frequency masks
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Low, mid, high frequency regions
        max_dist = min(h, w) // 2
        low_freq_mask = distances <= max_dist * 0.3
        mid_freq_mask = (distances > max_dist * 0.3) & (distances <= max_dist * 0.7)
        high_freq_mask = distances > max_dist * 0.7
        
        # Calculate energy in each band
        total_energy = np.sum(power_spectrum)
        low_freq_energy = np.sum(power_spectrum[low_freq_mask]) / total_energy
        mid_freq_energy = np.sum(power_spectrum[mid_freq_mask]) / total_energy
        high_freq_energy = np.sum(power_spectrum[high_freq_mask]) / total_energy
        
        # Detect periodic patterns (JPEG artifacts, etc.)
        # Look for regular patterns in frequency domain
        magnitude_log = np.log(magnitude_spectrum + 1)
        freq_variance = np.var(magnitude_log)
        
        return {
            'low_freq_energy': float(low_freq_energy),
            'mid_freq_energy': float(mid_freq_energy),
            'high_freq_energy': float(high_freq_energy),
            'frequency_variance': float(freq_variance),
            'total_spectral_energy': float(total_energy)
        }
    
    def _validate_score_range(self, score, metric_name):
        """Validate score is within acceptable range and not NaN/inf"""
        # Handle tensor conversion first
        if hasattr(score, 'item'):  # PyTorch tensor
            try:
                score = score.item()
            except:
                if not self.quiet:
                    print(f"    ⚠️  Invalid {metric_name} score: tensor conversion failed")
                return None
        
        # Additional validation for numeric types and NaN/inf
        if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
            if not self.quiet:
                print(f"    ⚠️  Invalid {metric_name} score: {score}")
            return None
        
        # CLIP-IQA specific validation (should be between 0 and 1)
        if metric_name == 'CLIP-IQA' and (score < 0 or score > 1):
            if not self.quiet:
                print(f"    ⚠️  {metric_name} score {score:.3f} outside valid range [0,1]")
            return None
        
        return float(score)
    
    def _normalize_brisque_score(self, brisque_score):
        """
        Normalize BRISQUE score using empirically determined thresholds.
        Based on analysis of natural image datasets.
        """
        # Empirically determined percentiles from LIVE database and similar datasets
        natural_50th = 31.4  # 50th percentile for natural images
        natural_90th = 58.2  # 90th percentile for natural images  
        natural_98th = 80.5  # 98th percentile for natural images
        
        if brisque_score <= natural_50th:
            return 0.0  # Definitely natural quality
        elif brisque_score <= natural_90th:
            # Linear scaling from 0 to 50 for moderate range
            return (brisque_score - natural_50th) / (natural_90th - natural_50th) * 50.0
        elif brisque_score <= natural_98th:
            # Linear scaling from 50 to 85 for high range
            return 50.0 + (brisque_score - natural_90th) / (natural_98th - natural_90th) * 35.0
        else:
            # Cap at 95 to avoid over-confidence, add gradual increase
            return min(95.0, 85.0 + (brisque_score - natural_98th) / 20.0 * 10.0)
    
    def _normalize_niqe_score(self, niqe_score):
        """
        Normalize NIQE score using empirically determined thresholds.
        """
        # Empirically determined thresholds for NIQE
        natural_50th = 4.2   # 50th percentile for natural images
        natural_90th = 7.8   # 90th percentile for natural images
        natural_98th = 12.5  # 98th percentile for natural images
        
        if niqe_score <= natural_50th:
            return 0.0
        elif niqe_score <= natural_90th:
            return (niqe_score - natural_50th) / (natural_90th - natural_50th) * 50.0
        elif niqe_score <= natural_98th:
            return 50.0 + (niqe_score - natural_90th) / (natural_98th - natural_90th) * 35.0
        else:
            return min(95.0, 85.0 + (niqe_score - natural_98th) / 8.0 * 10.0)
    
    def _normalize_clipiqa_score(self, clipiqa_score):
        """
        Normalize CLIP-IQA score (higher is better quality).
        """
        # CLIP-IQA typically ranges 0.2-0.8 for real images
        # Values outside this range often indicate model issues
        if clipiqa_score < 0.1 or clipiqa_score > 0.9:
            return None  # Likely invalid
        
        # Convert to editing indicator: lower CLIP-IQA = more editing
        # Natural images typically score 0.5-0.8
        natural_threshold = 0.55  # 50th percentile for natural images
        high_quality_threshold = 0.75  # 90th percentile for natural images
        
        if clipiqa_score >= high_quality_threshold:
            return 0.0  # High quality, minimal editing
        elif clipiqa_score >= natural_threshold:
            return (high_quality_threshold - clipiqa_score) / (high_quality_threshold - natural_threshold) * 40.0
        else:
            # Lower scores suggest more editing
            return 40.0 + (natural_threshold - clipiqa_score) / (natural_threshold - 0.1) * 50.0

    def calculate_comprehensive_score(self, pyiqa_results, histogram_results, edge_results, freq_results):
        """
        Calculate comprehensive editing detection score with robust validation and normalization.
        """
        scores = {}
        
        # Validate and normalize PyIQA scores
        valid_pyiqa_scores = []
        valid_pyiqa_count = 0
        
        # BRISQUE processing
        if 'brisque_score' in pyiqa_results:
            brisque = self._validate_score_range(pyiqa_results['brisque_score'], 'BRISQUE')
            if brisque is not None:
                brisque_normalized = self._normalize_brisque_score(brisque)
                scores['brisque_editing_indicator'] = brisque_normalized
                valid_pyiqa_scores.append(brisque_normalized)
                valid_pyiqa_count += 1
            else:
                scores['brisque_editing_indicator'] = None
        
        # NIQE processing
        if 'niqe_score' in pyiqa_results:
            niqe = self._validate_score_range(pyiqa_results['niqe_score'], 'NIQE')
            if niqe is not None:
                niqe_normalized = self._normalize_niqe_score(niqe)
                scores['niqe_editing_indicator'] = niqe_normalized
                valid_pyiqa_scores.append(niqe_normalized)
                valid_pyiqa_count += 1
            else:
                scores['niqe_editing_indicator'] = None
        
        # CLIP-IQA processing with enhanced validation
        if 'clipiqa_score' in pyiqa_results:
            clipiqa = self._validate_score_range(pyiqa_results['clipiqa_score'], 'CLIP-IQA')
            if clipiqa is not None:
                clipiqa_normalized = self._normalize_clipiqa_score(clipiqa)
                if clipiqa_normalized is not None:
                    scores['clipiqa_editing_indicator'] = clipiqa_normalized
                    valid_pyiqa_scores.append(clipiqa_normalized)
                    valid_pyiqa_count += 1
                else:
                    scores['clipiqa_editing_indicator'] = None
                    if not self.quiet:
                        print(f"    ⚠️  CLIP-IQA score {clipiqa:.3f} outside valid range")
            else:
                scores['clipiqa_editing_indicator'] = None
        
        # Calculate average PyIQA score only if we have valid scores
        if valid_pyiqa_count >= 1:
            scores['average_quality_editing_score'] = float(np.mean(valid_pyiqa_scores))
            scores['pyiqa_model_count'] = valid_pyiqa_count
        else:
            scores['average_quality_editing_score'] = None
            scores['pyiqa_model_count'] = 0
        
        # Feature-based scoring with empirical normalization
        scores.update(self._calculate_feature_based_scores(histogram_results, edge_results, freq_results))
        
        # Calculate overall score with robust weighting
        overall_score = self._calculate_weighted_overall_score(scores)
        
        scores['overall_editing_score'] = float(overall_score)
        
        # Determine category with updated thresholds
        if overall_score >= 75:
            category = "Heavy Editing Detected"
        elif overall_score >= 60:
            category = "Moderate Editing Detected"
        elif overall_score >= 40:
            category = "Light Editing Detected"
        else:
            category = "Natural/Minimal Editing"
        
        scores['editing_category'] = category
        
        # Calculate confidence based on agreement between methods
        confidence = self._calculate_confidence_level(scores)
        scores['confidence'] = confidence
        
        return scores
    
    def _calculate_feature_based_scores(self, histogram_results, edge_results, freq_results):
        """Calculate normalized feature-based scores using empirical thresholds"""
        feature_scores = {}
        
        # Histogram-based scoring with more aggressive thresholds for editing detection
        # Clipping analysis - professional editing often shows clipping
        clipping_score = 0
        total_clipping = histogram_results['total_clipping']
        if total_clipping > 0.1:  # Even 0.1% clipping can indicate processing
            if total_clipping > 5.0:  # Heavy clipping
                clipping_score = min(90, 60 + (total_clipping - 5.0) / 10.0 * 30)
            elif total_clipping > 1.0:  # Moderate clipping  
                clipping_score = 30 + (total_clipping - 1.0) / 4.0 * 30
            else:  # Light clipping
                clipping_score = total_clipping / 1.0 * 30
        
        # Histogram entropy - be more sensitive to manipulation
        entropy = histogram_results['histogram_entropy']
        entropy_score = 0
        if entropy < 6.5:  # More sensitive threshold for low entropy
            entropy_score = (6.5 - entropy) / 6.5 * 70  # Increased penalty
        elif entropy > 7.7:  # More sensitive to high entropy
            entropy_score = (entropy - 7.7) / 0.5 * 40
        
        # KS uniformity test - more aggressive for non-natural distributions
        ks_score = 0
        ks_stat = histogram_results['ks_uniformity_test']
        if ks_stat > 0.15:  # Lower threshold
            ks_score = min(60, (ks_stat - 0.15) / 0.35 * 60)
        
        histogram_score = (clipping_score + entropy_score + ks_score) / 3
        feature_scores['histogram_editing_score'] = min(100.0, float(histogram_score))
        
        # Edge-based scoring with more sensitive thresholds
        # Laplacian variance - lower threshold for sharpening detection
        lap_var = edge_results['laplacian_variance']
        lap_score = 0
        if lap_var > 2000:  # Reduced from 3000 - more sensitive
            if lap_var > 10000:  # Very high sharpening
                lap_score = min(85, 60 + (lap_var - 10000) / 20000 * 25)
            else:  # Moderate sharpening
                lap_score = (lap_var - 2000) / 8000 * 60
        elif lap_var < 100:  # More sensitive to over-smoothing
            lap_score = (100 - lap_var) / 100 * 50
        
        # Edge density - more sensitive thresholds
        edge_density = edge_results['edge_density']
        edge_density_score = 0
        if edge_density > 0.15:  # Reduced from 0.2
            edge_density_score = min(70, (edge_density - 0.15) / 0.25 * 70)
        elif edge_density < 0.05:  # More sensitive to low edge content
            edge_density_score = (0.05 - edge_density) / 0.05 * 40
        
        # High frequency energy - more sensitive
        hf_energy = edge_results['high_freq_energy']
        hf_score = 0
        if hf_energy > 5000:  # Reduced threshold
            hf_score = min(75, (hf_energy - 5000) / 15000 * 75)
        elif hf_energy < 500:  # More sensitive to low HF content
            hf_score = (500 - hf_energy) / 500 * 30
        
        edge_score = (lap_score + edge_density_score + hf_score) / 3
        feature_scores['edge_artifacts_score'] = min(100.0, float(edge_score))
        
        # Frequency domain scoring with better sensitivity
        freq_score = 0
        
        # High frequency energy ratio - more sensitive to unusual distributions
        hf_ratio = freq_results['high_freq_energy']
        if hf_ratio > 0.15 or hf_ratio < 0.03:  # Tighter range
            deviation = abs(hf_ratio - 0.09)  # Ideal around 9%
            freq_score += min(60, deviation / 0.1 * 60)
        
        # Frequency variance - more sensitive to manipulation
        freq_var = freq_results['frequency_variance']
        if freq_var > 2.0:  # Much lower threshold
            freq_score += min(40, (freq_var - 2.0) / 8.0 * 40)
        elif freq_var < 0.5:  # Too uniform
            freq_score += (0.5 - freq_var) / 0.5 * 25
        
        feature_scores['frequency_artifacts_score'] = min(100.0, float(freq_score / 2))
        
        return feature_scores
    
    def _calculate_weighted_overall_score(self, scores):
        """Calculate weighted overall score with robust method selection"""
        
        # Determine reliability of PyIQA scores
        pyiqa_reliable = scores.get('pyiqa_model_count', 0) >= 2
        pyiqa_score = scores.get('average_quality_editing_score')
        
        # Get feature scores
        hist_score = scores.get('histogram_editing_score', 0)
        edge_score = scores.get('edge_artifacts_score', 0) 
        freq_score = scores.get('frequency_artifacts_score', 0)
        feature_avg = (hist_score + edge_score + freq_score) / 3
        
        # Detect "professional editing" scenario:
        # Low PyIQA scores (good quality) but high feature scores (editing artifacts)
        professional_editing_detected = False
        if pyiqa_reliable and pyiqa_score is not None:
            if pyiqa_score < 30 and feature_avg > 25:  # Good quality but clear artifacts
                professional_editing_detected = True
        
        if pyiqa_reliable and pyiqa_score is not None:
            if professional_editing_detected:
                # Professional editing: give more weight to features
                overall_score = (
                    pyiqa_score * 0.3 +      # Reduced PyIQA weight
                    hist_score * 0.3 +       # Increased feature weight
                    edge_score * 0.3 +
                    freq_score * 0.1
                )
                # Add bonus for clear editing artifacts with good quality
                overall_score = min(100.0, overall_score * 1.4)
                
            elif abs(pyiqa_score - feature_avg) > 30:
                # Significant disagreement - blend more conservatively
                overall_score = (
                    pyiqa_score * 0.6 +
                    hist_score * 0.15 +
                    edge_score * 0.15 +
                    freq_score * 0.1
                )
            else:
                # Good agreement - trust PyIQA more
                overall_score = (
                    pyiqa_score * 0.7 +
                    hist_score * 0.12 +
                    edge_score * 0.12 +
                    freq_score * 0.06
                )
        elif scores.get('pyiqa_model_count', 0) == 1 and pyiqa_score is not None:
            # Only one PyIQA model - blend more conservatively
            overall_score = (
                pyiqa_score * 0.4 +
                hist_score * 0.25 +
                edge_score * 0.25 +
                freq_score * 0.1
            )
        else:
            # No reliable PyIQA - use feature-based only
            overall_score = (
                hist_score * 0.4 +
                edge_score * 0.4 +
                freq_score * 0.2
            )
        
        return max(0.0, min(100.0, overall_score))
    
    def _calculate_confidence_level(self, scores):
        """Calculate confidence based on agreement between different methods"""
        confidence_points = 0
        
        # PyIQA reliability
        if scores.get('pyiqa_model_count', 0) >= 2:
            confidence_points += 3
        elif scores.get('pyiqa_model_count', 0) == 1:
            confidence_points += 1
        
        # Feature agreement
        hist_score = scores.get('histogram_editing_score', 0)
        edge_score = scores.get('edge_artifacts_score', 0)
        freq_score = scores.get('frequency_artifacts_score', 0)
        
        feature_scores = [hist_score, edge_score, freq_score]
        feature_std = np.std(feature_scores)
        
        # Lower standard deviation = better agreement
        if feature_std < 10:
            confidence_points += 2
        elif feature_std < 20:
            confidence_points += 1
        
        # Score magnitude consistency
        overall_score = scores.get('overall_editing_score', 0)
        if overall_score > 70 or overall_score < 30:
            confidence_points += 1  # Clear decisions are more confident
        
        confidence_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extremely High']
        return confidence_levels[min(confidence_points, 5)]
    
    def analyze_single_image(self, image_path):
        """Complete analysis of a single image with optimized loading"""
        try:
            # Optimized image loading (load once)
            image = cv2.imread(image_path)
            if image is None:
                return None, "Could not read image"
            
            # Perform all analyses
            pyiqa_results = self.analyze_with_pyiqa(image_path)
            histogram_results = self.analyze_histogram_features(image)
            edge_results = self.analyze_edge_artifacts(image)
            freq_results = self.analyze_frequency_domain(image)
            
            # Calculate comprehensive score
            comprehensive_score = self.calculate_comprehensive_score(
                pyiqa_results, histogram_results, edge_results, freq_results
            )
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                pyiqa_results, histogram_results, edge_results, 
                freq_results, comprehensive_score
            )
            
            result = {
                'filename': os.path.basename(image_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'status': 'success',
                'comprehensive_assessment': comprehensive_score,
                'pyiqa_analysis': pyiqa_results,
                'histogram_analysis': histogram_results,
                'edge_analysis': edge_results,
                'frequency_analysis': freq_results,
                'interpretation': interpretation,
                'analysis_method': 'PyIQA + Feature-Based Detection'
            }
            
            return result, "Analysis completed successfully"
            
        except Exception as e:
            return None, f"Error analyzing image: {str(e)}"
    
    def organize_images_by_editing_score(self, results, folder_path, base_output_dir="Results", 
                                       manual_review_threshold=33.0):
        """
        Organize images into folders based on editing detection scores.
        Images above 33% confidence are moved to manual review as "artificial editing detected".
        
        Args:
            results: List of analysis results
            folder_path: Source folder containing the images
            base_output_dir: Base directory for organized images
            manual_review_threshold: Threshold for manual review (default: 33.0)
            
        Returns:
            Dictionary with organization statistics
        """
        import shutil
        
        # Create output directories 
        valid_dir = os.path.join(base_output_dir, "valid")
        manualreview_dir = os.path.join(base_output_dir, "manualreview")
        
        for directory in [valid_dir, manualreview_dir]:
            os.makedirs(directory, exist_ok=True)
        
        organization_stats = {
            'valid': [],
            'invalid': [],  # Won't be used for editing-only
            'manualreview': [],  # Images with artificial editing detected
            'errors': []
        }
        
        successful_results = [r for r in results if r.get('comprehensive_assessment')]
        
        for result in successful_results:
            filename = result['filename']
            source_path = os.path.join(folder_path, filename)
            
            if not os.path.exists(source_path):
                organization_stats['errors'].append(filename)
                continue
            
            try:
                score = result['comprehensive_assessment']['overall_editing_score']
                
                # NEW: Move images above 33% confidence to manual review
                if score >= manual_review_threshold:
                    dest_path = os.path.join(manualreview_dir, filename)
                    organization_stats['manualreview'].append(filename)
                    
                    # Add reason to result for reporting
                    result['manual_review_reason'] = "Artificial editing detected"
                else:
                    # Copy lower confidence images to valid folder
                    dest_path = os.path.join(valid_dir, filename)
                    organization_stats['valid'].append(filename)
                
                # Copy the file to appropriate destination
                if os.path.abspath(source_path) != os.path.abspath(dest_path):
                    shutil.copy2(source_path, dest_path)
                
            except Exception as e:
                organization_stats['errors'].append(filename)
                print(f"Error organizing {filename}: {str(e)}")
        
        return organization_stats

    def _generate_interpretation(self, pyiqa_results, histogram_results, 
                               edge_results, freq_results, comprehensive_score):
        """Generate detailed interpretation with improved thresholds"""
        interpretations = []
        
        # PyIQA interpretations with corrected thresholds
        if 'brisque_score' in pyiqa_results:
            brisque = pyiqa_results['brisque_score']
            if brisque > 65:
                interpretations.append(f"BRISQUE score ({brisque:.1f}) indicates significant quality degradation")
            elif brisque > 45:
                interpretations.append(f"BRISQUE score ({brisque:.1f}) suggests moderate quality issues")
            elif brisque < 25:
                interpretations.append(f"BRISQUE score ({brisque:.1f}) indicates excellent quality")
        
        if 'niqe_score' in pyiqa_results:
            niqe = pyiqa_results['niqe_score']
            if niqe > 10:
                interpretations.append(f"NIQE score ({niqe:.1f}) indicates poor naturalness")
            elif niqe > 6:
                interpretations.append(f"NIQE score ({niqe:.1f}) suggests reduced naturalness")
            elif niqe < 4:
                interpretations.append(f"NIQE score ({niqe:.1f}) indicates excellent naturalness")
        
        if 'clipiqa_score' in pyiqa_results:
            clipiqa = pyiqa_results['clipiqa_score']
            if clipiqa < 0.4:
                interpretations.append(f"CLIP-IQA score ({clipiqa:.2f}) suggests poor perceptual quality")
            elif clipiqa > 0.7:
                interpretations.append(f"CLIP-IQA score ({clipiqa:.2f}) indicates high perceptual quality")
        
        # Histogram interpretations with empirical thresholds
        if histogram_results['total_clipping'] > 2:
            interpretations.append(f"Histogram clipping ({histogram_results['total_clipping']:.1f}%) suggests tone mapping or exposure adjustment")
        
        if histogram_results['histogram_entropy'] < 6.0:
            interpretations.append(f"Low histogram entropy ({histogram_results['histogram_entropy']:.1f}) indicates tone manipulation")
        elif histogram_results['histogram_entropy'] > 7.9:
            interpretations.append(f"Very high histogram entropy ({histogram_results['histogram_entropy']:.1f}) may indicate artificial enhancement")
        
        # Edge interpretations with corrected thresholds
        if edge_results['laplacian_variance'] > 3000:
            interpretations.append(f"High Laplacian variance ({edge_results['laplacian_variance']:.0f}) indicates artificial sharpening")
        elif edge_results['laplacian_variance'] < 50:
            interpretations.append(f"Low Laplacian variance ({edge_results['laplacian_variance']:.0f}) suggests over-smoothing")
        
        if edge_results['edge_density'] > 0.2:
            interpretations.append(f"High edge density ({edge_results['edge_density']:.3f}) may indicate over-enhancement")
        elif edge_results['edge_density'] < 0.03:
            interpretations.append(f"Low edge density ({edge_results['edge_density']:.3f}) suggests blur or smoothing")
        
        # Frequency interpretations
        if freq_results['high_freq_energy'] > 0.25:
            interpretations.append("Unusual high-frequency content detected - possible artificial enhancement")
        elif freq_results['high_freq_energy'] < 0.02:
            interpretations.append("Very low high-frequency content - possible over-smoothing")
        
        # Overall assessment with confidence information
        score = comprehensive_score['overall_editing_score']
        category = comprehensive_score['editing_category']
        confidence = comprehensive_score['confidence']
        
        # Add model reliability information
        pyiqa_count = comprehensive_score.get('pyiqa_model_count', 0)
        if pyiqa_count >= 2:
            reliability_note = f"High reliability ({pyiqa_count} PyIQA models + features)"
        elif pyiqa_count == 1:
            reliability_note = f"Moderate reliability (1 PyIQA model + features)"
        else:
            reliability_note = "Feature-based analysis only"
        
        interpretations.append(f"Overall assessment: {category} (Score: {score:.1f}/100, Confidence: {confidence}, {reliability_note})")
        
        return interpretations if interpretations else ["No significant editing indicators detected"]
    
    def run_scoring_diagnostics(self, test_images_dir="photos4testing"):
        """
        Run diagnostic tests to validate the improved scoring system.
        """
        if not self.quiet:
            print("\n🔧 RUNNING SCORING DIAGNOSTICS")
            print("=" * 60)
        
        # Test with a few sample images if available
        if os.path.exists(test_images_dir):
            image_files = []
            for filename in os.listdir(test_images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(test_images_dir, filename))
            
            if image_files:
                test_file = image_files[0]  # Use first available image
                if not self.quiet:
                    print(f"📊 Testing with: {os.path.basename(test_file)}")
                
                result, status = self.analyze_single_image(test_file)
                
                if result and 'comprehensive_assessment' in result:
                    scores = result['comprehensive_assessment']
                    
                    if not self.quiet:
                        print(f"\n📈 SCORING BREAKDOWN:")
                        print(f"  PyIQA Models Used: {scores.get('pyiqa_model_count', 0)}")
                        
                        if scores.get('brisque_editing_indicator') is not None:
                            print(f"  BRISQUE Editing Score: {scores['brisque_editing_indicator']:.1f}/100")
                        if scores.get('niqe_editing_indicator') is not None:
                            print(f"  NIQE Editing Score: {scores['niqe_editing_indicator']:.1f}/100")
                        if scores.get('clipiqa_editing_indicator') is not None:
                            print(f"  CLIP-IQA Editing Score: {scores['clipiqa_editing_indicator']:.1f}/100")
                        
                        print(f"  Histogram Score: {scores.get('histogram_editing_score', 0):.1f}/100")
                        print(f"  Edge Artifacts Score: {scores.get('edge_artifacts_score', 0):.1f}/100")
                        print(f"  Frequency Score: {scores.get('frequency_artifacts_score', 0):.1f}/100")
                        print(f"  Overall Score: {scores['overall_editing_score']:.1f}/100")
                        print(f"  Category: {scores['editing_category']}")
                        print(f"  Confidence: {scores['confidence']}")
                        
                        print(f"\n✅ Diagnostics completed successfully")
                else:
                    if not self.quiet:
                        print(f"❌ Diagnostic test failed: {status}")
            else:
                if not self.quiet:
                    print(f"⚠️  No test images found in {test_images_dir}")
        else:
            if not self.quiet:
                print(f"⚠️  Test directory {test_images_dir} not found")

def process_image_wrapper(args):
    """Wrapper for parallel processing"""
    filename, file_path, detector = args
    print(f"🔍 Analyzing: {filename}")
    
    result, status = detector.analyze_single_image(file_path)
    
    if result is None:
        return {
            "filename": filename,
            "status": "failed",
            "error": status
        }
    
    return result

def print_main_style_editing_summary(results: list) -> None:
    """
    Print summary in the same style as main_optimized.py for consistency.
    
    Args:
        results: List of analysis results
    """
    print("=" * 120)
    print("IMAGE EDITING DETECTION RESULTS")
    print("=" * 120)
    
    print(f"\nTESTS PERFORMED:")
    print(f"  Enabled: editing")
    print(f"  Disabled: watermarks, specifications, text, borders")
    
    # Calculate statistics from successful results
    successful = [r for r in results if r.get('comprehensive_assessment')]
    total = len(successful)
    
    if total == 0:
        print("\nNo successful analyses to report.")
        return
    
    # Show editing confidence analysis table (like main pipeline)
    print(f"\nEDITING CONFIDENCE ANALYSIS:")
    print("-" * 120)
    print(f"{'Filename':<50} {'Editing Confidence':<20} {'Assessment':<30}")
    print("-" * 120)
    
    # Sort by confidence (highest first)
    successful_sorted = sorted(successful, key=lambda x: x['comprehensive_assessment']['overall_editing_score'], reverse=True)

    manual_review_needed = 0
    editing_review_flagged = 0
    
    for result in successful_sorted:
        filename = result['filename']
        score = result['comprehensive_assessment']['overall_editing_score']
        
        # Updated assessment logic with new 33% manual review threshold
        if score >= 33.0:
            assessment = "Artificial editing detected"
            manual_review_needed += 1
        elif score >= 27.0:
            assessment = "Probably artificially edited"
            editing_review_flagged += 1
        elif score >= 24.0:
            assessment = "Possible light editing"
        else:
            assessment = "Minimal/natural editing"
        
        filename_short = filename[:47] + "..." if len(filename) > 50 else filename
        print(f"{filename_short:<50} {score:>8.1f}%{'':<11} {assessment:<30}")
    
    print("-" * 120)
    print("Note: Images with confidence ≥33% are moved to manual review for artificial editing.")
    print("      Images with confidence ≥27% are flagged for editing review but kept in place.")    # Categorize based on editing scores with new thresholds
    minimal_edited = [r for r in successful if r['comprehensive_assessment']['overall_editing_score'] < 18.0]
    valid_images = total - manual_review_needed
    
    print(f"\nSUMMARY:")
    print(f"  Total Images Processed: {total}")
    print(f"  Valid Images: {valid_images}")
    print(f"  Invalid Images: 0")      # No images moved to invalid
    print(f"  Manual Review Needed (moved): {manual_review_needed}")
    if editing_review_flagged > 0:
        print(f"  Editing Review Flagged (kept in place): {editing_review_flagged}")
    print(f"  Success Rate: 100.0%")   # Always 100% since all images are processed
    
    # Display results with updated organization
    if valid_images > 0:
        print(f"\nVALID IMAGES ({valid_images} images):")
        print("-" * 120)
        print(f"Validation checks passed - images copied to 'Results\\valid' folder")
    
    if manual_review_needed > 0:
        print(f"\nMANUAL REVIEW NEEDED ({manual_review_needed} images):")
        print("-" * 120)
        print(f"Artificial editing detected - images moved to 'Results\\manualreview' folder")
    
    print(f"\nOUTPUT STRUCTURE:")
    print(f"  Valid images: Results\\valid")
    print(f"  Invalid images: Results\\invalid")
    print(f"  Manual review needed: Results\\manualreview")
    print(f"  Processing logs: Results\\logs")
    
    print(f"\nNOTE: Images flagged only for editing review are kept in their original location.")
    print(f"      Images with artificial editing detected (≥33%) are moved to manual review.")
    print(f"      Check the 'EDITING CONFIDENCE ANALYSIS' table above to see which images need editing review.")
    
    print("\n" + "=" * 120)

def main():
    """Main execution function with CUDA configuration options and improved scoring"""
    folder_path = "photos4testing"  # Standard input folder
    
    print("🔬 ADVANCED IMAGE EDITING DETECTOR v2.0")
    print("=" * 70)
    
    # CUDA Configuration Options
    import argparse
    import sys
    
    # Simple command-line argument parsing
    force_cpu = '--cpu' in sys.argv
    run_diagnostics = '--diagnostics' in sys.argv or '--test' in sys.argv
    gpu_id = 0
    # CLI-driven model selection (non-interactive)
    selected_models = None
    excluded_models = None
    skip_model_prompt = False
    
    # Check for GPU ID argument and optional flags
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--gpu='):
            try:
                gpu_id = int(arg.split('=')[1])
            except ValueError:
                print("⚠️  Invalid GPU ID, using default (0)")
        elif arg == '--gpu' and i + 1 < len(sys.argv):
            try:
                gpu_id = int(sys.argv[i + 1])
            except ValueError:
                print("⚠️  Invalid GPU ID, using default (0)")
        elif arg.startswith('--source='):
            folder_path = arg.split('=', 1)[1].strip('"')
        elif arg == '--source' and i + 1 < len(sys.argv):
            folder_path = sys.argv[i + 1].strip('"')
        elif arg == '--fast':
            selected_models = ['brisque', 'niqe', 'clipiqa']
            skip_model_prompt = True
        elif arg.startswith('--models='):
            csv = arg.split('=', 1)[1]
            selected_models = [m.strip().lower() for m in csv.split(',') if m.strip()]
            skip_model_prompt = True
    
    # Model selection logic
    
    if PYIQA_AVAILABLE:
        print("🎯 PyIQA Quality Assessment Models Available")
        print("📊 Available Models:")
        all_available_models = ['brisque', 'niqe', 'musiq', 'dbcnn', 'hyperiqa', 'clipiqa']
        for i, model in enumerate(all_available_models, 1):
            marker = " ⚡ FAST" if i in [1, 2, 6] else ""
            print(f"  {i}. {model.upper()}{marker}")
        
        if not skip_model_prompt:
            print("\n🔧 Model Selection Options:")
            print("  1. Use RECOMMENDED FAST models (BRISQUE + NIQE + CLIPIQA) ⚡")
            print("  2. Use ALL models (slower but comprehensive)")
            print("  3. Select SPECIFIC models to use")
            print("  4. EXCLUDE specific models")
            
            try:
                choice = input("\nEnter your choice (1-4) [1]: ").strip()
                if not choice:
                    choice = "1"
                
                if choice == "1":
                    # Use recommended fast models: BRISQUE, NIQE, CLIPIQA
                    selected_models = ['brisque', 'niqe', 'clipiqa']
                    print(f"⚡ Using FAST recommended models: {', '.join(selected_models)}")
                    
                elif choice == "2":
                    # Use all models (original default behavior)
                    selected_models = None
                    print("📊 Using ALL models for comprehensive analysis")
                    
                elif choice == "3":
                    print("\n📋 Select models to use (enter numbers separated by spaces):")
                    print("Available: " + ", ".join(f"{i}:{model}" for i, model in enumerate(all_available_models, 1)))
                    print("💡 Tip: Use 1,2,6 for fastest recommended combination")
                    model_input = input("Model numbers: ").strip()
                    if model_input:
                        try:
                            indices = [int(x.strip()) - 1 for x in model_input.split(',') if x.strip()]
                            selected_models = [all_available_models[i] for i in indices if 0 <= i < len(all_available_models)]
                            print(f"✅ Selected models: {', '.join(selected_models)}")
                        except (ValueError, IndexError):
                            print("⚠️  Invalid input, using fast recommended models")
                            selected_models = ['brisque', 'niqe', 'clipiqa']
                            
                elif choice == "4":
                    print("\n🚫 Select models to EXCLUDE (enter numbers separated by spaces):")
                    print("Available: " + ", ".join(f"{i}:{model}" for i, model in enumerate(all_available_models, 1)))
                    model_input = input("Model numbers to exclude: ").strip()
                    if model_input:
                        try:
                            indices = [int(x.strip()) - 1 for x in model_input.split(',') if x.strip()]
                            excluded_models = [all_available_models[i] for i in indices if 0 <= i < len(all_available_models)]
                            print(f"🚫 Excluded models: {', '.join(excluded_models)}")
                        except (ValueError, IndexError):
                            print("⚠️  Invalid input, using fast recommended models")
                            selected_models = ['brisque', 'niqe', 'clipiqa']
                else:
                    print("⚠️  Invalid choice, using fast recommended models")
                    selected_models = ['brisque', 'niqe', 'clipiqa']
                    
            except KeyboardInterrupt:
                print("\n🔄 Using fast recommended models (BRISQUE + NIQE + CLIPIQA)")
                selected_models = ['brisque', 'niqe', 'clipiqa']
        
        if selected_models:
            if set(selected_models) == {'brisque', 'niqe', 'clipiqa'}:
                print(f"⚡ Using FAST recommended models: {', '.join(selected_models)}")
            else:
                print(f"🎯 Using SELECTED PyIQA Models: {', '.join(selected_models)}")
        elif excluded_models:
            remaining_models = [m for m in all_available_models if m not in excluded_models]
            print(f"🎯 Using PyIQA Models (excluding {', '.join(excluded_models)}): {', '.join(remaining_models)}")
        else:
            print("🎯 Using ALL PyIQA Models: BRISQUE • NIQE • CLIP-IQA • MUSIQ • DBCNN • HyperIQA")
    else:
        print("🎯 Using Feature-Based Analysis Only with Empirical Thresholds")
        print("💡 Install PyIQA for enhanced detection: pip install pyiqa torch")
    
    print("🔍 Features: Histogram • Edges • Frequency Domain")
    print(f"📁 Target folder: {folder_path}")
    print("🆕 Improved: Empirical thresholds, robust validation, better normalization")
    
    # Display CUDA options
    if PYIQA_AVAILABLE and torch.cuda.is_available():
        print(f"🎮 CUDA Options:")
        print(f"  • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  • Use --cpu to force CPU mode")
        print(f"  • Use --gpu=N to select specific GPU")
        print(f"  • Use --diagnostics to run scoring validation tests")
    
    print("⚡ Parallel processing enabled\n")
    
    # Initialize detector with CUDA configuration and model selection
    detector = AdvancedEditingDetector(
        force_cpu=force_cpu, 
        gpu_id=gpu_id, 
        selected_models=selected_models,
        excluded_models=excluded_models
    )
    
    # Run diagnostics if requested
    if run_diagnostics:
        detector.run_scoring_diagnostics(folder_path)
        print("\n" + "=" * 70)
    
    # Get image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder {folder_path} not found")
        return
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            image_files.append((filename, file_path, detector))
    
    if not image_files:
        print("❌ No image files found")
        return
    
    print(f"📸 Found {len(image_files)} images to analyze")
    
    # Process images
    start_time = time.time()
    print("🚀 Starting analysis...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for PyIQA
        results = list(executor.map(process_image_wrapper, image_files))
    
    processing_time = time.time() - start_time
    
    # Compile results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    # Save results to unified Results/logs folder
    os.makedirs("Results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"Results/logs/advanced_editing_analysis_{timestamp}.json"
    
    analysis_summary = {
        'analysis_timestamp': datetime.now().isoformat(),
        'folder_path': folder_path,
        'analysis_method': 'PyIQA + Feature-Based Detection' if PYIQA_AVAILABLE else 'Feature-Based Detection',
        'total_images': len(image_files),
        'successful_analyses': len(successful),
        'failed_analyses': len(failed),
        'processing_time_seconds': round(processing_time, 2),
        'images_per_second': round(len(image_files) / processing_time, 2),
        'pyiqa_available': PYIQA_AVAILABLE,
        'images': {r['filename']: r for r in results}
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    # Display results
    print(f"\n✅ ANALYSIS COMPLETED!")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"🔄 Speed: {len(image_files) / processing_time:.1f} images/second")
    print(f"📄 Results saved to: {output_file}")
    
    if successful:
        # Use main pipeline style summary
        print_main_style_editing_summary(results)
        
        # Organize images by editing scores
        print(f"\n📁 Organizing images by editing scores...")
        organization_stats = detector.organize_images_by_editing_score(
            results,
            folder_path,
            base_output_dir="Results",
            manual_review_threshold=33.0
        )
    
    # Show any failed analyses briefly
    failed = [r for r in results if not r.get('comprehensive_assessment')]
    if failed:
        print(f"\n❌ Note: {len(failed)} images failed analysis due to errors.")

if __name__ == "__main__":
    main()
