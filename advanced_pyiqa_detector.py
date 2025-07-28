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
    
    def __init__(self, force_cpu=False, gpu_id=0, quiet=False):
        """
        Initialize the detector with CUDA configuration options.
        
        Args:
            force_cpu (bool): Force CPU usage even if CUDA is available
            gpu_id (int): Specific GPU ID to use (for multi-GPU systems)
            quiet (bool): Suppress initialization output for clean execution
        """
        self.quiet = quiet
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
        
        # Load models in a specific order to avoid conflicts
        # Load CLIP-IQA separately to avoid conflicts
        models_to_load = [
            'brisque',      # BRISQUE - excellent for unnatural distortions
            'niqe',         # NIQE - good for naturalness assessment
            'musiq',        # MUSIQ - multi-scale quality assessment
            'dbcnn',        # DB-CNN - deep learning based
            'hyperiqa',     # HyperIQA - hypernetwork based
        ]
        
        # Load non-CLIP models first
        for model_name in models_to_load:
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
        
        # Load CLIP-IQA separately with extra isolation
        try:
            if not self.quiet:
                print("  🔄 Loading CLIP-IQA with isolation...")
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            clipiqa_model = pyiqa.create_metric('clipiqa', device=self.device)
            
            # Test CLIP-IQA immediately
            test_path = os.path.join("photos4testing", "Normal_Image.jpg")
            if os.path.exists(test_path):
                with torch.no_grad():
                    test_score = clipiqa_model(test_path)
                if not self.quiet:
                    print(f"    🔍 CLIP-IQA test score: {test_score}")
                if torch.isnan(test_score).any():
                    if not self.quiet:
                        print(f"    ⚠️  CLIP-IQA model is corrupted - skipping")
                else:
                    self.quality_models['clipiqa'] = clipiqa_model
                    if not self.quiet:
                        print(f"  ✅ Loaded CLIPIQA")
            else:
                self.quality_models['clipiqa'] = clipiqa_model
                if not self.quiet:
                    print(f"  ✅ Loaded CLIPIQA")
                
        except Exception as e:
            if not self.quiet:
                print(f"  ❌ Failed to load CLIP-IQA: {str(e)}")
        
        if not self.quiet:
            print(f"🎯 Successfully loaded {len(self.quality_models)} quality models")
        
        # Print GPU memory usage after model loading
        if self.device.type == 'cuda' and not self.quiet:
            self._print_gpu_memory_usage("After model loading")
    
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
                    # PyIQA expects image path or PIL image
                    with torch.no_grad():  # Disable gradient computation for inference
                        score = model(image_path)
                    
                    # Convert tensor to float if needed
                    if torch.is_tensor(score):
                        # Handle different tensor shapes
                        if score.numel() == 1:
                            score = score.item()
                        else:
                            # For multi-dimensional tensors, take the mean or first element
                            score = score.flatten()[0].item()
                    
                    # Check for NaN values
                    if np.isnan(score) or np.isinf(score):
                        print(f"    ⚠️  {model_name} returned invalid value: {score}")
                        quality_scores[f'{model_name}_error'] = f"Invalid score: {score}"
                    else:
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
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
        peak_count = len(peaks)
        
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
    
    def calculate_comprehensive_score(self, pyiqa_results, histogram_results, edge_results, freq_results):
        """
        Calculate comprehensive editing detection score combining all analyses.
        """
        scores = {}
        
        # PyIQA-based scoring (if available)
        if 'error' not in pyiqa_results:
            quality_score = 0
            quality_count = 0
            
            # BRISQUE: Lower is better (natural images: 20-40, edited: 40-100)
            if 'brisque_score' in pyiqa_results:
                brisque = pyiqa_results['brisque_score']
                # Convert to 0-100 scale where higher = more editing
                brisque_editing_score = min(100, max(0, (brisque - 20) / 80 * 100))
                scores['brisque_editing_indicator'] = float(brisque_editing_score)
                quality_score += brisque_editing_score
                quality_count += 1
            
            # NIQE: Lower is better (natural images: 3-6, edited: 6-15)
            if 'niqe_score' in pyiqa_results:
                niqe = pyiqa_results['niqe_score']
                niqe_editing_score = min(100, max(0, (niqe - 3) / 12 * 100))
                scores['niqe_editing_indicator'] = float(niqe_editing_score)
                quality_score += niqe_editing_score
                quality_count += 1
            
            # CLIP-IQA: Higher is better (0-1 scale), so invert for editing detection
            if 'clipiqa_score' in pyiqa_results:
                clipiqa = pyiqa_results['clipiqa_score']
                if not (np.isnan(clipiqa) or np.isinf(clipiqa)):
                    # Convert to editing score: lower CLIP-IQA = more editing
                    clipiqa_editing_score = max(0, (1 - clipiqa) * 100)
                    scores['clipiqa_editing_indicator'] = float(clipiqa_editing_score)
                    quality_score += clipiqa_editing_score
                    quality_count += 1
                else:
                    print(f"    ⚠️  CLIP-IQA returned invalid value: {clipiqa}")
                    scores['clipiqa_editing_indicator'] = 0.0
            else:
                scores['clipiqa_editing_indicator'] = 0.0
            
            if quality_count > 0:
                scores['average_quality_editing_score'] = float(quality_score / quality_count)
            else:
                scores['average_quality_editing_score'] = 0.0
        else:
            scores['average_quality_editing_score'] = 0.0
        
        # Histogram-based scoring
        hist_score = 0
        hist_score += min(100, histogram_results['total_clipping'] * 10)  # Clipping indicator
        hist_score += min(100, (8 - histogram_results['histogram_entropy']) / 8 * 100)  # Low entropy
        hist_score += min(100, histogram_results['ks_uniformity_test'] * 200)  # Non-natural distribution
        scores['histogram_editing_score'] = float(hist_score / 3)
        
        # Edge-based scoring
        edge_score = 0
        edge_score += min(100, max(0, (edge_results['laplacian_variance'] - 500) / 2000 * 100))
        edge_score += min(100, edge_results['edge_density'] * 500)
        edge_score += min(100, edge_results['high_freq_energy'] / 10000)
        scores['edge_artifacts_score'] = float(edge_score / 3)
        
        # Frequency-based scoring
        freq_score = 0
        # Unnatural frequency distribution indicates editing
        freq_score += min(100, abs(freq_results['high_freq_energy'] - 0.1) / 0.1 * 50)
        freq_score += min(100, freq_results['frequency_variance'] / 50)
        scores['frequency_artifacts_score'] = float(freq_score / 2)
        
        # Overall score calculation
        if scores['average_quality_editing_score'] > 0:
            # Weight PyIQA heavily when available
            overall_score = (
                scores['average_quality_editing_score'] * 0.5 +
                scores['histogram_editing_score'] * 0.2 +
                scores['edge_artifacts_score'] * 0.2 +
                scores['frequency_artifacts_score'] * 0.1
            )
        else:
            # Fall back to feature-based analysis
            overall_score = (
                scores['histogram_editing_score'] * 0.4 +
                scores['edge_artifacts_score'] * 0.4 +
                scores['frequency_artifacts_score'] * 0.2
            )
        
        scores['overall_editing_score'] = float(overall_score)
        
        # Determine category
        if overall_score >= 70:
            category = "Heavy Editing Detected"
        elif overall_score >= 50:
            category = "Moderate Editing Detected"
        elif overall_score >= 25:
            category = "Light Editing Detected"
        else:
            category = "Natural/Minimal Editing"
        
        scores['editing_category'] = category
        
        # Calculate confidence
        confidence_indicators = 0
        if scores['average_quality_editing_score'] > 60:
            confidence_indicators += 2  # PyIQA is very reliable
        if scores['histogram_editing_score'] > 50:
            confidence_indicators += 1
        if scores['edge_artifacts_score'] > 50:
            confidence_indicators += 1
        
        confidence_levels = ['Low', 'Medium', 'High', 'Very High']
        scores['confidence'] = confidence_levels[min(confidence_indicators, 3)]
        
        return scores
    
    def analyze_single_image(self, image_path):
        """Complete analysis of a single image"""
        try:
            # Load image for traditional analysis
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
    
    def _generate_interpretation(self, pyiqa_results, histogram_results, 
                               edge_results, freq_results, comprehensive_score):
        """Generate detailed interpretation"""
        interpretations = []
        
        # PyIQA interpretations
        if 'brisque_score' in pyiqa_results:
            brisque = pyiqa_results['brisque_score']
            if brisque > 50:
                interpretations.append(f"BRISQUE score ({brisque:.1f}) indicates significant quality degradation")
            elif brisque > 30:
                interpretations.append(f"BRISQUE score ({brisque:.1f}) suggests moderate quality issues")
        
        if 'niqe_score' in pyiqa_results:
            niqe = pyiqa_results['niqe_score']
            if niqe > 8:
                interpretations.append(f"NIQE score ({niqe:.1f}) indicates poor naturalness")
            elif niqe > 5:
                interpretations.append(f"NIQE score ({niqe:.1f}) suggests reduced naturalness")
        
        # Histogram interpretations
        if histogram_results['total_clipping'] > 5:
            interpretations.append(f"Significant histogram clipping ({histogram_results['total_clipping']:.1f}%)")
        
        if histogram_results['histogram_entropy'] < 6:
            interpretations.append("Low histogram entropy suggests tone manipulation")
        
        # Edge interpretations
        if edge_results['laplacian_variance'] > 1000:
            interpretations.append("High Laplacian variance indicates artificial sharpening")
        
        if edge_results['edge_density'] > 0.15:
            interpretations.append("High edge density may indicate over-enhancement")
        
        # Frequency interpretations
        if freq_results['high_freq_energy'] > 0.2:
            interpretations.append("Unusual high-frequency content detected")
        
        # Overall assessment
        score = comprehensive_score['overall_editing_score']
        category = comprehensive_score['editing_category']
        confidence = comprehensive_score['confidence']
        
        interpretations.append(f"Overall assessment: {category} (Score: {score:.1f}/100, Confidence: {confidence})")
        
        return interpretations if interpretations else ["No significant editing indicators detected"]

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

def main():
    """Main execution function with CUDA configuration options"""
    folder_path = r"C:\Users\Public\Python\ittask\photos4testing"
    
    print("🔬 ADVANCED IMAGE EDITING DETECTOR")
    print("=" * 70)
    
    # CUDA Configuration Options
    import argparse
    import sys
    
    # Simple command-line argument parsing
    force_cpu = '--cpu' in sys.argv
    gpu_id = 0
    
    # Check for GPU ID argument
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--gpu='):
            try:
                gpu_id = int(arg.split('=')[1])
            except ValueError:
                print("⚠️  Invalid GPU ID, using default (0)")
    
    if PYIQA_AVAILABLE:
        print("🎯 Using PyIQA + Feature-Based Analysis")
        print("📊 Quality Models: BRISQUE • NIQE • CLIP-IQA • MUSIQ")
    else:
        print("🎯 Using Feature-Based Analysis Only")
        print("💡 Install PyIQA for enhanced detection: pip install pyiqa torch")
    
    print("🔍 Features: Histogram • Edges • Frequency Domain")
    print(f"📁 Target folder: {folder_path}")
    
    # Display CUDA options
    if PYIQA_AVAILABLE and torch.cuda.is_available():
        print(f"🎮 CUDA Options:")
        print(f"  • Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  • Use --cpu to force CPU mode")
        print(f"  • Use --gpu=N to select specific GPU")
    
    print("⚡ Parallel processing enabled\n")
    
    # Initialize detector with CUDA configuration
    detector = AdvancedEditingDetector(force_cpu=force_cpu, gpu_id=gpu_id)
    
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
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"advanced_editing_analysis_{timestamp}.json"
    
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
        print(f"\n📊 DETECTION SUMMARY:")
        categories = {}
        manual_review_count = 0
        
        for result in successful:
            category = result['comprehensive_assessment']['editing_category']
            score = result['comprehensive_assessment']['overall_editing_score']
            categories[category] = categories.get(category, 0) + 1
            
            # Count images requiring manual review
            if score >= 25.0:
                manual_review_count += 1
        
        for category, count in categories.items():
            percentage = (count / len(successful)) * 100
            print(f"  • {category}: {count} images ({percentage:.1f}%)")
        
        # Add manual review summary
        if manual_review_count > 0:
            manual_review_percentage = (manual_review_count / len(successful)) * 100
            print(f"\n⚠️  MANUAL REVIEW REQUIRED:")
            print(f"  • {manual_review_count} images (score ≥ 25/100) need manual verification ({manual_review_percentage:.1f}%)")
            print(f"  • These images show possible heavy editing - please check manually")
        else:
            print(f"\n✅ NO MANUAL REVIEW REQUIRED:")
            print(f"  • All images appear natural (scores < 25/100)")
        
        # Show all edited images ranked from most to least edited
        print(f"\n🏆 EDITED IMAGES (from most edited to lowest):")
        successful.sort(key=lambda x: x['comprehensive_assessment']['overall_editing_score'], reverse=True)
        
        for i, result in enumerate(successful):
            filename = result['filename']
            score = result['comprehensive_assessment']['overall_editing_score']
            category = result['comprehensive_assessment']['editing_category']
            confidence = result['comprehensive_assessment']['confidence']
            
            # Add manual review flag for scores ≥ 25/100
            manual_review_flag = "⚠️  REQUIRES MANUAL REVIEW" if score >= 25.0 else ""
            
            if manual_review_flag:
                print(f"  {i+1}. {filename}: {score:.1f}/100 ({category}, {confidence} confidence) {manual_review_flag}")
            else:
                print(f"  {i+1}. {filename}: {score:.1f}/100 ({category}, {confidence} confidence)")
    
    if failed:
        print(f"\n❌ FAILED ANALYSES ({len(failed)}):")
        for result in failed[:3]:  # Show first 3 failures
            print(f"  • {result['filename']}: {result['error']}")
    
    print(f"\n🎉 Advanced editing detection completed!")

if __name__ == "__main__":
    main()
