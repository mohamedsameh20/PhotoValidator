# Troubleshooting Guide - Advanced Image Processing Pipeline

This guide covers common issues and their solutions when setting up and using the Advanced Image Processing Pipeline.

## 🚨 Quick Diagnostics

### Run the Built-in Test
```bash
python quick_test.py
```
This will check all major components and report any issues.

## 📋 Installation Issues

### 1. Python Version Errors

**Problem**: `Python 3.8+ is required`
```
Error: Python 3.8+ is required. Found Python 3.7
```

**Solutions**:
- **Windows**: Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
- **macOS**: Use Homebrew: `brew install python@3.10`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3.10`
- **CentOS/RHEL**: `sudo yum install python3.10`

### 2. PyTorch Installation Issues

**Problem**: PyTorch installation fails
```
ERROR: Could not find a version that satisfies the requirement torch
```

**Solutions**:
```bash
# Option 1: Install with specific index
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

# Option 2: CPU-only version
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Option 3: Use conda
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. PaddleOCR Installation Issues

**Problem**: PaddlePaddle installation fails
```
ImportError: No module named 'paddle'
```

**Solutions**:
```bash
# For CPU version
pip install paddlepaddle

# For GPU version (requires CUDA)
pip install paddlepaddle-gpu

# Alternative: Use conda
conda install paddlepaddle -c paddle
```

**Common PaddleOCR Issues**:
- **Shape mismatch errors**: Ensure image is in RGB format
- **CUDA errors**: Install CPU version if GPU issues persist
- **Model download fails**: Check internet connection and firewall

### 4. OpenCV Installation Issues

**Problem**: OpenCV import fails
```
ImportError: No module named 'cv2'
```

**Solutions**:
```bash
# Standard installation
pip install opencv-python

# If still failing, try
pip install opencv-python-headless

# For development (includes extra modules)
pip install opencv-contrib-python

# System dependencies (Linux)
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

### 5. PyIQA Installation Issues

**Problem**: PyIQA model download fails
```
ConnectionError: Failed to download model
```

**Solutions**:
```bash
# Manual installation
pip install pyiqa --upgrade

# Check HuggingFace Hub access
python -c "from huggingface_hub import hf_hub_download; print('HF Hub accessible')"

# Use offline mode if internet issues persist
export HF_HUB_OFFLINE=1
```

## 🖥️ Runtime Issues

### 1. CUDA Out of Memory

**Problem**: GPU memory exhausted
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
```bash
# Reduce batch size
python main_optimized.py --batch-size 4

# Use CPU mode
python main_optimized.py --cpu-only

# Enable memory cleanup
python main_optimized.py --enable-memory-cleanup
```

**Memory optimization in code**:
```python
# In main_optimized.py, modify these settings:
BATCH_SIZE = 4  # Reduce from 16
ENABLE_MEMORY_CLEANUP = True
USE_MIXED_PRECISION = True  # If available
```

### 2. Slow Processing Speed

**Problem**: Processing is very slow

**Diagnostics**:
```bash
# Check GPU usage
nvidia-smi

# Run with profiling
python main_optimized.py --profile
```

**Solutions**:
- **Enable GPU**: Ensure CUDA is properly installed
- **Increase batch size**: If you have sufficient memory
- **Reduce image resolution**: For faster processing
- **Use multiple workers**: Increase `--num-workers`

### 3. Model Loading Errors

**Problem**: Models fail to load
```
OSError: Unable to load model from disk
```

**Solutions**:
```bash
# Clear model cache
rm -rf models_cache/
rm -rf ~/.cache/huggingface/
rm -rf ~/.paddlehub/

# Re-download models
python -c "import paddleocr; paddleocr.PaddleOCR(use_angle_cls=True, lang='en')"
python -c "import pyiqa; pyiqa.create_metric('brisque', device='cpu')"
```

### 4. File Permission Errors

**Problem**: Cannot create output directories
```
PermissionError: [Errno 13] Permission denied: 'Results'
```

**Solutions**:
```bash
# Linux/macOS
chmod 755 .
mkdir -p Results/{valid,invalid,manualreview}
chmod -R 755 Results/

# Windows (run as administrator)
icacls Results /grant Everyone:F /T
```

## 🐛 Common Processing Errors

### 1. Image Loading Failures

**Problem**: Images cannot be loaded
```
Error: Cannot load image: corrupted_image.jpg
```

**Solutions**:
- Check image file integrity
- Verify supported formats: JPG, PNG, TIFF, BMP, WebP
- Remove corrupted files from input folder
- Check file permissions

### 2. Text Detection Issues

**Problem**: PaddleOCR fails on specific images
```
PaddleOCR error: Shape mismatch
```

**Solutions**:
```python
# In paddle_text_detector.py, add preprocessing:
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    # Ensure minimum size
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        image = cv2.resize(image, (max(32, w), max(32, h)))
    return image
```

### 3. Watermark Detection Issues

**Problem**: Watermark detection gives inconsistent results

**Solutions**:
- Adjust confidence thresholds in config.json
- Ensure input images are RGB format
- Check if watermarks are too small/transparent

### 4. Border Detection False Positives

**Problem**: Natural image edges detected as artificial borders

**Solutions**:
```python
# Adjust border detection sensitivity
BORDER_THRESHOLD = 0.7  # Increase for fewer false positives
EDGE_DENSITY_THRESHOLD = 0.3  # Adjust based on your needs
```

## ⚙️ Configuration Issues

### 1. Config File Errors

**Problem**: Invalid configuration
```
JSONDecodeError: Expecting ',' delimiter
```

**Solution**: Use the default config template:
```json
{
    "processing": {
        "batch_size": 16,
        "num_workers": 4,
        "use_gpu": true,
        "debug_mode": false
    },
    "thresholds": {
        "text_confidence": 0.7,
        "watermark_confidence": 0.8,
        "border_confidence": 0.6,
        "quality_threshold": 0.5
    },
    "output": {
        "save_debug_images": true,
        "generate_report": true,
        "organize_by_type": true
    }
}
```

### 2. Path Configuration Issues

**Problem**: Input/output paths not found

**Solutions**:
```bash
# Use absolute paths
python main_optimized.py --input-folder /full/path/to/images

# Verify directory structure
ls -la photos4testing/
ls -la Results/
```

## 🔧 Platform-Specific Issues

### Windows

**Common Issues**:
1. **Long path names**: Enable long path support in Windows
2. **Antivirus interference**: Add project folder to antivirus exclusions
3. **Missing Visual C++**: Install Microsoft Visual C++ Redistributable

**Solutions**:
```cmd
# Enable long paths (run as administrator)
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Install Visual C++ Redistributable
# Download from Microsoft website
```

### macOS

**Common Issues**:
1. **Xcode Command Line Tools**: Required for compilation
2. **Permission issues**: Especially with system Python

**Solutions**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Use Homebrew Python instead of system Python
brew install python@3.10
```

### Linux

**Common Issues**:
1. **Missing system libraries**: Required for OpenCV
2. **CUDA driver issues**: Mismatched CUDA versions

**Solutions**:
```bash
# Ubuntu/Debian - Install OpenCV dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# CentOS/RHEL
sudo yum install mesa-libGL glib2 libSM libXext libXrender

# Check CUDA installation
nvidia-smi
cat /usr/local/cuda/version.txt
```

## 📊 Performance Optimization

### Memory Optimization

```python
# Reduce memory usage
import gc
import torch

# Clear cache regularly
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Use memory mapping for large datasets
import mmap
# Implementation depends on your specific use case
```

### Speed Optimization

```python
# Optimize batch processing
BATCH_SIZE = min(32, available_memory_gb * 2)  # Rule of thumb

# Use multiple workers
NUM_WORKERS = min(8, cpu_count())

# Enable mixed precision (if supported)
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
    with autocast():
        # Model inference code here
        pass
```

## 🆘 Getting Help

### Diagnostic Information Collection

When reporting issues, please provide:

```bash
# System information
python --version
pip list | grep -E "(torch|opencv|paddle|numpy)"

# GPU information (if applicable)
nvidia-smi

# Error logs
python main_optimized.py --debug 2>&1 | tee debug.log
```

### Debug Mode

Enable comprehensive debugging:
```bash
python main_optimized.py --debug --verbose
```

This will:
- Generate detailed logs
- Save intermediate processing results
- Create debug visualizations
- Show memory usage statistics

### Log Analysis

Check these log files for errors:
- `debug.log` - Main application log
- `paddle_ocr.log` - PaddleOCR specific issues
- `watermark_detection.log` - Watermark detection issues

### Community Support

1. **GitHub Issues**: [Report bugs](https://github.com/yourusername/advanced-image-processing-pipeline/issues)
2. **Discussions**: [Ask questions](https://github.com/yourusername/advanced-image-processing-pipeline/discussions)
3. **Documentation**: [Wiki](https://github.com/yourusername/advanced-image-processing-pipeline/wiki)

### Professional Support

For enterprise support or custom development:
- Email: support@yourcompany.com
- Priority support available for commercial users

---

## 📚 Additional Resources

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [OpenCV Installation Guide](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

*Last updated: July 28, 2025*
