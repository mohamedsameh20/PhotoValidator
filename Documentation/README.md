# PhotoValidator - Advanced Image Processing Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Overview

This project provides a comprehensive, high-performance image processing and filtering system designed for professional image validation, quality assessment, and automated sorting. The system combines state-of-the-art machine learning models with advanced computer vision techniques to detect artificial editing, watermarks, text overlays, and image quality issues.

### Key Features
- **Text Detection**: State-of-the-art PaddleOCR with DB (Differentiable Binarization) model
- **Watermark Detection**: CNN-based watermark identification using ConvNeXt architecture
- **Border Detection**: Multi-algorithm approach with adaptive thresholds
- **Quality Analysis**: PyIQA integration with BRISQUE, NIQE, and other metrics
- **Batch Processing**: Efficient processing of large image datasets
- **Smart Organization**: Automated sorting into valid/invalid/manual-review categories

## 🚀 Quick Start

### Manual Setup

#### 1. Install Python 3.8+
- Download from [python.org](https://www.python.org/downloads/)

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Basic Usage
```bash
# Add your images to the photos4testing folder
# Run the main processing pipeline
python main_optimized.py

# Results will be organized in the Results/ folder
```

## 📋 System Requirements

### Python Version
- **Python 3.8 or higher** (recommended: Python 3.10+)

### Hardware Requirements
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: Optional but recommended (CUDA-compatible for PyTorch and PaddlePaddle)
- **Storage**: At least 5GB free space for models and cache
- **CPU**: Multi-core processor recommended

### Operating Systems
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04+
- Other Linux distributions (tested on CentOS, Fedora)

## 📦 Installation Guide

### Step 1: Environment Setup

#### Option A: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv photovalidator-env

# Activate environment
# On Windows:
photovalidator-env\Scripts\activate
# On macOS/Linux:
source photovalidator-env/bin/activate
```

#### Option B: Using Conda
```bash
# Create a new conda environment
conda create -n photovalidator python=3.10
conda activate photovalidator
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages include:
- torch (PyTorch for neural networks)
- opencv-python (Computer vision operations)
- paddlepaddle and paddleocr (Text detection)
- pyiqa (Image quality assessment)
- PIL/Pillow (Image processing)
- numpy, matplotlib (Data processing and visualization)

## 🏗️ Project Structure

```
PhotoValidator/
├── main_optimized.py                 # Main processing controller
├── optimized_pipeline.py             # Core processing pipeline
├── paddle_text_detector.py           # PaddleOCR text detection
├── advanced_watermark_detector.py    # Watermark detection system
├── border_detector.py               # Border and frame detection
├── advanced_pyiqa_detector.py       # Image quality analysis
├── Spec_detector.py                 # Image specification analysis
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── QUICKSTART.md                    # Quick start guide
├── TROUBLESHOOTING.md               # Troubleshooting guide
├── photos4testing/                  # Input images folder
├── Results/                         # Output results folder (auto-created)
│   ├── valid/                      # Valid images
│   ├── invalid/                    # Invalid images
│   ├── manualreview/               # Images requiring manual review
│   └── logs/                       # Processing reports and logs
├── models/                         # Downloaded models cache
├── models_cache/                   # Model cache directory
└── watermark-detection/            # Watermark detection submodule
```

## 🎮 Usage Guide

### Basic Usage

1. **Prepare Your Images**
   - Add your images (JPG, PNG, TIFF, BMP, WEBP) to the `photos4testing/` folder
   - The system will automatically process all supported image formats

2. **Run Full Pipeline**
   ```bash
   python main_optimized.py
   ```

3. **Check Results**
   - Valid images: `Results/valid/`
   - Invalid images: `Results/invalid/`
   - Manual review needed: `Results/manualreview/`
   - Processing report: `Results/processing_report_[timestamp].json`

### Command Line Arguments

The main script `main_optimized.py` supports the following arguments:

```
usage: main_optimized.py [-h] [--source SOURCE] [--output OUTPUT] [--dry-run]
                         [--tests {specifications,text,borders,editing,watermarks} [{specifications,text,borders,editing,watermarks} ...]]
                         [--no-specs] [--no-text] [--no-borders] [--no-editing] [--no-watermarks] [--quiet]

Optimized Photo Filtering System

options:
  -h, --help            show this help message and exit
  --source SOURCE, -s SOURCE
                        Source directory containing images
  --output OUTPUT, -o OUTPUT
                        Output directory for results
  --dry-run             Process images but don't copy files
  --tests {specifications,text,borders,editing,watermarks} [{specifications,text,borders,editing,watermarks} ...], -t {specifications,text,borders,editing,watermarks} [{specifications,text,borders,editing,watermarks} ...]
                        Tests to run (text detection now uses PaddleOCR)
  --no-specs            Skip specifications check (ignore size/format requirements)
  --no-text             Skip text detection (disable PaddleOCR watermark detection)
  --no-borders          Skip border detection check
  --no-editing          Skip editing detection check (PyIQA-based analysis)
  --no-watermarks       Skip advanced watermark detection check
  --quiet, -q           Suppress detailed progress output
```

#### Basic Usage Examples

```bash
# Run with default settings (all tests enabled)
python main_optimized.py

# Specify custom source and output directories
python main_optimized.py --source /path/to/images --output /path/to/results

# Run only specific tests
python main_optimized.py --tests specifications borders

# Skip specific tests
python main_optimized.py --no-text --no-editing

# Dry run (analyze without moving files)
python main_optimized.py --dry-run

# Quiet mode (minimal output)
python main_optimized.py --quiet
```

#### Test Types Available

- **specifications**: Check image format, size, and basic requirements
- **text**: Detect text overlays using PaddleOCR
- **borders**: Detect artificial borders and frames
- **editing**: Analyze image quality and detect artificial editing (PyIQA-based)
- **watermarks**: Advanced watermark detection using CNN models

## 📊 Output Analysis

### File Organization

Images are automatically sorted into folders based on validation results:

- **`Results/valid/`** - Images that passed all validation checks
- **`Results/invalid/`** - Images that failed validation checks
- **`Results/manualreview/`** - Images that need manual review (non-editing issues)

### Special Editing Detection Behavior

**Important**: Images flagged only for editing review (confidence ≥25%) are **kept in their original location** and not moved to any output folder. This allows you to:
- Review the editing confidence analysis in the console output
- Keep original file organization intact
- Only move files that have actual validation failures

### Processing Reports

The system generates detailed reports in the `Results/logs/` folder:

- **`summary_report.txt`** - Comprehensive processing summary with:
  - Total statistics
  - List of valid images
  - List of invalid images with failure reasons
  - List of images flagged for editing review (kept in place)
  - Processing time and performance metrics

### Console Output

The system provides real-time feedback including:

1. **Processing Progress**: Shows each image being processed with test results
2. **Editing Confidence Analysis Table**: Detailed table showing editing confidence scores for all images
3. **Summary Statistics**: Final counts and success rates
4. **File Movement Information**: Clear indication of which files were moved vs kept in place

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: The system automatically uses CPU mode, but if you're forcing GPU usage and encountering memory issues, ensure you have sufficient GPU memory or let the system fallback to CPU.

#### 2. PaddleOCR Installation Issues
```
ImportError: No module named 'paddle'
```
**Solution**: Install PaddlePaddle separately
```bash
# For CPU
pip install paddlepaddle
# For GPU (if you have CUDA)
pip install paddlepaddle-gpu
```

#### 3. PyTorch Installation Issues
```
ImportError: No module named 'torch'
```
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

#### 4. OpenCV Installation Issues
```
ImportError: No module named 'cv2'
```
**Solution**: Install OpenCV
```bash
pip install opencv-python
```

### Getting Help

If you encounter issues:

1. **Check the console output** - The system provides detailed error messages
2. **Verify all dependencies** are installed from `requirements.txt`
3. **Ensure Python 3.8+** is being used
4. **Check that input images** are in supported formats (JPG, PNG, TIFF, BMP, WEBP)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Special Thanks

### Open Source Repositories
This project was made possible by incorporating code and techniques from these excellent repositories:

#### **PaddlePaddle/PaddleOCR**
- **Repository**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **Used for**: Text detection and recognition capabilities
- **License**: Apache 2.0
- **Contribution**: Core OCR functionality and pre-trained models

#### **chaofengc/IQA-PyTorch**
- **Repository**: [https://github.com/chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
- **Used for**: Image quality assessment metrics (PyIQA)
- **License**: MIT
- **Contribution**: BRISQUE, NIQE, CLIPIQA, and MUSIQ implementations

#### **facebookresearch/ConvNeXt**
- **Repository**: [https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- **Used for**: Watermark detection model architecture
- **License**: MIT
- **Contribution**: ConvNeXt backbone for feature extraction

#### **opencv/opencv**
- **Repository**: [https://github.com/opencv/opencv](https://github.com/opencv/opencv)
- **Used for**: Computer vision operations and image processing
- **License**: Apache 2.0
- **Contribution**: Border detection, morphological operations, and image utilities

#### **huggingface/transformers**
- **Repository**: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- **Used for**: Model hosting and distribution
- **License**: Apache 2.0
- **Contribution**: Model management and inference pipeline

### Research and Development
- **PaddlePaddle Team**: For outstanding OCR models and documentation
- **Computer Vision Community**: For open research and accessible implementations
- **PyTorch Team**: For the deep learning framework that powers our models
- **Hugging Face**: For democratizing access to machine learning models

### Community Contributors
Special appreciation to the open source community whose collective efforts make projects like this possible. The sharing of code, models, and knowledge accelerates innovation for everyone.