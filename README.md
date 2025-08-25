# PhotoValidator - Smart Image Validation Made Simple

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Windows Compatible](https://img.shields.io/badge/Windows-Compatible-blue.svg)](https://www.microsoft.com/windows)

## 🎯 What is PhotoValidator?

PhotoValidator is an **all-in-one image validation system** that automatically sorts your images into organized folders based on quality, authenticity, and professional standards. Whether you're processing hundreds of photos for a project or validating image collections, PhotoValidator does the heavy lifting for you.

**✨ Perfect for:**
- **Content creators** validating image collections
- **Digital agencies** processing client assets  
- **E-commerce teams** ensuring product image quality
- **Anyone** who needs to quickly sort and validate large image sets

## 🚀 Quick Start - Routine Scan (Recommended)

**The Routine Scan is our flagship feature** - it's the easiest and most comprehensive way to validate your images. Simply:

1. **Place your images** in the `photos4testing` folder
2. **Double-click** `PhotoValidator.bat` 
3. **Select option [1] Routine Scan**
4. **Let it run** - your images will be automatically organized!

### Why Routine Scan is the Best Choice:

- 🎯 **Complete validation** - runs all tests in the optimal sequence
- 🔄 **Smart organization** - automatically sorts results into clear folders
- 🚀 **One-click operation** - no complex setup or configuration needed
- 📊 **User-friendly output** - clear progress and easy-to-understand results
- ⚡ **Optimized performance** - uses the best settings for speed and accuracy

### What Routine Scan Checks:

| Test | What it Detects | Why it Matters |
|------|----------------|----------------|
| **� Specifications** | Image format, size, resolution | Ensures compatibility and standards compliance |
| **�️ Border Detection** | Artificial borders, frames, decorative edges | Identifies processed or edited images |
| **🏷️ Watermark Detection** | Text overlays, logos, copyright marks | Prevents copyright issues |
| **⚡ Quality Assessment** | Compression artifacts, editing signs | Maintains professional image quality |

### Your Results, Organized:

After Routine Scan completes, your images are automatically sorted into:

```
� Results/
├── ✅ Valid/                    - Clean, professional images ready to use
├── ❌ Invalid/
│   ├── Specifications/         - Format or size issues  
│   ├── Border/                 - Images with borders or frames
│   ├── Watermark/              - Images with watermarks detected
│   └── Quality/                - Images with quality issues
├── 🔍 ManualReview/            - Images needing your review
└── 📋 logs/                    - Detailed processing reports
```

---

## 🎮 User Guide - Getting Started

### Step 1: Setup (One-time only)

1. **Download** PhotoValidator to any folder
2. **Create** a `photos4testing` folder in the same location
3. **Install Python** if not already installed (the system will guide you)

### Step 2: Using Routine Scan

1. **Add your images** to the `photos4testing` folder
2. **Double-click** `PhotoValidator.bat`
3. **Select [1] Routine Scan** from the menu
4. **Wait** for processing to complete (progress is shown)
5. **Check the Results folder** - your images are now organized!

### Alternative Options

If you need specific testing only:

- **[2] Custom Scan** - Pick which tests to run (e.g., just watermark detection)
- **[3] Text Detection Only** - Find images with text overlays
- **[4] Border Detection** - Identify images with borders or frames
- **[5] Quality Detection** - Check for editing and compression issues
- **[6] Watermark Detection** - Find copyright marks and logos
- **[7] Specifications Check** - Validate format and size requirements

---

## 📊 Understanding Your Results

### Validation Categories Explained:

**✅ Valid Images**
- Passed all quality checks
- No watermarks or borders detected  
- Proper format and specifications
- Ready for professional use

**❌ Invalid Images**
- Failed one or more validation tests
- Organized by specific issue type
- Review each category to understand problems

**🔍 Manual Review**
- Borderline cases requiring human judgment
- May have minor issues worth checking
- Use your discretion for final decisions

### Reading the Output:

During processing, you'll see:
```
[STEP 1/5] Running Image Specifications Check...
✅ Specifications completed successfully
✗ image1.jpg → Invalid/Specifications
✓ image2.jpg → Valid (passed all tests)
```

- ✅ = Test completed successfully
- ✗ = Image failed this test
- ✓ = Image passed and is valid

---

## 🔧 Advanced Usage & Technical Details

### Custom Scan Mode

Want to run only specific tests? Use **Custom Scan**:

1. Select **[2] Custom Scan** from the main menu
2. Choose which tests to run by entering numbers (e.g., "1 2 4")
3. Only selected tests will be executed
4. Results are organized the same way

Available tests for Custom Scan:
- **[1]** Specifications Check
- **[2]** Border Detection  
- **[3]** Watermark Detection
- **[4]** Quality Assessment

### Command Line Usage (Advanced Users)

For automation or advanced workflows:

```bash
# Routine Scan via command line
python routine_scan_simple.py --input "your_images" --output "results"

# Custom Scan with specific tests (1=specs, 2=border, 3=watermark, 4=quality)
python routine_scan_simple.py --custom "1 2 4" --input "your_images" --output "results"

# Individual test scripts (for experts)
python advanced_pyiqa_detector.py --fast --workers=6 --source "your_images"
python border_detector.py --input "your_images" --output "results"
python advanced_watermark_detector.py --input "your_images" --output "results"
```

---

## 🖥️ System Requirements & Setup

### Requirements:
- **Windows 10/11** (primary support)
- **Python 3.8+** (automatically guided setup)
- **4GB RAM minimum** (8GB+ recommended for large batches)
- **2GB free disk space** (for models and processing)

### First-Time Setup:
1. **Download** PhotoValidator folder
2. **Run** `PhotoValidator.bat` - it will guide you through Python setup if needed
3. **Create** `photos4testing` folder for your images
4. **You're ready!** - Use Routine Scan for comprehensive validation

---

## 🏗️ Technical Architecture (For Developers)

### Core Technologies:
- **🚀 True Parallel Processing**: ThreadPoolExecutor for maximum performance
- **🔍 Text Detection**: PaddleOCR with DB (Differentiable Binarization) model
- **🏷️ Watermark Detection**: CNN-based identification using ConvNeXt architecture
- **🖼️ Border Detection**: Multi-algorithm approach with adaptive thresholds
- **⚡ Quality Analysis**: PyIQA model trio (BRISQUE + NIQE + CLIPIQA) with optional full suite
- **🎯 Adaptive Scoring**: Empirical normalization + multi-feature fusion
- **📁 Smart Organization**: Automated sorting with comprehensive logging

### Performance Metrics:
- **Parallel Processing**: 3-6x faster than sequential processing
- **Fast Model Set**: ~4.5-6 images/sec (BRISQUE + NIQE + CLIPIQA, 6 workers)
- **Full Model Set**: ~3.2-4 images/sec (adds MUSIQ, DBCNN, HyperIQA, 6 workers)
### Project Structure:

```
PhotoValidator/
├── PhotoValidator.bat               # 🎮 Main interface - START HERE
├── routine_scan_simple.py          # 🏗️ Routine & Custom scan engine
├── photos4testing/                 # 📁 Place your images here
├── Results/                        # 📊 Organized output (auto-created)
│   ├── Valid/                      # ✅ Clean images
│   ├── Invalid/                    # ❌ Problem images (by category)
│   ├── ManualReview/               # 🔍 Borderline cases
│   └── logs/                       # 📋 Processing reports
├── main_optimized.py              # 🔧 Advanced processing controller
├── advanced_pyiqa_detector.py     # ⚡ Quality & editing detection
├── border_detector.py             # 🖼️ Border detection
├── advanced_watermark_detector.py # 🏷️ Watermark detection
├── Spec_detector.py               # 📏 Specifications validation
└── requirements.txt               # 📦 Python dependencies
```

### Advanced CLI Commands (For Developers):

```bash
# Routine scan with custom settings
python routine_scan_simple.py --input "custom_folder" --output "results" --python "python"

# Custom scan with specific tests
python routine_scan_simple.py --custom "1 3" --input "images" --output "results"

# Individual test components
python advanced_pyiqa_detector.py --fast --workers=6 --source "images"
python border_detector.py --input "images" --output "results"
python advanced_watermark_detector.py --input "images" --output "results"
python main_optimized.py --tests specifications --source "images" --output "results"
```

---

## 📋 Installation & Dependencies (Technical)

### Python Requirements:
- **Python 3.8+** (3.10+ recommended)
- **Virtual environment recommended**

### Key Dependencies:
```bash
pip install -r requirements.txt
```

Core packages:
- `torch` - PyTorch for neural networks
- `opencv-python` - Computer vision operations  
- `paddlepaddle` & `paddleocr` - Text detection
- `pyiqa` - Image quality assessment
- `Pillow` - Image processing
- `numpy`, `matplotlib` - Data processing

### Hardware Recommendations:
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended (CUDA-compatible)
- **Storage**: 5GB+ free space for models and cache
- **CPU**: Multi-core processor recommended

### Supported Systems:
- Windows 10/11 (primary support)
- macOS 10.15+  
- Ubuntu 18.04+
---

## 🐛 Troubleshooting & Support

### Common Issues:

**❓ "Python not found" error**
- Run `PhotoValidator.bat` - it will guide you through Python installation
- Or manually install Python 3.8+ from python.org

**❓ "CUDA out of memory" warning**  
- The system automatically uses CPU mode - no action needed
- For better performance, reduce image batch size

**❓ Slow processing**
- Use Routine Scan (option 1) - it's optimized for performance
- Close other programs to free up system resources
- Consider using Custom Scan to run only needed tests

**❓ Images not organizing correctly**
- Check that your images are in `photos4testing` folder
- Ensure image formats are supported (JPG, PNG, BMP, TIFF)
- Check `Results/logs/` folder for detailed processing information

### Getting Help:

1. **Check the `Results/logs/` folder** for detailed error information
2. **Run Full System Validation** (option V) to check your setup
3. **Use Custom Scan** to isolate which specific test is causing issues

---

## � Quick Reference Card

### For Regular Users:
1. **Put images** in `photos4testing` folder
2. **Run** `PhotoValidator.bat`  
3. **Select [1] Routine Scan**
4. **Check** `Results` folder when done

### For Advanced Users:
- **Custom Scan**: Pick specific tests (option 2)
- **CLI Usage**: `python routine_scan_simple.py --help`
- **Individual Tests**: Run specific detection scripts directly

### Result Folders:
- **✅ Valid**: Ready to use
- **❌ Invalid**: Problems detected (organized by issue type)
- **🔍 ManualReview**: Check these yourself
- **📋 logs**: Detailed processing information

---

*PhotoValidator - Making image validation simple and reliable.*
portrait_enhanced.jpg        67.3%                 Invalid - heavily edited (>30%)
landscape_touched.jpg        27.1%                 Manual review needed (25-30%)
nature_photo.jpg             12.1%                 Valid - clean image (<25%)
────────────────────────────────────────────────────

SUMMARY:
  Total Images Processed: 177
  Processing Rate: 4.51 images/second
  Worker Efficiency: 87.3%
  Valid Images: 134 (moved to Results/valid/)
  Manual Review Needed: 28 (moved to Results/manualreview/)
  Invalid Images: 15 (moved to Results/invalid/)
```

#### File Organization Behavior

**All processing modes organize files into folders:**

- **`Results/valid/`** - Clean images with low editing confidence (<25% or <20% depending on mode)
- **`Results/manualreview/`** - Images requiring human review (borderline editing confidence)
- **`Results/invalid/`** - Images with high editing confidence (heavily processed)
- **`Results/logs/`** - Processing reports and analysis data

**Note**: Images are **moved** to appropriate folders, not kept in place. This provides clear organization for validation workflows.

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: The system automatically falls back to CPU mode, but you can also:
- Reduce worker count: `--workers=2` or `--workers=4`
- Force CPU mode: `--cpu`
- Use fast models only: `--fast`

#### 2. Parallel Processing Issues
```
High CPU usage or system slowdown
```
**Solutions**:
- Reduce worker count: `--workers=2` (for older systems)
- Check system resources before running large batches
- Use `PhotoValidator.bat` option 4 which is pre-optimized for parallel processing

#### 3. PaddleOCR Installation Issues
```
ImportError: No module named 'paddle'
```
**Solution**: Install PaddlePaddle separately
```powershell
# For CPU
pip install paddlepaddle

# For GPU (if you have CUDA)
pip install paddlepaddle-gpu

# Alternative: Use the setup script
.\setup_python_environment.ps1
```

#### 4. PyTorch Installation Issues
```
ImportError: No module named 'torch'
```
**Solution**: Install PyTorch
```bash
# CPU version
pip install torch torchvision

# GPU version (check pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Performance Issues
```
Slow processing or high memory usage
```
**Solutions**:
- Use PhotoValidator.bat for optimized settings
- Try `--fast` mode: `python advanced_pyiqa_detector.py --fast`
- Adjust worker count based on your CPU cores
- Monitor system resources during processing

#### 6. PhotoValidator.bat Not Working
```
'python' is not recognized as an internal or external command
```
**Solution**: Run the setup script first
```powershell
# Run as Administrator
.\setup_python_environment.ps1

# Or manually set Python path in PhotoValidator.bat
# Edit line: set "PYTHON_PATH=C:/Path/To/Your/Python.exe"
```

### Getting Help

If you encounter issues:

1. **🎮 Try PhotoValidator.bat first** - It has optimized settings and error handling
2. **📋 Check the console output** - The system provides detailed error messages and performance metrics
3. **🔧 Run the setup script** - `.\setup_python_environment.ps1` (Windows) handles most dependency issues
4. **📦 Verify dependencies** - Ensure all packages from `requirements.txt` are installed
5. **🐍 Check Python version** - Python 3.8+ is required (3.10+ recommended)
6. **🖼️ Validate input images** - Supported formats: JPG, PNG, TIFF, BMP, WEBP
7. **⚡ Test with fast mode** - Use `--fast` flag to isolate performance vs model loading issues
8. **👥 Check worker count** - Start with `--workers=2` on older systems

### Performance Optimization Tips

- **🚀 Use PhotoValidator.bat option 4** for fastest editing detection
- **⚡ Try `--fast` mode** for 40-50% speed improvement  
- **👥 Adjust `--workers=N`** based on your CPU cores (start with 4-6)
- **🖥️ Use `--cpu` on systems** without dedicated GPU
- **📁 Process smaller batches** if encountering memory issues
- **🔄 Monitor system resources** during large batch processing

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