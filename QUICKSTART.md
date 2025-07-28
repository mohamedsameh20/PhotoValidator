# Quick Start Guide - Get Running in 5 Minutes

## 🚀 1-Minute Setup (Automated)

### Windows
```cmd
git clone https://github.com/yourusername/advanced-image-processing-pipeline.git
cd advanced-image-processing-pipeline
setup.bat
```

### Linux/macOS
```bash
git clone https://github.com/yourusername/advanced-image-processing-pipeline.git
cd advanced-image-processing-pipeline
chmod +x setup.sh
./setup.sh
```

## ⚡ 2-Minute Manual Setup

### 1. Install Python 3.8+
- Download from [python.org](https://www.python.org/downloads/)
- Verify: `python --version`

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python pillow numpy matplotlib
pip install paddlepaddle paddleocr pyiqa timm scikit-learn pandas
pip install huggingface-hub tqdm
```

### 3. Create Folders
```bash
mkdir photos4testing Results debug_visualizations models
```

## 🖼️ 3-Minute Usage

### 1. Add Your Images
```bash
# Copy your images to the input folder
cp /path/to/your/images/* photos4testing/
```

### 2. Run Processing
```bash
python main_optimized.py
```

### 3. Check Results
```bash
ls Results/
# Results/valid/        - Clean images
# Results/invalid/      - Images with issues
# Results/manualreview/ - Borderline cases
```

## ⚙️ Command Line Options

```bash
# Basic usage
python main_optimized.py

# Custom input folder
python main_optimized.py --input-folder /path/to/images

# Adjust batch size (reduce if out of memory)
python main_optimized.py --batch-size 8

# CPU-only mode
python main_optimized.py --cpu-only

# Enable debug visualizations
python main_optimized.py --debug

# Adjust detection sensitivity
python main_optimized.py --text-threshold 0.8 --watermark-threshold 0.9
```

## 🎯 What Each Detector Does

| Detector | Purpose | Output |
|----------|---------|--------|
| **Text Detection** | Finds text/watermarks using PaddleOCR | `invalid/text/` |
| **Watermark Detection** | CNN-based watermark identification | `invalid/watermark/` |
| **Border Detection** | Artificial borders/frames | `invalid/border/` |
| **Quality Analysis** | Image quality issues (PyIQA metrics) | `invalid/quality/` |

## 📊 Understanding Results

### Folder Structure
```
Results/
├── valid/                    # ✅ Clean images
├── invalid/
│   ├── text/                # 📝 Text overlay detected
│   ├── watermark/           # 🏷️ Watermark detected
│   ├── border/              # 🖼️ Artificial border detected
│   └── quality/             # 📉 Quality issues detected
├── manualreview/            # 🤔 Uncertain cases
└── processing_report_*.json # 📋 Detailed statistics
```

### Processing Report
```json
{
    "timestamp": "2025-07-28T14:30:15",
    "total_images": 127,
    "processing_time": 45.2,
    "results": {
        "valid": 89,
        "invalid": 32,
        "manual_review": 6
    },
    "detection_summary": {
        "text_detected": 18,
        "watermarks_detected": 3,
        "borders_detected": 8,
        "quality_issues": 15
    }
}
```

## 🔧 Quick Troubleshooting

### Common Issues

#### Out of Memory Error
```bash
# Reduce batch size
python main_optimized.py --batch-size 4 --cpu-only
```

#### PaddleOCR Installation Failed
```bash
# Install CPU version
pip install paddlepaddle
pip install paddleocr
```

#### Slow Processing
```bash
# Check GPU usage
nvidia-smi

# Enable GPU if available
python main_optimized.py --batch-size 16
```

#### Images Not Processing
```bash
# Check supported formats: JPG, PNG, TIFF, BMP, WEBP
ls photos4testing/ | grep -E '\.(jpg|jpeg|png|tiff|tif|bmp|webp)$'

# Run test
python quick_test.py
```

### Quick Fixes

1. **Permission Errors**: Run with administrator/sudo privileges
2. **Network Issues**: Pre-download models with `python -c "import paddleocr; paddleocr.PaddleOCR()"`
3. **CUDA Issues**: Add `--cpu-only` flag
4. **Path Issues**: Use absolute paths: `--input-folder /full/path/to/images`

## 📈 Performance Tips

### Optimal Settings by Hardware

#### 8GB RAM / No GPU
```bash
python main_optimized.py --batch-size 4 --cpu-only --num-workers 2
```

#### 16GB RAM / GTX 1060 6GB
```bash
python main_optimized.py --batch-size 8 --num-workers 4
```

#### 32GB RAM / RTX 3080 10GB
```bash
python main_optimized.py --batch-size 16 --num-workers 8
```

#### High-end (64GB RAM / RTX 4090)
```bash
python main_optimized.py --batch-size 32 --num-workers 12
```

### Speed Optimization
- **Enable GPU**: Install CUDA-compatible PyTorch
- **Increase batch size**: More parallel processing
- **Use SSD storage**: Faster image loading
- **Reduce image size**: Resize large images before processing

### Memory Optimization
- **Reduce batch size**: Lower memory usage
- **Enable cleanup**: Automatic memory management
- **Close other applications**: Free up RAM
- **Use CPU mode**: If GPU memory insufficient

## 🎓 Example Workflows

### Workflow 1: Photo Collection Cleaning
```bash
# Process entire photo collection
python main_optimized.py --input-folder /Users/john/Photos

# Move valid photos to clean folder
cp Results/valid/* /Users/john/CleanPhotos/
```

### Workflow 2: Social Media Content Check
```bash
# High sensitivity for commercial use
python main_optimized.py --text-threshold 0.6 --watermark-threshold 0.7

# Review borderline cases manually
ls Results/manualreview/
```

### Workflow 3: Batch Processing with Quality Control
```bash
# Enable debug for detailed analysis
python main_optimized.py --debug --save-debug-images

# Check quality analysis charts
ls debug_visualizations/
```

### Workflow 4: Custom Processing Pipeline
```python
# Use individual components
from paddle_text_detector import PaddleTextDetector
from advanced_watermark_detector import WatermarkDetector

text_detector = PaddleTextDetector()
watermark_detector = WatermarkDetector()

# Process single image
text_result = text_detector.detect_text("image.jpg")
watermark_result = watermark_detector.detect_watermark("image.jpg")
```

## ❓ Need Help?

### Immediate Support
- **Test installation**: `python quick_test.py`
- **Check logs**: `tail -f debug.log`
- **Read troubleshooting**: `cat TROUBLESHOOTING.md`

### Community
- **GitHub Issues**: Bug reports and features
- **Discussions**: Questions and tips
- **Wiki**: Detailed documentation

### Next Steps
1. ✅ **Read full README**: `cat README.md`
2. 🔧 **Check configuration**: Edit `config.json`
3. 📊 **Analyze results**: Review processing reports
4. 🎯 **Fine-tune settings**: Adjust thresholds for your use case

---

**Happy processing! 🖼️✨**

*For detailed documentation, see [README.md](README.md)*
*For issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)*
