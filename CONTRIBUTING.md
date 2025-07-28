# Contributing to PhotoValidator

Thank you for your interest in contributing to PhotoValidator! 🎉

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of computer vision/image processing

### Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/PhotoValidator.git
   cd PhotoValidator
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv dev-env
   
   # Activate virtual environment
   # Windows:
   dev-env\Scripts\activate
   # Linux/macOS:
   source dev-env/bin/activate
   
   # Install dependencies
   python setup.py
   ```

3. **Set up upstream remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/PhotoValidator.git
   ```

## 🤝 How to Contribute

### 🐛 Reporting Bugs
- Use the GitHub issue tracker
- Use the bug report template
- Include detailed reproduction steps
- Provide system information and logs

### ✨ Suggesting Features
- Use the feature request template
- Describe the problem you're solving
- Explain your proposed solution
- Consider the impact on existing users

### 🔧 Code Contributions

#### Types of Contributions Welcome:
- Bug fixes
- Performance improvements
- New detection algorithms
- Documentation improvements
- Test coverage expansion
- UI/UX enhancements

#### Development Workflow:
1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Test with sample images
   python main_optimized.py --debug
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new watermark detection algorithm"
   # Use conventional commit format
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   # Then create PR on GitHub
   ```

## 📝 Coding Standards

### Python Style Guide
- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 88 characters
- Use meaningful variable and function names

### Documentation
- Add docstrings to all functions and classes
- Update README.md for new features
- Include inline comments for complex logic

### Example Code Style:
```python
from typing import List, Dict, Optional
import numpy as np

def detect_watermarks(
    image_path: str,
    confidence_threshold: float = 0.8,
    batch_size: int = 16
) -> Dict[str, Any]:
    """
    Detect watermarks in an image using deep learning.
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detection
        batch_size: Number of images to process simultaneously
        
    Returns:
        Dictionary containing detection results and confidence scores
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If confidence_threshold is not between 0 and 1
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")
    
    # Implementation here
    pass
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_watermark_detection.py

# Run with coverage
python -m pytest tests/ --cov=./ --cov-report=html
```

### Writing Tests
- Write tests for all new functions
- Use descriptive test names
- Include edge cases and error conditions
- Test with various image formats and sizes

### Test Structure:
```python
import pytest
from your_module import your_function

class TestWatermarkDetection:
    def test_detect_watermark_success(self):
        """Test successful watermark detection"""
        result = your_function("test_image.jpg")
        assert result["confidence"] > 0.5
        
    def test_detect_watermark_invalid_file(self):
        """Test error handling for invalid file"""
        with pytest.raises(FileNotFoundError):
            your_function("nonexistent.jpg")
```

## 📤 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)

### PR Title Format
Use conventional commit format:
- `feat: add new border detection algorithm`
- `fix: resolve memory leak in batch processing`
- `docs: update installation instructions`
- `test: add unit tests for quality analysis`

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots to help explain your changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

### Review Process
1. Automated checks must pass
2. At least one code review required
3. All discussions must be resolved
4. Final approval from maintainer

## 🏗️ Project Structure

Understanding the codebase:
```
PhotoValidator/
├── main_optimized.py           # Main entry point
├── optimized_pipeline.py       # Core processing pipeline
├── advanced_watermark_detector.py  # Watermark detection
├── paddle_text_detector.py     # Text detection
├── border_detector.py          # Border detection
├── advanced_pyiqa_detector.py  # Quality analysis
├── tests/                      # Test suite
├── docs/                       # Documentation
└── .github/                    # GitHub workflows
```

### Key Components
- **Detection Modules**: Independent modules for each detection type
- **Pipeline**: Orchestrates the detection workflow
- **Utilities**: Helper functions and configuration
- **Tests**: Comprehensive test suite

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

## 💬 Community

### Getting Help
- **GitHub Discussions**: General questions and community chat
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference

### Communication Guidelines
- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Share knowledge and experiences

## 📞 Contact

For questions about contributing:
- Open a GitHub Discussion
- Create an issue with the "question" label
- Check existing documentation and issues first

---

Thank you for contributing to PhotoValidator! 🙏

*Every contribution, no matter how small, makes a difference.*
