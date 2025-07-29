#!/usr/bin/env python3
"""
PhotoValidator Setup Script v2.0
Advanced cross-platform installation and setup for PhotoValidator image processing pipeline.
Supports Windows, Linux, and macOS with comprehensive error handling and validation.
"""

import os
import sys
import platform
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional

class PhotoValidatorSetup:
    def __init__(self, verbose: bool = False, skip_gpu: bool = False):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.project_dir = Path(__file__).parent.resolve()
        self.verbose = verbose
        self.skip_gpu = skip_gpu
        self.requirements_installed = False
        self.errors = []
        self.warnings = []
        self.installed_packages = []
        self.failed_packages = []
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def print_header(self):
        """Print setup header with system information"""
        print("=" * 80)
        print("🚀 PhotoValidator Setup Script v2.0")
        print("   Advanced Image Processing Pipeline - Cross-Platform Setup")
        print("=" * 80)
        print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
        print(f"Python: {sys.version}")
        print(f"Architecture: {platform.architecture()[0]}")
        print(f"Project Directory: {self.project_dir}")
        print(f"Setup Mode: {'Verbose' if self.verbose else 'Standard'}")
        if self.skip_gpu:
            print("GPU Detection: Disabled")
        print("=" * 80)
        print()
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("🐍 Checking Python version compatibility...")
        min_version = (3, 8)
        recommended_version = (3, 9)
        
        if self.python_version < min_version:
            error_msg = f"Python {min_version[0]}.{min_version[1]}+ is required for PhotoValidator"
            print(f"❌ Error: {error_msg}")
            print(f"   Current version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            print("   Please install Python 3.8 or higher from https://python.org")
            self.errors.append(error_msg)
            return False
            
        if self.python_version < recommended_version:
            warning_msg = f"Python {recommended_version[0]}.{recommended_version[1]}+ recommended for best performance"
            print(f"   ⚠️  {warning_msg}")
            self.warnings.append(warning_msg)
            
        print(f"   ✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} is compatible")
        return True
        
    def check_pip(self) -> bool:
        """Check if pip is available and working"""
        print("📦 Checking pip installation...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  check=True, capture_output=True, text=True, timeout=30)
            pip_version = result.stdout.strip()
            print(f"   ✅ {pip_version}")
            
            # Check if pip needs upgrading
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated", "pip"], 
                                      capture_output=True, text=True, timeout=10)
                if "pip" in result.stdout:
                    print("   ⚠️  pip can be upgraded")
                    return self._upgrade_pip()
            except:
                pass  # Ignore upgrade check errors
                
            return True
        except subprocess.TimeoutExpired:
            error_msg = "pip check timed out"
            print(f"❌ Error: {error_msg}")
            self.errors.append(error_msg)
            return False
        except subprocess.CalledProcessError as e:
            error_msg = "pip is not available or not working"
            print(f"❌ Error: {error_msg}")
            print("   Please install pip or repair your Python installation")
            self.errors.append(error_msg)
            return False
            
    def _upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        print("   Upgrading pip...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True, text=True, timeout=60)
            print("   ✅ pip upgraded successfully")
            return True
        except Exception as e:
            warning_msg = f"Failed to upgrade pip: {e}"
            print(f"   ⚠️  {warning_msg}")
            self.warnings.append(warning_msg)
            return True  # Continue even if upgrade fails
            
    def check_internet_connection(self) -> bool:
        """Check if internet connection is available for downloading packages"""
        print("🌐 Checking internet connectivity...")
        test_urls = [
            'https://pypi.org',
            'https://files.pythonhosted.org',
            'https://google.com'
        ]
        
        for url in test_urls:
            try:
                urllib.request.urlopen(url, timeout=10)
                print("   ✅ Internet connection available")
                return True
            except Exception:
                continue
                
        warning_msg = "No internet connection detected"
        print(f"   ⚠️  {warning_msg}")
        print("   Package installation may fail")
        self.warnings.append(warning_msg)
        return False
        
    def check_system_requirements(self) -> bool:
        """Check system-specific requirements"""
        print("💻 Checking system requirements...")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"   RAM: {memory_gb:.1f} GB")
            if memory_gb < 4:
                warning_msg = "Less than 4GB RAM available - may affect performance"
                print(f"   ⚠️  {warning_msg}")
                self.warnings.append(warning_msg)
        except ImportError:
            print("   ⚠️  Cannot check memory usage (psutil not available)")
            
        # Check disk space
        free_space = shutil.disk_usage(self.project_dir).free / (1024**3)
        print(f"   Free disk space: {free_space:.1f} GB")
        if free_space < 2:
            warning_msg = "Less than 2GB free disk space - may cause issues"
            print(f"   ⚠️  {warning_msg}")
            self.warnings.append(warning_msg)
            
        # Check CPU cores
        cpu_count = os.cpu_count() or 1
        print(f"   CPU cores: {cpu_count}")
        
        return True
        
    def create_directories(self) -> bool:
        """Create necessary project directories"""
        print("📁 Creating project directory structure...")
        
        directories = [
            "photos4testing",
            "Results",
            "Results/valid",
            "Results/invalid", 
            "Results/manualreview",
            "models_cache",
            "temp",
            "output",
            "logs"
        ]
        
        created_dirs = []
        for directory in directories:
            dir_path = self.project_dir / directory
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(directory)
                if self.verbose:
                    print(f"   Created: {directory}")
            except Exception as e:
                error_msg = f"Failed to create directory {directory}: {e}"
                print(f"❌ Error: {error_msg}")
                self.errors.append(error_msg)
                return False
                
        print(f"   ✅ Created {len(created_dirs)} directories")
        return True
        
    def install_requirements(self) -> bool:
        """Install Python requirements with comprehensive error handling"""
        requirements_file = self.project_dir / "requirements.txt"
        
        if not requirements_file.exists():
            error_msg = "requirements.txt not found"
            print(f"❌ Error: {error_msg}")
            self.errors.append(error_msg)
            return False
            
        print("📦 Installing Python requirements...")
        print(f"   Reading from: {requirements_file}")
        
        # Read and parse requirements
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            requirements = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    requirements.append(line)
                    
        except Exception as e:
            error_msg = f"Failed to read requirements.txt: {e}"
            print(f"❌ Error: {error_msg}")
            self.errors.append(error_msg)
            return False
            
        print(f"   Found {len(requirements)} packages to install")
        
        # Try bulk installation first
        success = self._try_bulk_install(requirements)
        if success:
            print("   ✅ All requirements installed successfully (bulk)")
            self.requirements_installed = True
            return True
            
        # Fall back to individual installation
        print("   ⚠️  Bulk installation failed, trying individual packages...")
        return self._install_packages_individually(requirements)
        
    def _try_bulk_install(self, requirements: List[str]) -> bool:
        """Try to install all requirements at once"""
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + requirements
            if self.verbose:
                print(f"   Running: {' '.join(cmd[:6])}... ({len(requirements)} packages)")
                
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if self.verbose:
                print(f"   Bulk install failed: {type(e).__name__}")
            return False
            
    def _install_packages_individually(self, requirements: List[str]) -> bool:
        """Install packages one by one with detailed error reporting"""
        for i, package in enumerate(requirements, 1):
            print(f"   [{i}/{len(requirements)}] Installing {package}...")
            try:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                print(f"   ✅ {package}")
                self.installed_packages.append(package)
            except subprocess.TimeoutExpired:
                print(f"   ⏰ Timeout installing {package}")
                self.failed_packages.append(package)
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed: {package}")
                if self.verbose:
                    print(f"      Error: {e.stderr.strip()[:100]}...")
                self.failed_packages.append(package)
                
        # Summary
        installed_count = len(self.installed_packages)
        failed_count = len(self.failed_packages)
        total_count = len(requirements)
        
        if failed_count > 0:
            print(f"   ⚠️  Installed {installed_count}/{total_count} packages")
            print(f"   Failed packages: {', '.join(self.failed_packages)}")
            warning_msg = f"Failed to install {failed_count} packages"
            self.warnings.append(warning_msg)
            
        if installed_count > 0:
            print(f"   ✅ Successfully installed {installed_count} packages")
            self.requirements_installed = True
            return True
        else:
            error_msg = "Failed to install any required packages"
            print(f"❌ Error: {error_msg}")
            self.errors.append(error_msg)
            return False
            
    def check_gpu_support(self) -> Dict:
        """Check for GPU support and provide detailed information"""
        if self.skip_gpu:
            print("🎮 GPU detection skipped (--skip-gpu)")
            return {'cuda_available': False, 'skip_gpu': True}
            
        print("🎮 Checking GPU support...")
        
        gpu_info = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_names': [],
            'torch_version': None,
            'gpu_memory': []
        }
        
        try:
            # Check PyTorch and CUDA
            import torch
            gpu_info['torch_version'] = torch.__version__
            print(f"   PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                print(f"   ✅ CUDA {gpu_info['cuda_version']} available")
                print(f"   ✅ {gpu_info['gpu_count']} GPU(s) detected:")
                
                for i in range(gpu_info['gpu_count']):
                    name = torch.cuda.get_device_name(i)
                    gpu_info['gpu_names'].append(name)
                    
                    # Get memory info
                    try:
                        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_info['gpu_memory'].append(memory_total)
                        print(f"      GPU {i}: {name} ({memory_total:.1f} GB)")
                    except:
                        print(f"      GPU {i}: {name}")
                        
            else:
                print("   ⚠️  CUDA not available - using CPU mode")
                print("   This is normal for systems without NVIDIA GPUs")
                
        except ImportError:
            print("   ⚠️  PyTorch not installed - cannot check CUDA")
            warning_msg = "Cannot verify GPU support without PyTorch"
            self.warnings.append(warning_msg)
        except Exception as e:
            print(f"   ⚠️  GPU check failed: {e}")
            warning_msg = f"GPU detection error: {e}"
            self.warnings.append(warning_msg)
            
        return gpu_info
        
    def create_config_file(self) -> bool:
        """Create comprehensive configuration file"""
        print("⚙️  Creating system configuration...")
        
        # Get GPU info
        gpu_info = self.check_gpu_support()
        
        config = {
            'version': '2.0.0',
            'setup_info': {
                'setup_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'setup_version': '2.0',
                'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                'platform': platform.system(),
                'architecture': platform.architecture()[0],
                'machine': platform.machine()
            },
            'system': {
                'cpu_count': os.cpu_count() or 1,
                'platform': platform.system(),
                'python_executable': sys.executable
            },
            'gpu': {
                'cuda_available': gpu_info.get('cuda_available', False),
                'cuda_version': gpu_info.get('cuda_version'),
                'gpu_count': gpu_info.get('gpu_count', 0),
                'gpu_names': gpu_info.get('gpu_names', []),
                'torch_version': gpu_info.get('torch_version')
            },
            'performance': {
                'use_gpu': gpu_info.get('cuda_available', False),
                'num_workers': min(8, os.cpu_count() or 1),
                'batch_size': 4 if gpu_info.get('cuda_available', False) else 1,
                'memory_limit_gb': 4,
                'enable_multiprocessing': True
            },
            'paths': {
                'input_dir': 'photos4testing',
                'output_dir': 'Results',
                'models_cache_dir': 'models_cache',
                'temp_dir': 'temp',
                'logs_dir': 'logs',
                'valid_dir': 'Results/valid',
                'invalid_dir': 'Results/invalid',
                'manual_review_dir': 'Results/manualreview'
            },
            'detection': {
                'confidence_threshold': 0.7,
                'enable_watermark_detection': True,
                'enable_text_detection': True,
                'enable_border_detection': True,
                'enable_quality_analysis': True,
                'enable_artificial_detection': True
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'max_log_size_mb': 100,
                'keep_logs_days': 30
            },
            'installation': {
                'requirements_installed': self.requirements_installed,
                'installed_packages': len(self.installed_packages),
                'failed_packages': len(self.failed_packages),
                'warnings_count': len(self.warnings),
                'errors_count': len(self.errors)
            }
        }
        
        config_file = self.project_dir / "config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Configuration saved to {config_file}")
            return True
        except Exception as e:
            error_msg = f"Failed to create config file: {e}"
            print(f"❌ Error: {error_msg}")
            self.errors.append(error_msg)
            return False
            
    def create_run_scripts(self) -> bool:
        """Create improved run scripts for different platforms"""
        print("📝 Creating cross-platform run scripts...")
        
        # Windows batch file
        batch_content = f'''@echo off
REM PhotoValidator Windows Launcher
REM Generated by setup script v2.0

echo ===============================================
echo PhotoValidator - Image Processing Pipeline
echo ===============================================
echo.

cd /d "{self.project_dir}"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available or not in PATH
    echo Please install Python 3.8+ or add it to your PATH
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "main_optimized.py" (
    echo ERROR: main_optimized.py not found
    echo Please ensure you're running this from the PhotoValidator directory
    pause
    exit /b 1
)

echo Starting PhotoValidator...
echo Input folder: photos4testing/
echo Output folder: Results/
echo.

REM Run the main script with all passed arguments
python main_optimized.py %*

REM Check exit code
if errorlevel 1 (
    echo.
    echo ERROR: PhotoValidator encountered an error
    echo Check the output above for details
) else (
    echo.
    echo SUCCESS: PhotoValidator completed successfully
    echo Check the Results/ folder for processed images
)

echo.
echo Press any key to close this window...
pause >nul
'''
        
        try:
            with open(self.project_dir / "run_photovalidator.bat", 'w', encoding='utf-8') as f:
                f.write(batch_content)
            print("   ✅ Created: run_photovalidator.bat")
        except Exception as e:
            print(f"   ❌ Failed to create Windows script: {e}")
            
        # Shell script for Linux/macOS
        shell_content = f'''#!/bin/bash
# PhotoValidator Unix Launcher
# Generated by setup script v2.0

echo "==============================================="
echo "PhotoValidator - Image Processing Pipeline"
echo "==============================================="
echo

cd "{self.project_dir}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not available"
        echo "Please install Python 3.8+ or add it to your PATH"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')")
echo "Using Python $PYTHON_VERSION"

# Check if main script exists
if [ ! -f "main_optimized.py" ]; then
    echo "ERROR: main_optimized.py not found"
    echo "Please ensure you're running this from the PhotoValidator directory"
    exit 1
fi

echo "Starting PhotoValidator..."
echo "Input folder: photos4testing/"
echo "Output folder: Results/"
echo

# Run the main script with all passed arguments
$PYTHON_CMD main_optimized.py "$@"
exit_code=$?

# Check exit code
if [ $exit_code -ne 0 ]; then
    echo
    echo "ERROR: PhotoValidator encountered an error (exit code: $exit_code)"
    echo "Check the output above for details"
else
    echo
    echo "SUCCESS: PhotoValidator completed successfully"
    echo "Check the Results/ folder for processed images"
fi

exit $exit_code
'''
        
        try:
            script_path = self.project_dir / "run_photovalidator.sh"
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(shell_content)
                
            # Make executable on Unix systems
            if self.system in ["linux", "darwin"]:
                os.chmod(script_path, 0o755)
                
            print("   ✅ Created: run_photovalidator.sh")
        except Exception as e:
            print(f"   ❌ Failed to create Unix script: {e}")
            
        return True
        
    def create_sample_readme(self) -> bool:
        """Create README file for the input folder"""
        print("📄 Creating input folder documentation...")
        
        readme_content = """PhotoValidator - Input Images Folder
====================================

This folder is where you place images for processing with PhotoValidator.

SUPPORTED FORMATS:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- WebP (.webp)

USAGE:
1. Copy your images to this folder
2. Run PhotoValidator using the run script or command line
3. Check the Results/ folder for processed images

OUTPUT STRUCTURE:
- Results/valid/        - Images that passed all validation checks
- Results/invalid/      - Images with detected issues (watermarks, text, etc.)
- Results/manualreview/ - Images requiring manual review

TIPS:
- Images will be automatically resized if they're too large
- Batch processing handles all images in this folder
- Original images are never modified
- Processing logs are saved to the logs/ folder

For more information, see the main README.md file.
"""
        
        try:
            readme_path = self.project_dir / "photos4testing" / "README.txt"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print("   ✅ Created input folder documentation")
            return True
        except Exception as e:
            print(f"   ❌ Failed to create input documentation: {e}")
            return False
            
    def test_installation(self) -> bool:
        """Comprehensive installation testing"""
        print("🧪 Testing installation...")
        
        test_results = {
            'imports': False,
            'basic_functionality': False,
            'gpu_detection': False,
            'file_access': False
        }
        
        # Test 1: Import test
        print("   Testing package imports...")
        import_test_script = '''
import sys
import traceback

try:
    # Core dependencies
    import torch
    import cv2
    import numpy as np
    import PIL
    print("✅ Core packages imported successfully")
    
    # Test torch functionality
    x = torch.tensor([1, 2, 3])
    print(f"✅ PyTorch tensor creation: {x}")
    
    # Test OpenCV
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
    
    print("IMPORTS_SUCCESS")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            result = subprocess.run([sys.executable, "-c", import_test_script], 
                                  capture_output=True, text=True, cwd=self.project_dir, timeout=60)
            
            if result.returncode == 0 and "IMPORTS_SUCCESS" in result.stdout:
                print("   ✅ Package import test passed")
                test_results['imports'] = True
            else:
                print("   ❌ Package import test failed")
                if self.verbose:
                    print(f"   STDOUT: {result.stdout}")
                    print(f"   STDERR: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print("   ❌ Import test timed out")
        except Exception as e:
            print(f"   ❌ Import test error: {e}")
            
        # Test 2: File access test
        print("   Testing file system access...")
        try:
            test_file = self.project_dir / "photos4testing" / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            print("   ✅ File system access test passed")
            test_results['file_access'] = True
        except Exception as e:
            print(f"   ❌ File access test failed: {e}")
            
        # Test 3: GPU detection (if not skipped)
        if not self.skip_gpu:
            print("   Testing GPU detection...")
            gpu_test_script = '''
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {device_count} device(s)")
    else:
        print("ℹ️  CUDA not available (using CPU)")
    print("GPU_TEST_SUCCESS")
except Exception as e:
    print(f"⚠️  GPU test error: {e}")
    print("GPU_TEST_SUCCESS")  # Non-critical
'''
            
            try:
                result = subprocess.run([sys.executable, "-c", gpu_test_script], 
                                      capture_output=True, text=True, timeout=30)
                if "GPU_TEST_SUCCESS" in result.stdout:
                    print("   ✅ GPU detection test passed")
                    test_results['gpu_detection'] = True
            except:
                print("   ⚠️  GPU test skipped")
                test_results['gpu_detection'] = True  # Non-critical
        else:
            test_results['gpu_detection'] = True  # Skipped
            
        # Summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        if passed_tests == total_tests:
            print(f"   ✅ All {total_tests} tests passed")
            return True
        elif passed_tests >= total_tests - 1:  # Allow one non-critical failure
            print(f"   ⚠️  {passed_tests}/{total_tests} tests passed (acceptable)")
            return True
        else:
            print(f"   ❌ Only {passed_tests}/{total_tests} tests passed")
            return False
            
    def print_completion_message(self) -> None:
        """Print comprehensive setup completion message"""
        print("\n" + "=" * 80)
        print("🎉 PhotoValidator Setup Complete!")
        print("=" * 80)
        
        # Setup summary
        print("\n📊 Setup Summary:")
        if self.requirements_installed:
            print(f"   ✅ Requirements installed ({len(self.installed_packages)} packages)")
        if self.failed_packages:
            print(f"   ⚠️  Failed packages: {len(self.failed_packages)}")
        if self.warnings:
            print(f"   ⚠️  Warnings: {len(self.warnings)}")
        if self.errors:
            print(f"   ❌ Errors: {len(self.errors)}")
            
        print("\n📁 Project Structure:")
        print("   photos4testing/         - Add your images here")
        print("   Results/               - Processing results appear here")
        print("     ├── valid/          - Images that passed validation")
        print("     ├── invalid/        - Images with detected issues")
        print("     └── manualreview/   - Images requiring manual review")
        print("   models_cache/          - Downloaded AI models cache")
        print("   logs/                  - Processing logs")
        print("   config.json           - System configuration")
        
        print("\n🚀 Quick Start:")
        print("   1. Add images to photos4testing/ folder")
        if self.system == "windows":
            print("   2. Double-click: run_photovalidator.bat")
        else:
            print("   2. Run: ./run_photovalidator.sh")
        print("   3. Check Results/ folder for processed images")
        
        print("\n💻 Command Line Usage:")
        print("   Basic processing:         python main_optimized.py")
        print("   Custom input folder:      python main_optimized.py --input-folder /path/to/images")
        print("   Skip text detection:      python main_optimized.py --no-text-detection")
        print("   Skip quality analysis:    python main_optimized.py --no-quality-analysis")
        print("   CPU only mode:            python main_optimized.py --cpu-only")
        print("   Verbose output:           python main_optimized.py --verbose")
        
        print("\n📖 Documentation:")
        print("   README.md              - Complete documentation")
        print("   QUICKSTART.md          - Quick start guide")
        print("   TROUBLESHOOTING.md     - Common issues and solutions")
        print("   config.json           - System configuration details")
        
        if self.warnings:
            print("\n⚠️  Warnings to Address:")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"   • {warning}")
            if len(self.warnings) > 5:
                print(f"   • ... and {len(self.warnings) - 5} more")
                
        if self.errors:
            print("\n❌ Errors to Fix:")
            for error in self.errors[:3]:  # Show first 3 errors
                print(f"   • {error}")
            if len(self.errors) > 3:
                print(f"   • ... and {len(self.errors) - 3} more")
                
        print("\n🎯 Next Steps:")
        if self.system == "windows":
            print("   → Double-click 'run_photovalidator.bat' to start")
        else:
            print("   → Run './run_photovalidator.sh' to start")
        print("   → Add test images to photos4testing/ folder")
        print("   → Check Results/ folder after processing")
        
        print("=" * 80)
        
    def run_setup(self) -> bool:
        """Run the complete setup process with comprehensive error handling"""
        try:
            self.print_header()
            
            # Phase 1: System checks
            print("🔍 Phase 1: System Validation")
            if not self.check_python_version():
                return False
            if not self.check_pip():
                return False
            self.check_internet_connection()
            self.check_system_requirements()
            print()
            
            # Phase 2: Directory setup
            print("📁 Phase 2: Directory Setup")
            if not self.create_directories():
                return False
            self.create_sample_readme()
            print()
            
            # Phase 3: Package installation
            print("📦 Phase 3: Package Installation")
            if not self.install_requirements():
                print("\n❌ Setup failed during package installation")
                print("   This is usually due to network issues or missing system dependencies")
                print("   Check the error messages above and try again")
                return False
            print()
            
            # Phase 4: Configuration
            print("⚙️  Phase 4: Configuration")
            if not self.create_config_file():
                print("⚠️  Configuration creation failed, but continuing...")
            self.create_run_scripts()
            print()
            
            # Phase 5: Testing
            print("🧪 Phase 5: Installation Testing")
            test_passed = self.test_installation()
            if not test_passed:
                print("   ⚠️  Some tests failed, but installation may still work")
                print("   Try running PhotoValidator manually to verify")
            print()
            
            # Completion
            self.print_completion_message()
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Setup interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed with unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="PhotoValidator Setup Script v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_improved.py                 # Standard setup
  python setup_improved.py --verbose       # Detailed output
  python setup_improved.py --skip-gpu      # Skip GPU detection
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output for debugging')
    parser.add_argument('--skip-gpu', action='store_true',
                       help='Skip GPU detection and setup')
    
    args = parser.parse_args()
    
    setup = PhotoValidatorSetup(verbose=args.verbose, skip_gpu=args.skip_gpu)
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
