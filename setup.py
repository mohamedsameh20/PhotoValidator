#!/usr/bin/env python3
"""
PhotoValidator Setup Script
Automated installation and setup for PhotoValidator image processing pipeline.
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

class PhotoValidatorSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = sys.version_info
        self.project_dir = Path(__file__).parent
        self.requirements_installed = False
        
    def print_header(self):
        """Print setup header"""
        print("=" * 60)
        print("🚀 PhotoValidator Setup Script")
        print("=" * 60)
        print(f"System: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print(f"Project Directory: {self.project_dir}")
        print("=" * 60)
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("✅ Checking Python version...")
        if self.python_version < (3, 8):
            print("❌ Error: Python 3.8+ is required")
            print("Please install Python 3.8 or higher from https://python.org")
            sys.exit(1)
        print(f"✅ Python {self.python_version.major}.{self.python_version.minor} is compatible")
        
    def check_pip(self):
        """Check if pip is available"""
        print("✅ Checking pip installation...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            print("✅ pip is available")
        except subprocess.CalledProcessError:
            print("❌ Error: pip is not available")
            print("Please install pip or use the --user flag")
            sys.exit(1)
            
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating project directories...")
        
        directories = [
            "photos4testing",
            "Results",
            "Results/valid",
            "Results/invalid", 
            "Results/manualreview",
            "models",
            "models_cache"
        ]
        
        for directory in directories:
            dir_path = self.project_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
            
    def download_sample_images(self):
        """Download sample images for testing"""
        print("🖼️  Setting up sample images...")
        
        # Create sample images info file
        sample_info = {
            "info": "Add your images to this folder for processing",
            "supported_formats": ["jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"],
            "recommended_size": "Images will be automatically resized if needed",
            "batch_processing": "All images in this folder will be processed together"
        }
        
        info_file = self.project_dir / "photos4testing" / "README.txt"
        with open(info_file, 'w') as f:
            f.write("PhotoValidator - Image Input Folder\n")
            f.write("=" * 40 + "\n\n")
            f.write("This folder is where you place images for processing.\n\n")
            f.write("Supported formats: JPG, PNG, TIFF, BMP, WebP\n")
            f.write("The system will automatically process all images in this folder.\n\n")
            f.write("Results will be saved to the Results/ folder:\n")
            f.write("- Results/valid/ - Images that passed all checks\n")
            f.write("- Results/invalid/ - Images with detected issues\n")
            f.write("- Results/manualreview/ - Images requiring manual review\n")
            
        print("   ✅ Sample folder setup complete")
        
    def install_requirements(self):
        """Install Python requirements"""
        print("📦 Installing Python dependencies...")
        
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("❌ Error: requirements.txt not found")
            return False
            
        try:
            # Upgrade pip first
            print("   Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            print("   Installing dependencies (this may take several minutes)...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print("❌ Error installing requirements:")
                print(result.stderr)
                return False
                
            print("   ✅ Dependencies installed successfully")
            self.requirements_installed = True
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
            
    def setup_gpu_support(self):
        """Check and setup GPU support"""
        print("🔧 Checking GPU support...")
        
        try:
            # Check if CUDA is available
            result = subprocess.run([
                sys.executable, "-c", 
                "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("   " + result.stdout.strip().replace('\n', '\n   '))
                if "CUDA: True" in result.stdout:
                    print("   ✅ GPU acceleration available")
                else:
                    print("   ℹ️  GPU not available, will use CPU")
            else:
                print("   ℹ️  Could not detect GPU, will use CPU")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            print("   ℹ️  GPU check timeout, will use CPU")
            
    def create_run_script(self):
        """Create convenient run scripts"""
        print("📝 Creating run scripts...")
        
        # Windows batch file
        if self.system == "windows":
            batch_content = f'''@echo off
echo Starting PhotoValidator...
cd /d "{self.project_dir}"
python main_optimized.py %*
pause
'''
            with open(self.project_dir / "run_photovalidator.bat", 'w') as f:
                f.write(batch_content)
            print("   Created: run_photovalidator.bat")
            
        # Shell script for Linux/macOS
        shell_content = f'''#!/bin/bash
echo "Starting PhotoValidator..."
cd "{self.project_dir}"
python3 main_optimized.py "$@"
'''
        with open(self.project_dir / "run_photovalidator.sh", 'w') as f:
            f.write(shell_content)
            
        # Make shell script executable on Unix systems
        if self.system in ["linux", "darwin"]:
            os.chmod(self.project_dir / "run_photovalidator.sh", 0o755)
            print("   Created: run_photovalidator.sh")
            
    def create_config_file(self):
        """Create default configuration file"""
        print("⚙️  Creating default configuration...")
        
        config = {
            "version": "1.0.0",
            "setup_date": "2025-07-28",
            "system_info": {
                "platform": platform.system(),
                "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                "architecture": platform.architecture()[0]
            },
            "paths": {
                "input_folder": "photos4testing",
                "output_folder": "Results",
                "models_cache": "models_cache"
            },
            "processing": {
                "default_batch_size": 16,
                "auto_detect_gpu": True,
                "save_debug_logs": False
            }
        }
        
        config_file = self.project_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        print("   ✅ Configuration file created")
        
    def test_installation(self):
        """Test if installation works"""
        print("🧪 Testing installation...")
        
        try:
            # Test imports
            test_script = '''
import sys
try:
    import torch
    import cv2
    import numpy as np
    print("✅ Core dependencies imported successfully")
    
    # Test if main modules can be imported
    sys.path.append('.')
    
    print("✅ Installation test passed")
    print("🎉 PhotoValidator is ready to use!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run([sys.executable, "-c", test_script], 
                                  capture_output=True, text=True, cwd=self.project_dir)
            
            if result.returncode == 0:
                print("   " + result.stdout.strip().replace('\n', '\n   '))
                return True
            else:
                print("   ❌ Test failed:")
                print("   " + result.stderr.strip().replace('\n', '\n   '))
                return False
                
        except Exception as e:
            print(f"   ❌ Test error: {e}")
            return False
            
    def print_completion_message(self):
        """Print setup completion message"""
        print("\n" + "=" * 60)
        print("🎉 PhotoValidator Setup Complete!")
        print("=" * 60)
        print()
        print("📁 Project Structure:")
        print("   photos4testing/     - Add your images here")
        print("   Results/           - Processing results will appear here")
        print("   models/            - Downloaded models cache")
        print()
        print("🚀 Getting Started:")
        print("   1. Add images to the photos4testing/ folder")
        print("   2. Run: python main_optimized.py")
        print("   3. Check Results/ folder for processed images")
        print()
        print("💡 Quick Commands:")
        print("   Basic processing:           python main_optimized.py")
        print("   Skip text detection:        python main_optimized.py --no-text-detection")
        print("   Skip quality analysis:      python main_optimized.py --no-quality-analysis")
        print("   CPU only mode:              python main_optimized.py --cpu-only")
        print("   Custom input folder:        python main_optimized.py --input-folder /path/to/images")
        print()
        print("📖 Documentation:")
        print("   README.md          - Full documentation")
        print("   QUICKSTART.md      - Quick start guide")
        print("   TROUBLESHOOTING.md - Common issues and solutions")
        print()
        if self.system == "windows":
            print("🎯 Quick Start: Double-click 'run_photovalidator.bat'")
        else:
            print("🎯 Quick Start: ./run_photovalidator.sh")
        print("=" * 60)
        
    def run_setup(self):
        """Run the complete setup process"""
        try:
            self.print_header()
            self.check_python_version()
            self.check_pip()
            self.create_directories()
            self.download_sample_images()
            
            # Install requirements
            if not self.install_requirements():
                print("\n❌ Setup failed during dependency installation")
                print("Please check the error messages above and try again")
                sys.exit(1)
                
            self.setup_gpu_support()
            self.create_run_script()
            self.create_config_file()
            
            # Test installation
            if self.test_installation():
                self.print_completion_message()
            else:
                print("\n⚠️  Setup completed but tests failed")
                print("Please check the error messages and try running manually")
                
        except KeyboardInterrupt:
            print("\n❌ Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Setup failed with error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = PhotoValidatorSetup()
    setup.run_setup()
