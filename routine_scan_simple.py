#!/usr/bin/env python3
"""
PhotoValidator - Routine Scan
Complete image validation pipeline with organized output
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Set

class ImageProcessor:
    """Image processing system for PhotoValidator routine scan"""
    
    def __init__(self, input_dir: str, output_dir: str, python_path: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.python_path = python_path
        
        # Image tracking files
        self.logs_dir = self.output_dir / "logs"
        self.image_list_file = self.logs_dir / "image_list.txt"
        self.flagged_files_file = self.logs_dir / "flagged_files.txt"
        
        # Image extensions to process
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        
        # Master lists
        self.master_image_list: List[str] = []
        self.flagged_images: Set[str] = set()
        
    def print_header(self):
        """Print the application header"""
        print("\n" + "=" * 66)
        print("║" + " " * 18 + "PHOTOVALIDATOR ROUTINE SCAN" + " " * 17 + "║")
        print("=" * 66)
        print(f"\nInput Directory:  {self.input_dir}")
        print(f"Output Directory: {self.output_dir}")
        print("\nStarting comprehensive image validation...")
        print("\nScan Order:")
        print("  1. Image Specifications Check")
        print("  2. Border & Frame Detection")
        print("  3. Watermark Detection")
        print("  4. Quality & Editing Detection")
        print("  5. Results Organization")
        print("\n" + "=" * 66 + "\n")
        
    def initialize_scan(self):
        """Initialize scan structure and create master image list"""
        print("Initializing scan structure...")
        
        # Clean previous results
        if self.output_dir.exists():
            try:
                shutil.rmtree(self.output_dir)
            except PermissionError:
                # Try to remove individual files first
                for root, dirs, files in os.walk(self.output_dir, topdown=False):
                    for file in files:
                        try:
                            os.remove(os.path.join(root, file))
                        except:
                            pass
                    for dir in dirs:
                        try:
                            os.rmdir(os.path.join(root, dir))
                        except:
                            pass
                # Try final removal
                try:
                    os.rmdir(self.output_dir)
                except:
                    pass
            
        # Create output structure
        directories = [
            self.output_dir / "Valid",
            self.output_dir / "Invalid" / "Specifications", 
            self.output_dir / "Invalid" / "Border",
            self.output_dir / "Invalid" / "Watermark",
            self.output_dir / "Invalid" / "Quality",
            self.output_dir / "ManualReview",
            self.logs_dir,
            self.output_dir / "_temp_specs",
            self.output_dir / "_temp_border", 
            self.output_dir / "_temp_watermark",
            self.output_dir / "_temp_pyiqa"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize image tracking system
        self.create_master_image_list()
        
        print("✅ Scan structure initialized")
        print()
        
    def create_master_image_list(self):
        """Create master list of all images in input directory"""
        print("Creating master list of images...")
        
        # Initialize files
        if self.image_list_file.exists():
            self.image_list_file.unlink()
        if self.flagged_files_file.exists():
            self.flagged_files_file.unlink()
        
        # Scan for images
        self.master_image_list = []
        
        if not self.input_dir.exists():
            print(f"❌ Error: Input directory '{self.input_dir}' does not exist!")
            sys.exit(1)
            
        for file_path in self.input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                self.master_image_list.append(file_path.name)
                
        # Save master list to file
        with open(self.image_list_file, 'w') as f:
            for image in self.master_image_list:
                f.write(f"{image}\n")
                
        # Create empty flagged files list
        self.flagged_files_file.touch()
        
        print(f"✅ Found {len(self.master_image_list)} images to process")
        
    def run_test(self, test_name: str, script_name: str, temp_folder: str, invalid_folder: str, extra_args: List[str] = None):
        """Run a specific test and process results"""
        print(f"Running {test_name}...")
        print("─" * 61)
        
        # Create temp directories
        temp_path = self.output_dir / temp_folder
        for subdir in ["valid", "invalid", "manualreview"]:
            (temp_path / subdir).mkdir(parents=True, exist_ok=True)
            
        # Build command
        cmd = [self.python_path, script_name]
        if extra_args:
            cmd.extend(extra_args)
            
        # Add standard arguments based on script
        if "main_optimized.py" in script_name:
            cmd.extend(["--workers=6", "--tests", "specifications", "--source", str(self.input_dir), "--output", str(temp_path)])
        elif "border_detector.py" in script_name:
            cmd.extend(["--input", str(self.input_dir), "--output", str(temp_path)])
        elif "advanced_watermark_detector.py" in script_name:
            cmd.extend(["--input", str(self.input_dir), "--output", str(temp_path)])
        elif "advanced_pyiqa_detector.py" in script_name:
            cmd.extend(["--fast", "--workers=6", "--source", str(self.input_dir), "--output", str(temp_path)])
        
        # Run the test with progress indication for quality assessment
        try:
            if "Quality" in test_name and "advanced_pyiqa_detector.py" in script_name:
                # Show progress for quality assessment
                print(f"Analyzing {len(self.master_image_list)} images with AI quality metrics...")
                print("This may take a few minutes depending on image count and system performance.")
                print("Progress: Starting quality analysis...")
                
                # Run with real-time output for PyIQA
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         text=True, cwd=self.input_dir.parent, bufsize=1, universal_newlines=True)
                
                output_lines = []
                error_lines = []
                
                # Read output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        output_lines.append(output.strip())
                        # Show progress indicators
                        if "Processing image" in output or "Analyzing" in output:
                            print(".", end="", flush=True)
                        elif "Score:" in output or "Quality:" in output:
                            print("✓", end="", flush=True)
                        elif "Invalid:" in output or "Failed:" in output:
                            print("✗", end="", flush=True)
                        elif "Completed" in output or "Finished" in output:
                            print(" Done!")
                
                # Get any remaining output
                stdout, stderr = process.communicate()
                if stdout:
                    output_lines.extend(stdout.strip().split('\n'))
                if stderr:
                    error_lines.extend(stderr.strip().split('\n'))
                
                result_returncode = process.returncode
                result_stderr = '\n'.join(error_lines) if error_lines else ''
                
                print()  # New line after progress indicators
                
            else:
                # Regular execution for other tests
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.input_dir.parent)
                result_returncode = result.returncode
                result_stderr = result.stderr
            
            if result_returncode != 0:
                print(f"⚠️ {test_name} completed with warnings")
                if result_stderr:
                    print(f"   Note: {result_stderr.strip()}")
            else:
                print(f"✅ {test_name} completed successfully")
        except Exception as e:
            print(f"❌ Error running {test_name}: {e}")
            return
            
        # Process results
        self.process_results(temp_folder, invalid_folder, test_name)
        
    def process_results(self, temp_folder: str, invalid_folder: str, test_name: str):
        """Process test results and flag invalid images"""
        print(f"Processing {test_name} results...")
        
        temp_path = self.output_dir / temp_folder
        
        # Process invalid files first (highest priority)
        invalid_temp_path = temp_path / "invalid"
        if invalid_temp_path.exists() and any(invalid_temp_path.iterdir()):
            print(f"Moving invalid files to Invalid/{invalid_folder}...")
            
            # Create the nested invalid folder path
            invalid_dest_path = self.output_dir / "Invalid" / invalid_folder
            invalid_dest_path.mkdir(parents=True, exist_ok=True)
            
            for file_path in invalid_temp_path.iterdir():
                if file_path.is_file():
                    try:
                        dest_path = invalid_dest_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                        if dest_path.exists():
                            self.flag_image(file_path.name, f"{test_name}_INVALID")
                            print(f"✗ {file_path.name} → Invalid/{invalid_folder}")
                        else:
                            print(f"⚠️ Failed to copy {file_path.name}")
                    except Exception as e:
                        print(f"⚠️ Error processing {file_path.name}: {e}")
            
        # Process manual review files (only if not already flagged as invalid)
        manual_temp_path = temp_path / "manualreview"
        if manual_temp_path.exists() and any(manual_temp_path.iterdir()):
            print("Processing manual review files...")
            
            manual_review_path = self.output_dir / "ManualReview"
            for file_path in manual_temp_path.iterdir():
                if file_path.is_file():
                    # Check if this image is already flagged as invalid from any previous test
                    if not self.is_image_flagged(file_path.name):
                        try:
                            dest_path = manual_review_path / file_path.name
                            shutil.copy2(file_path, dest_path)
                            if dest_path.exists():
                                self.flag_image(file_path.name, f"{test_name}_MANUAL")
                                print(f"? {file_path.name} → ManualReview")
                            else:
                                print(f"⚠️ Failed to copy {file_path.name}")
                        except Exception as e:
                            print(f"⚠️ Error processing {file_path.name}: {e}")
                    else:
                        print(f"- {file_path.name} → Skipped ManualReview (already flagged as invalid)")
                        
        # Copy any logs generated to our logs folder
        temp_logs_path = temp_path / "logs"
        if temp_logs_path.exists():
            for log_file in temp_logs_path.iterdir():
                if log_file.is_file():
                    shutil.copy2(log_file, self.logs_dir / f"{test_name}_{log_file.name}")
                    
        # Clean up temp folder
        if temp_path.exists():
            shutil.rmtree(temp_path)
            
        print(f"✅ {test_name} results processed")
        print()
        
    def flag_image(self, filename: str, reason: str):
        """Flag an image with a reason"""
        # Check if already flagged
        if filename not in self.flagged_images:
            self.flagged_images.add(filename)
            with open(self.flagged_files_file, 'a') as f:
                f.write(f"{filename}\n")
            
    def is_image_flagged(self, filename: str) -> bool:
        """Check if an image is flagged"""
        # Load flagged images if not already loaded
        if not self.flagged_images:
            if self.flagged_files_file.exists():
                with open(self.flagged_files_file, 'r') as f:
                    self.flagged_images = {line.strip() for line in f if line.strip()}
                    
        return filename in self.flagged_images
        
    def organize_final_results(self):
        """Organize final results - move unflagged images to Valid folder"""
        print("Organizing final results...")
        
        # Load flagged images
        flagged_images = set()
        if self.flagged_files_file.exists():
            with open(self.flagged_files_file, 'r') as f:
                flagged_images = {line.strip() for line in f if line.strip()}
                
        # Count total and flagged images
        total_images = len(self.master_image_list)
        flagged_count = len(flagged_images)
        
        # Move unflagged images to Valid folder
        valid_count = 0
        valid_path = self.output_dir / "Valid"
        
        for filename in self.master_image_list:
            if filename not in flagged_images:
                source_path = self.input_dir / filename
                dest_path = valid_path / filename
                
                if source_path.exists():
                    try:
                        shutil.copy2(source_path, dest_path)
                        if dest_path.exists():
                            valid_count += 1
                            print(f"✓ {filename} → Valid (passed all tests)")
                        else:
                            print(f"⚠️ Failed to copy {filename}")
                    except Exception as e:
                        print(f"⚠️ Error copying {filename}: {e}")
                else:
                    print(f"⚠️ Source file not found: {filename}")
            else:
                print(f"- {filename} → Skipped (failed tests)")
                
        # Generate final report
        print("\n✅ Final organization completed:")
        print(f"├─ Valid: {valid_count} images passed all tests")
        
        # Count files in each category
        categories = ["Specifications", "Border", "Watermark", "Quality"]
        
        for category in categories:
            category_path = self.output_dir / "Invalid" / category
            count = len([f for f in category_path.iterdir() if f.is_file()]) if category_path.exists() else 0
            print(f"├─ Invalid/{category}: {count} files")
            
        manual_review_path = self.output_dir / "ManualReview"
        manual_count = len([f for f in manual_review_path.iterdir() if f.is_file()]) if manual_review_path.exists() else 0
        print(f"└─ ManualReview: {manual_count} files")
        
        print()
        
    def run_custom_scan(self, selected_tests: List[int]):
        """Run a custom scan with selected tests only"""
        # Check if input directory exists
        if not self.input_dir.exists():
            print(f"❌ Error: Input directory '{self.input_dir}' does not exist!")
            print("Please create the directory or update the input directory path.")
            return 1
            
        # Define all available tests
        all_tests = [
            (1, "Specifications", "main_optimized.py", "_temp_specs", "Specifications"),
            (2, "Border Detection", "border_detector.py", "_temp_border", "Border"),
            (3, "Watermark Detection", "advanced_watermark_detector.py", "_temp_watermark", "Watermark"),
            (4, "Quality Detection", "advanced_pyiqa_detector.py", "_temp_pyiqa", "Quality")
        ]
        
        # Filter tests based on user selection
        tests_to_run = [test for test in all_tests if test[0] in selected_tests]
        
        if not tests_to_run:
            print("❌ Error: No valid tests selected!")
            return 1
            
        self.print_custom_header(tests_to_run)
        self.initialize_scan()
        
        # Run selected tests sequentially
        for i, (test_num, test_name, script_name, temp_folder, invalid_folder) in enumerate(tests_to_run, 1):
            print(f"[STEP {i}/{len(tests_to_run)}] Running {test_name}...")
            print("─" * 61)
            self.run_test(test_name, script_name, temp_folder, invalid_folder)
        
        print(f"[STEP {len(tests_to_run) + 1}/{len(tests_to_run) + 1}] Organizing Final Results...")
        print("─" * 61)
        self.organize_final_results()
        
        print("=" * 66)
        print("🎉 CUSTOM SCAN COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 66)
        print(f"\n📁 Results Organization:")
        print("  ├── Valid\\                    - Images that passed all selected tests")
        print("  ├── Invalid\\")
        
        # Show only categories for tests that were run
        for test_num, test_name, _, _, invalid_folder in tests_to_run:
            if invalid_folder == "Specifications":
                print("  │   ├── Specifications\\       - Images with format/size issues")
            elif invalid_folder == "Border":
                print("  │   ├── Border\\               - Images with borders or frames")
            elif invalid_folder == "Watermark":
                print("  │   ├── Watermark\\            - Images with watermarks")
            elif invalid_folder == "Quality":
                print("  │   └── Quality\\              - Images with quality issues")
                
        print("  └── ManualReview\\              - Images requiring manual verification")
        print(f"\nOutput saved to: {self.output_dir}")
        
        return 0
        
    def print_custom_header(self, tests_to_run):
        """Print the custom scan header"""
        print("\n" + "=" * 66)
        print("║" + " " * 18 + "PHOTOVALIDATOR CUSTOM SCAN" + " " * 18 + "║")
        print("=" * 66)
        print(f"\nInput Directory:  {self.input_dir}")
        print(f"Output Directory: {self.output_dir}")
        print("\nSelected Tests:")
        
        for i, (test_num, test_name, _, _, _) in enumerate(tests_to_run, 1):
            print(f"  {i}. {test_name}")
            
        print(f"\nRunning {len(tests_to_run)} selected tests...")
        print("\n" + "=" * 66 + "\n")
        
    def run_routine_scan(self):
        """Run the complete routine scan"""
        # Check if input directory exists
        if not self.input_dir.exists():
            print(f"❌ Error: Input directory '{self.input_dir}' does not exist!")
            print("Please create the directory or update the input directory path.")
            return 1
            
        self.print_header()
        self.initialize_scan()
        
        # Run all tests sequentially
        print("[STEP 1/5] Running Image Specifications Check...")
        print("─" * 61)
        self.run_test("Specifications", "main_optimized.py", "_temp_specs", "Specifications")
        
        print("[STEP 2/5] Running Border & Frame Detection...")
        print("─" * 61)
        self.run_test("Border Detection", "border_detector.py", "_temp_border", "Border")
        
        print("[STEP 3/5] Running Watermark Detection...")
        print("─" * 61)
        self.run_test("Watermark Detection", "advanced_watermark_detector.py", "_temp_watermark", "Watermark")
        
        print("[STEP 4/5] Running Quality & Editing Detection...")
        print("─" * 61)
        self.run_test("Quality Detection", "advanced_pyiqa_detector.py", "_temp_pyiqa", "Quality")
        
        print("[STEP 5/5] Organizing Final Results...")
        print("─" * 61)
        self.organize_final_results()
        
        print("=" * 66)
        print("🎉 ROUTINE SCAN COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 66)
        print("\n📁 Results Organization:")
        print("  ├── Valid\\                    - Images that passed all tests")
        print("  ├── Invalid\\")
        print("  │   ├── Specifications\\       - Images with format/size issues")
        print("  │   ├── Border\\               - Images with borders or frames")
        print("  │   ├── Watermark\\            - Images with watermarks")
        print("  │   └── Quality\\              - Images with quality issues")
        print("  └── ManualReview\\              - Images requiring manual verification")
        print(f"\nOutput saved to: {self.output_dir}")
        
        return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PhotoValidator - Routine Scan")
    parser.add_argument("--input", default="photos4testing", help="Input directory (default: photos4testing)")
    parser.add_argument("--output", default="Results", help="Output directory (default: Results)")
    parser.add_argument("--python", default="C:/Users/Public/Python/MyPy/Scripts/python.exe", help="Python executable path")
    parser.add_argument("--custom", help="Custom test selection (e.g., '1 2 4' for tests 1, 2, and 4)")
    
    args = parser.parse_args()
    
    # Create processor
    processor = ImageProcessor(args.input, args.output, args.python)
    
    # Check for custom scan mode
    if args.custom:
        try:
            # Parse the custom test selection
            selected_tests = [int(x.strip()) for x in args.custom.split()]
            
            # Validate test numbers
            valid_tests = [1, 2, 3, 4]
            invalid_tests = [t for t in selected_tests if t not in valid_tests]
            
            if invalid_tests:
                print(f"❌ Error: Invalid test numbers: {invalid_tests}")
                print("Valid test numbers are: 1 (Specifications), 2 (Border), 3 (Watermark), 4 (Quality)")
                return 1
                
            if not selected_tests:
                print("❌ Error: No tests selected!")
                return 1
                
            # Remove duplicates and sort
            selected_tests = sorted(list(set(selected_tests)))
            
            return processor.run_custom_scan(selected_tests)
            
        except ValueError:
            print("❌ Error: Invalid custom test format!")
            print("Please use format like: --custom '1 2 4' (space-separated numbers)")
            return 1
    else:
        # Run regular routine scan
        return processor.run_routine_scan()

if __name__ == "__main__":
    sys.exit(main())
