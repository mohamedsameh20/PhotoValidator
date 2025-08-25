@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
title PhotoValidator - Image Processing Pipeline

REM ===== COLOR THEMING SYSTEM =====
REM Color codes: 0=Black 1=Blue 2=Green 3=Aqua 4=Red 5=Purple 6=Yellow 7=White
REM             8=Gray  9=Light Blue A=Light Green B=Light Aqua C=Light Red 
REM             D=Light Purple E=Light Yellow F=Bright White
set "COLOR_MAIN=0B"
set "COLOR_CONFIG=0E"
set "COLOR_ANALYSIS=0A"
set "COLOR_ERROR=0C"
set "COLOR_SUCCESS=0F"
set "COLOR_INFO=09"
color %COLOR_MAIN%

REM ===== PYTHON CONFIGURATION =====
set "PYTHON_PATH=C:/Users/Public/Python/MyPy/Scripts/python.exe"

REM ===== GPU CONFIGURATION =====
set "GPU_ID=0"
set "GPU_FLAGS="

REM ===== DEFAULT PATHS =====
set "DEFAULT_INPUT=photos4testing"
set "DEFAULT_OUTPUT=Results"
set "INPUT_DIR=%DEFAULT_INPUT%"
set "OUTPUT_DIR=%DEFAULT_OUTPUT%"

REM ===== BOOLEAN NORMALIZATION =====
set "USE_GPU=true"

REM ═══════════════════════════════════════════════════════════════════════════
REM INTERACTIVE MODE
REM ═══════════════════════════════════════════════════════════════════════════

:MAIN_MENU
cls

REM Perform system validation before showing menu
call :VALIDATE_SYSTEM_QUICK

echo.
echo ╔══════════════════════════════════════════════════════════════════════════╗
echo ║                                                                          ║
echo ║  ██████╗ ██╗  ██╗ ██████╗ ████████╗ ██████╗                              ║
echo ║  ██╔══██╗██║  ██║██╔═══██╗╚══██╔══╝██╔═══██╗                             ║
echo ║  ██████╔╝███████║██║   ██║   ██║   ██║   ██║                             ║
echo ║  ██╔═══╝ ██╔══██║██║   ██║   ██║   ██║   ██║                             ║
echo ║  ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝                             ║
echo ║  ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝                              ║
echo ║                                                                          ║
echo ║  ██╗   ██╗ █████╗ ██╗     ██╗██████╗  █████╗ ████████╗ ██████╗ ██████╗   ║
echo ║  ██║   ██║██╔══██╗██║     ██║██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗  ║
echo ║  ██║   ██║███████║██║     ██║██║  ██║███████║   ██║   ██║   ██║██████╔╝  ║
echo ║  ╚██╗ ██╔╝██╔══██║██║     ██║██║  ██║██╔══██║   ██║   ██║   ██║██╔══██╗  ║
echo ║   ╚████╔╝ ██║  ██║███████╗██║██████╔╝██║  ██║   ██║   ╚██████╔╝██║  ██║  ║
echo ║    ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝  ║
echo ╚══════════════════════════════════════════════════════════════════════════╝
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                     CURRENT CONFIGURATION                    │
echo └──────────────────────────────────────────────────────────────┘
echo   Input:  %INPUT_DIR%
echo   Output: %OUTPUT_DIR%
echo   Python: %PYTHON_PATH%
echo   GPU:    %USE_GPU% (ID: %GPU_ID%)
if defined SYSTEM_STATUS (
    if defined SYSTEM_ISSUES (
        echo   Status: %SYSTEM_STATUS% - %SYSTEM_ISSUES%
    ) else (
        echo   Status: %SYSTEM_STATUS% Ready
    )
)
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                     CORE VALIDATION TESTS                    │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Routine Scan (All Tests Sequential)
echo   [2] Custom Scan (Select Specific Tests)
echo   [3] Text Detection Only                 
echo   [4] Border ^& Frame Detection         
echo   [5] Quality ^& Editing Detection      
echo   [6] Watermark Detection               
echo   [7] Image Specifications Check        
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                      ADVANCED OPTIONS                        │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [V] Full System Validation           
echo   [P] Configure Paths ^& Settings      
echo   [G] GPU Configuration                
echo.
echo ═══════════════════════════════════════════════════════════════
echo   Press Ctrl+C to exit anytime
echo ═══════════════════════════════════════════════════════════════
set /p choice="Please select an option: "

if "%choice%"=="1" goto RUN_ROUTINE_SCAN
if "%choice%"=="2" goto RUN_CUSTOM_SCAN
if "%choice%"=="3" goto RUN_TEXT_DETECTION
if "%choice%"=="4" goto RUN_BORDER_DETECTION
if "%choice%"=="5" goto RUN_QUALITY_DETECTION
if "%choice%"=="6" goto RUN_WATERMARK_DETECTION
if "%choice%"=="7" goto RUN_SPECS_CHECK
if /i "%choice%"=="V" goto FULL_SYSTEM_VALIDATION
if /i "%choice%"=="P" goto CONFIGURE_PATHS
if /i "%choice%"=="G" goto GPU_CONFIGURATION
goto INVALID_CHOICE

:FULL_SYSTEM_VALIDATION
color %COLOR_INFO%
cls
call :VALIDATE_SYSTEM_FULL
echo.
echo [B] Back to Main Menu
echo.
set /p validation_choice="Press B to return to main menu or any other key to continue: "
if /i "%validation_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:CONFIGURE_PATHS
color %COLOR_CONFIG%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  CONFIGURATION CENTER                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Current Configuration:
echo   ┌─ Input Folder:  %INPUT_DIR%
echo   └─ Output Folder: %OUTPUT_DIR%
echo.
echo Python Environment:
echo   └─ Executable: %PYTHON_PATH%
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                      CONFIGURATION OPTIONS                   │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Change Input Folder
echo   [2] Change Output Folder  
echo   [3] Reset to Defaults
echo   [4] Browse for Input Folder
echo   [5] Configure Python Path
echo   [B] Back to Main Menu
echo.
echo ═══════════════════════════════════════════════════════════════
set /p path_choice="Select an option: "

if "%path_choice%"=="1" goto CHANGE_INPUT
if "%path_choice%"=="2" goto CHANGE_OUTPUT
if "%path_choice%"=="3" goto RESET_PATHS
if "%path_choice%"=="4" goto BROWSE_INPUT
if "%path_choice%"=="5" goto PYTHON_PATH_CONFIG
if /i "%path_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
goto CONFIGURE_PATHS

:CHANGE_INPUT
echo.
echo Enter new input folder path:
echo (Examples: "photos4testing", "C:\Images", "D:\TestImages")
echo [Press Enter to keep current: %INPUT_DIR%]
echo.
set /p new_input="Input folder: "
if not "%new_input%"=="" (
    REM Remove quotes if present and validate
    set "cleaned_path=%new_input:"=%"
    
    REM Check for invalid characters (allow colon for drive letters)
    echo "!cleaned_path!" | findstr /r "[<>\"|?*]" >nul
    if not errorlevel 1 (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Path contains invalid characters ^(<^> " ^| ? *^)
        echo Please enter a valid directory path.
        echo.
        pause
        color %COLOR_CONFIG%
        goto CHANGE_INPUT
    )
    
    REM Check if directory exists
    if exist "!cleaned_path!" (
        set "INPUT_DIR=!cleaned_path!"
        echo.
        echo ✅ Valid directory: !cleaned_path!
        
        REM Count image files
        for /f %%i in ('dir /b "!cleaned_path!\*.jpg" "!cleaned_path!\*.png" "!cleaned_path!\*.jpeg" "!cleaned_path!\*.bmp" "!cleaned_path!\*.tiff" 2^>nul ^| find /c /v ""') do (
            if %%i GTR 0 (
                echo ✅ Found %%i image files
            ) else (
                echo ⚠️  Warning: No common image files found ^(jpg, png, jpeg, bmp, tiff^)
            )
        )
    ) else (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Directory does not exist: !cleaned_path!
        echo Please enter a valid directory path.
        echo.
        choice /c YN /m "Try again? (Y/N)"
        if errorlevel 2 (
            color %COLOR_CONFIG%
            goto CONFIGURE_PATHS
        )
        color %COLOR_CONFIG%
        goto CHANGE_INPUT
    )
) else (
    echo.
    echo Keeping current input folder: %INPUT_DIR%
)
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto CONFIGURE_PATHS
pause
goto CONFIGURE_PATHS

:CHANGE_OUTPUT
echo.
echo Enter new output folder path:
echo (Examples: "Results", "C:\Output", "D:\ProcessedImages")
echo [Press Enter to keep current: %OUTPUT_DIR%]
echo.
set /p new_output="Output folder: "
if not "%new_output%"=="" (
    REM Remove quotes if present and validate
    set "cleaned_path=%new_output:"=%"
    
    REM Check for invalid characters (allow colon for drive letters)
    echo "!cleaned_path!" | findstr /r "[<>\"|?*]" >nul
    if not errorlevel 1 (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Path contains invalid characters ^(<^> " ^| ? *^)
        echo Please enter a valid directory path.
        echo.
        pause
        color %COLOR_CONFIG%
        goto CHANGE_OUTPUT
    )
    
    REM Check if parent directory exists (for creation validation)
    for %%p in ("!cleaned_path!") do set "parent_dir=%%~dpp"
    if exist "!parent_dir!" (
        set "OUTPUT_DIR=!cleaned_path!"
        echo.
        echo ✅ Valid output path: !cleaned_path!
        if not exist "!cleaned_path!" (
            echo ℹ️  Directory will be created when needed
        ) else (
            echo ✅ Directory already exists
        )
    ) else (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Parent directory does not exist: !parent_dir!
        echo Cannot create output directory at this location.
        echo.
        choice /c YN /m "Try again? (Y/N)"
        if errorlevel 2 (
            color %COLOR_CONFIG%
            goto CONFIGURE_PATHS
        )
        color %COLOR_CONFIG%
        goto CHANGE_OUTPUT
    )
) else (
    echo.
    echo Keeping current output folder: %OUTPUT_DIR%
)
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto CONFIGURE_PATHS
pause
goto CONFIGURE_PATHS

:RESET_PATHS
set "INPUT_DIR=%DEFAULT_INPUT%"
set "OUTPUT_DIR=%DEFAULT_OUTPUT%"
echo.
echo Paths reset to defaults:
echo   Input:  %INPUT_DIR%
echo   Output: %OUTPUT_DIR%
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto CONFIGURE_PATHS
pause
goto CONFIGURE_PATHS

:BROWSE_INPUT
echo.
echo Note: This will open Windows Explorer to select a folder.
echo Close the Explorer window after selecting your folder.
echo.
pause
for /f "delims=" %%i in ('powershell -command "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = 'Select Input Folder for Images'; $f.ShowDialog(); $f.SelectedPath"') do set "browsed_path=%%i"
if not "%browsed_path%"=="" (
    set "INPUT_DIR=%browsed_path%"
    echo.
    echo Input folder set to: %INPUT_DIR%
) else (
    echo.
    echo No folder selected, keeping current: %INPUT_DIR%
)
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto CONFIGURE_PATHS
pause
goto CONFIGURE_PATHS

:PYTHON_PATH_CONFIG
color %COLOR_CONFIG%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  PYTHON CONFIGURATION                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Current Python Executable: %PYTHON_PATH%
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                        OPTIONS                               │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Auto-detect Python (System Default)
echo   [2] Specify Custom Python Path
echo   [3] Use Python in MyPy Environment  
echo   [4] Test Current Python Installation
echo   [B] Back to Configuration Menu
echo.
set /p py_choice="Select an option: "

if "%py_choice%"=="1" goto AUTO_DETECT_PYTHON
if "%py_choice%"=="2" goto CUSTOM_PYTHON_PATH
if "%py_choice%"=="3" goto USE_MYPY_PYTHON
if "%py_choice%"=="4" goto TEST_PYTHON
if /i "%py_choice%"=="B" (
    color %COLOR_CONFIG%
    goto CONFIGURE_PATHS
)
goto PYTHON_PATH_CONFIG

:AUTO_DETECT_PYTHON
set "PYTHON_PATH=python"
echo.
echo Python path set to system default: python
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" (
    color %COLOR_CONFIG%
    goto CONFIGURE_PATHS
)
pause
color %COLOR_CONFIG%
goto CONFIGURE_PATHS

:CUSTOM_PYTHON_PATH
echo.
set /p "new_python_path=Enter full path to Python executable: "
if exist "%new_python_path%" (
    set "PYTHON_PATH=%new_python_path%"
    echo.
    echo Python path updated to: %PYTHON_PATH%
) else (
    echo.
    echo Error: Python executable not found at specified path
    echo Please verify the path and try again.
)
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" (
    color %COLOR_CONFIG%
    goto CONFIGURE_PATHS
)
pause
color %COLOR_CONFIG%
goto CONFIGURE_PATHS

:USE_MYPY_PYTHON
set "PYTHON_PATH=C:\Users\Public\AppData\Local\Programs\Python\MyPy\python.exe"
echo.
echo Python path set to MyPy environment: %PYTHON_PATH%
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" (
    color %COLOR_CONFIG%
    goto CONFIGURE_PATHS
)
pause
color %COLOR_CONFIG%
goto CONFIGURE_PATHS

:TEST_PYTHON
echo.
echo Testing Python installation...
echo.
"%PYTHON_PATH%" --version
if errorlevel 1 (
    echo Error: Python not accessible at: %PYTHON_PATH%
    echo Please check your Python installation or path.
) else (
    echo Python is working correctly!
)
echo.
echo [B] Back to Configuration Menu
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" (
    color %COLOR_CONFIG%
    goto CONFIGURE_PATHS
)
pause
color %COLOR_CONFIG%
goto CONFIGURE_PATHS

:RUN_TEXT_DETECTION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 TEXT DETECTION ANALYSIS                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running PaddleOCR text detection analysis...
echo This identifies text overlays, watermarks, and labels in images.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Processing images with AI text recognition...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --workers=6 --tests text --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Text detection analysis complete! 
echo ═══════════════════════════════════════════════════════════════
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Text Analysis
echo.
set /p text_choice="Select option [1-3]: "
if "%text_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%text_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%text_choice%"=="3" goto RUN_TEXT_DETECTION
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_BORDER_DETECTION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 BORDER ^& FRAME DETECTION                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running border and frame detection...
echo Identifying artificial borders, frames, and decorative edges.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Analyzing images for border elements...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" border_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Border detection analysis complete!
echo ═══════════════════════════════════════════════════════════════
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Border Analysis
echo.
set /p border_choice="Select option [1-3]: "
if "%border_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%border_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%border_choice%"=="3" goto RUN_BORDER_DETECTION
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_QUALITY_DETECTION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              IMAGE QUALITY ^& EDITING DETECTION              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running image quality and editing detection...
echo Analyzing image artifacts, compression, and artificial enhancements.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo Model Set: BRISQUE + NIQE + CLIPIQA (Optimized)
echo.
echo Processing images with AI quality metrics...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" advanced_pyiqa_detector.py --fast --workers=6 --source "%INPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Quality analysis complete!
echo ═══════════════════════════════════════════════════════════════
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Quality Analysis
echo.
set /p quality_choice="Select option [1-3]: "
if "%quality_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%quality_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%quality_choice%"=="3" goto RUN_QUALITY_DETECTION
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_WATERMARK_DETECTION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   WATERMARK DETECTION                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running CNN-based watermark detection...
echo Identifying subtle watermarks and copyright marks.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Analyzing images for watermark patterns...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" advanced_watermark_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Watermark analysis complete!
echo ═══════════════════════════════════════════════════════════════
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Watermark Analysis
echo.
set /p watermark_choice="Select option [1-3]: "
if "%watermark_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%watermark_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%watermark_choice%"=="3" goto RUN_WATERMARK_DETECTION
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_SPECS_CHECK
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 IMAGE SPECIFICATIONS CHECK                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running image specifications validation...
echo Checking format, size, and compliance requirements.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Validating image specifications...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --workers=6 --tests specifications --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Specifications analysis complete!
echo ═══════════════════════════════════════════════════════════════
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Specifications Check
echo.
set /p specs_choice="Select option [1-3]: "
if "%specs_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%specs_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%specs_choice%"=="3" goto RUN_SPECS_CHECK
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_ROUTINE_SCAN
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     ROUTINE SCAN MODE                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running complete routine scan using optimized Python routine...
echo This integrates all tests with debug output and proper organization.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo This will analyze all images and organize them by test results.
echo Each test checks for different image quality issues.
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting comprehensive routine scan...
echo.
"%PYTHON_PATH%" routine_scan_simple.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --python "%PYTHON_PATH%"

if errorlevel 1 (
    echo.
    echo ⚠️ Some tests encountered issues but processing completed
    echo Please check the Results folder for details.
) else (
    echo.
    echo ✅ All tests completed successfully!
)

echo ═══════════════════════════════════════════════════════════════
echo 🎉 ROUTINE SCAN COMPLETED! 🎉
echo ═══════════════════════════════════════════════════════════════
echo.
echo Your images have been processed and organized into:
echo.
echo 📁 Results Structure:
echo   ├── Valid\                    - Images that passed all tests
echo   ├── Invalid\
echo   │   ├── Specifications\       - Images with format/size issues
echo   │   ├── Border\               - Images with borders or frames
echo   │   ├── Watermark\            - Images with watermarks detected
echo   │   └── Quality\              - Images with quality issues
echo   ├── ManualReview\             - Images needing your review
echo   └── logs\                     - Processing logs
echo.
echo Check the Results folder to see your organized images!
echo.
echo ═══════════════════════════════════════════════════════════════
echo Press any key to continue...
echo ═══════════════════════════════════════════════════════════════
pause
goto ROUTINE_POSTMENU

:ROUTINE_POSTMENU
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Routine Scan
echo.
set "routine_choice="
set /p routine_choice="Select option [1-3]: "

if /i "%routine_choice%"=="1" goto MAIN_MENU

if /i "%routine_choice%"=="2" (
  if exist "%OUTPUT_DIR%" (
    start "" "%OUTPUT_DIR%"
  ) else (
    echo Results folder not found.
    pause
  )
  goto MAIN_MENU
)

if /i "%routine_choice%"=="3" goto RUN_ROUTINE_SCAN

color %COLOR_MAIN%
goto MAIN_MENU

:RUN_CUSTOM_SCAN
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     CUSTOM SCAN MODE                         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Available Tests:
echo   [1] Specifications Check (Format/Size validation)
echo   [2] Border Detection (Artificial borders and frames)
echo   [3] Watermark Detection (Text and logo watermarks)
echo   [4] Quality Assessment (PyIQA quality analysis)
echo.
echo Enter the numbers of tests you want to run separated by spaces.
echo Example: 1 2 3  (runs specs, border, and watermark tests)
echo Example: 2 4    (runs border and quality tests only)
echo.
set "test_selection="
set /p test_selection="Enter test numbers: "

if "%test_selection%"=="" (
    echo No tests selected. Returning to main menu.
    pause
    goto MAIN_MENU
)

echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo Selected Tests: %test_selection%
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting custom scan with selected tests...
echo.
"%PYTHON_PATH%" routine_scan_simple.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --python "%PYTHON_PATH%" --custom "%test_selection%"

if errorlevel 1 (
    echo.
    echo ⚠️ Some tests encountered issues but processing completed
    echo Please check the Results folder for details.
) else (
    echo.
    echo ✅ Custom scan completed successfully!
)

echo ═══════════════════════════════════════════════════════════════
echo 🎉 CUSTOM SCAN COMPLETED! 🎉
echo ═══════════════════════════════════════════════════════════════
echo.
echo Your selected tests have been completed and results organized.
echo Check the Results folder to see your organized images!
echo.
echo ═══════════════════════════════════════════════════════════════
echo Press any key to continue...
echo ═══════════════════════════════════════════════════════════════
pause
goto CUSTOM_POSTMENU

:CUSTOM_POSTMENU
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Custom Scan
echo.
set "custom_choice="
set /p custom_choice="Select option [1-3]: "

if /i "%custom_choice%"=="1" goto MAIN_MENU

if /i "%custom_choice%"=="2" (
  if exist "%OUTPUT_DIR%" (
    start "" "%OUTPUT_DIR%"
  ) else (
    echo Results folder not found.
    pause
  )
  goto MAIN_MENU
)

if /i "%custom_choice%"=="3" goto RUN_CUSTOM_SCAN

color %COLOR_MAIN%
goto MAIN_MENU

:INVALID_CHOICE
color %COLOR_ERROR%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                       INVALID SELECTION                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Please choose from the available options.
echo.
pause
color %COLOR_MAIN%
goto MAIN_MENU

:GPU_CONFIGURATION
color %COLOR_CONFIG%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                     GPU CONFIGURATION                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Current GPU Settings:
echo   Use GPU:  %USE_GPU%
echo   GPU ID:   %GPU_ID%
echo   Flags:    %GPU_FLAGS%
echo.

REM Check GPU availability
"%PYTHON_PATH%" gpu_diagnostic.py --quiet >gpu_status.tmp 2>&1
if exist gpu_status.tmp (
    findstr "CUDA Available: True" gpu_status.tmp >nul
    if errorlevel 1 (
        echo ❌ No CUDA-capable GPU detected
        echo    CPU mode will be used regardless of settings
    ) else (
        echo ✅ CUDA-capable GPU detected
        for /f "tokens=*" %%i in ('findstr "GPU.*:" gpu_status.tmp') do echo    %%i
    )
    del gpu_status.tmp
) else (
    echo ⚠️  Unable to check GPU status
)

echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                         OPTIONS                              │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Enable GPU Processing            
echo   [2] Disable GPU (Force CPU)          
echo   [3] Select GPU Device ID             
echo   [4] Test GPU Performance             
echo   [5] View GPU Diagnostic Report       
echo   [B] Back to Main Menu
echo.
echo ═══════════════════════════════════════════════════════════════
set /p gpu_choice="Select an option: "

if "%gpu_choice%"=="1" goto ENABLE_GPU
if "%gpu_choice%"=="2" goto DISABLE_GPU
if "%gpu_choice%"=="3" goto SELECT_GPU_ID
if "%gpu_choice%"=="4" goto TEST_GPU_PERFORMANCE
if "%gpu_choice%"=="5" goto VIEW_GPU_DIAGNOSTIC
if /i "%gpu_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
goto GPU_CONFIGURATION

:ENABLE_GPU
set "USE_GPU=true"
set "GPU_FLAGS=--gpu"
echo.
echo ✅ GPU processing enabled
echo    Detectors will attempt to use CUDA acceleration
echo    Note: Fallback to CPU will occur if GPU is unavailable
echo.
echo [B] Back to GPU Configuration
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto GPU_CONFIGURATION
pause
goto GPU_CONFIGURATION

:DISABLE_GPU
set "USE_GPU=false"
set "GPU_FLAGS=--cpu"
echo.
echo 💻 GPU processing disabled
echo    All detectors will use CPU processing
echo    This ensures maximum compatibility but slower performance
echo.
echo [B] Back to GPU Configuration
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto GPU_CONFIGURATION
pause
goto GPU_CONFIGURATION

:SELECT_GPU_ID
echo.
echo Select GPU Device ID:
echo   0 - Primary GPU (default)
echo   1 - Secondary GPU (if available)
echo   2 - Third GPU (if available)
echo.
set /p new_gpu_id="Enter GPU ID [0-3]: "

REM Validate input
echo %new_gpu_id%| findstr /r "^[0-3]$" >nul
if errorlevel 1 (
    echo ❌ Invalid GPU ID. Must be 0, 1, 2, or 3
    pause
    goto SELECT_GPU_ID
)

set "GPU_ID=%new_gpu_id%"
if /i "%USE_GPU%"=="true" (
    set "GPU_FLAGS=--gpu=%GPU_ID%"
) else (
    set "GPU_FLAGS=--cpu"
)

echo.
echo ✅ GPU ID set to: %GPU_ID%
if "%USE_GPU%"=="true" (
    echo    Will use GPU %GPU_ID% for processing
) else (
    echo    Note: GPU is currently disabled, enable it to use GPU %GPU_ID%
)
echo.
echo [B] Back to GPU Configuration
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto GPU_CONFIGURATION
pause
goto GPU_CONFIGURATION

:TEST_GPU_PERFORMANCE
echo.
echo Testing GPU performance...
echo This will run a quick benchmark to compare CPU vs GPU processing
echo.
"%PYTHON_PATH%" performance_benchmark.py --gpu-test
echo.
echo [B] Back to GPU Configuration
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto GPU_CONFIGURATION
pause
goto GPU_CONFIGURATION

:VIEW_GPU_DIAGNOSTIC
echo.
echo Running detailed GPU diagnostic...
echo.
"%PYTHON_PATH%" gpu_diagnostic.py
echo.
echo [B] Back to GPU Configuration
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto GPU_CONFIGURATION
pause
goto GPU_CONFIGURATION

REM ═══════════════════════════════════════════════════════════════════════════
REM SYSTEM VALIDATION AND UTILITY FUNCTIONS
REM ═══════════════════════════════════════════════════════════════════════════

:VALIDATE_SYSTEM_QUICK
REM Quick validation for main menu display
set "SYSTEM_STATUS=✅"
set "SYSTEM_ISSUES="

REM Check Python
"%PYTHON_PATH%" --version >nul 2>&1
if errorlevel 1 (
    set "SYSTEM_STATUS=⚠️"
    set "SYSTEM_ISSUES=Python"
)

REM Check input directory
if not exist "%INPUT_DIR%" (
    set "SYSTEM_STATUS=⚠️"
    if defined SYSTEM_ISSUES (
        set "SYSTEM_ISSUES=%SYSTEM_ISSUES%, Input Dir"
    ) else (
        set "SYSTEM_ISSUES=Input Dir"
    )
)
goto :EOF

:VALIDATE_SYSTEM_FULL
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    SYSTEM VALIDATION                         ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Checking system requirements...
echo.

set "validation_failed=0"

REM Check Python
echo [1/4] Checking Python installation...
"%PYTHON_PATH%" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not accessible
    echo    Please check Python configuration in settings.
    set "validation_failed=1"
) else (
    for /f "tokens=*" %%i in ('"%PYTHON_PATH%" --version 2^>^&1') do echo ✅ %%i
)

REM Check directories
echo [2/4] Checking input directory...
if not exist "%INPUT_DIR%" (
    echo ❌ Input directory not found: %INPUT_DIR%
    set "validation_failed=1"
) else (
    echo ✅ Input Directory: %INPUT_DIR%
    for /f %%i in ('dir /b "%INPUT_DIR%\*.jpg" "%INPUT_DIR%\*.png" "%INPUT_DIR%\*.jpeg" 2^>nul ^| find /c /v ""') do (
        if %%i GTR 0 (
            echo    └─ Found %%i image files
        ) else (
            echo    └─ ⚠️  No image files found
        )
    )
)

REM Check output directory writability
echo [3/4] Checking output directory access...
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%" 2>nul
    if errorlevel 1 (
        echo ❌ Cannot create output directory: %OUTPUT_DIR%
        set "validation_failed=1"
    ) else (
        echo ✅ Output Directory: %OUTPUT_DIR% (created)
        rmdir "%OUTPUT_DIR%" 2>nul
    )
) else (
    echo ✅ Output Directory: %OUTPUT_DIR%
)

REM Check main script
echo [4/4] Checking analysis scripts...
if exist "main_optimized.py" (
    echo ✅ Analysis scripts found
) else (
    echo ❌ Analysis scripts missing
    set "validation_failed=1"
)

echo.
if "%validation_failed%"=="1" (
    color %COLOR_ERROR%
    echo ❌ VALIDATION FAILED
    echo    Please fix the issues above before running analysis.
    echo.
    echo [C] Continue anyway
    echo [S] Go to Settings  
    echo.
    choice /c CS /m "Select option"
    if errorlevel 2 (
        color %COLOR_CONFIG%
        goto CONFIGURE_PATHS
    )
    color %COLOR_ANALYSIS%
) else (
    echo ✅ SYSTEM VALIDATION COMPLETE
    echo    Ready for image analysis.
    echo.
)
goto :EOF

:CREATE_SESSION_LOG
REM Create session log with timestamp
set "TIMESTAMP=%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%_%TIME:~0,2%-%TIME:~3,2%-%TIME:~6,2%"
set "TIMESTAMP=%TIMESTAMP: =0%"
set "SESSION_LOG=%OUTPUT_DIR%\logs\session_%TIMESTAMP%.log"

REM Create logs directory
if not exist "%OUTPUT_DIR%\logs" mkdir "%OUTPUT_DIR%\logs" 2>nul

REM Initialize session log
echo === PhotoValidator Session Log === > "%SESSION_LOG%" 2>nul
echo Date/Time: %DATE% %TIME% >> "%SESSION_LOG%" 2>nul
echo User: %USERNAME% >> "%SESSION_LOG%" 2>nul
echo Computer: %COMPUTERNAME% >> "%SESSION_LOG%" 2>nul
echo Input Directory: %INPUT_DIR% >> "%SESSION_LOG%" 2>nul
echo Output Directory: %OUTPUT_DIR% >> "%SESSION_LOG%" 2>nul
echo Python Path: %PYTHON_PATH% >> "%SESSION_LOG%" 2>nul
echo ================================== >> "%SESSION_LOG%" 2>nul
goto :EOF

:LOG_EVENT
REM Usage: call :LOG_EVENT "message"
if defined SESSION_LOG (
    echo %TIME% - %~1 >> "%SESSION_LOG%" 2>nul
)
goto :EOF

:EXECUTE_WITH_ERROR_HANDLING
REM Usage: call :EXECUTE_WITH_ERROR_HANDLING "command" "operation_name"
set "retry_count=0"

:RETRY_OPERATION
set /a retry_count+=1
if %retry_count% GEQ 4 (
    color %COLOR_ERROR%
    echo ❌ Maximum retry attempts reached. Operation failed.
    echo Please check your configuration and try again.
    call :LOG_EVENT "Operation failed after 3 retries: %~2"
    pause
    color %COLOR_MAIN%
    goto MAIN_MENU
)

if %retry_count% GTR 1 (
    echo.
    echo Attempt %retry_count% of 3...
    call :LOG_EVENT "Retry attempt %retry_count% for: %~2"
)

call :LOG_EVENT "Starting operation: %~2"
%~1
if errorlevel 1 (
    color %COLOR_ERROR%
    echo.
    echo ❌ Operation failed: %~2
    echo Error code: %errorlevel%
    call :LOG_EVENT "Operation failed with error code %errorlevel%: %~2"
    
    if %retry_count% LSS 3 (
        echo.
        choice /c YN /m "Retry operation? (Y/N)"
        if errorlevel 2 (
            color %COLOR_MAIN%
            goto MAIN_MENU
        )
        color %COLOR_ANALYSIS%
        goto RETRY_OPERATION
    ) else (
        echo.
        echo Maximum retry attempts reached.
        pause
        color %COLOR_MAIN%
        goto MAIN_MENU
    )
) else (
    echo.
    echo ✅ Operation completed successfully: %~2
    call :LOG_EVENT "Operation completed successfully: %~2"
)
goto :EOF

REM ═══════════════════════════════════════════════════════════════════════════
REM ROUTINE SCAN FUNCTIONS FOR ORGANIZED RESULTS
REM ═══════════════════════════════════════════════════════════════════════════

:INITIALIZE_ROUTINE_SCAN
echo Initializing folder structure...

REM Clean folders from previous runs
for %%D in ("Valid" "Invalid" "ManualReview" "valid" "invalid" "manualreview") do (
  if exist "%OUTPUT_DIR%\%%~D" rmdir /s /q "%OUTPUT_DIR%\%%~D" 2>nul
)

REM Clean staging folders from previous runs
for %%D in ("Invalid\_spec_failures" "Invalid\_border_failures" "Invalid\_watermark_failures" "Invalid\_quality_failures") do (
  if exist "%OUTPUT_DIR%\%%~D" rmdir /s /q "%OUTPUT_DIR%\%%~D" 2>nul
)

REM Create timestamp for this scan session
set "SCAN_TIMESTAMP=%DATE:~-4,4%-%DATE:~-10,2%-%DATE:~-7,2%_%TIME:~0,2%-%TIME:~3,2%-%TIME:~6,2%"
set "SCAN_TIMESTAMP=%SCAN_TIMESTAMP: =0%"

REM Create organized folder structure with subcategories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%\Valid"
mkdir "%OUTPUT_DIR%\Invalid"
mkdir "%OUTPUT_DIR%\ManualReview"
mkdir "%OUTPUT_DIR%\Reports"
mkdir "%OUTPUT_DIR%\logs"

REM Create staging folders for each test type INSIDE the Invalid directory
mkdir "%OUTPUT_DIR%\Invalid\_spec_failures"
mkdir "%OUTPUT_DIR%\Invalid\_border_failures"
mkdir "%OUTPUT_DIR%\Invalid\_watermark_failures"
mkdir "%OUTPUT_DIR%\Invalid\_quality_failures"

REM Initialize scan log
set "ROUTINE_LOG=%OUTPUT_DIR%\Reports\Routine_Scan_Log_%SCAN_TIMESTAMP%.txt"
echo =============================================================================== > "%ROUTINE_LOG%"
echo                    PHOTOVALIDATOR ROUTINE SCAN LOG                           >> "%ROUTINE_LOG%"
echo =============================================================================== >> "%ROUTINE_LOG%"
echo Scan Started: %DATE% %TIME% >> "%ROUTINE_LOG%"
echo User: %USERNAME% >> "%ROUTINE_LOG%"
echo Computer: %COMPUTERNAME% >> "%ROUTINE_LOG%"
echo Input Directory: %INPUT_DIR% >> "%ROUTINE_LOG%"
echo Output Directory: %OUTPUT_DIR% >> "%ROUTINE_LOG%"
echo Python Path: %PYTHON_PATH% >> "%ROUTINE_LOG%"
echo GPU Enabled: %USE_GPU% >> "%ROUTINE_LOG%"
echo =============================================================================== >> "%ROUTINE_LOG%"
echo. >> "%ROUTINE_LOG%"

REM Initialize excluded files tracking
set "EXCLUDED_FILES_LIST=%OUTPUT_DIR%\logs\excluded_files.txt"
if exist "%EXCLUDED_FILES_LIST%" del "%EXCLUDED_FILES_LIST%" 2>nul

echo ✅ Folder structure initialized
echo ✅ Scan log created: %ROUTINE_LOG%
echo ✅ Exclusion tracking initialized
echo.
goto :EOF

:PROCESS_TEST_RESULTS
REM Usage: call :PROCESS_TEST_RESULTS "temp_folder_name" "failure_folder_name" "test_name"
setlocal EnableDelayedExpansion
set "temp_folder=%~1"
set "failure_folder=%~2"
set "test_name=%~3"

echo Processing %test_name% test results...

REM Ensure target directories exist
if not exist "%OUTPUT_DIR%\Invalid\%failure_folder%" mkdir "%OUTPUT_DIR%\Invalid\%failure_folder%" 2>nul
if not exist "%OUTPUT_DIR%\ManualReview" mkdir "%OUTPUT_DIR%\ManualReview" 2>nul

REM Process invalid results
if exist "%OUTPUT_DIR%\%temp_folder%\invalid\*.*" (
    echo Moving invalid results from %test_name% test...
    for %%F in ("%OUTPUT_DIR%\%temp_folder%\invalid\*.*") do (
        copy "%%F" "%OUTPUT_DIR%\Invalid\%failure_folder%\" >nul 2>&1
        echo %%~nxF>>"%EXCLUDED_FILES_LIST%"
        echo ✗ %%~nxF → Invalid\%failure_folder%
    )
)

REM Process manual review results
if exist "%OUTPUT_DIR%\%temp_folder%\manualreview\*.*" (
    echo Moving manual review results from %test_name% test...
    for %%F in ("%OUTPUT_DIR%\%temp_folder%\manualreview\*.*") do (
        copy "%%F" "%OUTPUT_DIR%\ManualReview\" >nul 2>&1
        echo %%~nxF>>"%EXCLUDED_FILES_LIST%"
        echo ? %%~nxF → ManualReview
    )
)

REM Clean up temporary folder
rmdir /s /q "%OUTPUT_DIR%\%temp_folder%" 2>nul

echo ✅ %test_name% results processed
endlocal
goto :EOF

:PROCESS_PYIQA_RESULTS
setlocal EnableDelayedExpansion
echo Processing PyIQA quality test results...

REM Ensure target directories exist
if not exist "%OUTPUT_DIR%\Invalid\_quality_failures" mkdir "%OUTPUT_DIR%\Invalid\_quality_failures" 2>nul
if not exist "%OUTPUT_DIR%\ManualReview" mkdir "%OUTPUT_DIR%\ManualReview" 2>nul

REM Process invalid results from PyIQA default Results folder
if exist "Results\invalid\*.*" (
    echo Moving invalid results from quality test...
    for %%F in ("Results\invalid\*.*") do (
        copy "%%F" "%OUTPUT_DIR%\Invalid\_quality_failures\" >nul 2>&1
        echo %%~nxF>>"%EXCLUDED_FILES_LIST%"
        echo ✗ %%~nxF → Invalid\_quality_failures
    )
)

REM Process manual review results from PyIQA
if exist "Results\manualreview\*.*" (
    echo Moving manual review results from quality test...
    for %%F in ("Results\manualreview\*.*") do (
        copy "%%F" "%OUTPUT_DIR%\ManualReview\" >nul 2>&1
        echo %%~nxF>>"%EXCLUDED_FILES_LIST%"
        echo ? %%~nxF → ManualReview
    )
)

REM Try to extract invalid filenames from PyIQA logs if images weren't copied
if exist "Results\logs\pyiqa_scores.txt" (
    echo Checking PyIQA log for additional invalid files...
    for /f "tokens=1 delims=," %%F in ('findstr /i "score_above" "Results\logs\pyiqa_scores.txt" 2^>nul') do (
        if exist "%INPUT_DIR%\%%F" (
            if not exist "%OUTPUT_DIR%\Invalid\_quality_failures\%%F" (
                echo Copying %%F to quality_failures from log analysis
                copy "%INPUT_DIR%\%%F" "%OUTPUT_DIR%\Invalid\_quality_failures\" >nul 2>&1
                echo %%F>>"%EXCLUDED_FILES_LIST%"
                echo ✗ %%F → Invalid\_quality_failures ^(from logs^)
            )
        )
    )
)

REM Copy logs if present
if exist "Results\logs" (
    echo Copying PyIQA logs...
    if not exist "%OUTPUT_DIR%\logs" mkdir "%OUTPUT_DIR%\logs" 2>nul
    xcopy "Results\logs\*" "%OUTPUT_DIR%\logs\" /e /i /y >nul 2>&1
)

REM Clean PyIQA default output
if exist "Results" rmdir /s /q "Results" 2>nul

echo ✅ PyIQA results processed
endlocal
goto :EOF

:LOG_ROUTINE_EVENT
REM Log events to scan log
if defined ROUTINE_LOG (
    echo %TIME% - %~1 >> "%ROUTINE_LOG%"
)
goto :EOF

:ORGANIZE_ROUTINE_RESULTS
setlocal EnableDelayedExpansion
echo Organizing results into final categorized structure...

REM Ensure required folders exist
if not exist "%OUTPUT_DIR%\Valid" mkdir "%OUTPUT_DIR%\Valid" 2>nul
if not exist "%OUTPUT_DIR%\Invalid" mkdir "%OUTPUT_DIR%\Invalid" 2>nul
if not exist "%OUTPUT_DIR%\ManualReview" mkdir "%OUTPUT_DIR%\ManualReview" 2>nul

REM Load excluded files list into memory for faster checking
set "excluded_count=0"
if exist "%EXCLUDED_FILES_LIST%" (
    for /f "usebackq delims=" %%F in ("%EXCLUDED_FILES_LIST%") do (
        set /a excluded_count+=1
        set "excluded[!excluded_count!]=%%F"
    )
)

echo Found !excluded_count! files excluded from previous tests

REM Build Valid folder by copying any input image that is NOT in the excluded list
echo Determining images that passed all tests...
set "valid_count=0"
for %%G in ("%INPUT_DIR%\*.jpg" "%INPUT_DIR%\*.jpeg" "%INPUT_DIR%\*.png" "%INPUT_DIR%\*.bmp" "%INPUT_DIR%\*.tif" "%INPUT_DIR%\*.tiff") do (
  if exist "%%~fG" (
    set "fn=%%~nxG"
    set "is_excluded=false"
    
    REM Check if this file is in the excluded list
    for /l %%i in (1,1,!excluded_count!) do (
        if "!fn!"=="!excluded[%%i]!" (
            set "is_excluded=true"
            goto :next_file
        )
    )
    
    :next_file
    if "!is_excluded!"=="false" (
        copy "%%~fG" "%OUTPUT_DIR%\Valid\" >nul 2>&1
        set /a valid_count+=1
        echo ✓ !fn! → Valid
    ) else (
        echo - !fn! → Excluded ^(failed one or more tests^)
    )
  )
)

echo.
echo ✅ Results organization completed:
echo ├─ Valid: !valid_count! images that passed ALL tests
echo ├─ Invalid: Contains subcategories for different failure types
echo └─ ManualReview: Images requiring human verification

REM Show summary of invalid categories
echo.
echo Invalid Categories Summary:
for %%D in ("_spec_failures" "_border_failures" "_watermark_failures" "_quality_failures") do (
    set "cat_count=0"
    if exist "%OUTPUT_DIR%\Invalid\%%~D" (
        for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Invalid\%%~D\*" 2^>nul ^| find /c /v ""') do set "cat_count=%%c"
    )
    echo ├─ %%~D: !cat_count! files
)

set "manual_count=0"
if exist "%OUTPUT_DIR%\ManualReview" (
    for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\ManualReview\*" 2^>nul ^| find /c /v ""') do set "manual_count=%%c"
)
echo └─ ManualReview: !manual_count! files

endlocal & exit /b 0

:GENERATE_RESULTS_REPORT
setlocal EnableDelayedExpansion
echo Generating comprehensive Results Report...
set "RESULTS_REPORT=%OUTPUT_DIR%\Reports\Results_Report.txt"
if not exist "%OUTPUT_DIR%\Reports" mkdir "%OUTPUT_DIR%\Reports" 2>nul

(
echo ===============================================================================
echo RESULTS REPORT
echo PHOTOVALIDATOR ROUTINE SCAN
echo ===============================================================================
echo.
echo SCAN INFORMATION:
echo Date/Time: %DATE% %TIME%
echo Operator: %USERNAME%
echo Source Directory: %INPUT_DIR%
echo Output Directory: %OUTPUT_DIR%
echo.
) > "%RESULTS_REPORT%"

REM Count total input files safely
set "TOTAL_INPUT=0"
for /f %%c in ('dir /a-d /b "%INPUT_DIR%\*.jpg" "%INPUT_DIR%\*.jpeg" "%INPUT_DIR%\*.png" "%INPUT_DIR%\*.bmp" "%INPUT_DIR%\*.tif" "%INPUT_DIR%\*.tiff" 2^>nul ^| find /c /v ""') do set "TOTAL_INPUT=%%c"

REM Count Valid
set "VALID_COUNT=0"
if exist "%OUTPUT_DIR%\Valid" (
  for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Valid\*" 2^>nul ^| find /c /v ""') do set "VALID_COUNT=%%c"
)

REM Count ManualReview
set "MANUAL_COUNT=0"
if exist "%OUTPUT_DIR%\ManualReview" (
  for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\ManualReview\*" 2^>nul ^| find /c /v ""') do set "MANUAL_COUNT=%%c"
)

REM Count category failures
for %%# in (SPECS BORDER WATERMARK QUALITY) do set "%%#_COUNT=0"
if exist "%OUTPUT_DIR%\Invalid\_spec_failures" for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Invalid\_spec_failures\*" 2^>nul ^| find /c /v ""') do set "SPECS_COUNT=%%c"
if exist "%OUTPUT_DIR%\Invalid\_border_failures" for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Invalid\_border_failures\*" 2^nul ^| find /c /v ""') do set "BORDER_COUNT=%%c"
if exist "%OUTPUT_DIR%\Invalid\_watermark_failures" for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Invalid\_watermark_failures\*" 2^>nul ^| find /c /v ""') do set "WATERMARK_COUNT=%%c"
if exist "%OUTPUT_DIR%\Invalid\_quality_failures" for /f %%c in ('dir /a-d /b "%OUTPUT_DIR%\Invalid\_quality_failures\*" 2^>nul ^| find /c /v ""') do set "QUALITY_COUNT=%%c"

REM Compute unique invalids
set "UNIQUE_INVALID_COUNT=0"
set "temp_list=%OUTPUT_DIR%\temp_invalid_list.txt"
set "temp_sorted=%OUTPUT_DIR%\temp_sorted.txt"
if exist "%temp_list%" del "%temp_list%" 2>nul
if exist "%temp_sorted%" del "%temp_sorted%" 2>nul
for %%F in ("%OUTPUT_DIR%\Invalid\_spec_failures\*.*") do if exist "%%~fF" echo %%~nxF>>"%temp_list%"
for %%F in ("%OUTPUT_DIR%\Invalid\_border_failures\*.*") do if exist "%%~fF" echo %%~nxF>>"%temp_list%"
for %%F in ("%OUTPUT_DIR%\Invalid\_watermark_failures\*.*") do if exist "%%~fF" echo %%~nxF>>"%temp_list%"
for %%F in ("%OUTPUT_DIR%\Invalid\_quality_failures\*.*") do if exist "%%~fF" echo %%~nxF>>"%temp_list%"
if exist "%temp_list%" (
  sort "%temp_list%" > "%temp_sorted%" 2>nul
  for /f %%C in ('findstr /r "." "%temp_sorted%" 2^>nul ^| find /c /v ""') do set "UNIQUE_INVALID_COUNT=%%C"
  del "%temp_list%" 2>nul
  del "%temp_sorted%" 2>nul
)

REM Write summary
>>"%RESULTS_REPORT%" echo SUMMARY:
>>"%RESULTS_REPORT%" echo Total Images Processed: !TOTAL_INPUT!
>>"%RESULTS_REPORT%" echo Images Passed All Tests: !VALID_COUNT!
>>"%RESULTS_REPORT%" echo Images Failed Tests ^(unique count^): !UNIQUE_INVALID_COUNT!
>>"%RESULTS_REPORT%" echo ├─ Specifications Issues: !SPECS_COUNT!
>>"%RESULTS_REPORT%" echo ├─ Border Detection Issues: !BORDER_COUNT!
>>"%RESULTS_REPORT%" echo ├─ Watermark Detection Issues: !WATERMARK_COUNT!
>>"%RESULTS_REPORT%" echo ├─ Quality Issues: !QUALITY_COUNT!
>>"%RESULTS_REPORT%" echo Images Needing Manual Review: !MANUAL_COUNT!
>>"%RESULTS_REPORT%" echo.

set /a SUCCESS_RATE=0
if !TOTAL_INPUT! gtr 0 set /a SUCCESS_RATE=(!VALID_COUNT!*100)/!TOTAL_INPUT!
>>"%RESULTS_REPORT%" echo Overall Success Rate: !SUCCESS_RATE!%%
>>"%RESULTS_REPORT%" echo.

call :LOG_ROUTINE_EVENT "Results Report generated: %RESULTS_REPORT%"
echo ✅ Results Report generated: %RESULTS_REPORT%
echo ✅ Success Rate: !SUCCESS_RATE!%%
echo.

endlocal & exit /b 0
