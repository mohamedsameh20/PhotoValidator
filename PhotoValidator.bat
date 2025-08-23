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
set "USE_GPU=True"
set "GPU_ID=0"
set "GPU_FLAGS="

REM ===== DEFAULT PATHS =====
set "DEFAULT_INPUT=photos4testing"
set "DEFAULT_OUTPUT=Results"
set "INPUT_DIR=%DEFAULT_INPUT%"
set "OUTPUT_DIR=%DEFAULT_OUTPUT%"

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
        echo   Status: %SYSTEM_STATUS% Issues: %SYSTEM_ISSUES%
    ) else (
        echo   Status: %SYSTEM_STATUS% All systems operational
    )
) else (
    echo   Status: Checking...
)
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                     CORE VALIDATION TESTS                    │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Complete Pipeline Analysis        
echo   [2] Text Detection Only                 
echo   [3] Border ^& Frame Detection         
echo   [4] Quality ^& Editing Detection      
echo   [5] Watermark Detection               
echo   [6] Image Specifications Check        
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                      ADVANCED OPTIONS                        │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [7] Custom Test Combination           
echo   [8] System Information              
echo   [9] View Analysis Reports            
echo   [A] Advanced PyIQA Editing Detection  
echo   [V] Full System Validation           
echo   [P] Configure Paths ^& Settings      
echo   [M] Model Cache Management           
echo   [G] GPU Configuration                
echo.
echo ═══════════════════════════════════════════════════════════════
echo   Press Ctrl+C to exit anytime
echo ═══════════════════════════════════════════════════════════════
set /p choice="Please select an option: "

if "%choice%"=="1" goto RUN_COMPLETE
if "%choice%"=="2" goto RUN_TEXT_DETECTION
if "%choice%"=="3" goto RUN_BORDER_DETECTION
if "%choice%"=="4" goto RUN_QUALITY_DETECTION
if "%choice%"=="5" goto RUN_WATERMARK_DETECTION
if "%choice%"=="6" goto RUN_SPECS_CHECK
if "%choice%"=="7" goto CUSTOM_COMBINATION
if "%choice%"=="8" goto SYSTEM_INFO
if "%choice%"=="9" goto VIEW_REPORTS
if /i "%choice%"=="A" goto RUN_ADVANCED_PYIQA
if /i "%choice%"=="V" goto FULL_SYSTEM_VALIDATION
if /i "%choice%"=="P" goto CONFIGURE_PATHS
if /i "%choice%"=="M" goto MODEL_CACHE_MANAGEMENT
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
    
    REM Check for invalid characters
    echo "!cleaned_path!" | findstr /r "[<>:\"|?*]" >nul
    if not errorlevel 1 (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Path contains invalid characters ^(<^> : " ^| ? *^)
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
    
    REM Check for invalid characters
    echo "!cleaned_path!" | findstr /r "[<>:\"|?*]" >nul
    if not errorlevel 1 (
        color %COLOR_ERROR%
        echo.
        echo ❌ Error: Path contains invalid characters ^(<^> : " ^| ? *^)
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

:RUN_COMPLETE
color %COLOR_ANALYSIS%
cls

REM Perform full system validation before running
call :VALIDATE_SYSTEM_FULL

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              COMPLETE IMAGE ANALYSIS - ALL TESTS             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running comprehensive analysis with all available tests:
echo   • Image Specifications Check
echo   • Text Detection (PaddleOCR)
echo   • Border ^& Frame Detection  
echo   • Image Quality ^& Editing Detection
echo   • Watermark Detection
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Create session log
call :CREATE_SESSION_LOG
call :LOG_EVENT "Starting complete analysis"

echo ═══════════════════════════════════════════════════════════════
echo [1/3] Warming up models for optimal performance...
echo ═══════════════════════════════════════════════════════════════
"%PYTHON_PATH%" model_cache_optimizer.py --warm-cache %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo [2/3] Starting comprehensive analysis...
echo ═══════════════════════════════════════════════════════════════
echo Please wait, this may take several minutes...
echo Processing... (this window will update when complete)

REM Execute with error handling and retry logic
call :EXECUTE_WITH_ERROR_HANDLING '"%PYTHON_PATH%" main_optimized.py --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%' "Complete Analysis"

echo.
echo ═══════════════════════════════════════════════════════════════
echo [3/3] Analysis complete! 
echo ═══════════════════════════════════════════════════════════════ 
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Analysis
echo [4] View Session Log
echo.
set /p complete_choice="Select option [1-4]: "
if "%complete_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%complete_choice%"=="2" (
    start "" "%OUTPUT_DIR%" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%complete_choice%"=="3" goto RUN_COMPLETE
if "%complete_choice%"=="4" (
    if exist "%SESSION_LOG%" (
        start notepad "%SESSION_LOG%" 2>nul
    ) else (
        echo No session log found.
        pause
    )
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_TEXT_DETECTION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 TEXT DETECTION ANALYSIS                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running PaddleOCR-based text detection analysis...
echo This will identify images containing text overlays, watermarks, and labels.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Initializing text detection models and processing images...
echo Using PaddleOCR for advanced text recognition
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --tests text --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Text detection analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Text Analysis
echo [B] Back to Main Menu
echo.
set /p text_choice="Select option [1-3, B]: "
if "%text_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%text_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%text_choice%"=="3" goto RUN_TEXT_DETECTION
if /i "%text_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
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
echo Running advanced border and frame detection...
echo Identifying artificial borders, frames, and decorative edges
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Processing images with computer vision algorithms...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" border_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Border detection analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Border Analysis
echo [B] Back to Main Menu
echo.
set /p border_choice="Select option [1-3, B]: "
if "%border_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%border_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%border_choice%"=="3" goto RUN_BORDER_DETECTION
if /i "%border_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
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
echo Running PyIQA-based quality and editing detection...
echo Analyzing artifacts, compression, and artificial enhancements
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo Default Model Set: BRISQUE + NIQE + CLIPIQA (FAST)
echo.
echo Processing images with AI quality metrics...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" advanced_pyiqa_detector.py --fast --workers=6 --source "%INPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Quality analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Quality Analysis
echo [B] Back to Main Menu
echo.
set /p quality_choice="Select option [1-3, B]: "
if "%quality_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%quality_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%quality_choice%"=="3" goto RUN_QUALITY_DETECTION
if /i "%quality_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
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
echo Running advanced CNN-based watermark detection...
echo This will identify subtle watermarks and copyright marks.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Loading models and analyzing images...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" advanced_watermark_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Watermark analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Watermark Analysis
echo [B] Back to Main Menu
echo.
set /p watermark_choice="Select option [1-3, B]: "
if "%watermark_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%watermark_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%watermark_choice%"=="3" goto RUN_WATERMARK_DETECTION
if /i "%watermark_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
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
echo Running image format, size, and specification validation...
echo This will check compliance with image requirements.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Validating image specifications...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --tests specifications --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Specifications analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Specifications Check
echo [B] Back to Main Menu
echo.
set /p specs_choice="Select option [1-3, B]: "
if "%specs_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%specs_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%specs_choice%"=="3" goto RUN_SPECS_CHECK
if /i "%specs_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:RUN_ADVANCED_PYIQA
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║            ADVANCED PYIQA EDITING DETECTION                 ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running advanced PyIQA-based image editing detection...
echo This uses multiple AI quality metrics with customizable model selection.
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Features:
echo   - Multiple PyIQA quality models (BRISQUE, NIQE, MUSIQ, etc.)
echo   - User-selectable model combinations
echo   - Empirical thresholds and robust scoring
echo   - Feature-based analysis integration
echo.
echo Starting analysis with model selection prompt...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Use fast recommended models by default in non-interactive mode
"%PYTHON_PATH%" advanced_pyiqa_detector.py --fast --workers=6 --source "%INPUT_DIR%" %GPU_FLAGS%

echo.
echo ═══════════════════════════════════════════════════════════════
echo Advanced PyIQA analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Advanced Analysis
echo [4] Run PyIQA Model Combinations Test
echo [B] Back to Main Menu
echo.
set /p advanced_choice="Select option [1-4, B]: "
if "%advanced_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%advanced_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%advanced_choice%"=="3" goto RUN_ADVANCED_PYIQA
if "%advanced_choice%"=="4" (
    echo.
    echo Running PyIQA Model Combinations Test...
    "%PYTHON_PATH%" pyiqa_model_combinations_test.py
    pause
    goto RUN_ADVANCED_PYIQA
)
if /i "%advanced_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:CUSTOM_COMBINATION
color %COLOR_ANALYSIS%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    CUSTOM TEST COMBINATION                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                       PROCESSING MODES                       │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Parallel Mode    - Run all selected tests (standard)
echo   [2] Sequential Mode  - Stop on first failure (specs-^>borders-^>watermarks-^>editing-^>text)
echo   [B] Back to Main Menu
echo.
set /p mode_choice="Choose processing mode [1-2, B]: "
if "%mode_choice%"=="1" set "processing_mode=parallel"
if "%mode_choice%"=="2" set "processing_mode=sequential"
if /i "%mode_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if not defined processing_mode goto CUSTOM_COMBINATION

echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                    TEST SELECTION METHOD                     │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Include Tests    - Select which tests TO RUN
echo   [2] Exclude Tests    - Select which tests to SKIP
echo   [B] Back to Main Menu
echo.
set /p selection_method="Choose selection method [1-2, B]: "
if "%selection_method%"=="1" set "method=include"
if "%selection_method%"=="2" set "method=exclude"
if /i "%selection_method%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if not defined method goto CUSTOM_COMBINATION

cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  CUSTOM TEST COMBINATION                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Processing Mode: !processing_mode!
if "!processing_mode!"=="sequential" (
    echo Test Order: specifications -^> borders -^> watermarks -^> editing -^> text
    echo Note: Processing stops on first failure in sequential mode
)
echo Selection Method: !method! tests
echo.
echo Available Tests:
echo   [1] specifications - Image format and size validation
echo   [2] text           - Text detection using PaddleOCR
echo   [3] borders        - Border and frame detection
echo   [4] editing        - Image quality and editing detection
echo   [5] watermarks     - Advanced watermark detection
echo.
if "!method!"=="include" (
    echo Examples:
    echo   - "specifications text" for specs + text detection
    echo   - "borders editing" for borders + quality analysis
    echo   - "text borders watermarks" for multiple tests
    echo.
    set /p custom_tests="Enter test names to RUN (space-separated): "
) else (
    echo Examples:
    echo   - "text" to run all tests EXCEPT text detection
    echo   - "editing watermarks" to run all EXCEPT editing and watermarks
    echo   - "specifications borders" to run text, editing, watermarks only
    echo.
    set /p excluded_tests="Enter test names to SKIP (space-separated): "
)

echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo [B] Back to main menu
echo.

REM Handle back option
if /i "!custom_tests!"=="B" goto MAIN_MENU
if /i "!excluded_tests!"=="B" goto MAIN_MENU

REM Handle exclude method - convert to include list
if "!method!"=="exclude" (
    set "all_tests=specifications text borders editing watermarks"
    set "custom_tests="
    
    REM Build include list by excluding specified tests
    for %%t in (!all_tests!) do (
        set "skip_test="
        for %%e in (!excluded_tests!) do (
            if /i "%%t"=="%%e" set "skip_test=1"
        )
        if not defined skip_test (
            if defined custom_tests (
                set "custom_tests=!custom_tests! %%t"
            ) else (
                set "custom_tests=%%t"
            )
        )
    )
    
    if "!custom_tests!"=="" (
        echo Error: All tests excluded! At least one test must run.
        echo Press any key to try again...
        pause >nul
        goto CUSTOM_COMBINATION
    )
)

REM Validate that we have tests to run
if "!custom_tests!"=="" (
    echo Error: No tests specified!
    echo Press any key to try again...
    pause >nul
    goto CUSTOM_COMBINATION
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                    EXECUTION SUMMARY                         │
echo └──────────────────────────────────────────────────────────────┘
echo Mode: !processing_mode!
echo Tests: !custom_tests!
if "!method!"=="exclude" echo Originally Excluded: !excluded_tests!
echo.
echo ═══════════════════════════════════════════════════════════════

REM Run with appropriate mode
if "!processing_mode!"=="sequential" (
    echo Running tests in sequential mode ^(stop on first failure^)...
    "%PYTHON_PATH%" main_optimized.py --tests !custom_tests! --source "%INPUT_DIR%" --output "%OUTPUT_DIR%" --sequential-mode
) else (
    echo Running tests in parallel mode ^(run all tests^)...
    "%PYTHON_PATH%" main_optimized.py --tests !custom_tests! --source "%INPUT_DIR%" --output "%OUTPUT_DIR%"
)

echo.
echo ═══════════════════════════════════════════════════════════════
echo Custom analysis complete! Check Results folder for output.
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Custom Analysis
echo [B] Back to Main Menu
echo.
set /p custom_complete_choice="Select option [1-3, B]: "
if "%custom_complete_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%custom_complete_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%custom_complete_choice%"=="3" goto CUSTOM_COMBINATION
if /i "%custom_complete_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:SYSTEM_INFO
color %COLOR_INFO%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║             SYSTEM INFORMATION ^& DIAGNOSTICS                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo System Information:
echo ═══════════════════
echo Computer: %COMPUTERNAME%
echo User: %USERNAME%
echo Date: %DATE%
echo Time: %TIME%
echo OS: %OS%
echo.
echo Python Environment:
echo ═════════════════════
"%PYTHON_PATH%" --version 2>nul && (
    echo [OK] Python is installed
    "%PYTHON_PATH%" -c "import cv2; print('[OK] OpenCV version:', cv2.__version__)" 2>nul || echo [ERROR] OpenCV not available
    "%PYTHON_PATH%" -c "import paddleocr; print('[OK] PaddleOCR is available')" 2>nul || echo [ERROR] PaddleOCR not available
    "%PYTHON_PATH%" -c "import torch; print('[OK] PyTorch version:', torch.__version__)" 2>nul || echo [ERROR] PyTorch not available
    "%PYTHON_PATH%" -c "import numpy; print('[OK] NumPy version:', numpy.__version__)" 2>nul || echo [ERROR] NumPy not available
) || (
    echo [ERROR] Python is not installed or not in PATH
)
echo.
echo Project Structure:
echo ════════════════════
if exist "main_optimized.py" (
    echo [OK] Main pipeline: main_optimized.py
) else (
    echo [ERROR] Main pipeline: main_optimized.py not found
)
if exist "border_detector.py" (
    echo [OK] Border detector: border_detector.py
) else (
    echo [ERROR] Border detector: border_detector.py not found
)
if exist "advanced_watermark_detector.py" (
    echo [OK] Watermark detector: advanced_watermark_detector.py
) else (
    echo [ERROR] Watermark detector: advanced_watermark_detector.py not found
)
if exist "photos4testing" (
    echo [OK] Test images directory: photos4testing
) else (
    echo [ERROR] Test images directory: photos4testing not found
)
echo.
echo [B] Back to Main Menu
echo.
set /p sys_choice="Press B to return to main menu or any other key to continue: "
if /i "%sys_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:VIEW_REPORTS
color %COLOR_INFO%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                       ANALYSIS REPORTS                       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Available Reports:
echo ═══════════════════
echo.
if exist "Results" (
    echo Results Directory:
    dir /b "Results" 2>nul
    echo.
)
if exist "FrameDetectionResults" (
    echo Border Detection Results:
    dir /b "FrameDetectionResults" 2>nul
    echo.
)
if exist "CODE_QUALITY_ISSUES.md" (
    echo Code Quality Report: CODE_QUALITY_ISSUES.md
    echo.
)
if exist "*.json" (
    echo JSON Reports:
    dir /b "*.json" 2>nul
    echo.
)
echo [1] Open Results folder
echo [2] View latest processing log
echo [3] Open code quality report
echo [B] Back to main menu
echo.
set /p report_choice="Select option: "
if "%report_choice%"=="1" start "" "Results" 2>nul
if "%report_choice%"=="2" (
    if exist "Results\logs" (
        dir /b /o:d "Results\logs\*.log" 2>nul | findstr /r ".*" >nul && (
            for /f %%i in ('dir /b /o:d "Results\logs\*.log"') do set latest_log=%%i
            start notepad "Results\logs\!latest_log!"
        ) || echo No log files found.
    ) else echo No logs directory found.
)
if "%report_choice%"=="3" start notepad "CODE_QUALITY_ISSUES.md" 2>nul
if /i "%report_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
echo.
echo [B] Back to Main Menu
echo.
set /p back_choice="Press B to return to main menu or any other key to continue: "
if /i "%back_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
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
echo The option you selected is not valid.
echo Please choose from the available options (1-9, P).
echo.
echo [B] Back to Main Menu
echo.
set /p invalid_choice="Press B to return to main menu or any other key to continue: "
if /i "%invalid_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
color %COLOR_MAIN%
goto MAIN_MENU

:MODEL_CACHE_MANAGEMENT
color %COLOR_CONFIG%
cls
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   MODEL CACHE MANAGEMENT                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Model caching improves performance by keeping AI models loaded
echo between runs, reducing startup time from ~30s to ~5s.
echo.
echo ┌──────────────────────────────────────────────────────────────┐
echo │                         OPTIONS                              │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   [1] Warm Cache (Pre-load Models)     
echo   [2] Clear Model Cache                
echo   [3] Check Cache Status               
echo   [4] Optimize for Speed               
echo   [5] Optimize for Memory              
echo   [B] Back to Main Menu
echo.
echo ═══════════════════════════════════════════════════════════════
set /p cache_choice="Select an option: "

if "%cache_choice%"=="1" goto WARM_CACHE
if "%cache_choice%"=="2" goto CLEAR_CACHE
if "%cache_choice%"=="3" goto CHECK_CACHE_STATUS
if "%cache_choice%"=="4" goto OPTIMIZE_SPEED
if "%cache_choice%"=="5" goto OPTIMIZE_MEMORY
if /i "%cache_choice%"=="B" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
goto MODEL_CACHE_MANAGEMENT

:WARM_CACHE
echo.
echo Warming up model cache...
echo This will pre-load AI models for faster subsequent runs.
if "%USE_GPU%"=="true" (
    echo Using GPU acceleration: GPU %GPU_ID%
) else (
    echo Using CPU processing
)
echo.
"%PYTHON_PATH%" model_cache_optimizer.py --warm-cache %GPU_FLAGS%
echo.
echo [B] Back to Cache Management
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto MODEL_CACHE_MANAGEMENT
pause
goto MODEL_CACHE_MANAGEMENT

:CLEAR_CACHE
echo.
echo Clearing model cache...
echo This will free up disk space but slow down the next run.
echo.
"%PYTHON_PATH%" model_cache_optimizer.py --clear-cache
echo.
echo [B] Back to Cache Management
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto MODEL_CACHE_MANAGEMENT
pause
goto MODEL_CACHE_MANAGEMENT

:CHECK_CACHE_STATUS
echo.
echo Checking cache status...
echo.
if exist "%TEMP%\photovalidator_cache\model_cache.pkl" (
    echo ✅ Model cache exists
    for %%f in ("%TEMP%\photovalidator_cache\model_cache.pkl") do (
        echo    Size: %%~zf bytes
        echo    Modified: %%~tf
    )
) else (
    echo ❌ No model cache found
    echo    First run will be slower as models load from scratch
)
echo.
echo [B] Back to Cache Management
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto MODEL_CACHE_MANAGEMENT
pause
goto MODEL_CACHE_MANAGEMENT

:OPTIMIZE_SPEED
echo.
echo Optimizing for maximum speed...
echo - Enabling model caching
echo - Pre-loading essential models
if "%USE_GPU%"=="true" (
    echo - Using GPU acceleration: GPU %GPU_ID%
) else (
    echo - Using CPU for consistent performance
)
echo.
"%PYTHON_PATH%" model_cache_optimizer.py --warm-cache %GPU_FLAGS%
echo.
echo Speed optimization complete!
echo Next analysis runs will be significantly faster.
echo.
echo [B] Back to Cache Management
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto MODEL_CACHE_MANAGEMENT
pause
goto MODEL_CACHE_MANAGEMENT

:OPTIMIZE_MEMORY
echo.
echo Optimizing for memory usage...
echo - Clearing cached models
echo - Models will load on-demand
echo - Slower but uses less memory
echo.
"%PYTHON_PATH%" model_cache_optimizer.py --clear-cache
echo.
echo Memory optimization complete!
echo System will use minimal memory but runs may be slower.
echo.
echo [B] Back to Cache Management
echo.
set /p back_choice="Press B to go back or any other key to continue: "
if /i "%back_choice%"=="B" goto MODEL_CACHE_MANAGEMENT
pause
goto MODEL_CACHE_MANAGEMENT

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
if "%USE_GPU%"=="true" (
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
echo Validating system requirements...
echo.

set "validation_failed=0"

REM Check Python
echo [1/5] Checking Python installation...
"%PYTHON_PATH%" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found at: %PYTHON_PATH%
    echo    Please configure Python path in settings.
    set "validation_failed=1"
) else (
    for /f "tokens=*" %%i in ('"%PYTHON_PATH%" --version 2^>^&1') do echo ✅ %%i
)

REM Check core dependencies
echo [2/5] Checking Python dependencies...
"%PYTHON_PATH%" -c "import sys; print('Python', sys.version.split()[0])" >nul 2>&1
if errorlevel 1 (
    echo ❌ Python interpreter error
    set "validation_failed=1"
) else (
    echo ✅ Python interpreter: OK
)

"%PYTHON_PATH%" -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ NumPy not available
    set "validation_failed=1"
) else (
    echo ✅ NumPy: Available
)

"%PYTHON_PATH%" -c "import PIL; print('Pillow:', PIL.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ Pillow not available
    set "validation_failed=1"
) else (
    echo ✅ Pillow: Available
)

REM Check directories
echo [3/5] Checking directories...
if not exist "%INPUT_DIR%" (
    echo ❌ Input directory does not exist: %INPUT_DIR%
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
echo [4/5] Checking output directory access...
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
    echo ✅ Output Directory: %OUTPUT_DIR% (exists)
    echo test > "%OUTPUT_DIR%\write_test.tmp" 2>nul
    if errorlevel 1 (
        echo ❌ Output directory is not writable
        set "validation_failed=1"
    ) else (
        echo    └─ Write access: OK
        del "%OUTPUT_DIR%\write_test.tmp" 2>nul
    )
)

REM Check main script
echo [5/5] Checking main script...
if exist "main_optimized.py" (
    echo ✅ Main script: main_optimized.py found
) else (
    echo ❌ Main script: main_optimized.py not found
    set "validation_failed=1"
)

echo.
if "%validation_failed%"=="1" (
    color %COLOR_ERROR%
    echo ❌ VALIDATION FAILED
    echo    Please fix the issues above before running analysis.
    echo.
    echo [C] Continue anyway (may fail)
    echo [S] Go to Settings
    echo [B] Back to main menu
    echo.
    choice /c CSB /m "Select option"
    if errorlevel 3 (
        color %COLOR_MAIN%
        goto MAIN_MENU
    )
    if errorlevel 2 (
        color %COLOR_CONFIG%
        goto CONFIGURE_PATHS
    )
    REM Continue anyway if errorlevel 1
    color %COLOR_ANALYSIS%
) else (
    echo ✅ SYSTEM VALIDATION COMPLETE
    echo    All requirements met. Ready for analysis.
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
