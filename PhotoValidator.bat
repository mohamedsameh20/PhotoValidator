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

REM ===== DEFAULT PATHS =====
set "DEFAULT_INPUT=photos4testing"
set "DEFAULT_OUTPUT=Results"
set "INPUT_DIR=%DEFAULT_INPUT%"
set "OUTPUT_DIR=%DEFAULT_OUTPUT%"

:MAIN_MENU
cls
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
echo   [P] Configure Paths ^& Settings      
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
if /i "%choice%"=="P" goto CONFIGURE_PATHS
goto INVALID_CHOICE

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
    set "INPUT_DIR=%new_input%"
    echo.
    echo Input folder updated to: %INPUT_DIR%
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
    set "OUTPUT_DIR=%new_output%"
    echo.
    echo Output folder updated to: %OUTPUT_DIR%
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
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              COMPLETE IMAGE ANALYSIS - ALL TESTS             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo Running comprehensive analysis with all available tests:
echo   • Image Specifications Check
echo   • Text Detection (PaddleOCR)
echo   • Border & Frame Detection  
echo   • Image Quality & Editing Detection
echo   • Watermark Detection
echo.
echo Input Folder:  %INPUT_DIR%
echo Output Folder: %OUTPUT_DIR%
echo.
echo Starting comprehensive analysis...
echo Please wait, this may take several minutes...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --source "%INPUT_DIR%" --output "%OUTPUT_DIR%"

echo.
echo ═══════════════════════════════════════════════════════════════
echo Analysis complete! 
echo.
echo [1] Return to Main Menu
echo [2] View Results Folder
echo [3] Run Another Analysis
echo.
set /p complete_choice="Select option [1-3]: "
if "%complete_choice%"=="1" (
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%complete_choice%"=="2" (
    start "" "Results" 2>nul
    color %COLOR_MAIN%
    goto MAIN_MENU
)
if "%complete_choice%"=="3" goto RUN_COMPLETE
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

"%PYTHON_PATH%" main_optimized.py --tests text --source "%INPUT_DIR%" --output "%OUTPUT_DIR%"

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

"%PYTHON_PATH%" border_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%"

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
echo.
echo Processing images with AI quality metrics...
echo.
echo ═══════════════════════════════════════════════════════════════

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON_PATH%" main_optimized.py --tests editing --source "%INPUT_DIR%" --output "%OUTPUT_DIR%"

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

"%PYTHON_PATH%" advanced_watermark_detector.py --input "%INPUT_DIR%" --output "%OUTPUT_DIR%"

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

"%PYTHON_PATH%" main_optimized.py --tests specifications --source "%INPUT_DIR%" --output "%OUTPUT_DIR%"

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
