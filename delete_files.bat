@echo off
setlocal enabledelayedexpansion

set "BASE_DIR=c:\Users\youss\Downloads\PFE\M4.1"

set "FILES=pile_stiffness_model.py app.py train_model.py run_app.bat README.md analyze_excel.py feature_names.pkl pile_model.pth scaler_X.pkl scaler_y.pkl"

for %%F in (%FILES%) do (
    if exist "!BASE_DIR!\%%F" (
        del "!BASE_DIR!\%%F"
        echo ✓ SUCCESS: %%F deleted
    ) else (
        echo ✗ FAILED: %%F not found
    )
)
