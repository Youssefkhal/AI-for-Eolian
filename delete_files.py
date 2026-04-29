import os

base_dir = r'c:\Users\youss\Downloads\PFE\M4.1'
files_to_delete = [
    'pile_stiffness_model.py',
    'app.py',
    'train_model.py',
    'run_app.bat',
    'README.md',
    'analyze_excel.py',
    'feature_names.pkl',
    'pile_model.pth',
    'scaler_X.pkl',
    'scaler_y.pkl'
]

for filename in files_to_delete:
    filepath = os.path.join(base_dir, filename)
    try:
        os.remove(filepath)
        print(f'✓ SUCCESS: {filename} deleted')
    except FileNotFoundError:
        print(f'✗ FAILED: {filename} not found')
    except Exception as e:
        print(f'✗ FAILED: {filename} - {str(e)}')
