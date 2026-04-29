import joblib
import os

base = r'c:\Users\youss\Downloads\PFE'
for ver in ['M5', 'M6']:
    path = os.path.join(base, ver, 'model_metrics.pkl')
    m = joblib.load(path)
    print(f'[{ver}]')
    ov = m['overall']
    print(f"overall R2={ov['r2']:.6f} RMSE={ov['rmse']:.4e} MAE={ov['mae']:.4e}")
    for name in ['KL', 'KR', 'KLR']:
        v = m['per_variable'][name]
        print(f"{name} R2={v['r2']:.6f} RMSE={v['rmse']:.4e} MAE={v['mae']:.4e}")
    first = m['per_slot'][0]
    last = m['per_slot'][-1]
    print(f"slot1 R2={first['r2']:.6f} RMSE={first['rmse']:.4e} MAE={first['mae']:.4e}")
    print(f"slot21 R2={last['r2']:.6f} RMSE={last['rmse']:.4e} MAE={last['mae']:.4e}")
    print()
