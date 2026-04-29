import joblib
import os

for ver in ['M5', 'M6']:
    path = os.path.join(r'c:\Users\youss\Downloads\PFE', ver, 'model_metrics.pkl')
    m = joblib.load(path)
    print(f"\n{'='*60}")
    print(f"  {ver}")
    print(f"{'='*60}")
    print(f"  Overall: R2={m['overall']['r2']:.6f}  RMSE={m['overall']['rmse']:.4e}")
    print(f"  Per-var:")
    for name, v in m['per_variable'].items():
        print(f"    {name:>3}: R2={v['r2']:.6f}  RMSE={v['rmse']:.4e}")
    print(f"  Per-slot R2 trend:")
    for s in m['per_slot']:
        bar = '#' * int((s['r2'] - 0.95) * 1000)
        print(f"    Slot {s['slot']:>2}: R2={s['r2']:.6f}  RMSE={s['rmse']:.4e}  {bar}")
