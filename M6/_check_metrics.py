import joblib
m = joblib.load('model_metrics.pkl')

print("=== OVERALL ===")
print(f"  R2={m['overall']['r2']:.6f}  RMSE={m['overall']['rmse']:.4e}  MAE={m['overall']['mae']:.4e}")

print("\n=== PER VARIABLE ===")
for name, v in m['per_variable'].items():
    print(f"  {name:>3}: R2={v['r2']:.6f}  RMSE={v['rmse']:.4e}  MAE={v['mae']:.4e}")

print("\n=== PER SLOT ===")
for s in m['per_slot']:
    print(f"  Slot {s['slot']:>2} ({s['type']:>7}): R2={s['r2']:.6f}  RMSE={s['rmse']:.4e}  MAE={s['mae']:.4e}")
    pv = s.get('per_variable', {})
    for vn, vm in pv.items():
        print(f"         {vn:>3}: R2={vm['r2']:.6f}  RMSE={vm['rmse']:.4e}")
