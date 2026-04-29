import torch
import time
import importlib.util
import os

def load_model_class(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SlotAttentionDegradation

M5Cls = load_model_class(r'c:\Users\youss\Downloads\PFE\M5\train.py', 'm5_train')
M6Cls = load_model_class(r'c:\Users\youss\Downloads\PFE\M6\train.py', 'm6_train')

m5 = M5Cls(input_size=8, d_model=64, num_heads=4, num_slots=21, max_seq_len=21, dropout=0.1, num_iterations=2)
m6 = M6Cls(input_size=8, d_model=64, num_heads=4, num_slots=21, max_seq_len=21, dropout=0.1, num_iterations=3)

p5 = sum(p.numel() for p in m5.parameters())
p6 = sum(p.numel() for p in m6.parameters())

print("=== PARAMETER COUNT ===")
print(f"M5: {p5:,}")
print(f"M6: {p6:,}")
print(f"Reduction: {p5 - p6:,} params ({(1 - p6/p5)*100:.1f}%)")

# Module-level breakdown
print("\n=== M5 MODULE BREAKDOWN ===")
for name, p in m5.named_parameters():
    if p.requires_grad:
        print(f"  {name}: {p.numel():,}")

print(f"\n=== M6 MODULE BREAKDOWN ===")
for name, p in m6.named_parameters():
    if p.requires_grad:
        print(f"  {name}: {p.numel():,}")

# Inference speed
x = torch.randn(32, 8)
m5.eval(); m6.eval()
with torch.no_grad():
    for _ in range(100):
        m5(x, seq_len=21); m6(x, seq_len=21)

N = 500
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(N):
        m5(x, seq_len=21)
    t5 = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        m6(x, seq_len=21)
    t6 = (time.perf_counter() - t0) / N * 1000

print(f"\n=== INFERENCE SPEED (batch=32, CPU, {N} runs) ===")
print(f"M5: {t5:.2f} ms/batch")
print(f"M6: {t6:.2f} ms/batch")
print(f"Speedup: {t5/t6:.2f}x ({(1 - t6/t5)*100:.1f}% faster)")

# Model file size
s5 = os.path.getsize(r'c:\Users\youss\Downloads\PFE\M5\pile_model.pth') / 1024
s6 = os.path.getsize(r'c:\Users\youss\Downloads\PFE\M6\pile_model.pth') / 1024
print(f"\n=== MODEL FILE SIZE ===")
print(f"M5: {s5:.1f} KB")
print(f"M6: {s6:.1f} KB")
print(f"Reduction: {(1 - s6/s5)*100:.1f}%")
