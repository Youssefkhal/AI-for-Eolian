import torch
import time
import sys, os

sys.path.insert(0, os.path.join(r'c:\Users\youss\Downloads\PFE', 'M5'))
from train import SlotAttentionDegradation as M5Model

sys.path.insert(0, os.path.join(r'c:\Users\youss\Downloads\PFE', 'M6'))
from train import SlotAttentionDegradation as M6Model

m5 = M5Model(input_size=8, d_model=64, num_heads=4, num_slots=21, max_seq_len=21, dropout=0.1, num_iterations=2)
m6 = M6Model(input_size=8, d_model=64, num_heads=4, num_slots=21, max_seq_len=21, dropout=0.1, num_iterations=3)

p5 = sum(p.numel() for p in m5.parameters())
p6 = sum(p.numel() for p in m6.parameters())

print(f"M5 parameters: {p5:,}")
print(f"M6 parameters: {p6:,}")
print(f"M6 is {(1 - p6/p5)*100:.1f}% smaller")
print(f"Ratio: M5/M6 = {p5/p6:.2f}x")

# Inference speed
x = torch.randn(32, 8)
m5.eval()
m6.eval()

# Warmup
with torch.no_grad():
    for _ in range(50):
        m5(x, seq_len=21)
        m6(x, seq_len=21)

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

print(f"\nInference (batch=32, CPU, avg of {N} runs):")
print(f"M5: {t5:.2f} ms")
print(f"M6: {t6:.2f} ms")
print(f"M6 is {(1 - t6/t5)*100:.1f}% faster")
print(f"Speedup: {t5/t6:.2f}x")

# Memory (model size on disk)
import joblib
m5_path = os.path.join(r'c:\Users\youss\Downloads\PFE', 'M5', 'pile_model.pth')
m6_path = os.path.join(r'c:\Users\youss\Downloads\PFE', 'M6', 'pile_model.pth')
s5 = os.path.getsize(m5_path) / 1024
s6 = os.path.getsize(m6_path) / 1024
print(f"\nModel file size:")
print(f"M5: {s5:.1f} KB")
print(f"M6: {s6:.1f} KB")
print(f"M6 is {(1 - s6/s5)*100:.1f}% smaller on disk")
