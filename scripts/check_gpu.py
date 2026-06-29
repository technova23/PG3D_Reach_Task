from __future__ import annotations

import sys

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"Failed to import torch: {exc}")
    sys.exit(1)

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print(f"device: {torch.cuda.get_device_name(idx)}")
    print(f"capability: {torch.cuda.get_device_capability(idx)}")
    x = torch.randn((1024, 1024), device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    print(f"matmul ok; checksum={float(y[0, 0]):.6f}")
else:
    sys.exit(2)
