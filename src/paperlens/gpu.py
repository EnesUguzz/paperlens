"""GPU sanity check: is CUDA visible to torch, and does an fp16 matmul work?

Purpose: The RTX 5080 (Blackwell, sm_120) needs CUDA 12.8+ wheels; this catches a wrong
torch build before an hour-long indexing job fails.
Flow: `paperlens check-gpu` -> gpu_report() -> printed dict.
Params: none.
Try: Run it after any torch upgrade.
"""

from __future__ import annotations

from typing import Any

MATMUL_SIZE = 1024


def gpu_report() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if info["cuda_available"]:
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = torch.cuda.get_device_capability(0)
        a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda", dtype=torch.float16)
        info["fp16_matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
    return info
