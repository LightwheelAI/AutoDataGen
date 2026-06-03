from typing import Any

import numpy as np
import torch

try:
    import warp as wp
except Exception:  # pragma: no cover
    wp = None  # type: ignore[assignment]


def as_torch(x: Any) -> torch.Tensor:
    """View sim buffers as ``torch.Tensor`` (no-op if already a tensor)."""
    if isinstance(x, torch.Tensor):
        return x
    if wp is None:
        raise RuntimeError("warp is required to convert non-tensor sim buffers to torch")
    try:
        return wp.to_torch(x)  # type: ignore[no-any-return]
    except AttributeError as exc:
        if "is_cpu" not in str(exc):
            raise
        device = getattr(x, "device", None)
        if isinstance(device, torch.device):
            if device.type == "cpu":
                return torch.as_tensor(np.asarray(x))
            return torch.as_tensor(x, device=device)
        raise
