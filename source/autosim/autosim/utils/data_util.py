from typing import Any, Literal

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


def convert_quat(quat: torch.Tensor | np.ndarray, to: Literal["xyzw", "wxyz"] = "xyzw") -> torch.Tensor | np.ndarray:
    """Converts quaternion from one convention to another.

    The convention to convert TO is specified as an optional argument. If to == 'xyzw',
    then the input is in 'wxyz' format, and vice-versa.

    Args:
        quat: The quaternion of shape (..., 4).
        to: Convention to convert quaternion to.. Defaults to "xyzw".

    Returns:
        The converted quaternion in specified convention.

    Raises:
        ValueError: Invalid input argument `to`, i.e. not "xyzw" or "wxyz".
        ValueError: Invalid shape of input `quat`, i.e. not (..., 4,).
    """
    # check input is correct
    if quat.shape[-1] != 4:
        msg = f"Expected input quaternion shape mismatch: {quat.shape} != (..., 4)."
        raise ValueError(msg)
    if to not in ["xyzw", "wxyz"]:
        msg = f"Expected input argument `to` to be 'xyzw' or 'wxyz'. Received: {to}."
        raise ValueError(msg)
    # check if input is numpy array (we support this backend since some classes use numpy)
    if isinstance(quat, np.ndarray):
        # use numpy functions
        if to == "xyzw":
            # wxyz -> xyzw
            return np.roll(quat, -1, axis=-1)
        else:
            # xyzw -> wxyz
            return np.roll(quat, 1, axis=-1)
    else:
        # convert to torch (sanity check)
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=float)
        # convert to specified quaternion type
        if to == "xyzw":
            # wxyz -> xyzw
            return quat.roll(-1, dims=-1)
        else:
            # xyzw -> wxyz
            return quat.roll(1, dims=-1)
