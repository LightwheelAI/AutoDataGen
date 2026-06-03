from abc import ABC, abstractmethod

import torch


class PoseSampler(ABC):
    @abstractmethod
    def sample(self, base_pose_oe: torch.Tensor) -> torch.Tensor:
        """
        Sample candidate poses around a base pose in object frame.

        Args:
            base_pose_oe: Tensor shape [7] in object frame [x,y,z,qx,qy,qz,qw]

        Returns:
            poses_oe: Tensor [K, 7] of sampled poses in object frame
        """


class OffsetSampler(PoseSampler):
    """Samples poses by adding uniform random dx/dy/dz/yaw offsets around a base pose."""

    def __init__(
        self,
        num_samples: int = 64,
        dx_range: tuple[float, float] = (-0.03, 0.03),
        dy_range: tuple[float, float] = (-0.03, 0.03),
        dz_range: tuple[float, float] = (-0.02, 0.02),
        yaw_range_rad: tuple[float, float] = (-0.35, 0.35),
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.dz_range = dz_range
        self.yaw_range_rad = yaw_range_rad
        self.seed = seed

    def sample(self, base_pose_oe: torch.Tensor) -> torch.Tensor:
        if base_pose_oe.shape != (7,):
            raise ValueError(f"base_pose_oe must have shape [7], got {tuple(base_pose_oe.shape)}")

        g = torch.Generator(device=base_pose_oe.device)
        g.manual_seed(int(self.seed))

        k = self.num_samples
        dx = torch.empty((k,), device=base_pose_oe.device).uniform_(*self.dx_range, generator=g)
        dy = torch.empty((k,), device=base_pose_oe.device).uniform_(*self.dy_range, generator=g)
        dz = torch.empty((k,), device=base_pose_oe.device).uniform_(*self.dz_range, generator=g)
        dyaw = torch.empty((k,), device=base_pose_oe.device).uniform_(*self.yaw_range_rad, generator=g)

        pos = base_pose_oe[:3].view(1, 3).repeat(k, 1) + torch.stack([dx, dy, dz], dim=-1)

        # Apply yaw delta around object local +Z axis: q_new = q_base ⊗ q_delta
        base_quat = base_pose_oe[3:].view(1, 4).repeat(k, 1)
        half = dyaw * 0.5
        q_delta = torch.stack(
            [torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)], dim=-1
        )
        quat = self._quat_mul(base_quat, q_delta)

        return torch.cat([pos, quat], dim=-1)

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )
