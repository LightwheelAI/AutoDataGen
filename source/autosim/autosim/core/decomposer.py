import json
from abc import ABC, abstractmethod
from dataclasses import MISSING, asdict
from pathlib import Path

from dacite import from_dict
from isaaclab.utils import configclass

from autosim.core.logger import AutoSimLogger

from .types import DecomposeResult, EnvExtraInfo


@configclass
class DecomposerCfg:
    """Configuration for the decomposer."""

    class_type: type = MISSING
    """The class type of the decomposer."""

    cache_dir: str = "~/.cache/autosim/decomposer_cache"
    """The cache directory for the decomposer."""


class Decomposer(ABC):
    def __init__(self, cfg: DecomposerCfg) -> None:
        self.cfg = cfg
        self._cache_dir = Path(self.cfg.cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = AutoSimLogger("Decomposer")

    @abstractmethod
    def decompose(self, extra_info: EnvExtraInfo) -> DecomposeResult:
        """Decompose the task with the given extra information."""

        raise NotImplementedError(f"{self.__class__.__name__}.decompose() must be implemented.")

    def is_cache_hit(self, task_name: str) -> bool:
        """Check if the cache hit for the given task name."""
        return (self._cache_dir / f"{task_name}.json").exists()

    def write_cache(self, task_name: str, decompose_result: DecomposeResult) -> None:
        """Write the cache for the given task name."""

        with open(self._cache_dir / f"{task_name}.json", "w") as f:
            json.dump(asdict(decompose_result), f, indent=4)

    def read_cache(self, task_name: str) -> DecomposeResult:
        """Read the cache for the given task name."""

        if not (self._cache_dir / f"{task_name}.json").exists():
            raise FileNotFoundError(f"Cache file not found for task name: {task_name} in storage: {self.cfg.cache_dir}")
        with open(self._cache_dir / f"{task_name}.json") as f:
            return from_dict(DecomposeResult, json.load(f))
