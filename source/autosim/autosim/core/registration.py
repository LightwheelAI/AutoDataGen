"""AutoSim registration system."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from autosim.core.pipeline import AutoSimPipeline as Pipeline
    from autosim.core.skill import Skill, SkillExtraCfg


"""Pipeline Registration System

This section provides the registration and instantiation system for
pipelines. Pipelines can be registered with an ID and entry points, then
created using make_pipeline().

Usage:
    # 1. Register a pipeline
    >>> register_pipeline(
    >>>     id="MyPipeline-v0",
    >>>     entry_point="autosim.pipelines:MyPipeline",
    >>>     cfg_entry_point="autosim.pipelines:MyPipelineCfg",
    >>> )

    # 2. Create a pipeline instance
    >>> pipeline = make_pipeline("MyPipeline-v0")
    >>> pipeline.run()

    # 3. List all registered pipelines
    >>> pipeline_ids = list_pipelines()

    # 4. Unregister a pipeline
    >>> unregister_pipeline("MyPipeline-v0")
"""


class PipelineCreator(Protocol):
    """Function that creates a pipeline instance."""

    def __call__(self, **kwargs: Any) -> Pipeline: ...


class ConfigCreator(Protocol):
    """Function that creates a configuration instance."""

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class PipelineEntry:
    """Entry for a pipeline in the registry.

    Attributes:
        id: Unique identifier for the pipeline (e.g., "MyPipeline-v0").
        entry_point: String pointing to the Pipeline class or a callable that creates a pipeline instance. Format: "module.path:ClassName".
        cfg_entry_point: String pointing to the configuration class or a callable that creates a config instance. Format: "module.path:ConfigClass".
    """

    id: str
    entry_point: PipelineCreator | str | None = field(default=None)
    cfg_entry_point: ConfigCreator | str | None = field(default=None)


# Global registry for pipelines
pipeline_registry: dict[str, PipelineEntry] = {}


def register_pipeline(
    id: str,
    entry_point: PipelineCreator | str | None = None,
    cfg_entry_point: ConfigCreator | str | None = None,
) -> None:
    """Register a pipeline in the global registry."""

    assert entry_point is not None, "Entry point must be provided."
    assert cfg_entry_point is not None, "Configuration entry point must be provided."

    if id in pipeline_registry:
        raise ValueError(
            f"Pipeline with id '{id}' is already registered. To register a new version, use a different id (e.g.,"
            f" '{id}-v1')."
        )

    entry = PipelineEntry(
        id=id,
        entry_point=entry_point,
        cfg_entry_point=cfg_entry_point,
    )
    pipeline_registry[entry.id] = entry


def _load_entry_point(entry_point: str) -> Any:
    """Load a class or function from an entry point string."""

    try:
        mod_name, attr_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr_name)
        return obj
    except (ValueError, ModuleNotFoundError, AttributeError) as e:
        raise ValueError(
            f"Could not resolve entry point '{entry_point}'. Expected format: 'module.path:ClassName'. Error: {e}"
        ) from e


def _load_creator(creator: str | PipelineCreator | ConfigCreator) -> PipelineCreator | ConfigCreator:

    if isinstance(creator, str):
        return _load_entry_point(creator)
    else:
        return creator


def make_pipeline(
    id: str,
) -> Pipeline:
    """Create a pipeline instance from the registry."""

    if id not in pipeline_registry:
        raise ValueError(
            f"Pipeline '{id}' not found in registry. You can list all registered pipelines with list_pipelines()."
        )

    entry = pipeline_registry[id]

    pipeline_creator = _load_creator(entry.entry_point)
    cfg_creator = _load_creator(entry.cfg_entry_point)

    # Instantiate the pipeline
    try:
        cfg = cfg_creator()
        pipeline = pipeline_creator(cfg=cfg)
    except TypeError as e:
        entry_point_str = entry.entry_point if isinstance(entry.entry_point, str) else str(entry.entry_point)
        raise TypeError(
            f"Failed to instantiate pipeline '{id}' with entry point '{entry_point_str}'. Error: {e}"
        ) from e

    return pipeline


def list_pipelines() -> list[str]:
    """List all registered pipeline IDs."""

    return sorted(pipeline_registry.keys())


def unregister_pipeline(id: str) -> None:
    """Unregister a pipeline from the registry."""

    if id not in pipeline_registry:
        raise ValueError(f"Pipeline '{id}' not found in registry.")
    del pipeline_registry[id]


"""Skill Registration System
This section provides the registration and instantiation system for skills.
Skills can be registered manually, then created using SkillRegistry.create().

Usage:
    # 1. Using decorator (recommended)
    >>> @register_skill("reach", "Reach to target pose", ["curobo"])
    >>> class ReachSkill(Skill):
    >>>    ...

    # 2. Manual registration
    >>> class MySkill(Skill):
    >>>     cfg = SkillCfg(name="my_skill", description="My custom skill")
    >>> SkillRegistry.register(MySkill)

    # 3. Create a skill instance
    >>> skill = SkillRegistry.create("reach", extra_cfg=SkillExtraCfg(param="value"))

    # 4. List all registered skills
    >>> skill_configs = SkillRegistry.list_skills()
"""


class SkillRegistry:
    """Skill Registry - Supports manual registration of skills."""

    _instance: SkillRegistry | None = None
    _skills: dict[str, Skill] = dict()

    def __new__(cls) -> SkillRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, skill_cls: type) -> type:
        """Register a skill in the registry."""
        from autosim.core.skill import Skill

        if not issubclass(skill_cls, Skill):
            raise TypeError(f"Skill class '{skill_cls.__name__}' must inherit from Skill.")
        if skill_cls.cfg.name in cls._skills and cls._skills[skill_cls.cfg.name] != skill_cls:
            raise ValueError(
                f"Skill with name '{skill_cls.cfg.name}' already registered with different class:"
                f" {cls._skills[skill_cls.cfg.name]}"
            )

        cls._skills[skill_cls.cfg.name] = skill_cls
        return skill_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Get a skill from the registry."""

        if name not in cls._skills:
            raise ValueError(f"Skill '{name}' not found in registry.")
        return cls._skills[name]

    @classmethod
    def create(cls, name: str, extra_cfg: SkillExtraCfg) -> Skill:
        """Create a skill instance from the registry, extra_cfg will overwrite the default value in the skill configuration."""

        skill_cls = cls.get(name)
        return skill_cls(extra_cfg)

    @classmethod
    def list_skills(cls) -> list[str]:
        """List all registered skill names."""

        return [skill_cls.get_cfg() for skill_cls in cls._skills.values()]


def register_skill(name: str, description: str, cfg_type: type) -> type:
    """Decorator: Simplify skill definition."""

    def decorator(cls: type) -> type:
        cls.cfg = cfg_type(name=name, description=description)
        SkillRegistry.register(cls)
        return cls

    return decorator
