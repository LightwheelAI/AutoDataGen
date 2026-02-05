from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dacite import from_dict
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from autosim import SkillRegistry
from autosim.core.decomposer import Decomposer
from autosim.core.types import DecomposeResult, EnvExtraInfo

if TYPE_CHECKING:
    from .llm_decomposer_cfg import LLMDecomposerCfg


class LLMBackend:
    """
    LLM Backend, using OpenAI-compatible interface including GPT, DeepSeek, Claude, etc.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initialize LLM backend

        Args:
            api_key: API key
            base_url: API endpoint URL
            model: Model name, supported models:
                - GPT: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.
                - DeepSeek: deepseek-chat, deepseek-reasoner
                - Claude: claude-3-5-sonnet-20241022, etc.
        """

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate response from LLM

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class LLMDecomposer(Decomposer):
    def __init__(self, cfg: LLMDecomposerCfg) -> None:
        super().__init__(cfg)

        self._llm_backend = LLMBackend(api_key=self.cfg.api_key, base_url=self.cfg.base_url, model=self.cfg.model)

        self._atomic_skills = [skill_cfg.name for skill_cfg in SkillRegistry.list_skills()]

        jinja_env = Environment(
            loader=FileSystemLoader(str(Path(__file__).parent / "prompts")),
            autoescape=False,
        )
        self._prompt_template = jinja_env.get_template("task_decompose.jinja")

    def decompose(self, extra_info: EnvExtraInfo) -> DecomposeResult:

        task_code = self._load_task_code(extra_info.task_name)
        prompt = self._build_prompt(task_code, extra_info)
        self._logger.debug(f"prompt for llm composer: \n{prompt}")

        self._logger.info("generate response from llm...")
        response = self._llm_backend.generate(
            prompt=prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens
        )

        # parse json response
        try:
            results = self._extract_json(response)
            self._validate_result(results)

            return from_dict(DecomposeResult, results)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")

    def _load_task_code(self, task_name: str) -> str:
        """
        Load task code from gymnasium registry.

        Args:
            task_name: The name of the task.

        Returns:
            The task code.
        """

        module_path, class_name = self._find_task_in_gym_registry(task_name)
        if module_path is None or class_name is None:
            raise ValueError(f"Task {task_name} not found in gymnasium registry")
        module = importlib.import_module(module_path)
        task_cls = getattr(module, class_name)

        cls_source_code = inspect.getsource(task_cls)
        module_source_code = inspect.getsource(module)

        # extract import statements
        import_lines = []
        for line in module_source_code.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                import_lines.append(line)
            elif line and not line.startswith("#"):
                # stop at the first non-import, non-comment line
                break
        full_code = "\n".join(import_lines) + f"\n\n{cls_source_code}"

        return full_code

    def _find_task_in_gym_registry(self, task_name: str) -> tuple:
        """
        Find task in gymnasium registry and extract module path and class name

        Args:
            task_name: Task cfg class name in gymnasium registry

        Returns:
            Tuple of (module_path, class_name) or (None, None) if not found
        """

        import gymnasium as gym

        for task_spec in gym.registry.values():
            if task_spec.id == task_name and task_spec.kwargs:
                env_cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
                if env_cfg_entry_point:
                    module_path, class_name = env_cfg_entry_point.split(":")
                    return module_path, class_name
        return None, None

    def _build_prompt(self, task_code: str, extra_info: EnvExtraInfo) -> str:
        """
        Build the prompt for the LLM decomposer.

        Args:
            task_code: The code of the task.
            extra_info: The extra information of the environment.

        Returns:
            The prompt for the LLM decomposer.
        """

        skills = {skill_cfg.name: skill_cfg.description for skill_cfg in SkillRegistry.list_skills()}

        return self._prompt_template.render(
            task_code=task_code,
            task_name=extra_info.task_name,
            skills=skills,
            objects=extra_info.objects,
            additional_prompt_contents=extra_info.additional_prompt_contents,
        )

    def _extract_json(self, response: str) -> dict:
        """
        Extract JSON from LLM response (handles markdown code blocks)

        Args:
            response: LLM response string

        Returns:
            Parsed JSON dictionary
        """

        # Try direct parsing
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(response)

        # Try markdown code blocks
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if matches:
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(matches[0])

        # Try finding JSON object
        matches = re.findall(r"\{.*\}", response, re.DOTALL)
        for match in sorted(matches, key=len, reverse=True):
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(match)

        raise json.JSONDecodeError("No valid JSON found in response", response, 0)

    def _validate_result(self, result: dict) -> None:
        """
        Validate decomposition result structure

        Args:
            result: Decomposition result dictionary

        Raises:
            ValueError: If validation fails
        """

        # Check required fields
        required_fields = [
            "task_name",
            "task_description",
            "parent_classes",
            "objects",
            "fixtures",
            "subtasks",
            "success_conditions",
            "total_steps",
            "skill_sequence",
        ]

        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate skill types
        for subtask in result["subtasks"]:
            for skill in subtask["skills"]:
                if skill["skill_type"] not in self._atomic_skills:
                    raise ValueError(f"Invalid skill type: {skill['skill_type']}. Must be one of {self._atomic_skills}")
