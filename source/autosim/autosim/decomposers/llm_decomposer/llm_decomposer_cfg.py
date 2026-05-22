import os
from dataclasses import MISSING

from isaaclab.utils import configclass

from autosim.core.decomposer import DecomposerCfg

from .llm_decomposer import LLMDecomposer


@configclass
class LLMDecomposerCfg(DecomposerCfg):
    """Configuration for the LLM decomposer."""

    class_type: type = LLMDecomposer
    """The class type of the LLM decomposer."""

    api_key: str = MISSING
    """The API key for the LLM."""

    base_url: str = "https://api.openai.com/v1"
    """The base URL for the LLM API."""

    model: str = "gpt-5.4"
    """The model name for the LLM."""

    temperature: float = 0.3
    """The temperature for the LLM."""

    max_tokens: int = 4000
    """The maximum number of tokens to generate."""

    max_decompose_retries: int = 3
    """Maximum number of retries if decomposition fails (JSON parse error or validation error)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        api_key = os.environ.get("AUTOSIM_LLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "Please set the AUTOSIM_LLM_API_KEY environment variable when using the LLMDecomposer, e.g. export"
                " AUTOSIM_LLM_API_KEY=your_api_key"
            )
        self.api_key = api_key
