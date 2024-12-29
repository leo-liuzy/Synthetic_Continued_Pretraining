import json
from dataclasses import dataclass

from llm_scripts.config import Config


@dataclass
class OpenAIGenerateConfig(Config):
    input_file: str
    output_dir: str
    model: str
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_completion_tokens: int = 16
    temperature: float = 0.0
    top_p: float | None = None
    n: int = 1
    response_format: str | None = None  # Hint: json_object
    stop: str | None = None
    strip_trailing_assistant_messages: bool = True
    org: str = ''

    def __post_init__(self):
        if self.stop is not None:
            try:
                self.stop = json.loads(self.stop)
            except json.JSONDecodeError:
                pass

    @property
    def openai_params(self) -> dict:
        return {
            'model': self.model,
            'frequency_penalty': self.frequency_penalty,
            'logprobs': self.logprobs,
            'top_logprobs': self.top_logprobs,
            'max_completion_tokens': self.max_completion_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'response_format': self.response_format,
            'stop': self.stop,
        }
