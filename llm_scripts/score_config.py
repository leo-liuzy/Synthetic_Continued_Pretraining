from dataclasses import dataclass

from llm_scripts.chat import load_chat_template
from llm_scripts.config import Config


@dataclass
class ScoreConfig(Config):
    input_file: str
    output_dir: str
    model_id: str
    checkpoint_dir: str | None = None
    chat_template: str | None = None
    pad_token: str | None = None
    batch_size: int = 8

    def __post_init__(self):
        if self.chat_template:
            self.chat_template = load_chat_template(self.chat_template)
