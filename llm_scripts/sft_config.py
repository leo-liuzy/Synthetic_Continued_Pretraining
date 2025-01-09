import json
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_scripts.config import Config


LORA_TARGET_MODULES = [
    'up_proj',
    'down_proj',
    'k_proj',
    'q_proj',
    'v_proj',
]


@dataclass
class SFTConfig(Config):
    # ---- #
    # Data #
    # ---- #
    train_file: str
    eval_file: str | None = None
    eval_split_size: float | None = None

    # ------------------------- #
    # Additional run-level info #
    # ------------------------- #
    run_info_json: str | None = None

    # ------------------ #
    # Tokenizer and Chat #
    # ------------------ #
    max_length: int = 4096
    # If setup_chat_model is True, then special tokens are added to
    # the tokenizer, and the chat_template (if present) is overridden.
    # If setup_chat_model is False, the tokenizer is not modified.
    # If setup_chat_model is None, then setup_chat_model is treated as True
    # if the loaded tokenizer does not have a chat template and False otherwise.
    setup_chat_model: bool | None = None
    special_tokens_to_add_json: str | None = None
    chat_template: str | None = None
    pad_token: str | None = None
    mask_input_segments: bool = True
    eot_token: str | None = None

    # -------- #
    # Collator #
    # -------- #
    padding: bool = True
    label_pad_token_id: int = -100
    # NVIDIA recommends padding to multiple of 16 for 8bit or 8 for 16bit
    # https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape
    pad_to_multiple_of: int = 16

    # ----- #
    # Model #
    # ----- #
    model_id: str = 'meta-llama/Meta-Llama-3-8B'
    device_map: str | None = None
    torch_dtype_name: str | None = 'bfloat16'
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    attn_implementation: str = 'flash_attention_2'

    # ---- #
    # LoRA #
    # ---- #
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = 'none'
    lora_task_type: str = 'CAUSAL_LM'
    lora_target_modules_json: str | None = json.dumps(LORA_TARGET_MODULES)

    # -------- #
    # Training #
    # -------- #
    output_dir: str | None = None  # REQUIRED!
    logging_dir: str | None = None
    # Training -- Optimization
    num_train_epochs: int = 2
    optim: str = 'adamw_torch_fused'
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    neftune_noise_alpha: float | None = 5
    gradient_accumulation_steps: int = 8
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    auto_find_batch_size: bool = True
    group_by_length: bool = True
    # Training -- Eval, logging, and saving
    evaluation_strategy: str = 'steps'
    eval_steps: int = 50
    save_strategy: str = 'steps'
    save_steps: int | None = None
    logging_strategy: str = 'steps'
    logging_steps: int | None = None
    save_total_limit: int | None = None
    # Training -- precision
    bf16: bool = False
    fp16: bool = False
    # Training -- Other
    ddp_backend: str = 'nccl'
    report_to: str = 'tensorboard'
    remove_unused_columns: bool = True
    load_best_model_at_end: bool = False
    gradient_checkpointing: bool = False
    ddp_find_unused_parameters: bool = False
    save_on_each_node: bool = True
    # By default we generate a new output_dir every run
    # If you want to set this to True,
    # you will also need to specify output_dir to a previous run
    resume_from_checkpoint: bool = False

    @property
    def use_cache(self) -> bool:
        return not self.gradient_checkpointing

    @property
    def lora_target_modules(self) -> list[str]:
        return json.loads(self.lora_target_modules_json)

    @property
    def special_tokens_to_add(self) -> dict[str, str]:
        if not self.special_tokens_to_add_json:
            return {}

        return json.loads(self.special_tokens_to_add_json)

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.torch_dtype_name:
            return getattr(torch, self.torch_dtype_name)

        return None

    def __post_init__(self):
        if self.train_file is None:
            raise ValueError('train_file is required')

        if self.output_dir is None:
            raise ValueError('output_dir is required')

        if self.logging_dir is None:
            self.logging_dir = str(Path(self.output_dir, 'logs'))

        if self.eval_steps is not None:
            self.save_steps = self.save_steps or self.eval_steps
            self.logging_steps = self.logging_steps or self.eval_steps

        if self.torch_dtype == torch.float16:
            self.bf16 = False
            self.fp16 = True
        elif self.torch_dtype == torch.bfloat16:
            self.bf16 = True
            self.fp16 = False
        else:
            self.bf16 = self.fp16 = False

        if not self.neftune_noise_alpha or self.neftune_noise_alpha <= 0:
            self.neftune_noise_alpha = None
