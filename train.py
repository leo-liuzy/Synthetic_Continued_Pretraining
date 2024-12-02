from dataclasses import dataclass, field, asdict
from typing import Optional, List
import transformers
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from data.cptdata import get_task_data_module

from peft import get_peft_model, LoraConfig, TaskType


@dataclass
class TrainingConfig:
    task_name: str
    block_size: int
    rehersal_rate: float
    model_name: str
    subsample_ratio: float
    trimE: bool
    no_single: bool
    no_pair: bool
    no_triplet: bool
    train_split: Optional[str] = "1doc"
    valid_split: Optional[str] = "valid"

    sample_triplet_ratio: Optional[float] = None
    specified_bin: Optional[str] = None

    wandb_project: Optional[str] = field(default="synthetic-continued-pretraining")
    use_peft: bool = False
    lora_r: int = 8
    lora_dropout: int = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["all-linear"])

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    # parsing input

    os.chdir(os.path.dirname(__file__))
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    # loading dataset
    data_module = get_task_data_module(**asdict(config))

    if config.task_name == "real-jd-vance":
        # args.dataloader_drop_last = False
        args.output_dir += f"_{config.train_split}"
    else:
        if config.trimE:
            args.output_dir += "_trimE"
        if config.no_single:
            args.output_dir += "_no1"
        if config.no_pair:
            args.output_dir += "_no2"

        if config.sample_triplet_ratio is not None:
            assert not config.no_triplet
            args.output_dir += f"_sub3={config.sample_triplet_ratio}"
        elif config.no_triplet:
            args.output_dir += "_no3"

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
    )
    if config.use_peft:
        args.output_dir += "_lora"
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=2 * config.lora_r,  # this is recommended by Atula
            lora_dropout=config.lora_dropout,
        )
        logging.info(f"Using LoRA: {peft_config}")
        model = get_peft_model(model, peft_config)
        args.output_dir += f"_r={config.lora_r}"
        args.output_dir += f"_dropout={config.lora_dropout}"
        model.print_trainable_parameters()

    logging.info(f"Output dir: {args.output_dir}")

    # setting up trainer
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    trainer.model.save_pretrained(save_directory=args.output_dir)
    # trainer.save_model(output_dir=args.output_dir)
    # trainer.save_model()
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
