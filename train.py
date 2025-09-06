from dataclasses import dataclass, field, asdict
from typing import Optional
import transformers
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from data.cptdata import get_task_data_module


@dataclass
class TrainingConfig:
    task_name: str
    block_size: int
    rehersal_rate: float
    model_name: str
    subsample_ratio: float
    split: Optional[str] = field(default="naive")
    wandb_project: Optional[str] = field(default="synthetic-continued-pretraining")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    # loading model
    assert config.split in ["naive", "entigraph", "meta_aug-one_stage-naive", "meta_aug-one_stage-ice", "meta_aug-two_stage-naive", "meta_aug-two_stage-ice", "active_reading-task_agnostic", "active_reading-task_specific", "active_reading-task_agnostic-task_specific"]
    model_name_base = os.path.basename(config.model_name)
    config.split = f"{config.split}-{model_name_base}"
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    # loading dataset
    data_module = get_task_data_module(**asdict(config))
    # import pdb; pdb.set_trace()
    # setting up trainer
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()