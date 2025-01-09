from dataclasses import dataclass, field, asdict
from typing import Optional, List
import transformers
from transformers import AutoTokenizer, GenerationConfig
import os
import warnings

# from data.cptdata import MemmapDataset, _MemmapDataset
import hydra
import gc
from typing import Dict, Optional
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from data.cptdata import get_task_data_module
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.utils import io
from experiments.musique.inference_only import eval_inferencer, macro_averaging
import torch
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np


from time import sleep
import math

from accelerate.logging import get_logger
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    task_name: str
    example_id: str
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
    run_train: Optional[bool] = None

    sample_triplet_ratio: Optional[float] = None
    specified_bin: Optional[str] = None

    wandb_project: Optional[str] = field(default="synthetic-continued-pretraining")
    use_peft: bool = False
    lora_r: int = 8
    lora_dropout: int = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["all-linear"])

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


class MemmapDataset(Dataset):
    def __init__(self, block_size: int, token_ids, eos_token_id, pad_token_id):
        logger.info(f"block_size: {block_size}")
        logger.info(f"len(token_ids): {len(token_ids)}")
        logger.info(f"eos_token_id: {eos_token_id}")
        logger.info(f"pad_token_id: {pad_token_id}")
        self.block_size = block_size
        self.ids = token_ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        logger.info(f"len(self): {math.ceil(len(self.ids) / self.block_size)}")

    def __len__(self):
        return math.ceil(len(self.ids) / self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        if x_id[-1] != self.eos_token_id:
            x_id = np.concatenate([x_id, [self.eos_token_id]])

        if len(x_id) < self.block_size:
            # pad
            x_id = np.concatenate([x_id, [self.pad_token_id] * (self.block_size - len(x_id) + 1)])
        try:
            return dict(input_ids=torch.from_numpy(x_id).long(), labels=torch.from_numpy(x_id).long())
        except Exception as e:
            print(x_id)

class CPTDataset(Dataset):
    def __init__(
        self,
        target_data: MemmapDataset,
        rehersal_data: MemmapDataset,
        rehersal_rate: float,
    ):
        assert rehersal_rate <= 1.0
        self.target_data = target_data
        self.rehersal_data = rehersal_data
        self.rehersal_rate = rehersal_rate

    def __len__(self):
        return self.target_data.__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return self.target_data[i]


def train():
    # parsing input

    os.chdir(os.path.dirname(__file__))
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    # loading dataset
    # data_module = get_task_data_module(**asdict(config))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # START: special operation for Llama3 for missing padding token
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Just to suppress tokenizer's warning. Supposedly do nothing.
    tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token
    # END: special operation for Llama3 for missing padding token
    
    target_tokens = np.memmap(f"data/dataset/bins/{config.task_name}/{config.example_id}.bin", dtype=np.int32, mode="r")
    logging.info(f"# target_tokens: {len(target_tokens)}")

    rehersal_tokens = np.memmap("data/dataset/bins/RedPajama_Data_1T_Sample_train.bin", dtype=np.int32, mode="r")
    rehersal_dataset = MemmapDataset(config.block_size, rehersal_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id)

    # args.max_steps = 1 # debug
    if config.task_name == "musique":
        target_dataset = MemmapDataset(config.block_size, target_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id)
        logging.info(f"# target_dataset example: {len(target_dataset)}")
        train = CPTDataset(target_dataset, rehersal_dataset, config.rehersal_rate)
        # train = target_dataset

        data_module = dict(train_dataset=train, eval_dataset=None)
        args.eval_strategy = "no"
    else:
        assert config.task_name == "musique_page"

        target_dataset = MemmapDataset(
            config.block_size, 
            target_tokens[: int(len(target_tokens) * 0.9)], 
            tokenizer.eos_token_id, 
            tokenizer.pad_token_id
        )
        logging.info(f"# target_dataset example: {len(target_dataset)}")
        train = CPTDataset(target_dataset, rehersal_dataset, config.rehersal_rate)
        # train = target_dataset

        val = MemmapDataset(
            config.block_size, 
            target_tokens[int(len(target_tokens) * 0.9) :], 
            tokenizer.eos_token_id,
            tokenizer.pad_token_id
        )
        data_module = dict(train_dataset=train, eval_dataset=val)
        args.eval_strategy = "epoch"
    args.eval_on_start = data_module["eval_dataset"] is not None
    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Model: {model}")

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
    # model.train()
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    # method 1
    trainer.save_model(output_dir=args.output_dir + "/tmp_ckpt") # original line
    
    # trainer.save_model()
    trainer.accelerator.wait_for_everyone()
    # sleep(200)

    # logger.info(result_df.sort_values(by=["question_type"], inplace=False))


if __name__ == "__main__":
    train()
