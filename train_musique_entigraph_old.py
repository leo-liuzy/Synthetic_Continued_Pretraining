from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import transformers
import os
import warnings
from torch.utils.data import Dataset
import hydra
import gc

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from data.cptdata import CPTDataset, _MemmapDataset
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import torch
import math
from knowledge_propagation.modules.inferencers import QAInferencer
from transformers import AutoTokenizer, GenerationConfig
from knowledge_propagation.utils import io
from experiments.musique.inference_only import eval_inferencer, macro_averaging


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

    sample_triplet_ratio: Optional[float] = None
    specified_bin: Optional[str] = None

    wandb_project: Optional[str] = field(default="synthetic-continued-pretraining")
    use_peft: bool = False
    lora_r: int = 8
    lora_dropout: int = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["all-linear"])

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode="r")
        self.ids = self.ids[: int(len(self.ids) * subsample_ratio)]

    def __len__(self):
        return math.ceil(len(self.ids) / self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(), labels=torch.from_numpy(x_id).long())


class CPTDataset(_MemmapDataset):
    def __init__(self, block_size: int, rehersal_rate: float, subsample_ratio: float, task_name, **kwargs):
        assert rehersal_rate <= 1.0
        self.rehersal_rate = rehersal_rate
        self.rehersal_data = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-train"), 1.0)
        super().__init__(block_size, _get_bin(task_name, "entigraph", **kwargs), subsample_ratio)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return super().__getitem__(i)


def _get_bin(task_name: str, split: str, **kwargs):
    # assert task_name in ["quality", "rehersal", "instruct", "jd-vance", "real-jd-vance"]
    bin_data_dir = "data/dataset/bins"

    if task_name == "jd-vance":
        bin_fname = f"{task_name}_entigraph_gpt-4-turbo"
        if kwargs.get("trimE", False):
            bin_fname += "_trimE"
        if kwargs.get("no_single", False):
            bin_fname += "_no1"
        if kwargs.get("no_pair", False):
            bin_fname += "_no2"

        if kwargs.get("sample_triplet_ratio", None) is not None:
            assert not kwargs.get("no_triplet", False)
            bin_fname += f"_sub3={kwargs['sample_triplet_ratio']}"
        elif kwargs.get("no_triplet", False):
            bin_fname += "_no3"
    elif task_name == "real-jd-vance":
        bin_fname = f"{task_name}-{split}"
    elif task_name == "musique_entigraph":
        bin_fname = f"musique_entigraph_gpt-4-turbo_sample8/{kwargs['example_id']}"
    else:
        bin_fname = "quality_all-entigraphgpt-4-turbo"

    implemented_quality_split = {
        "entigraph": f"{bin_data_dir}/{bin_fname}.bin",
    }
    implemented_rehersal_split = {
        "rpj-train": f"{bin_data_dir}/RedPajama_Data_1T_Sample_train.bin",
        "rpj-test": f"{bin_data_dir}/RedPajama_Data_1T_Sample_test.bin",
    }
    implemented_instruct_split = {
        "ultrachat-train": f"{bin_data_dir}/ultrachat_train.bin",
        "ultrachat-test": f"{bin_data_dir}/ultrachat_test.bin",
    }
    if task_name in ["quality", "jd-vance", "musique_entigraph"]:
        assert split in implemented_quality_split
        return implemented_quality_split[split]
    elif task_name in "real-jd-vance":
        return f"{bin_data_dir}/{bin_fname}.bin"
    elif task_name == "rehersal":
        assert split in implemented_rehersal_split
        return implemented_rehersal_split[split]
    elif task_name == "instruct":
        assert split in implemented_instruct_split
        return implemented_instruct_split[split]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


def get_task_data_module(
    task_name: str,
    block_size: int,
    rehersal_rate: float,
    subsample_ratio: float,
    specified_bin: Optional[str] = None,
    **kwargs,
):
    if task_name == "musique_entigraph":
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio, task_name, **kwargs)
        val = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)

    if task_name == "jd-vance":
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio, task_name, **kwargs)
        val = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)

    if task_name == "real-jd-vance":
        assert "train_split" in kwargs and isinstance(kwargs["train_split"], str)
        assert "valid_split" in kwargs and isinstance(kwargs["valid_split"], str)
        train = _MemmapDataset(block_size, _get_bin(task_name, kwargs["train_split"]), 1.0)
        val = _MemmapDataset(block_size, _get_bin(task_name, kwargs["valid_split"]), 1.0)
        return dict(train_dataset=train, eval_dataset=val)

    if task_name == "quality":
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio)
        val = _MemmapDataset(block_size, _get_bin("rehersal", "rpj-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == "instruct":
        train = _MemmapDataset(block_size, _get_bin("instruct", "ultrachat-train"), 1.0)
        val = _MemmapDataset(block_size, _get_bin("instruct", "ultrachat-test"), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


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

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")
    # ! This is important
    # leave a model pointer to the model in trainer
    model = trainer.model
    # clear internal pointer in trainer/accelerator
    trainer.accelerator.free_memory(trainer.model, trainer.optimizer, trainer.lr_scheduler)
    del trainer.model, trainer.optimizer, trainer.lr_scheduler
    del trainer
    # clear cache to make spaces in GPU and CPU
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # sleep(200)
    logging.info("Starting inferencer")

    question_types = [
        "single_hop_efficacy",
        "multi_hop_efficacy",
        "single_hop_specificity",
        "multi_hop_specificity",
    ]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    generation_config = GenerationConfig(
        do_sample=cfg.generation.do_sample,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        temperature=cfg.generation.temperature,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=cfg.generation.max_new_tokens,
        num_return_sequences=cfg.generation.n_decoding_example,
    )
    raw_instance = io.load_json(f"data/dataset/raw/id2{config.task_name}.json")[config.example_id]
    all_results = []
    for question_type in question_types:
        questions = raw_instance[question_type]
        logging.info(f"Question type: {question_type}")
        inferencer = QAInferencer(
            cfg.evaluator.inferencers[0],
            cfg.seed,
            rag_model=None,
            queries=questions,
        )
        result_df = eval_inferencer(
            inferencer,
            model,
            tokenizer=tokenizer,
            generation_cfg=generation_config,
        )
        result_df.insert(0, "question_type", question_type)
        result_df.insert(0, "id", raw_instance["id"])
        all_results.append(result_df)

    all_results = pd.concat(all_results)

    all_results.to_excel(
        f"{args.output_dir}/{raw_instance['id']}_inferencer_results.xlsx",
        index=False,
    )
    metrics = ["rouge1", "llm_accuracy"]
    multi_level_averaging = ["question_type", "id", "question"]
    result_df = macro_averaging(all_results, metrics, multi_level_averaging).round(2)
    q_cat_dtype = pd.CategoricalDtype(
        categories=question_types,
        ordered=True,
    )

    result_df["question_type"] = result_df["question_type"].astype(q_cat_dtype)


if __name__ == "__main__":
    train()
