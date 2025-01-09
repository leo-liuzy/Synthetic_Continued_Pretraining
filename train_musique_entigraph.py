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
    
    target_tokens = np.memmap(
        f"data/dataset/bins/musique_entigraph_gpt-4-turbo_sample8/{config.example_id}.bin", dtype=np.int32, mode="r"
    )
    logging.info(f"# target_tokens: {len(target_tokens)}")

    rehersal_tokens = np.memmap("data/dataset/bins/RedPajama_Data_1T_Sample_train.bin", dtype=np.int32, mode="r")
    rehersal_dataset = MemmapDataset(
        config.block_size, 
        rehersal_tokens,
        tokenizer.eos_token_id, 
        tokenizer.pad_token_id
    )

    assert config.task_name == "musique_entigraph"

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
    from time import time
    start = time()
    trainer.save_model(output_dir=args.output_dir + "/tmp_ckpt") # original line
    logger.info(f"save_model time: {time() - start}")
    # trainer.model.save_pretrained(save_directory=args.output_dir)
    # trainer.save_model(output_dir=args.output_dir)
    # trainer.save_model()
    trainer.accelerator.wait_for_everyone()

    # with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
    #     cfg = hydra.compose(config_name="fft.yaml")
    # # ! This is important
    # # leave a model pointer to the model in trainer
    # model = trainer.model
    # # clear internal pointer in trainer/accelerator
    # trainer.accelerator.free_memory(trainer.model, trainer.optimizer, trainer.lr_scheduler)
    # del trainer.model, trainer.optimizer, trainer.lr_scheduler
    # del trainer
    # # clear cache to make spaces in GPU and CPU
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # sleep(200)
    # logging.info("Starting inferencer")

    # question_types = [
    #     "single_hop_efficacy",
    #     "multi_hop_efficacy",
    #     "single_hop_specificity",
    #     "multi_hop_specificity",
    # ]
    # generation_config = GenerationConfig(
    #     do_sample=cfg.generation.do_sample,
    #     top_k=cfg.generation.top_k,
    #     top_p=cfg.generation.top_p,
    #     temperature=cfg.generation.temperature,
    #     pad_token_id=tokenizer.pad_token_id,
    #     bos_token_id=tokenizer.bos_token_id,
    #     eos_token_id=tokenizer.eos_token_id,
    #     max_new_tokens=cfg.generation.max_new_tokens,
    #     num_return_sequences=cfg.generation.n_decoding_example,
    # )
    # raw_instance = io.load_json("data/dataset/raw/id2musique.json")[config.example_id]
    # all_results = []
    # for question_type in question_types:
    #     questions = raw_instance[question_type]
    #     logging.info(f"Question type: {question_type}")
    #     inferencer = QAInferencer(
    #         cfg.evaluator.inferencers[0],
    #         cfg.seed,
    #         rag_model=None,
    #         queries=questions,
    #     )
    #     result_df = eval_inferencer(
    #         inferencer,
    #         model,
    #         tokenizer=tokenizer,
    #         generation_cfg=generation_config,
    #     )
    #     result_df.insert(0, "question_type", question_type)
    #     result_df.insert(0, "id", raw_instance["id"])
    #     all_results.append(result_df)

    # all_results = pd.concat(all_results)

    # all_results.to_excel(
    #     f"{args.output_dir}/{raw_instance['id']}_inferencer_results.xlsx",
    #     index=False,
    # )
    # metrics = ["rouge1", "llm_accuracy"]
    # multi_level_averaging = ["question_type", "id", "question"]
    # result_df = macro_averaging(all_results, metrics, multi_level_averaging).round(2)
    # q_cat_dtype = pd.CategoricalDtype(
    #     categories=question_types,
    #     ordered=True,
    # )

    # result_df["question_type"] = result_df["question_type"].astype(q_cat_dtype)
    # # logger.info(result_df.sort_values(by=["question_type"], inplace=False))


if __name__ == "__main__":
    train()
