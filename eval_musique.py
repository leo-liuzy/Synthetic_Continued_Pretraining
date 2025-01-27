from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import transformers
from transformers import AutoTokenizer, GenerationConfig, pipeline
import os
import warnings

from copy import deepcopy

# from data.cptdata import MemmapDataset, _MemmapDataset
import hydra
import math
import gc
from time import sleep

warnings.filterwarnings("ignore", category=FutureWarning)
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from tqdm.auto import tqdm
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.utils import io

import numpy as np

from experiments.musique.inference_only import eval_inferencer, macro_averaging
import torch
from torch.utils.data import DataLoader
from time import time, sleep
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import datasets
from accelerate.logging import get_logger
from accelerate.utils import set_seed

logger = get_logger(__name__)


@dataclass
class EvalConfig:
    task_name: str
    example_id: str
    model_name: str


def evaluate():
    # parsing input
    os.chdir(os.path.dirname(__file__))
    parser = transformers.HfArgumentParser((EvalConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    set_seed(args.seed)

    # loading dataset
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # loading model
    model = AutoModelForCausalLM.from_pretrained(
        args.output_dir + "/tmp_ckpt",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto"
    )
    logger.info(f"Model: {model}")

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Just to suppress tokenizer's warning. Supposedly do nothing.
    tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info("Starting inferencer")
    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")
    # model = unwrapped_model.to("cuda:0")
    model.eval()

    all_questions = []
    all_results = []
    question_types = [
        "single_hop_efficacy",
        "multi_hop_efficacy",
        "single_hop_specificity",
        "multi_hop_specificity",
    ]
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
    
    raw_instance = io.load_json(f"data/dataset/raw/id2musique.json")[config.example_id]
    for question_type in question_types:
        questions = raw_instance[question_type]

        logging.info(f"Question type: {question_type}")
        all_questions.extend(questions)
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
    os.makedirs(f"{args.output_dir}/inference_results", exist_ok=True)
    all_results.to_excel(
        f"{args.output_dir}/inference_results/{raw_instance['id']}_inferencer_results.xlsx",
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
    evaluate()
