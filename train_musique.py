from dataclasses import dataclass, field, asdict
from typing import Optional, List
import transformers
from transformers import AutoTokenizer, GenerationConfig
import os
import warnings
from data.cptdata import MemmapDataset, _MemmapDataset
import hydra

warnings.filterwarnings("ignore", category=FutureWarning)
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from data.cptdata import get_task_data_module
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.utils import io

from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

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
    token_ids = np.memmap(f"data/dataset/bins/{config.task_name}/{config.example_id}.bin", dtype=np.int32, mode="r")
    
    args.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    # args.max_steps = 1 # debug
    if config.task_name == "musique":
        train = MemmapDataset(config.block_size, token_ids, tokenizer.eos_token_id)
        data_module = dict(train_dataset=train, eval_dataset=None)
        args.eval_strategy = "no"
    else:
        assert config.task_name == "musique_page"
        train = MemmapDataset(config.block_size, token_ids[:int(len(token_ids) * 0.9)], tokenizer.eos_token_id)
        val = train = MemmapDataset(config.block_size, token_ids[int(len(token_ids) * 0.9):], tokenizer.eos_token_id)
        data_module = dict(train_dataset=train, eval_dataset=val)
        args.eval_strategy = "epoch"

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
    )
    
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Just to suppress tokenizer's warning. Supposedly do nothing.
    tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token

    model.resize_token_embeddings(len(tokenizer))
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

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")
    # setting up trainer
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    # trainer.model.save_pretrained(save_directory=args.output_dir)
    # trainer.save_model(output_dir=args.output_dir)
    # trainer.save_model()
    trainer.accelerator.wait_for_everyone()

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
            trainer.model,
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
    # logger.info(result_df.sort_values(by=["question_type"], inplace=False))

if __name__ == "__main__":
    train()
