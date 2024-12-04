from dataclasses import dataclass, field, asdict
from typing import Optional, List
import transformers
from transformers import AutoTokenizer, GenerationConfig
import os
import warnings
from data.cptdata import MemmapDataset, _MemmapDataset
import hydra
import math
import gc
from time import sleep
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
from tqdm.auto import tqdm
from data.cptdata import get_task_data_module
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.utils import io

from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

from experiments.musique.inference_only import eval_inferencer, macro_averaging
import torch
from torch.utils.data import DataLoader

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import datasets
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from pathlib import Path

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


def init_accelerator(args):
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    return accelerator


def train():
    # parsing input

    os.chdir(os.path.dirname(__file__))
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")
    
    accelerator = init_accelerator(args)
    
    # loading dataset
    # data_module = get_task_data_module(**asdict(config))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    token_ids = np.memmap(f"data/dataset/bins/{config.task_name}/{config.example_id}.bin", dtype=np.int32, mode="r")
    
    args.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    # args.max_steps = 1 # debug
    if config.task_name == "musique":
        train = MemmapDataset(config.block_size, token_ids, tokenizer.eos_token_id)
        val = None
        args.eval_strategy = "no"
    else:
        assert config.task_name == "musique_page"
        train = MemmapDataset(config.block_size, token_ids[:int(len(token_ids) * 0.9)], tokenizer.eos_token_id)
        val = train = MemmapDataset(config.block_size, token_ids[int(len(token_ids) * 0.9):], tokenizer.eos_token_id)
        args.eval_strategy = "epoch"

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
    )
    
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # Just to suppress tokenizer's warning. Supposedly do nothing.
    tokenizer.sep_token = tokenizer.cls_token = tokenizer.mask_token = tokenizer.pad_token

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

    with hydra.initialize(config_path="../KE-by-CP/configs", version_base=None):
        cfg = hydra.compose(config_name="fft.yaml")
    # setting up trainer
    
    
    
    
    train_dataloader = DataLoader(
        train, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        val, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    ) if val else None
    # trainer.train()
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = np.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps is None or args.max_steps < 0:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if not args.warmup_steps:
        args.warmup_steps = np.floor(args.max_steps * args.warmup_ratio)
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # https://github.com/huggingface/transformers/issues/26827
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_steps:
                break

        model.eval()
        losses = []
        log_info = {
            "train_loss": total_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        }
        if eval_dataloader:
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
            log_info.update(
                {
                    "eval_ppl": perplexity,
                    "eval_loss": eval_loss,
                }
            )
        accelerator.log(
            log_info,
            step=completed_steps,
        )
    del loss
    if eval_dataloader:
        del losses, eval_loss
    
    accelerator.clear(optimizer, lr_scheduler,)
    # sleep(100)
    gc.collect()
    logger.info("Starting inferencer")

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
    # logger.info(result_df.sort_values(by=["question_type"], inplace=False))

if __name__ == "__main__":
    train()
