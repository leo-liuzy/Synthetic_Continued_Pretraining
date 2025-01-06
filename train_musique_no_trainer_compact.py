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
from data.cptdata import get_task_data_module
from knowledge_propagation.modules.inferencers import QAInferencer
from knowledge_propagation.utils import io, extractor

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
from torch.utils.data import Dataset
from accelerate import Accelerator, DistributedType, dispatch_model
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object

from pathlib import Path
import pdb

logger = get_logger(__name__)
from torch.cuda.amp import autocast


class MemmapDataset(Dataset):
    def __init__(self, block_size: int, token_ids, eos_token_id):
        logger.info(f"block_size: {block_size}")
        logger.info(f"len(token_ids): {len(token_ids)}")
        logger.info(f"eos_token_id: {eos_token_id}")
        self.block_size = block_size
        self.ids = token_ids
        self.eos_token_id = eos_token_id
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
        return dict(input_ids=torch.from_numpy(x_id).long(), labels=torch.from_numpy(x_id).long())


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
    # logger.info(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    # sleep(1000)
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
        set_seed(args.seed + accelerator.process_index)

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
    target_tokens = np.memmap(f"data/dataset/bins/{config.task_name}/{config.example_id}.bin", dtype=np.int32, mode="r")
    logger.info(f"total # tokens: {len(target_tokens)}")

    # args.gradient_accumulation_steps = max(1, args.gradient_accumulation_steps)
    # args.max_steps = 1 # debug
    if config.task_name == "musique":
        train = MemmapDataset(config.block_size, target_tokens, tokenizer.eos_token_id)
        val = None
        args.eval_strategy = "no"
    else:
        assert config.task_name == "musique_page"
        logger.info(f"Train dataset")
        train = MemmapDataset(config.block_size, target_tokens[: int(len(target_tokens) * 0.9)], tokenizer.eos_token_id)
        logger.info(f"Eval dataset")
        val = MemmapDataset(config.block_size, target_tokens[int(len(target_tokens) * 0.9) :], tokenizer.eos_token_id)
        args.eval_strategy = "epoch"
    logger.info(f"block size: {config.block_size}")
    logger.info(f"# train instances: {len(train)}")
    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
    )
    logger.info(f"Model: {model}")

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
    logger.info(f"per_device_train_batch_size: {args.per_device_train_batch_size}")
    logger.info(f"len(train_dataloader): {len(train_dataloader)}")
    eval_dataloader = (
        DataLoader(val, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size) if val else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    overrode_max_train_steps = False
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    num_update_steps_per_epoch = np.ceil(len(train_dataloader) / total_batch_size)
    if args.max_steps is None or args.max_steps < 0:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if not args.warmup_steps:
        args.warmup_steps = np.floor(args.max_steps * args.warmup_ratio)

    logger.info(f"warmup_steps: {args.warmup_steps}")
    logger.info(f"max_steps: {args.max_steps}")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # https://github.com/huggingface/transformers/issues/26827
        # num_warmup_steps=args.warmup_steps,
        # num_training_steps=args.max_steps,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_steps * accelerator.num_processes,
    )
    logger.info(f"len(train_dataloader) [before prepare()]: {len(train_dataloader)}")
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info(f"len(train_dataloader) [after prepare()]: {len(train_dataloader)}")
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
    # scaler = GradScaler()

    if eval_dataloader:
        model.eval()
        losses = []
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

        logger.info(
            f"Eval before training: perplexity: {perplexity} eval_loss: {eval_loss}",
        )

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                logger.info(f"[rank{accelerator.local_process_index}] loss: {loss.detach().float()}")

                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    grad_norm = (
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )
                        .detach()
                        .float()
                    )
                    logger.info(f"[rank{accelerator.local_process_index}] grad_norm: {grad_norm}")
                logger.info(f"[rank{accelerator.local_process_index}] lr: {lr_scheduler.get_last_lr()}")
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

            logger.info(
                f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}",
            )
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
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # accelerator.save_model(model, args.output_dir, max_shard_size="1GB", safe_serialization=True)
    unwrapped_model.save_pretrained(
        args.output_dir,
        is_main_process=accelerator.is_main_process,
        state_dict=accelerator.get_state_dict(model),
    )
    exit(0)

    if eval_dataloader:
        del losses, eval_loss
    model.zero_grad()
    optimizer.zero_grad()
    accelerator.clear(
        model,
        optimizer,
        lr_scheduler,
    )
    del optimizer, lr_scheduler

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # exit(0)
    logger.info("Starting inferencer")
    # model = unwrapped_model.to("cuda:0")
    model.eval()

    unwrapped_model = accelerator.unwrap_model(model)
    from collections import defaultdict

    tmp = defaultdict(list)
    for n, param in unwrapped_model.named_parameters():
        if param.grad is not None:
            tmp[str(param.grad.data.device) + "[grad]"].append(n)
        tmp[str(param.data.device)].append(n)
    print(tmp.keys())
    print(tmp)
    #     if param.grad is not None:
    #         param.grad.data = param.grad.data.cpu()
    #     param.data = param.data.cpu()
    # unwrapped_model = unwrapped_model.cpu().to("cuda:0")

    # unwrapped_model = unwrapped_model.cpu()

    # Clear any existing gradients
    # 2. Detach and move parameters to CPU first

    # for param in unwrapped_model.parameters():
    #     if param.grad is not None:
    #         param.grad.data = param.grad.data.cpu()
    #     param.data = param.data.cpu()
    # unwrapped_model = unwrapped_model.cpu().to("cuda:0")
    all_questions = []
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
    for question_type in question_types:
        questions = raw_instance[question_type]

        logging.info(f"Question type: {question_type}")
        all_questions.extend(questions)
        # inferencer = QAInferencer(
        #     cfg.evaluator.inferencers[0],
        #     cfg.seed,
        #     rag_model=None,
        #     queries=questions,
        # )
        # result_df = eval_inferencer(
        #     inferencer,
        #     model,
        #     tokenizer=tokenizer,
        #     generation_cfg=generation_config,
        # )
        # result_df.insert(0, "question_type", question_type)
        # result_df.insert(0, "id", raw_instance["id"])
        # all_results.append(result_df)
    inferencer = QAInferencer(
        cfg.evaluator.inferencers[0],
        cfg.seed,
        rag_model=None,
        queries=all_questions,
    )
    question = all_questions[0]
    context = inferencer.prepare_context(question)

    # inputs = tokenizer(context, return_tensors="pt")
    # inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    # generation_output = model.generate(
    #     **inputs,
    # )

    generator = pipeline("text-generation", model=unwrapped_model, tokenizer=tokenizer)
    generated_texts = generator(context, max_length=generation_config.max_new_tokens)

    # all_results = []

    set_seed(cfg.seed)
    logger.info("model: ", model)
    # logger.info("unwrapped_model: ", unwrapped_model)
    question = all_questions[0]
    # unwrapped_model.device = "cuda"
    # model = dispatch_model(model, device_map={"": accelerator.process_index})
    # context = inferencer.prepare_context(question)
    accelerator.wait_for_everyone()
    # generator = pipeline("text-generation", model=unwrapped_model, tokenizer=tokenizer)
    # generated_texts = generator(context, max_length=generation_config.max_new_tokens)
    # logger.info(f"generated_texts: {generated_texts}")
    # from accelerate import PartialState

    # distributed_state = PartialState()
    # from accelerate.inference import prepare_pippy

    # model = prepare_pippy(
    #     model,
    #     example_kwargs=dict(
    #         generation_config=generation_config,
    #         pad_token_id=tokenizer.pad_token_id,
    #         return_dict_in_generate=True,
    #     ),
    # )
    with accelerator.split_between_processes(all_questions) as questions:
        answered_qs = []

        logger.info(f"[rank{accelerator.local_process_index}] len(questions): {len(questions)}")
        with torch.no_grad():
            for question in questions[:1]:
                context = inferencer.prepare_context(question)
                inputs = tokenizer(context, return_tensors="pt")
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                generation_output = model.generate(
                    **inputs,
                )
                # generated_texts = generator(context, max_length=generation_config.max_new_tokens)
                logger.info(f"generated_texts: {generation_output}")

    #         for g_i, generated_text in enumerate(generated_texts):
    #             answered_q = deepcopy(question)
    #             answered_q["predicted_answer_idx"] = g_i
    #             answered_q["predicted_answer"] = extractor.urial_ans_extractor(generated_text)
    #             answered_q["predicted_response"] = generated_text
    #             answered_qs.append(answered_q)

    # answered_qs_gathered = gather_object(answered_qs)
    # answered_qs_gathered = pd.concat(answered_qs_gathered)
    # if inferencer.has_answer:
    #     answered_qs_gathered = inferencer.score_df(answered_qs_gathered)

    # answered_qs_gathered.to_excel(
    #     f"{args.output_dir}/{raw_instance['id']}_inferencer_results.xlsx",
    #     index=False,
    # )
    # metrics = ["rouge1", "llm_accuracy"]
    # multi_level_averaging = ["question_type", "id", "question"]
    # result_df = macro_averaging(answered_qs_gathered, metrics, multi_level_averaging).round(2)
    # q_cat_dtype = pd.CategoricalDtype(
    #     categories=question_types,
    #     ordered=True,
    # )

    # result_df["question_type"] = result_df["question_type"].astype(q_cat_dtype)
    # logger.info(result_df.sort_values(by=["question_type"], inplace=False))


if __name__ == "__main__":
    train()
