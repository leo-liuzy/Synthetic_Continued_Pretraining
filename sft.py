import time
from pathlib import Path

import accelerate
import bitsandbytes
import datasets
import flash_attn
import peft
import torch
import transformers
import trl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from llm_scripts import chat, env
from llm_scripts.sft_config import SFTConfig
from llm_scripts.utils import demo


def load_tokenizer(config: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if env.is_primary():
        print("initial chat_template:", tokenizer.chat_template)
        print("initial special_tokens_map:", tokenizer.special_tokens_map)

    tokenizer.model_max_length = config.max_length

    setup_chat_model = config.setup_chat_model or tokenizer.chat_template is None
    if setup_chat_model:
        if env.is_primary():
            print("configuring tokenizer for chat")

        chat.configure_tokenizer_for_chat(
            tokenizer,
            chat_template=config.chat_template,
        )

    if tokenizer.pad_token is None:
        if config.pad_token:
            tokenizer.pad_token = config.pad_token
        else:
            if env.is_primary():
                print("!!! warning: setting pad_token to eos_token")

            tokenizer.pad_token = tokenizer.eos_token

    elif config.pad_token and env.is_primary():
        print("!!! warning: tokenizer already has a pad_token, " f"ignoring config.pad_token={repr(config.pad_token)}")

    assert tokenizer.pad_token

    if env.is_primary():
        messages = demo.messages
        s1 = tokenizer.apply_chat_template(messages, tokenize=False)
        s2 = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        print("Tokenizer info")
        print("  chat_template:", tokenizer.chat_template)
        print("  special_tokens_map:", tokenizer.special_tokens_map)
        print('"""' + s1 + '"""')
        print("-----")
        print('"""' + s2 + '"""')

    return tokenizer


def preprocess_dataset(
    config: SFTConfig,
    dataset: list[dict],
    tokenizer: PreTrainedTokenizerBase,
) -> datasets.Dataset:
    def generate_dataset():
        total = skipped = 0
        for d in dataset:
            messages = d["messages"]
            segments = chat.create_train_segments(
                messages,
                tokenizer=tokenizer,
                eot_token=config.eot_token,
            )
            input_ids, labels = chat.convert_to_input_ids_and_labels(
                segments,
                tokenizer=tokenizer,
                label_pad_token_id=config.label_pad_token_id,
                mask_input_segments=config.mask_input_segments,
            )
            total += 1
            if len(input_ids) > config.max_length:
                skipped += 1
                continue

            yield {"input_ids": input_ids, "labels": labels}

        if skipped > 0 and env.is_primary():
            print(f"Skipped {skipped} of {total} ({skipped / total:.2%})")

    features = datasets.Features(
        input_ids=[datasets.Value(dtype="int64")],
        labels=[datasets.Value(dtype="int64")],
    )
    dataset = datasets.Dataset.from_generator(generate_dataset, features=features)
    return dataset


def load_dataset(config: SFTConfig, tokenizer: PreTrainedTokenizerBase):
    train_dataset = datasets.load_dataset(
        "json",
        split="train",
        data_files=config.train_file,
    )

    eval_dataset = None
    if config.eval_file:
        eval_dataset = datasets.load_dataset(
            "json",
            split="train",
            data_files=config.eval_file,
        )
    elif config.eval_split_size is not None:
        test_size = config.eval_split_size
        if test_size > 1:
            test_size = int(test_size)

        splits = train_dataset.train_test_split(test_size=test_size, shuffle=False)
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

    train_dataset = preprocess_dataset(config, train_dataset, tokenizer=tokenizer)
    if eval_dataset is not None:
        eval_dataset = preprocess_dataset(config, eval_dataset, tokenizer=tokenizer)

    dataset = {"train": train_dataset, "eval": eval_dataset}
    if env.is_primary():
        for k, v in dataset.items():
            v = v or []
            print(f"number of {k} instances:", len(v))

    return dataset


def train(config: SFTConfig):
    tokenizer = load_tokenizer(config)
    dataset = load_dataset(config, tokenizer=tokenizer)

    # Load model and handle quantization

    quantize_base_model = config.load_in_4bit or config.load_in_8bit
    quantization_config = None
    if quantize_base_model:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_8bit=config.load_in_8bit,
            load_in_4bit=config.load_in_4bit,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map={"": env.rank()},
        attn_implementation=config.attn_implementation,
        torch_dtype=config.torch_dtype,
        quantization_config=quantization_config,
    )
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=config.pad_to_multiple_of,
    )
    if quantize_base_model:
        model = peft.prepare_model_for_kbit_training(model)

    # Setup LoRA

    lora_config = peft.LoraConfig(
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=config.lora_task_type,
        target_modules=config.lora_target_modules,
    )
    model = peft.get_peft_model(model, lora_config)
    if env.is_primary():
        model.print_trainable_parameters()

    # Setup trainer

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=config.padding,
        pad_to_multiple_of=config.pad_to_multiple_of,
        label_pad_token_id=config.label_pad_token_id,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        num_train_epochs=config.num_train_epochs,
        optim=config.optim,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        neftune_noise_alpha=config.neftune_noise_alpha,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        auto_find_batch_size=config.auto_find_batch_size,
        group_by_length=config.group_by_length,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        logging_strategy=config.logging_strategy,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        ddp_backend=config.ddp_backend,
        report_to=config.report_to,
        remove_unused_columns=config.remove_unused_columns,
        load_best_model_at_end=config.load_best_model_at_end,
        gradient_checkpointing=config.gradient_checkpointing,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        save_on_each_node=config.save_on_each_node,
        disable_tqdm=not env.is_primary(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )

    print(f"[GPU {env.rank()}] starting training")
    start_time = time.time()
    trainer.train()
    stop_time = time.time()
    total_time = (stop_time - start_time) / 60
    print(f"[GPU {env.rank()}] completed training (time: {total_time:.2f} minutes)")


def main():
    if env.is_primary():
        print(f"accelerate version: {accelerate.__version__}")
        print(f"bitsandbytes version: {bitsandbytes.__version__}")
        print(f"datasets version: {datasets.__version__}")
        print(f"flash_attn version: {flash_attn.__version__}")
        print(f"peft version: {peft.__version__}")
        print(f"torch version: {torch.__version__}")
        print(f"transformers version: {transformers.__version__}")
        print(f"trl version: {trl.__version__}")

    config = SFTConfig.load()
    if env.is_primary():
        print(config)
        config.dump(Path(config.output_dir, "sft_config.json"))

    train(config)


if __name__ == "__main__":
    main()
