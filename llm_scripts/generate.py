import traceback
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_scripts.chat import preprocess_messages
from llm_scripts.generate_config import GenerateConfig
from llm_scripts.utils import BatchIter, jsonlines, move_to_device


def run(config: GenerateConfig):
    print('loading data')
    data = jsonlines.load(config.input_file)

    print('loading tokenizer')
    if config.checkpoint_dir:
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    if config.chat_template:
        tokenizer.chat_template = config.chat_template

    tokenizer.padding_side = 'left'
    if config.eos_token is None:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id, = tokenizer.convert_tokens_to_ids([config.eos_token])

    if config.pad_token:
        tokenizer.pad_token = config.pad_token
        if tokenizer.pad_token_id is None:
            raise ValueError(f'requested pad token "{config.pad_token}" is not in vocab')

    if not tokenizer.pad_token:
        raise ValueError('missing pad_token')

    print('creating prompts')
    prompts = []
    for d in data:
        messages = d['messages']
        messages = preprocess_messages(
            messages,
            strip_trailing_assistant_messages=config.strip_trailing_assistant_messages,
        )

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=config.add_generation_prompt,
        )
        if config.add_generation_prefix:
            prompt += config.add_generation_prefix

        prompts.append(prompt)

    print('loading model')
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map='cpu',
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    if config.checkpoint_dir:
        peft_model = PeftModel.from_pretrained(model, config.checkpoint_dir, device_map='cpu')
        print('merging PEFT parameters')
        model = peft_model.merge_and_unload()

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print('moving model to GPU')
    model = model.to(config.device)

    print('begin generation')
    outputs = []
    output_keys_to_keep = config.output_keys_to_keep
    batches = BatchIter(prompts, config.batch_size)

    with torch.no_grad():
        try:
            for batch_index, prompt_batch in enumerate(tqdm(batches)):
                input_batch = tokenizer(
                    prompt_batch,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                )
                input_length = input_batch['input_ids'].shape[1]

                output_batch = model.generate(
                    **input_batch.to(model.device),
                    eos_token_id=eos_token_id,
                    stop_strings=config.stop_strings,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.temperature is not None,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    num_return_sequences=config.num_return_sequences,
                    output_attentions=config.output_attentions,
                    output_hidden_states=config.output_hidden_states,
                    output_logits=True,  # Always output logits to compute transition scores
                    output_scores=config.output_scores,
                    return_dict_in_generate=True,
                    tokenizer=tokenizer,
                )
                output_batch = move_to_device(output_batch, 'cpu')

                output_sequences = output_batch['sequences']
                output_sequences = output_sequences[:, input_length:]
                output_sequences = output_sequences.reshape(len(prompt_batch), config.num_return_sequences, -1)

                transition_scores = model.compute_transition_scores(
                    output_batch['sequences'],
                    output_batch['logits'],
                    normalize_logits=True,
                )
                transition_scores = transition_scores.reshape(
                    len(prompt_batch),
                    config.num_return_sequences,
                    -1,
                )

                output_responses = [
                    [tokenizer.decode(x, skip_special_tokens=False).strip() for x in xs]
                    for xs in output_sequences
                ]

                elements = prompt_batch, output_responses, output_sequences.tolist(), transition_scores.tolist()
                for prompt, responses, tokens, scores in zip(*elements):
                    outputs.append({'prompt': prompt, 'responses': responses, 'tokens': tokens, 'scores': scores})

                if output_keys_to_keep:
                    d = {}
                    for k, v in output_batch.items():
                        if k not in output_keys_to_keep:
                            continue

                        if torch.is_tensor(v):
                            d[k] = v.cpu()
                        elif isinstance(v, (list, tuple)):
                            d[k] = [e.cpu() for e in v]
                        else:
                            raise ValueError(f'unknown batch item with key: "{k}" and type: {type(v)}')

                    file = Path(config.output_dir, 'batch_data', f'batch_{batch_index:02d}.npz')
                    file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(file, **d)

        except (Exception, KeyboardInterrupt):
            print(traceback.format_exc())
            pass

    print(f'generated {len(outputs)} outputs')
    return outputs


def main(config: GenerateConfig):
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        raise ValueError(f'output_dir already exists: "{output_dir}"')

    output_dir.mkdir(parents=True, exist_ok=True)
    config.dump(file=output_dir / 'generation_config.json')

    outputs = run(config)
    jsonlines.dump(outputs, file=output_dir / 'generation_output.jsonl')


if __name__ == '__main__':
    config = GenerateConfig.load()
    main(config)
