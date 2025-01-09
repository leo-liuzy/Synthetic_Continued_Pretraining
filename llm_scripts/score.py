import traceback
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_scripts.score_config import ScoreConfig
from llm_scripts.utils import BatchIter, jsonlines


def get_tokens_and_logprobs(model, tokenizer, prompts: list[str]):
    """Compute the predicted log probs for a sequence of tokens.

    Mostly taken from https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
    """  # noqa
    input_batch = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    output = model(**input_batch.to(model.device))
    logprob_mat = torch.log_softmax(output.logits, dim=-1).detach().cpu()

    # Collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    logprob_mat = logprob_mat[:, :-1, :]
    input_id_mat = input_batch.input_ids[:, 1:].cpu()
    gen_logprobs = torch.gather(logprob_mat, 2, input_id_mat[:, :, None]).squeeze(-1)

    results = []
    for prompt, input_ids, input_logprobs in zip(prompts, input_id_mat, gen_logprobs):
        tokens = []
        token_ids = []
        logprobs = []
        for token_id, logprob in zip(input_ids, input_logprobs):
            tokens.append(tokenizer.decode(token_id))
            token_ids.append(token_id.item())
            logprobs.append(logprob.item())

        d = {
            'prompt': prompt,
            'tokens': tokens,
            'token_ids': token_ids,
            'logprobs': logprobs,
        }
        results.append(d)

    return results


def run(config: ScoreConfig):
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
    if config.pad_token:
        tokenizer.pad_token = config.pad_token
        if tokenizer.pad_token_id is None:
            raise ValueError(f'requested pad token "{config.pad_token}" is not in vocab')

    print('creating prompts')
    prompts = []
    for d in data:
        messages = d['messages']
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
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
    model = model.to('cuda')

    print('begin generation')
    outputs = []
    batches = BatchIter(prompts, config.batch_size)

    with torch.no_grad():
        try:
            for prompt_batch in tqdm(batches):
                output_batch = get_tokens_and_logprobs(model, tokenizer, prompt_batch)
                outputs.extend(output_batch)

        except Exception:
            print(traceback.format_exc())
            pass

    print(f'generated {len(outputs)} outputs')
    return outputs


def main():
    config = ScoreConfig.load()
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        raise ValueError(f'output_dir already exists: "{output_dir}"')

    output_dir.mkdir(parents=True, exist_ok=True)
    config.dump(file=output_dir / 'score_config.json')

    outputs = run(config)
    jsonlines.dump(outputs, file=output_dir / 'score_output.jsonl')


if __name__ == '__main__':
    main()
