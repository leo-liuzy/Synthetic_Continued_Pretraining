# LLM Scripts

Some useful reusable scripts.
Everything is designed around the following standards.

1. **OpenAI Chat Format:** JSON lines input and output files, each object should have a "messages" key.
2. **Hugging Chat Templates:** String formatting uses the chat templates library.


Example SFT usage

```sh
accelerate launch run_script.py sft \
--train_file path/to/train.jsonl \
--eval_file path/to/eval.jsonl \
--output_dir ./runs/test \
--model_id meta-llama/Meta-Llama-3.1-8B-Instruct
```

Tested on p4d.24xlarge with the following package versions

```
accelerate: 0.33.0
bitsandbytes: 0.43.2
datasets: 2.20.0
flash_attn: 2.6.3
peft version: 0.12.0
torch: 2.4.0+cu121
transformers: 4.43.3
trl: 0.9.6
```

You'll also need to install tensorboard (or disable tensorboard logging).
There's probably some other things I forgot too 🐝

```sh
pip install tensorboard
```

## Other Examples

Generate scores for all logits (teacher forcing style)

```shell
python run_script.py generate \
--input_file ./qampari-small.jsonl \
--output_dir qampari-small-output \
--model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_scores True \
--output_logits True
```

Output scores for all logits (teacher forcing style)

```shell
python run_script.py score \
--input_file data/inputs/wikipedia-dev.chat.idk.jsonl \
--output_dir output-scores/wiki-10k-3shot/llama-3-min_conf_em-7/checkpoint-298/wikipedia-dev.chat.idk \
--model_id meta-llama/Meta-Llama-3-8B \
--checkpoint_dir runs/wiki-10k-3shot/llama-3-min_conf_em-7/checkpoint-298
```



## Random Stuff Ignore

```python
# Number of params when using the following settings

model_id = 'meta-llama/Meta-Llama-3-8B'
lora_target_modules = [
    'up_proj',
    'down_proj',
    'k_proj',
    'q_proj',
    'v_proj',
]


from safetensors import safe_open

tensors = {}
with safe_open('./runs/wiki-10k-3shot/llama-3-censor-min_conf_em-7/checkpoint-196/adapter_model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(f'{sum(v.numel() for v in tensors.values()):,d}')
# Output: 1,079,115,776
```
