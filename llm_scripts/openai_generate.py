import traceback
from pathlib import Path

from llm_scripts import openai_api
from llm_scripts.chat import preprocess_messages
from llm_scripts.openai_generate_config import OpenAIGenerateConfig
from llm_scripts.utils import jsonlines


def run(config: OpenAIGenerateConfig):
    print('loading data')
    raw_data = jsonlines.load(config.input_file)
    message_data = [
        preprocess_messages(
            obj['messages'],
            strip_trailing_assistant_messages=config.strip_trailing_assistant_messages,
        )
        for obj in raw_data
    ]

    client = openai_api.configure_async_client(org=config.org)
    print('begin generation')
    try:
        outputs = openai_api.send_async_chat_request_batch(
            client=client,
            data=message_data,
            **config.openai_params,
        )
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        pass

    print(f'generated {len(outputs)} outputs')
    num_failed = sum(1 for output in outputs if not output['ok'])
    total = len(raw_data)
    rate = num_failed / total
    print(f'failures: {num_failed} of {total} ({rate:.3%})')

    for output in outputs:
        if output['ok']:
            choices = output['completion']['choices']
            responses = [choice['message']['content'] for choice in choices]
            output['responses'] = responses
        else:
            output['responses'] = []

    return outputs


def main(config: OpenAIGenerateConfig):
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        raise ValueError(f'output_dir already exists: "{output_dir}"')

    output_dir.mkdir(parents=True, exist_ok=True)
    config.dump(file=output_dir / 'generation_config.json')

    outputs = run(config)
    jsonlines.dump(outputs, file=output_dir / 'generation_output.jsonl')


if __name__ == '__main__':
    config = OpenAIGenerateConfig.load()
    main(config)
