import asyncio
import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

DEFAULT_MAX_RETRY = 5
DEFAULT_SLEEP_TIME = 2
DEFAULT_MAX_CONCURRENCY = 10
DEFAULT_STATUS_FREQUENCY = 20
DEFAULT_OPENAI_MODEL = 'gpt-4o-mini-2024-07-18'

ORG_TAUR = 'TAUR'
ORG_INDEED = 'INDEED'


def configure_client(org: str = ORG_INDEED) -> OpenAI:
    return _configure_client(OpenAI, org)


def configure_async_client(org: str = ORG_INDEED) -> AsyncOpenAI:
    return _configure_client(AsyncOpenAI, org)


def _configure_client(cls, org):
    org = org.upper()
    if org == ORG_TAUR:
        return cls(
            api_key=os.environ['OPENAI_API_KEY_PERSONAL'],
            organization=os.environ['OPENAI_ORGANIZATION_TAUR'],
        )
    elif org == ORG_INDEED:
        return cls(
            base_url=os.environ['LLM_PROXY_QA_BASE_URL_OAI'],
            api_key=os.environ['LLM_PROXY_API_KEY_QA_PLM'],
            organization=os.environ['OPENAI_ORGANIZATION_INDEED'],
        )

    raise ValueError(f'Unknown org: "{org}" (Valid options are "{ORG_TAUR}" or "{ORG_INDEED}")')


def get_rate_limit_sleep_time(error: RateLimitError) -> int:
    words = str(error).split(' ')
    try:
        return int(words[words.index('after') + 1])
    except ValueError:
        return DEFAULT_SLEEP_TIME


def send_chat_request(
        client: OpenAI,
        messages: list[dict[str, str]],
        max_retry: int = DEFAULT_MAX_RETRY,
        model: str = DEFAULT_OPENAI_MODEL,
        **openai_params: dict[str, Any],
) -> ChatCompletion:
    for i in range(max_retry):
        try:
            completion: ChatCompletion = client.chat.completions.create(
                messages=messages,
                model=model,
                **openai_params,
            )
            completion = completion.model_dump()
            return {'ok': True, 'completion': completion}
        except RateLimitError as e:
            if i == max_retry - 1:
                return {'ok': False, 'error': str(e)}

            sleep_time = get_rate_limit_sleep_time(e)
            time.sleep(sleep_time)

    raise RuntimeError('Should not see me!')


async def send_async_chat_request(
        client: AsyncOpenAI,
        messages: list[dict[str, str]],
        max_retry: int = DEFAULT_MAX_RETRY,
        model: str = DEFAULT_OPENAI_MODEL,
        **openai_params: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    for i in range(max_retry):
        try:
            completion: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                model=model,
                **openai_params,
            )
            completion = completion.model_dump()
            return {'ok': True, 'completion': completion}
        except RateLimitError as e:
            if i == max_retry - 1:
                return {'ok': False, 'error': str(e)}

            sleep_time = get_rate_limit_sleep_time(e)
            time.sleep(sleep_time)
        except (Exception, KeyboardInterrupt) as e:
            return {'ok': False, 'error': str(e)}

    raise RuntimeError('Should not see me!')


def send_async_chat_request_batch(
        client: AsyncOpenAI,
        data: list[list[dict[str, str]]],
        max_retry: int = DEFAULT_MAX_RETRY,
        max_concurrency: int | None = DEFAULT_MAX_CONCURRENCY,
        status_frequency: int | None = DEFAULT_STATUS_FREQUENCY,
        model: str = DEFAULT_OPENAI_MODEL,
        **openai_params: dict[str, Any],
) -> list[ChatCompletion]:
    coroutine = _send_async_chat_request_batch(
        client=client,
        data=data,
        max_retry=max_retry,
        max_concurrency=max_concurrency,
        status_frequency=status_frequency,
        model=model,
        **openai_params,
    )
    result = asyncio.run(coroutine)
    return result


async def _send_async_chat_request_batch(
        client: AsyncOpenAI,
        data: list[list[dict[str, str]]],
        max_retry: int = DEFAULT_MAX_RETRY,
        max_concurrency: int | None = DEFAULT_MAX_CONCURRENCY,
        status_frequency: int | None = DEFAULT_STATUS_FREQUENCY,
        model: str = DEFAULT_OPENAI_MODEL,
        **openai_params: dict[str, Any],
) -> list[dict[str, Any]]:
    total = len(data)
    coroutines = (
        send_async_chat_request(
            client,
            messages=messages,
            max_retry=max_retry,
            model=model,
            **openai_params,

        ) for messages in data
    )

    run_semaphore: asyncio.Semaphore | None = None
    if max_concurrency is not None:
        run_semaphore = asyncio.Semaphore(max_concurrency)

    async def wrapper(i: int, coroutine):
        if run_semaphore is not None:
            async with run_semaphore:
                result = await coroutine
        else:
            result = await coroutine

        if status_frequency is not None and (i + 1) % status_frequency == 0:
            print(f'[{(i+1)/total:.2%}] completed {i+1} of {total}')

        return result

    wrapped_coroutines = (wrapper(i, c) for i, c in enumerate(coroutines))
    outputs = await asyncio.gather(*wrapped_coroutines)
    return outputs
