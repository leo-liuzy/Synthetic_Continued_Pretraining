"""Chat template utilities and definitions."""

ROLE_SYSTEM = 'system'
ROLE_USER = 'user'
ROLE_ASSISTANT = 'assistant'

CHAT = """
{{ bos_token }}
{% for message in messages %}
    {{ '<|header_start|>' }}{{ message['role'].lower() }}
    {{ '<|message_start|>\\n' }}{{ message['content'].strip() }}{{ '<|end|>' }}
    {% if not loop.last %}
        {{ '\\n' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '\\n<|header_start|>assistant<|message_start|>\\n' }}
{% endif %}"""

CHAT_PAD_TOKEN = '<|pad|>'
CHAT_HEADER_TOKEN = '<|header_start|>'
CHAT_MESSAGE_TOKEN = '<|message_start|>'
CHAT_END_TOKEN = '<|end|>'

CHAT_SPECIAL_TOKENS_TO_ADD = {
    'pad_token': CHAT_PAD_TOKEN,
    'additional_special_tokens': [
        CHAT_HEADER_TOKEN,
        CHAT_MESSAGE_TOKEN,
        CHAT_END_TOKEN,
    ],
}

#
# Old version which doesn't tokenize well
#
# SIMPLE = """
# {{ bos_token }}
# {% for message in messages %}
#     {{ '<message role="' }}{{ message['role'].lower() }}{{ '">\\n' }}
#     {{ message['content'].strip() }}
#     {{ '\\n</message>' }}
#     {% if not loop.last %}
#         {{ '\\n' }}
#     {% endif %}
# {% endfor %}
# {% if add_generation_prompt %}
#     {{ '\\n<message role="assistant">\\n' }}
# {% endif %}"""

SIMPLE = """
{{ bos_token }}
{% for message in messages %}
    {{ '<message role="' }}{{ message['role'].lower() }}{{ '">\\n' }}
    {{ message['content'].strip() }}
    {{ '</message>' }}
    {% if not loop.last %}
        {{ '\\n' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ '\\n<message role="assistant">\\n' }}
{% endif %}"""


URIAL = """
{% set templates = namespace() %}
{% set templates.user = '# Query:\\n```\\n%s\\n```\\n\\n' %}
{% set templates.assistant = '# Answer:\\n```\\n%s\\n```\\n\\n' %}
{% if messages[0].role == 'system' %}
    {% set system = messages[0]['content'] %}
    {% set messages = messages[1:] %}
    {{ '# Instructions:\\n\\n%s\\n\\n'|format(system) }}
{% endif %}
{% for message in messages %}
    {% set role = message['role'] %}
    {% set content = message['content'].strip() %}
    {% set template = templates|attr(role) %}
    {% if template is undefined %}
        {{ raise_exception('Message role must be "user" or "assistant", got "' ~ message['role'] ~ '"') }}
    {% endif %}
    {{ template|format(content) }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '# Answer:\\n```\\n' }}
{% endif %}"""


URIAL_CHAT = """
{% set template = '# %s:\\n```\\n%s\\n```\\n\\n' %}
{% for message in messages %}
    {{ template|format(message['role'].strip().upper(), message['content'].strip()) }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '# ASSISTANT:\\n```\\n' }}
{% endif %}"""


CHAT_TEMPLATE_MAP = {
    'chat': CHAT,
    'simple': SIMPLE,
    'urial': URIAL,
    'urial_chat': URIAL_CHAT,
}

SPECIAL_TOKENS_MAP = {
    'chat': CHAT_SPECIAL_TOKENS_TO_ADD,
}

EOT_TOKEN_MAP = {
    'chat': CHAT_END_TOKEN,
    'simple': '\n</message>',
    'urial': '\n```',
    'urial_chat': '\n```',
}


def load_chat_template(value_or_key_or_path: str, strip_whitespace: bool = True) -> str:
    chat_template = CHAT_TEMPLATE_MAP.get(value_or_key_or_path, None)
    if not chat_template:
        try:
            with open(value_or_key_or_path) as fp:
                chat_template = fp.read()
        except FileNotFoundError:
            chat_template = value_or_key_or_path

    if strip_whitespace:
        chat_template = ''.join(line.strip() for line in chat_template.split('\n'))

    return chat_template


def configure_tokenizer_for_chat(
        tokenizer,
        chat_template: str | None = None,
        special_tokens_to_add: dict | None = None,
):
    chat_template = chat_template or 'chat'
    _chat_template = load_chat_template(chat_template)
    tokenizer.chat_template = _chat_template

    if not special_tokens_to_add:
        special_tokens_to_add = dict(SPECIAL_TOKENS_MAP.get(chat_template, {}))

    if special_tokens_to_add:
        tokenizer.add_special_tokens(special_tokens_to_add)


def strip_trailing_messages(
        messages: list[dict[str, str]],
        role: str = ROLE_ASSISTANT,
) -> list[dict[str, str]]:
    messages = list(messages)
    while messages:
        if messages[-1]['role'] == role:
            messages.pop()
        else:
            break

    return messages


def preprocess_messages(
        messages: list[dict[str, str]],
        strip_trailing_assistant_messages: bool = False,
) -> list[dict[str, str]]:
    first_message = messages[0]
    assert isinstance(first_message, dict)

    if strip_trailing_assistant_messages:
        messages = strip_trailing_messages(messages=messages, role=ROLE_ASSISTANT)

    if not messages:
        raise ValueError('no messages')

    return messages


def create_train_segments(
        messages: list[dict[str, str]],
        tokenizer,
        output_role: str = 'assistant',
        eot_token: str | None = None,
) -> list[str]:
    """Convert a list of messages to formatted input/output segments.

    To be compatible with this function ``chat_template`` must
    support the keyword argument ``messages``.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        The list of messages to format.
        Assuming OpenAI Chat format, each message will typically
        have the keys ``'role'`` and ``'content'``. The optional
        key ``'template'`` can be included to specify a message
        specific template. If ``'template'`` is not present,
        ``'role'`` is used instead.
    tokenizer : PreTrainedTokenizer
        The tokenizer to use for formatting messages.
        Must have a ``chat_template`` assigned.
    output_role : str
        The message role corresponding to output segments.
    eot_token : str
        An optional end of turn token. If given will be used instead
        of ``tokenizer.eos_token`` to locate segment boundaries.

    Returns
    -------
    list[str]
        A list of alternating input/output segments.
        Input segments may correspond to multiple messages,
        but each output_segments corresponds to exactly one message.

    Raises
    ------
    ValueError
        If ``"eos_token"`` is not in ``special_token_map``
        and ``include_eos_token`` is ``True``.

        If ``include_eos_token`` is ``True`` and the eos
        token is not found in the text preceding output content.
    """
    eot_token = EOT_TOKEN_MAP.get(eot_token, eot_token)
    eot_token = eot_token or tokenizer.special_tokens_map.get('eos_token', '')
    if not eot_token:
        raise ValueError('missing eot token')

    # Create a copy of messages where message[i]['content']
    # has been replaced with the placeholder '{0[i][content]}'.
    # Placeholders are used to identify output segments
    # and when substituting formatted message content into the
    # partially formatted text segments.
    def _create_placeholder(i: int) -> str:
        """Create a format string placeholder."""
        return '{{0[{i}][content]}}'.format(i=i)

    mock_messages = [
        {**message, 'content': _create_placeholder(i)}
        for i, message in enumerate(messages)
    ]

    # Render the conversation with placeholder content.
    remain = tokenizer.apply_chat_template(
        mock_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Loop through the messages.
    # Whenever messages[i]['role'] matches the argument output_role:
    # 1. The remaining text is split on the placeholder '{0[i][content]}'
    # 2. The preceding content is fully formatted and appended to segments
    # 3. (Optional) Perform eos_token cleanup and add eos_token to output
    # 4. The output is appended to segments
    segments: list[str] = []
    for i, message in enumerate(messages):
        if message['role'] != output_role:
            continue

        placeholder = mock_messages[i]['content']
        prefix, remain = remain.split(placeholder)
        prefix = prefix.rstrip(' ')

        output_segment = str(message['content']).strip()
        output_segment += eot_token

        if not remain.startswith(eot_token):
            raise ValueError(
                f'eot token ({repr(eot_token)}) not found after "{output_role}" content.'
            )

        remain = remain[len(eot_token):]

        # stripped_remain = remain.lstrip()
        # if not stripped_remain.startswith(eot_token):
        #     raise ValueError(
        #         f'eot token ("{eot_token}") not found after "{output_role}" content.'
        #     )

        # remain = stripped_remain[len(eot_token):]
        input_segment = prefix.format(messages)
        segments.append(input_segment)
        segments.append(output_segment)

    # Handle any leftover text.
    # This shouldn't typically happen as training conversations
    # normally end with an output (assistant) message.
    if remain:
        final_segment = remain.format(messages)
        segments.append(final_segment)

    return segments


def convert_to_input_ids_and_labels(
        segments: list[str],
        tokenizer,
        label_pad_token_id: int = -100,
        mask_input_segments: bool = True,
) -> tuple[list[int], list[int]]:
    concatenated_input_ids = []
    concatenated_label_ids = []

    for i, text in enumerate(segments):
        is_input_segment = i % 2 == 0

        d = tokenizer(text, add_special_tokens=False)
        input_ids = d['input_ids']
        concatenated_input_ids.extend(input_ids)

        if mask_input_segments and is_input_segment:
            label_ids = len(input_ids) * [label_pad_token_id]
            concatenated_label_ids.extend(label_ids)
        else:
            concatenated_label_ids.extend(input_ids)

    return concatenated_input_ids, concatenated_label_ids
