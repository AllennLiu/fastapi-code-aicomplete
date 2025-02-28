import io
import re
import json
import datetime
import textwrap
import colorama
import traceback
import numpy as np
import jieba.posseg as pseg
from PIL import Image
from zoneinfo import ZoneInfo
from operator import itemgetter
from chatglm_cpp import Pipeline, ChatMessage
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Buffer
from typing import Any, Dict, List, Callable, Final, Generator, Iterable, cast

colorama.init(autoreset=True)

try:
    from chatglm_cpp import Image as CImage
except:
    import chatglm_cpp
    print(f'{colorama.Fore.YELLOW}WARNING: module chatglm_cpp {chatglm_cpp.__version__} ' +
        'is not support the Image object yet.')

from .toolkit import Tools, register_tool
from .utils import print_process, remove_punctuation, set_similarity

OBSERVATION_MAX_LENGTH: Final[int] = 1024
PATTERN_CODE_BLOCK: Final = re.compile(r'^```[^\S\r\n]*[a-z]*(?:\n(?!```$).*)*\n```', flags=re.MULTILINE)
PATTERN_FUNC_CALL: Final = re.compile('get_[\w_]+\n+\{\S.*\}\n+')
PATTERN_UNUSED_NEWLINE: Final = re.compile(r'(\n){2,}', flags=re.MULTILINE)
REGISTERED_TOOLS: Final = register_tool(Tools)

def build_tool_system_prompt(content: str, tools: List[dict] = REGISTERED_TOOLS) -> ChatMessage:
    """
    Appending all available :class:`~Tools` to system prompt
    message, this method is fitting with the ``GLM-4`` model.

    Args:
        content (str): System prompt message.
        tools (List[dict]): Available tools data. Defaults to \
            `REGISTERED_TOOLS`.

    Returns:
        ChatMessage: ChatGLM message instance with system prompt.
    """
    content += '\n\n# 可用工具'
    for tool in tools:
        content += f'\n\n## {tool["name"]}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}'
        content += '\n在调用上述函数时，请使用 Json 格式表示调用的参数。'
    return ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=content)

SYSTEM_PROMPT: Final[ChatMessage] = ChatMessage(
    role=ChatMessage.ROLE_SYSTEM,
    content=textwrap.dedent("""\
    你是人工智能 AI 助手，你叫 Black.Milan，你是由 SIT TA 团队创造的。
    你是基于 BU6 SIT TA 团队开发与发明的，你被该团队长期训练而成为优秀的助理。
    """)
)

class ChatOption(BaseModel):
    max_length: int = Field(default=2500, le=2500, description='Response length maximum is `2500`')
    top_p: float = Field(default=.8, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(default=.8, description='Higher will make **outputs** more `random` and `diverse`')
    repetition_penalty: float = Field(default=1., le=1., description='Higher values bot will not be repeating')

class ChatResponse(BaseModel):
    response: str | List[Any] | Dict[str, Any]
    history: List[Dict[str, str]] | None = []
    datetime: str = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time: int | float | None = 0

    @field_validator('elapsed_time', mode='before')
    def round_decimal_places(cls, value):
        return round(value, 4)

def func_called(message: ChatMessage) -> bool:
    """
    Determine whether the first line of message content which name is
    included in `REGISTERED_TOOLS`. _(this available :class:`~Tools`)_

    Args:
        message (ChatMessage): ChatGLM message instance.

    Returns:
        bool: Whether the first line of message content which name is \
            included in `REGISTERED_TOOLS`.
    """
    return message.content.splitlines()[0] in map(itemgetter('name'), REGISTERED_TOOLS)

def escape_json_values(json_string: str) -> str:
    """
    If the values of the escaped JSON data contain double quotes, it
    may cause :class:`~SyntaxError` when the value is passed in a
    function call.

    Therefore, this function is used to attempt to parse the values
    into valid data that can be used by the :func:`~eval` function.

    Args:
        json_string (str): JSON string.

    Returns:
        str: Escaped JSON string.
    """
    arg_splits = [ re.split(r'":\s{0,}"', arg) for arg in re.split(r'",\s{0,}"', json_string) ]
    escape_double_quote_values = [ f'{k}": "' + re.sub(r'"', '\\"', v) for k, v in arg_splits ]
    return re.sub(r'\\"}$', '"}', '", "'.join(escape_double_quote_values))

def run_func(name: str, arguments: str) -> str:
    """
    Run `observation` mode with assistant which function name or
    arguments were been passed.

    Finally, it returns the response of **stringify** :class:`~dict`
    to next round conversation.

    Args:
        name (str): Tool function name.
        arguments (str): Tool function arguments.

    Returns:
        str: The response of tool function call.
    """
    print_process(f'Calling tool {colorama.Fore.MAGENTA}{name}{colorama.Fore.RESET}, args: {colorama.Fore.BLUE}{arguments}')
    def tool_call(**kwargs: Any) -> Dict[str, Any]:
        return kwargs
    func: Callable[..., Any] = getattr(Tools, name)
    try:
        kwargs = eval(arguments, dict(tool_call=tool_call))
    except SyntaxError:
        kwargs = eval(escape_json_values(arguments), dict(tool_call=tool_call))
    except Exception as _:
        return f'Invalid arguments {arguments}'
    try:
        return str(func(**kwargs))
    except Exception as _:
        err = traceback.format_exc()
        print_process(f'Called tool exception: {err}')
        return err

async def observe(messages: List[ChatMessage]) -> List[ChatMessage]:
    """
    While the tool function is included in chat messages, it parsing
    the contents to separate it into the function name and arguments,
    then call the :func:`~run_func` to invoke specified function.

    Args:
        messages (List[ChatMessage]): ChatGLM message instances.

    Returns:
        List[ChatMessage]: ChatGLM message instances including \
            the observation.
    """
    tool_call_contents = messages[-1].content.splitlines()
    func_name, func_arg = tool_call_contents[0], '\n'.join(tool_call_contents[1:])
    observation = run_func(func_name, func_arg)
    messages.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content=observation))
    return messages

def remove_tool_calls(messages: List[ChatMessage]) -> Generator[ChatMessage, None, None]:
    """
    Removing wether the :class:`~ChatMessage` which role is `observation`
    or `tool calls by assistant` then pop out of the list.

    Args:
        messages (List[ChatMessage]): ChatGLM message instances.

    Yields:
        Generator[ChatMessage, None, None]: ChatGLM message generator.
    """
    for message in messages:
        if message.role == ChatMessage.ROLE_OBSERVATION:
            continue
        elif message.role == ChatMessage.ROLE_ASSISTANT and func_called(message):
            continue
        yield message

def ai_sentences_similarity(pipeline: Pipeline, *sentences: str) -> str:
    """
    This function is going to compare the similarity between of
    **multiple sentences**, it using loaded ``LLM`` to check the
    **intent** in each sentence, finally the result will follow
    the describe of `ChatMessage.ROLE_SYSTEM` prompt to answer
    it is ``yes`` or ``no``.

    Args:
        pipeline (Pipeline): ChatGLM pipeline instance.
        sentences (str): Multiple sentences.

    Returns:
        str: The result of similarity is ``yes`` or ``no``.
    """
    system_prompt = '你需要比对用户输入两句话的语境是否相似，最后只能回答yes或no'
    sentence_concat = '"；"'.join(sentences)
    message = f'这两句话 "{sentence_concat}" 是否为相似语境和语意，只能回答yes或no'
    response = pipeline.chat([
        ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=system_prompt),
        ChatMessage(role=ChatMessage.ROLE_USER, content=message)
    ], top_p=.1, temperature=.1)
    return cast(ChatMessage, response).content.strip().lower() or 'no'

def select_tool_call(
    message: ChatMessage, pipeline: Pipeline
) -> Dict[str, str | List[Dict[str, str | bool]]] | None:
    """
    Selecting tool calls by `ChatMessage.ROLE_USER` only, and then
    iterate to compare the similarity as following selector:
    - The last message and tool's description without any
        sentence **punctuations**, get the most similarly
        tool which value is large equal as ``10%``.
    - If these items are satisfy above mentioned condition,
        it'll be validated by :func:`~ai_sentences_similarity`
        then response the **largest similarity** of tool function.

    Args:
        message (ChatMessage): ChatGLM message instance.
        pipeline (Pipeline): ChatGLM pipeline instance.

    Returns:
        Dict[str, str | List[Dict[str, str | bool]]] | None: Tool \
            called function.
    """
    if message.role != ChatMessage.ROLE_USER: return
    similarities, results = [], []
    for tool in REGISTERED_TOOLS:
        msg_generator = pseg.cut(remove_punctuation(message.content))
        desc_generator = pseg.cut(remove_punctuation(str(tool["description"])))
        _, percent = set_similarity((e.word for e in msg_generator), (e.word for e in desc_generator))
        if percent >= 10:
            similarities.append((tool, percent))
    if not similarities: return
    for tool, percent in similarities:
        result = ai_sentences_similarity(pipeline, tool["description"], message.content)
        print_process(f'Func: {tool["name"]: <25} Similarity: {f"{percent}%": <8} AI: {result: >3}')
        if result == 'yes':
            results.append((tool, percent))
    return max(results, key=itemgetter(1))[0] if results else None

def compress_message(
    messages: List[ChatMessage], pipeline: Pipeline, max_length: int = 2048, called: bool = False
) -> List[ChatMessage]:
    """
    When tool function has been called, the messages is going to
    compress into **system prompt** + **the last message which is
    function calls**, and **system prompt** will be the only one
    with most similarly tool in `REGISTERED_TOOLS`.

    Compress the each message recursively if the content length
    is large over than `max_length`, then shift out the first two
    items _(user + assistant)_ from messages.

    At the mean time, these message being iterated to remove as
    following contents:
        - Markdown code block including the context
        - Function calls name and passed arguments
        - After the two previous operations, clear the unused newline

    If a message content length is large over than ``512`` length,
    truncate it to ``128`` length before the content, it just for
    collecting the characteristic of message content makes history
    easy with memory reference.

    Args:
        messages (List[ChatMessage]): ChatGLM message instances.
        pipeline (Pipeline): ChatGLM pipeline instance.
        max_length (int): Maximum length of message content is \
            allowed. Defaults to 2048.
        called (bool): Whether tool function has been called. \
            Defaults to False.

    Returns:
        List[ChatMessage]: Compressed ChatGLM message instances.
    """
    if not called:
        if tool := select_tool_call(messages[-1], pipeline):
            return [ build_tool_system_prompt(SYSTEM_PROMPT.content, [ tool ]), messages[-1] ]
    content = ''
    for message in messages:
        if message.role != ChatMessage.ROLE_SYSTEM:
            message.content = PATTERN_CODE_BLOCK.sub('', message.content, 0)
            message.content = PATTERN_FUNC_CALL.sub('', message.content)
            message.content = PATTERN_UNUSED_NEWLINE.sub('\n', message.content, 0)
            if len(message.content) > 512:
                message.content = message.content[:128]
        content += message.content
    if len(content) > max_length and len(messages) > 1:
        return compress_message([ messages[0], *messages[3:] ], pipeline, max_length, True)
    return messages

def to_chat_messages(messages: Iterable[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert each entry which type is :class:`~dict` to \
    :class:`~ChatMessage`.

    Args:
        messages (Iterable[Dict[str, Any]]): :class:`~dict` entries.

    Returns:
        List[ChatMessage]: converted ChatGLM message instances.
    """
    return [ ChatMessage(**m) for m in messages ]

def to_dict_messages(messages: Iterable[ChatMessage]) -> List[Dict[str, str]]:
    """Convert each entry which type is :class:`~ChatMessage` to \
    :class:`~dict`.

    Args:
        messages (Iterable[ChatMessage]): ChatGLM message instances.

    Returns:
        List[Dict[str, str]]: converted :class:`~dict` entries.
    """
    return [ dict(role=m.role, content=m.content) for m in messages ]

def insert_image(message: ChatMessage, file_bytes: bytes) -> ChatMessage:
    """
    Convert the **image** :class:`~bytes` data to :class:`~CImage` \
    object with ``RGB`` mode, and this object argument required \
    type is the :class:`~np.ndarray` to be \
        :class:`~chatglm_cpp.Image` aka *(:class:`~CImage`)* buffer.

    #### NOTE:
        Use :class:`~chatglm_cpp.Image` require module ``chatglm_cpp>=0.4.1``

    Args:
        message (ChatMessage): ChatGLM message instance.
        file_bytes (bytes): read file bytes.

    Returns:
        ChatMessage: ChatGLM message instance with image.
    """
    img_obj = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    image = CImage(cast(Buffer, np.asarray(img_obj)))
    return ChatMessage(role=message.role, content=message.content, image=image)
