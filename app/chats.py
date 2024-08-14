import io, re, json, textwrap, traceback
import numpy as np
import jieba.posseg as pseg
from PIL import Image
from operator import itemgetter
from chatglm_cpp import Pipeline, ChatMessage
from typing import Any, Dict, List, Callable, Coroutine, Generator

try:
    from chatglm_cpp import Image as CImage
except:
    import chatglm_cpp
    print(f'WARNING: module chatglm_cpp {chatglm_cpp.__version__} is not support the Image object yet.')

from .toolkit import Tools, register_tool
from .utils import remove_punctuation, set_similarity

OBSERVATION_MAX_LENGTH: int = 1024
REGEX_FUNC_CALL: str = 'get_[\w_]+\n+\{\S.*\}\n+'
REGISTERED_TOOLS = register_tool(Tools)

def build_tool_system_prompt(content: str, tools: List[dict] = REGISTERED_TOOLS) -> ChatMessage:
    """
    Appending all available :class:`~Tools` to system prompt
    message, this method is fitting with the ``GLM-4`` model.
    """
    content += '\n\n# 可用工具'
    for tool in tools:
        content += f'\n\n## {tool["name"]}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}'
        content += '\n在调用上述函数时，请使用 Json 格式表示调用的参数。'
    return ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=content)

SYSTEM_PROMPT = ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=textwrap.dedent("""\
你是人工智能 AI 助手，你叫 Black.Milan，你是由 SIT TA 团队创造的。
你是基于 BU6 SIT TA 团队开发与发明的，你被该团队长期训练而成为优秀的助理。\
"""))

def func_called(message: ChatMessage) -> bool:
    """
    Determine whether the first line of message content which name is
    included in `REGISTERED_TOOLS` _(this available :class:`~Tools`)_.
    """
    return message.content.splitlines()[0] in map(itemgetter('name'), REGISTERED_TOOLS)

def run_func(name: str, arguments: str) -> str:
    """
    Run `observation` mode with assistant which function name or
    arguments were been passed.
    Finally, it returns the response of **stringify** :class:`~dict`
    to next round conversation.
    """
    print(f'Calling tool {name}, args: {arguments}')
    def tool_call(**kwargs: Any) -> Dict[str, Any]:
        return kwargs
    func: Callable[..., Any] = getattr(Tools, name)
    try:
        kwargs = eval(arguments, dict(tool_call=tool_call))
    except Exception as _:
        return f'Invalid arguments {arguments}'
    try:
        return str(func(**kwargs))
    except Exception as _:
        return traceback.format_exc()

async def observe(messages: List[ChatMessage]) -> Coroutine[None, None, List[ChatMessage]]:
    """
    While the tool function is included in chat messages, it parsing
    the contents to separate it into the function name and arguments,
    then call the :func:`~run_func` to invoke specified function.
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
    """
    system_prompt = '你需要比对用户输入两句话的语境是否相似，最后只能回答yes或no'
    sentence_concat = '"；"'.join(sentences)
    message = f'这两句话 "{sentence_concat}" 是否为相似语境和语意，只能回答yes或no'
    return pipeline.chat([
        ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=system_prompt),
        ChatMessage(role=ChatMessage.ROLE_USER, content=message)
    ], top_p=.1, temperature=.1).content.strip().lower() or 'no'

def select_tool_call(
    message: ChatMessage, pipeline: Pipeline
) -> Dict[str, str | List[Dict[str, str | bool]]] | None:
    """
    Selecting tool calls by `ChatMessage.ROLE_USER` only, and then
    iterate to compare the similarity as following selector:
    - The last message and tool's description without any sentence
    **punctuations**, get the most similarly tool which value is
    large equal as ``10%``.
    - If these items are satisfy above mentioned condition, it'll
    be validated by :func:`~ai_sentences_similarity` then response
    the **largest similarity** of tool function.
    """
    if message.role != ChatMessage.ROLE_USER: return
    similarities, results = [], []
    for tool in REGISTERED_TOOLS:
        msg_generator = pseg.cut(remove_punctuation(message.content))
        desc_generator = pseg.cut(remove_punctuation(tool["description"]))
        _, percent = set_similarity((e.word for e in msg_generator), (e.word for e in desc_generator))
        percent >= 10 and similarities.append((tool, percent))
    if not similarities: return
    for tool, percent in similarities:
        result = ai_sentences_similarity(pipeline, tool["description"], message.content)
        print(f'Func: {tool["name"]: <25} Similarity: {f"{percent}%": <8} AI: {result: >3}')
        result == 'yes' and results.append((tool, percent))
    return max(results, key=itemgetter(1))[0] if results else None

def compress_message(
    messages: List[ChatMessage], pipeline: Pipeline, max_length: int = 2048, called: bool = False
) -> List[ChatMessage]:
    """
    When tool function has been called, the messages is going to
    compress into **system prompt** + **the last message which is
    function calls**, and the **system prompt** will be the only
    one with most similarly tool in `REGISTERED_TOOLS`.

    Compress the each message recursively if the content length
    is large over than `max_length` then shift out the first two
    items _(user + assistant)_ from messages, at the mean time,
    these message being iterated to remove following contents:
    - Markdown code block including the context
    - Function calls name and passed arguments
    - After the two previous operations, clear the unused newline

    If a message content length is large over than ``512`` bytes,
    truncate it to ``128`` bytes before the content, it just for
    collecting the characteristic of message content.
    """
    if not called:
        if tool := select_tool_call(messages[-1], pipeline):
            return [ build_tool_system_prompt(SYSTEM_PROMPT.content, [ tool ]), messages[-1] ]
    content = ''
    for message in messages:
        if message.role != ChatMessage.ROLE_SYSTEM:
            regex_code_block = r'^```[^\S\r\n]*[a-z]*(?:\n(?!```$).*)*\n```'
            regex_unused_newline = r'(\n){2,}'
            message.content = re.sub(regex_code_block, '', message.content, 0, re.MULTILINE)
            message.content = re.sub(REGEX_FUNC_CALL, '', message.content)
            message.content = re.sub(regex_unused_newline,'\n', message.content, 0, re.MULTILINE)
            if len(message.content) > 512:
                message.content = message.content[:128]
        content += message.content
    if len(content) > max_length and len(messages) > 1:
        return compress_message([ messages[0], *messages[3:] ], pipeline, max_length, True)
    return messages

def convert_message(
    messages: List[ChatMessage | Dict[str, str]], to_object: ChatMessage | dict
) -> List[ChatMessage | Dict[str, str]]:
    """
    Converting each data between type is :class:`~ChatMessage` or
    type is :class:`~dict` in messages.
    """
    return (
        [ ChatMessage(**m) for m in messages ] if isinstance(to_object, type(ChatMessage))
        else [ dict(role=m.role, content=m.content) for m in messages ]
    )

def insert_image(message: ChatMessage, file_bytes: bytes) -> ChatMessage:
    """
    Converting the **image** :class:`~bytes` data to :class:`~CImage`\
    object with ``RGB`` mode, and this object argument require type\
    is the :class:`~np.ndarray` to be :class:`~chatglm_cpp.Image`\
    _(:class:`~CImage`)_ buffer.

    _(Using :class:`~chatglm_cpp.Image` require module ``chatglm_cpp>=0.4.1``)_
    """
    image = CImage(np.asarray(Image.open(io.BytesIO(file_bytes)).convert('RGB')))
    return ChatMessage(role=message.role, content=message.content, image=image)
