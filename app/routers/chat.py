import json, textwrap, datetime
from operator import itemgetter
from asyncio import sleep as asleep
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket, Body, Path
from chatglm_cpp import Pipeline, ChatMessage, DeltaMessage
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse
from typing import Dict, List, Annotated, AsyncIterator, Coroutine, Generator, Iterable

from ..config import get_settings
from ..utils import RedisContextManager, websocket_catch, block_bad_words, tw_to_cn
from ..toolkit import Tools, run_function, build_tool_system_prompt, is_function_call, remove_tool_calls_message

SYSTEM_PROMPT = build_tool_system_prompt(textwrap.dedent("""\
你是人工智能 AI 助手，你叫 Black.Milan，你是由 SIT TA 团队创造的。
你是基于 BU6 SIT TA 团队开发与发明的，你被该团队长期训练而成为优秀的助理。\
"""))

settings = get_settings()
router = APIRouter(prefix='/chat', tags=[ 'Chat' ], responses={ 404: { "description": "Not found" }})

class ChatParam(BaseModel):
    max_length: int = Field(8192, le=131072, description='Response length maximum is `128k`')
    top_p: float = Field(.8, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(.8, description='Higher will make **outputs** more `random` and `diverse`')
    repetition_penalty: float = Field(1., le=1., description='Higher values bot will not be repeating')

class ChatInfo(ChatParam):
    uuid: str | None = Field('', description='User `UUID` _(empty will be without Redis cache)_')
    query: str = Field(..., examples=[ '你好' ], description='Message content')
    history: List[Dict[str, str]] = Field([], description='Conversation history list for assistant reference')
    system: str = Field(SYSTEM_PROMPT.content, description='Role `system` content for declare bot')
    html: bool = Field(False, description='Response to `HTML` context directly')

def merge_chat_history(chat_info: ChatInfo) -> List[ChatMessage]:
    """Retrieving the conversation history by **Redis**, then convert
    it as a list of type :class:`~ChatMessage` object.\n
    This function also to join the ``system prompt`` if it not exists
    in ``first system declared message`` or it even not exists ever."""
    prompt = ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=chat_info.system or SYSTEM_PROMPT.content)
    if not chat_info.uuid:
        return [ ChatMessage(**e) for e in chat_info.history ] or [ prompt ]
    messages: List[ChatMessage] = []
    with RedisContextManager(settings.db.redis) as r:
        query: List[Dict[str, str]] = json.loads(r.get(f'talk-history-{chat_info.uuid}') or '[]')
    for history in query:
        if history["role"] == ChatMessage.ROLE_SYSTEM and prompt.content not in history["content"]:
            history |= dict(content=prompt.content + history["content"])
        messages.append(ChatMessage(**history))
    if ChatMessage.ROLE_SYSTEM not in map(itemgetter('role'), query):
        messages[0: 0] = [ prompt ]
    return messages or [ prompt ]

def save_chat_history(user_uuid: str, messages: Iterable[ChatMessage]) -> None:
    """Converting all the :class:`~ChatMessage` object to :class:`~dict`
    then save it to **Redis** with it owns user's ``uuid``."""
    if not user_uuid: return
    data = [ dict(role=m.role, content=m.content) for m in messages ]
    with RedisContextManager(settings.db.redis) as r:
        r.set(f'talk-history-{user_uuid}', json.dumps(data, ensure_ascii=False))

async def observe(messages: List[ChatMessage]) -> Coroutine[None, None, List[ChatMessage]]:
    """While the tool function is included in chat messages, this is
    gonna parsing the contents to separate it into the function name
    and arguments, then using the :func:`~run_function` to call the
    specified function."""
    tool_call_contents = messages[-1].content.splitlines()
    func_name, func_arg = tool_call_contents[0], '\n'.join(tool_call_contents[1:])
    observation = run_function(func_name, func_arg)
    messages.append(ChatMessage(role=ChatMessage.ROLE_OBSERVATION, content=observation))
    return messages

async def create_conversation(
    pipeline: Pipeline, chat_info: ChatInfo, messages: List[ChatMessage]
) -> Coroutine[None, None, List[ChatMessage]]:
    """Using ``chatglm_cpp`` to create a chat pipeline, it is going to
    handling if tool function has been called by chatting message, then
    response the observation recursively to chat again, let assistant
    to assess tool function response and recommend user how to do."""
    response: ChatMessage = pipeline.chat(
        messages, **dict(p := ChatParam(**dict(chat_info))), do_sample=p.temperature > 0)
    response.content = block_bad_words(response.content).lstrip()
    messages.append(response)
    if is_function_call(messages[-1]):
        return await create_conversation(pipeline, chat_info, await observe(messages))
    return messages

async def create_socket(
    websocket: WebSocket, pipeline: Pipeline, messages: List[ChatMessage]
) -> Coroutine[None, None, List[ChatMessage]]:
    """Similar as :func:`~create_conversation` the only difference is
    that chat pipeline is created by streaming :class:`~WebSocket`."""
    streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
        messages, **dict(p := ChatParam()), do_sample=p.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    for chunk in streaming:
        chunks.append(chunk)
        if (content := pipeline.merge_streaming_messages(chunks).content).strip():
            await websocket.send_json(dict(content=block_bad_words(content).lstrip(), completion=False))
            await asleep(.01)
    messages.append(pipeline.merge_streaming_messages(chunks))
    if is_function_call(messages[-1]):
        return await create_socket(websocket, await observe(messages))
    return messages

@router.post('/conversation', response_model=None, response_description='Chat Conversation')
async def create_chat_conversation(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> HTMLResponse | JSONResponse:
    """
    Create a **Single Round** chat conversation.\n
    _(More parameters usage please refer to `Schema`)_
    """
    begin = datetime.datetime.now()
    messages = merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=tw_to_cn.convert(chat_info.query)))
    messages = await create_conversation(request.app.chat.model, chat_info, messages)
    messages = list(remove_tool_calls_message(messages))
    save_chat_history(chat_info.uuid, messages)
    if chat_info.html:
        return HTMLResponse(status_code=200, content=messages[-1].content)
    return JSONResponse(status_code=200, content=dict(
        response     = messages[-1].content,
        history      = [ dict(role=e.role, content=e.content) for e in messages ],
        datetime     = (now := datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time = (now - begin).total_seconds()
    ))

async def create_streaming(
    pipeline: Pipeline, chat_info: ChatInfo, messages: List[ChatMessage]
) -> AsyncIterator[str]:
    print(messages)
    streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
        messages, **dict(p := ChatParam(**dict(chat_info))), do_sample=p.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    for chunk in streaming:
        chunks.append(chunk)
        if pipeline.merge_streaming_messages(chunks).content.strip():
            yield block_bad_words(chunk.content)
            await asleep(.01)
    messages.append(pipeline.merge_streaming_messages(chunks))
    print(messages[-1])
    if is_function_call(messages[-1]):
        async for chunk in create_streaming(pipeline, chat_info, await observe(messages)):
            yield chunk
            await asleep(.01)
    save_chat_history(chat_info.uuid, remove_tool_calls_message(messages))

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Chat')
async def create_chat_stream(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> StreamingResponse:
    """
    Create a **Single Round** streaming chat.\n
    _(More parameters usage please refer to `Schema`)_
    """
    messages = merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=tw_to_cn.convert(chat_info.query)))
    resp = create_streaming(request.app.chat.model, chat_info, messages)
    return StreamingResponse(resp, media_type='text/plain')

@router.websocket('/ws')
@websocket_catch
async def create_chat_websocket(websocket: WebSocket):
    await websocket.accept()
    messages: List[ChatMessage] = [ SYSTEM_PROMPT ]
    while True:
        input_content = tw_to_cn.convert(await websocket.receive_text())
        messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=input_content))
        messages = await create_socket(websocket, websocket.app.chat.model, messages)
        await websocket.send_json(dict(content=messages[-1].content, completion=True))
        await asleep(1)
        messages = list(remove_tool_calls_message(messages))

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_chat_demo(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('chat.html', context=dict(request=request))

@router.get('/help')
async def show_toolkit() -> JSONResponse:
    resp = list(map(itemgetter('description'), Tools.tools))
    return JSONResponse(status_code=200, content=resp)

@router.delete('/forget/{uuid}/{range}')
async def clean_chat_history(
    uuid : Annotated[str, Path(description='User `UUID`')],
    range: Annotated[int, Path(description=textwrap.dedent("""\
    Chat history **forget range**: the range of `pair entry` is **count from the end**\n
    **Pair entry** means the history which is the role both of user and assistant.
    - `0 = let it go` _(all the history)_
    - `-1 = the last one`
    - `-2 = the last two`
    - and so on ...
    """))]
) -> JSONResponse:
    with RedisContextManager(settings.db.redis) as r:
        history: List[Dict[str, str]] = json.loads(r.get(f'talk-history-{uuid}') or '[]')
        r.set(f'talk-history-{uuid}', json.dumps(history[: range * 2], ensure_ascii=False))
    return JSONResponse(status_code=200, content=dict(ok=True))
