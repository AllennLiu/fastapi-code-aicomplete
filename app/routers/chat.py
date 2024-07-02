import json, textwrap, datetime
from operator import itemgetter
from asyncio import sleep as asleep
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket, Body, Path
from chatglm_cpp import Pipeline, ChatMessage, DeltaMessage
from typing import Dict, List, Annotated, Coroutine, Generator
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse

from ..config import get_settings
from ..toolkit import Tools, run_function, OBSERVATION_MAX_LENGTH
from ..utils import RedisContextManager, websocket_catch, block_bad_words, tw_to_cn

SYSTEM_PROMPT = ChatMessage(role='system', content=textwrap.dedent("""\
你是人工智能 AI 助手，你叫 Black.Milan，你是由 SIT TA 团队创造的。
你是基于 BU6 SIT TA 团队开发与发明的，你被该团队长期训练而成为优秀的助理。
在使用 Python 解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。\
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
    """Retrieving the conversation history by Redis, and then convert
    it as a list of type :class:`~ChatMessage` object.\n
    This function also to join the ``system prompt`` if it not exists
    in ``first system declared message`` or it even not exists ever."""
    system_prompt = ChatMessage(role='system', content=chat_info.system or SYSTEM_PROMPT.content)
    if not chat_info.uuid:
        return [ ChatMessage(**e) for e in chat_info.history ] or [ system_prompt ]
    messages: List[ChatMessage] = []
    with RedisContextManager(settings.db.redis) as r:
        query: List[Dict[str, str]] = json.loads(r.get(f'talk-history-{chat_info.uuid}') or '[]')
    for history in query:
        if history.get('role') == 'system' and system_prompt.content not in history.get('content'):
            history |= dict(content=system_prompt.content + history.get('content', ''))
        messages.append(ChatMessage(**history))
    if 'system' not in map(itemgetter('role'), query):
        messages[0:0] = [ system_prompt ]
    return messages or [ system_prompt ]

def save_chat_history(user_uuid: str, messages: List[ChatMessage]) -> None:
    """Converting all the :class:`~ChatMessage` object to :class:`~dict`
    then save it to Redis with it owns user's ``uuid``."""
    if not user_uuid: return
    data = [ dict(role=m.role, content=m.content) for m in messages ]
    with RedisContextManager(settings.db.redis) as r:
        r.set(f'talk-history-{user_uuid}', json.dumps(data, ensure_ascii=False))

async def observe(messages: List[ChatMessage]) -> Coroutine[None, None, List[ChatMessage]]:
    ( tool_call, ) = messages[-1].tool_calls
    observation = run_function(tool_call.function.name, tool_call.function.arguments)
    if isinstance(observation, str) and len(observation) > OBSERVATION_MAX_LENGTH:
        observation = f'{observation[:OBSERVATION_MAX_LENGTH]} [TRUNCATED]'
    messages.append(ChatMessage(role='observation', content=observation))
    return messages

async def create_conversation(
    pipeline: Pipeline, messages: List[ChatMessage]
) -> Coroutine[None, None, List[ChatMessage]]:
    """Using ``chatglm_cpp`` to create a chat pipeline, and handling
    if tool function called by chatting message, then response the
    observation to chat again, let assistant to assess tool function
    response and recommend user how to do."""
    response: ChatMessage = pipeline.chat(
        messages, **dict(p := ChatParam()), do_sample=p.temperature > 0)
    response.content = block_bad_words(response.content).lstrip()
    messages.append(response)
    if messages[-1].tool_calls:
        return await create_conversation(pipeline, await observe(messages))
    return messages

def create_streaming(pipeline: Pipeline, chat_info: ChatInfo) -> Generator[str, None, None]:
    messages = merge_chat_history(chat_info)
    messages.append(ChatMessage(role='user', content=tw_to_cn.convert(chat_info.query)))
    streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
        messages, **dict(p := ChatParam(**dict(chat_info))), do_sample=p.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    for chunk in streaming:
        chunks.append(chunk)
        yield block_bad_words(chunk.content)
    messages.append(pipeline.merge_streaming_messages(chunks))
    save_chat_history(chat_info.uuid, messages)

async def create_socket(
    websocket: WebSocket, pipeline: Pipeline, messages: List[ChatMessage]
) -> Coroutine[None, None, List[ChatMessage]]:
    """Similar as :func:`~create_conversation` the only difference is
    that chat pipeline is created by streaming :class:`~WebSocket`."""
    response = ''
    chunks: List[DeltaMessage] = []
    streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
        messages, **dict(p := ChatParam()), do_sample=p.temperature > 0, stream=True)
    for chunk in streaming:
        response += chunk.content
        chunks.append(chunk)
        await websocket.send_json(dict(content=block_bad_words(response).lstrip(), completion=False))
        await asleep(.01)
    messages.append(pipeline.merge_streaming_messages(chunks))
    if messages[-1].tool_calls:
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
    messages.append(ChatMessage(role='user', content=tw_to_cn.convert(chat_info.query)))
    messages: List[ChatMessage] = await create_conversation(request.app.chat.model, messages)
    save_chat_history(chat_info.uuid, messages)
    if chat_info.html:
        return HTMLResponse(status_code=200, content=messages[-1].content)
    return JSONResponse(status_code=200, content=dict(
        response     = messages[-1].content,
        history      = [ dict(role=e.role, content=e.content) for e in messages ],
        datetime     = (now := datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time = (now - begin).total_seconds()
    ))

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Chat')
async def create_chat_stream(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> StreamingResponse:
    """
    Create a **Single Round** streaming chat.\n
    _(More parameters usage please refer to `Schema`)_
    """
    resp = create_streaming(request.app.chat.model, chat_info)
    return StreamingResponse(resp, media_type='text/plain')

@router.websocket('/ws')
@websocket_catch
async def create_chat_websocket(websocket: WebSocket):
    await websocket.accept()
    messages: List[ChatMessage] = [ SYSTEM_PROMPT ]
    while True:
        input_content = tw_to_cn.convert(await websocket.receive_text())
        messages.append(ChatMessage(role='user', content=input_content))
        messages = await create_socket(websocket, websocket.app.chat.model, messages)
        await websocket.send_json(dict(content=messages[-1].content, completion=True))
        await asleep(1)

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
