import json, copy, textwrap, datetime, markdown
from operator import itemgetter
from asyncio import sleep as asleep
from pydantic import BaseModel, Field
from fastapi.templating import Jinja2Templates
from chatglm_cpp import Pipeline, ChatMessage, DeltaMessage
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse
from typing import Dict, List, Annotated, AsyncIterator, Coroutine, Generator, Iterable
from fastapi import APIRouter, Request, WebSocket, HTTPException, UploadFile, File, Form, Body, Path

from ..config import get_settings
from ..utils import RedisContextManager, websocket_catch, block_bad_words, remove_punctuation, file_to_text, tw_to_cn
from ..toolkit import run_func, func_called, remove_tool_calls, compress_message, convert_message, SYSTEM_PROMPT, REGISTERED_TOOLS

settings = get_settings()
router = APIRouter(prefix='/chat', tags=[ 'Chat' ], responses={ 404: { "description": "Not found" }})
PATH_UUID = Path(description='User `UUID`')
PATH_DATE = Path(description='Tab created **datetime**')

class ChatTab(BaseModel):
    label    : str = ''
    datetime : str = ''

class ChatRename(BaseModel):
    uuid: str = Field(..., description='User `UUID`')
    name: str = Field(..., description='The `Tab Name` which is going to change')
    datetime: str = Field(..., description='Tab created **datetime**')

class ChatParam(BaseModel):
    max_length: int = Field(2500, le=2500, description='Response length maximum is `2500`')
    top_p: float = Field(.8, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(.8, description='Higher will make **outputs** more `random` and `diverse`')
    repetition_penalty: float = Field(1., le=1., description='Higher values bot will not be repeating')

class ChatInfo(ChatParam):
    uuid: str | None = Field('', description='User `UUID` _(empty will be without Redis cache)_')
    query: str = Field(..., examples=[ '你好' ], description='Message content')
    history: List[Dict[str, str]] = Field([], description='Conversation history list for assistant reference')
    label: Dict[str, str] | None = Field({}, description='Label data for entry details')
    system: str = Field(SYSTEM_PROMPT.content, description='Role `system` content for declare bot')
    html: bool = Field(False, description='Response to `HTML` context directly')

def merge_chat_history(chat_info: ChatInfo) -> List[ChatMessage]:
    """Retrieving the conversation history by **Redis**, then convert
    it as a list of type :class:`~ChatMessage` object.\n
    This function also to join the ``system prompt`` if it not exists
    in ``first system declared message`` or it even not exists ever."""
    prompt = ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=chat_info.system or SYSTEM_PROMPT.content)
    if not chat_info.uuid:
        messages = convert_message(chat_info.history, ChatMessage)
        if ChatMessage.ROLE_SYSTEM not in set(map(itemgetter('role'), chat_info.history)):
            messages[0: 0] = [ prompt ]
        return messages
    messages: List[ChatMessage] = []
    with RedisContextManager(settings.db.redis) as r:
        if chat_info.label:
            query = r.hget(f'talk-history-hash-{chat_info.uuid}', chat_info.label.get('datetime'))
            histories = json.loads(query or '{}').get('history', [])
        else:
            histories = json.loads(r.get(f'talk-history-{chat_info.uuid}') or '[]')
    for history in histories:
        if history["role"] == ChatMessage.ROLE_SYSTEM and prompt.content not in history["content"]:
            history |= dict(content=prompt.content + history["content"])
        messages.append(ChatMessage(**history))
    if ChatMessage.ROLE_SYSTEM not in set(map(itemgetter('role'), histories)):
        messages[0: 0] = [ prompt ]
    return messages or [ prompt ]

def save_chat_history(chat_info: ChatInfo, messages: Iterable[ChatMessage]) -> None:
    """Converting all the :class:`~ChatMessage` object to :class:`~dict`
    then save it to **Redis** with it owns user's ``uuid``."""
    if not chat_info.uuid: return
    data = convert_message(messages, dict)
    with RedisContextManager(settings.db.redis) as r:
        if chat_info.label:
            stringify = json.dumps(dict(**chat_info.label, history=data), ensure_ascii=False)
            r.hset(f'talk-history-hash-{chat_info.uuid}', chat_info.label.get('datetime'), stringify)
        else:
            r.set(f'talk-history-{chat_info.uuid}', json.dumps(data, ensure_ascii=False))

def save_chat_record(prompt: str, response: str) -> None:
    with RedisContextManager(settings.db.redis) as r:
        records: List[Dict[str, str]] = json.loads(r.get(f'talk-history-records') or '[]')
        records.append(dict(user=prompt, assistant=response))
        r.set(f'talk-history-records', json.dumps(records, ensure_ascii=False))

async def observe(messages: List[ChatMessage]) -> Coroutine[None, None, List[ChatMessage]]:
    """While the tool function is included in chat messages, it parsing
    the contents to separate it into the function name and arguments,
    then call the :func:`~run_func` to invoke specified function."""
    tool_call_contents = messages[-1].content.splitlines()
    func_name, func_arg = tool_call_contents[0], '\n'.join(tool_call_contents[1:])
    observation = run_func(func_name, func_arg)
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
    if func_called(messages[-1]):
        return await create_conversation(pipeline, chat_info, await observe(messages))
    return messages

async def create_streaming(
    pipeline: Pipeline, chat_info: ChatInfo, messages: List[ChatMessage]
) -> AsyncIterator[str]:
    copies = copy.deepcopy(convert_message(messages, dict))
    compresses = compress_message(convert_message(copies, ChatMessage), chat_info.max_length)
    print(f' ~~~~~ Num => {len(compresses)} Len => {sum( len(m.content) for m in compresses )}')
    streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
        compresses,
        **dict(p := ChatParam(**dict(chat_info))),
        do_sample=p.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    for chunk in streaming:
        chunks.append(chunk)
        if pipeline.merge_streaming_messages(chunks).content.strip():
            yield block_bad_words(chunk.content)
            await asleep(.01)
    messages.append(pipeline.merge_streaming_messages(chunks))
    if func_called(messages[-1]):
        yield '\n\n'
        async for chunk in create_streaming(pipeline, chat_info, await observe(messages)):
            yield chunk
            await asleep(.01)
    save_chat_history(chat_info, remove_tool_calls(messages))

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
    if func_called(messages[-1]):
        return await create_socket(websocket, await observe(messages))
    return messages

@router.post('/conversation', response_model=None, response_description='Chat Conversation')
async def create_chat_conversation(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> JSONResponse:
    """
    Create a **Single Round** chat conversation.\n
    _(More parameters usage please refer to `Schema`)_
    """
    begin = datetime.datetime.now()
    prompt = tw_to_cn.convert(chat_info.query)
    messages = merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=prompt))
    messages = await create_conversation(request.app.chat.model, chat_info, messages)
    messages = list(remove_tool_calls(messages))
    save_chat_history(chat_info, messages)
    save_chat_record(prompt, content := messages[-1].content)
    return JSONResponse(status_code=200, content=dict(
        response     = markdown.markdown(content) if chat_info.html else content,
        history      = convert_message(messages, dict),
        datetime     = (now := datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time = (now - begin).total_seconds()
    ))

@router.post('/subject', response_model=None, response_description='Chat Subject')
async def get_chat_conversation_subject(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> JSONResponse:
    chat_info.query = f'10个字以内描述大意:{chat_info.query[:128]}'
    prompt = ChatMessage(role=ChatMessage.ROLE_USER, content=chat_info.query)
    messages = await create_conversation(request.app.chat.model, chat_info, [ prompt ])
    resp = dict(subject=remove_punctuation(messages[-1].content).capitalize()[:10])
    return JSONResponse(status_code=200, content=resp)

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Chat')
async def create_chat_stream(
    request: Request, chat_info: Annotated[ChatInfo, Body()]
) -> StreamingResponse:
    """
    Create a **Multiple Round** streaming chat.\n
    _(More parameters usage please refer to `Schema`)_
    """
    messages = merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=tw_to_cn.convert(chat_info.query)))
    resp = create_streaming(request.app.chat.model, chat_info, messages)
    return StreamingResponse(resp, media_type='text/plain')

@router.post('/file/stream', response_class=StreamingResponse, response_description='Streaming File')
async def create_chat_file_stream(
    request: Request,
    uuid: Annotated[str, Form(..., description='User `UUID`')],
    label: Annotated[str, Form(..., description='**Stringify** `JSON` label data')],
    file: Annotated[UploadFile, File(description='A file read as UploadFile')],
    system: Annotated[str, Form(description='`System` content')] = '你需要对用户输入的长文本内容进行解读'
) -> StreamingResponse:
    """
    Create a **Multiple Round** file streaming analysis.
    """
    chat = ChatInfo(uuid=uuid, query=await file_to_text(file, uuid), label=json.loads(label), system=system)
    if not chat.query:
        raise HTTPException(status_code=422, detail='Invalid File Content')
    async def stream(pipeline: Pipeline) -> AsyncIterator[str]:
        messages = merge_chat_history(chat)
        messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=chat.query))
        streaming: Generator[DeltaMessage, None, None] = pipeline.chat(
            messages, **dict(ChatParam(**dict(chat))), do_sample=True, stream=True)
        chunks: List[DeltaMessage] = []
        for chunk in streaming:
            chunks.append(chunk)
            if pipeline.merge_streaming_messages(chunks).content.strip():
                yield block_bad_words(chunk.content)
                await asleep(.01)
        messages.append(pipeline.merge_streaming_messages(chunks))
        messages[-2].content = f'file=[{file.filename}] {messages[-2].content}'
        save_chat_history(chat, messages)
    return StreamingResponse(stream(request.app.chat.model), media_type='text/plain')

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
        messages = list(remove_tool_calls(messages))

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_chat_demo(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('chat.html', context=dict(request=request))

@router.get('/help')
async def show_toolkit() -> JSONResponse:
    resp = list(map(itemgetter('description'), REGISTERED_TOOLS))
    return JSONResponse(status_code=200, content=resp)

@router.get('/tabs/{uuid}')
async def get_chat_tabs(uuid: Annotated[str, Path(description='User `UUID`')]) -> JSONResponse:
    with RedisContextManager(settings.db.redis) as r:
        name = f'talk-history-hash-{uuid}'
        tabs = [ dict(ChatTab(**json.loads(r.hget(name, k) or '{}'))) for k in r.hkeys(name) ]
    resp = sorted(tabs, key=itemgetter('datetime'), reverse=True)
    return JSONResponse(status_code=200, content=resp)

@router.put('/tab/rename')
async def rename_chat_tab(chat_tab: Annotated[ChatRename, Body()]) -> JSONResponse:
    with RedisContextManager(settings.db.redis) as r:
        args = ( f'talk-history-hash-{chat_tab.uuid}', chat_tab.datetime )
        data = json.loads(r.hget(*args) or '{}') | dict(label=chat_tab.name)
        r.hset(*args, json.dumps(data, ensure_ascii=False))
        resp = dict(ChatTab(**json.loads(r.hget(*args))))
    return JSONResponse(status_code=200, content=resp)

@router.get('/history/{uuid}/{datetime}')
async def get_chat_history_by_datetime(
    uuid: Annotated[str, PATH_UUID], datetime: Annotated[str, PATH_DATE]
) -> JSONResponse:
    with RedisContextManager(settings.db.redis) as r:
        data = json.loads(r.hget(f'talk-history-hash-{uuid}', datetime) or '{}')
    return JSONResponse(status_code=200, content=data.get('history', []))

@router.delete('/history/{uuid}/{datetime}')
async def delete_chat_history_by_datetime(
    uuid: Annotated[str, PATH_UUID], datetime: Annotated[str, PATH_DATE]
) -> JSONResponse:
    with RedisContextManager(settings.db.redis) as r:
        args = ( f'talk-history-hash-{uuid}', datetime )
        r.hdel(*args)
        resp = dict(ok=r.hget(*args) is None)
    return JSONResponse(status_code=200, content=resp)

@router.delete('/forget/{uuid}/{range}')
async def clean_chat_history(
    uuid : Annotated[str, PATH_UUID],
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
