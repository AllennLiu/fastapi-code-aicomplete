import json
import copy
import asyncio
import colorama
import datetime
import operator
import contextlib
from pydantic import Field
from chatglm_cpp import Pipeline, ChatMessage, DeltaMessage
from starlette.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Annotated, AsyncIterator, Final, Generator, Iterable, cast
from fastapi import APIRouter, Request, WebSocket, HTTPException, UploadFile, status, File, Form, Body
from ..config import get_settings
from ..db import RedisAsynchronous
from ..catch import load_model_catch, websocket_catch
from ..utils import print_process, block_bad_words, remove_punctuation, read_file, md_to_html, tw_to_cn
from ..chats import ChatOption, ChatResponse, observe, func_called, remove_tool_calls, compress_message, to_chat_messages, to_dict_messages, insert_image, SYSTEM_PROMPT

colorama.init(autoreset=True)
settings = get_settings()
router = APIRouter(prefix='/chat', tags=[ 'Chat' ], responses={ 404: dict(description='Not found') })
STREAM_WAIT_TIME: Final[float] = .05
FORM_UUID: Final = Form(..., description='User `UUID`')
FORM_LABEL: Final = Form(..., description='**Stringify** `JSON` label data')

class ChatInfo(ChatOption):
    uuid: str | None = Field(default='', description='User `UUID` _(empty will be without Redis cache)_')
    query: str = Field(..., examples=[ '你好' ], description='Message content')
    history: List[Dict[str, str]] = Field(default=[], description='Conversation history list for assistant reference')
    label: Dict[str, str] | None = Field(default={}, description='Label data for entry details')
    system: str = Field(default=SYSTEM_PROMPT.content, description='Role `system` content for declare bot')
    html: bool = Field(default=False, description='Response to `HTML` context directly')

async def merge_chat_history(chat_info: ChatInfo) -> List[ChatMessage]:
    """
    Retrieving the conversation history by **Redis**, then convert
    it as a list of type :class:`~ChatMessage` object.\n
    This function also to join the ``system prompt`` if it not exists
    in ``first system declared message`` or it even not exists ever.
    """
    prompt = ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=chat_info.system or SYSTEM_PROMPT.content)
    if not chat_info.uuid:
        messages = to_chat_messages(chat_info.history)
        if ChatMessage.ROLE_SYSTEM not in set(map(operator.itemgetter('role'), chat_info.history)):
            messages[0: 0] = [ prompt ]
        return messages
    messages: List[ChatMessage] = []
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        if chat_info.label:
            query = await r.hget(f'talk-history-hash-{chat_info.uuid}', chat_info.label.get('datetime', ''))
            histories = json.loads(query or '{}').get('history', [])
        else:
            histories = json.loads(await r.get(f'talk-history-{chat_info.uuid}') or '[]')
    for history in histories:
        if history["role"] == ChatMessage.ROLE_SYSTEM and prompt.content not in history["content"]:
            history |= dict(content=prompt.content + history["content"])
        messages.append(ChatMessage(**history))
    if ChatMessage.ROLE_SYSTEM not in set(map(operator.itemgetter('role'), histories)):
        messages[0: 0] = [ prompt ]
    return messages or [ prompt ]

async def save_chat_history(chat_info: ChatInfo, messages: Iterable[ChatMessage]) -> None:
    """
    Converting all the :class:`~ChatMessage` object to :class:`~dict`
    then save it to **Redis** with it owns user's ``uuid``.
    """
    if not chat_info.uuid: return
    data = to_dict_messages(messages)
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        if chat_info.label:
            stringify = json.dumps(dict(**chat_info.label, history=data), ensure_ascii=False)
            await r.hset(f'talk-history-hash-{chat_info.uuid}', chat_info.label.get('datetime', ''), stringify)
        else:
            await r.set(f'talk-history-{chat_info.uuid}', json.dumps(data, ensure_ascii=False))

async def save_chat_record(prompt: str, response: str) -> None:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        records: List[Dict[str, str]] = json.loads(await r.get(f'talk-history-records') or '[]')
        records.append(dict(user=prompt, assistant=response))
        await r.set(f'talk-history-records', json.dumps(records, ensure_ascii=False))

async def create_conversation(
    pipeline: Pipeline, chat_info: ChatInfo, messages: List[ChatMessage]
) -> List[ChatMessage]:
    """
    Using ``chatglm_cpp`` to create a chat pipeline, it is going to
    handling if tool function has been called by chatting message,
    then response the observation recursively and chat again, let
    assistant to assess tool function response for recommending that
    user how to do.
    """
    param = ChatOption(**chat_info.model_dump())
    response = cast(ChatMessage, pipeline.chat(
        messages, **param.model_dump(), do_sample=param.temperature > 0))
    response.content = block_bad_words(response.content).lstrip()
    messages.append(response)
    if func_called(messages[-1]):
        return await create_conversation(pipeline, chat_info, await observe(messages))
    return messages

async def create_streaming(
    request: Request, chat_info: ChatInfo, messages: List[ChatMessage]
) -> AsyncIterator[str]:
    """
    Similar as :func:`~create_conversation` the only difference is
    that chat pipeline stream is being enabled, and it responses
    the streaming generator for :class:`~StreamingResponse`.
    """
    pipeline: Pipeline = request.app.chatbot.model
    copies = copy.deepcopy(to_dict_messages(messages))
    compresses = compress_message(to_chat_messages(copies), pipeline, chat_info.max_length)
    print_process(f'Num: {colorama.Fore.RED}{len(compresses)}{colorama.Fore.RESET} ' +
        f'Len: {colorama.Fore.RED}{sum(len(m.content) for m in compresses)}')
    param = ChatOption(**chat_info.model_dump())
    streaming = pipeline.chat(
        compresses, **param.model_dump(), do_sample=param.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    with contextlib.suppress(asyncio.CancelledError):
        for chunk in cast(Generator[DeltaMessage, None, None], streaming):
            chunks.append(chunk)
            if pipeline.merge_streaming_messages(chunks).content.strip():
                yield block_bad_words(chunk.content)
                await asyncio.sleep(STREAM_WAIT_TIME)
    messages.append(pipeline.merge_streaming_messages(chunks))
    if func_called(messages[-1]):
        yield '\n\n'
        with contextlib.suppress(asyncio.CancelledError):
            async for chunk in create_streaming(request, chat_info, await observe(messages)):
                yield chunk
                await asyncio.sleep(STREAM_WAIT_TIME)
    await save_chat_history(chat_info, remove_tool_calls(messages))

async def file_stream(
    request: Request,
    pipeline: Pipeline,
    chat: ChatInfo,
    messages: List[ChatMessage],
    tag: str,
    file: UploadFile
) -> AsyncIterator[str]:
    chat.query = tw_to_cn.convert(chat.query)
    streaming = pipeline.chat(
        messages, **ChatOption(**chat.model_dump()).model_dump(), do_sample=True, stream=True)
    chunks: List[DeltaMessage] = []
    with contextlib.suppress(asyncio.CancelledError):
        for chunk in cast(Generator[DeltaMessage, None, None], streaming):
            chunks.append(chunk)
            if await request.is_disconnected(): break
            if pipeline.merge_streaming_messages(chunks).content.strip():
                yield block_bad_words(chunk.content)
                await asyncio.sleep(STREAM_WAIT_TIME)
    messages.append(pipeline.merge_streaming_messages(chunks))
    messages[-2].content = f'{tag}=[{file.filename}] {messages[-2].content}'
    await save_chat_history(chat, messages)

async def create_socket(
    websocket: WebSocket, pipeline: Pipeline, messages: List[ChatMessage]
) -> List[ChatMessage]:
    """
    Similar as :func:`~create_conversation` the only difference is
    that chat pipeline is created by streaming :class:`~WebSocket`.
    """
    param = ChatOption()
    streaming = pipeline.chat(messages, **param.model_dump(), do_sample=param.temperature > 0, stream=True)
    chunks: List[DeltaMessage] = []
    with contextlib.suppress(asyncio.CancelledError):
        for chunk in cast(Generator[DeltaMessage, None, None], streaming):
            chunks.append(chunk)
            if (content := pipeline.merge_streaming_messages(chunks).content).strip():
                await websocket.send_json(dict(
                    content=block_bad_words(content).lstrip(), completion=False))
                await asyncio.sleep(STREAM_WAIT_TIME)
    messages.append(pipeline.merge_streaming_messages(chunks))
    if func_called(messages[-1]):
        return await create_socket(websocket, pipeline, await observe(messages))
    return messages

@router.post('/conversation', response_model=ChatResponse, response_description='Chat Conversation')
async def create_chat_conversation(
    request: Request, chat_info: Annotated[ChatInfo, Body(...)]
) -> ChatResponse:
    """
    Create a **Single Round** chat conversation.\n
    _(More parameters usage please refer to `Schema`)_
    """
    begin = datetime.datetime.now(settings.timezone)
    prompt = tw_to_cn.convert(chat_info.query)
    messages = await merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=prompt))
    messages = await create_conversation(request.app.chatbot.model, chat_info, messages)
    messages = list(remove_tool_calls(messages))
    await save_chat_history(chat_info, messages)
    await save_chat_record(prompt, content := messages[-1].content)
    return ChatResponse(
        response     = md_to_html(content) if chat_info.html else content,
        history      = to_dict_messages(messages),
        datetime     = (now := datetime.datetime.now(settings.timezone)).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time = (now - begin).total_seconds()
    )

@router.post('/subject', response_model=None, response_description='Conversation Subject')
async def get_chat_subject(request: Request, chat_info: Annotated[ChatInfo, Body(...)]) -> JSONResponse:
    """
    Get the chat **Conversation Subject** `within **10** words`.
    """
    chat_info.system = '用户将输入一段文本，你需要用不含标点符号中英文加起来共 10 个字以内来描述大意！'
    messages = [ ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=chat_info.system) ]
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=chat_info.query[:128]))
    subjects = await create_conversation(request.app.chatbot.model, chat_info, messages)
    resp = dict(subject=remove_punctuation(subjects[-1].content).capitalize()[:10])
    return JSONResponse(resp)

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Chat')
@load_model_catch
async def create_chat_stream(
    request: Request, chat_info: Annotated[ChatInfo, Body(...)]
) -> StreamingResponse:
    """
    Create a stream of **Multiple Round** conversations.\n
    _(More parameters usage please refer to `Schema`)_
    """
    chat_info.query = tw_to_cn.convert(chat_info.query)
    messages = await merge_chat_history(chat_info)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=chat_info.query))
    return StreamingResponse(create_streaming(request, chat_info, messages), media_type='text/plain')

@router.post('/file/stream', response_class=StreamingResponse, response_description='Streaming File')
@load_model_catch
async def create_chat_file_stream(
    request: Request,
    uuid: Annotated[str, FORM_UUID],
    label: Annotated[str, FORM_LABEL],
    file: Annotated[UploadFile, File(..., description='A file read as UploadFile')],
    system: Annotated[str, Form(..., description='`System` content')] = '你需要对用户输入的长文本内容进行解读'
) -> StreamingResponse:
    """
    Create a stream of **Multiple Round** file analysis.
    """
    chat = ChatInfo(uuid=uuid, query=await read_file(file, uuid), label=json.loads(label), system=system)
    if not chat.query:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, 'Invalid File Content')
    messages = await merge_chat_history(chat)
    messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=chat.query))
    return StreamingResponse(file_stream(
        request, request.app.chatbot.model, chat, messages, 'file', file), media_type='text/plain')

@router.post('/image/stream', response_class=StreamingResponse, response_description='Streaming Image')
@load_model_catch
async def create_chat_image_stream(
    request: Request,
    uuid: Annotated[str, FORM_UUID],
    label: Annotated[str, FORM_LABEL],
    image: Annotated[UploadFile, File(..., description='A image read as UploadFile')],
    query: Annotated[str, Form(..., description='Message content')] = '请描述这张图片'
) -> StreamingResponse:
    """
    Create a stream of **Multiple Round** image analysis.
    """
    chat = ChatInfo(uuid=uuid, query=query, label=json.loads(label))
    messages = [ *await merge_chat_history(chat), ChatMessage(role=ChatMessage.ROLE_USER, content=query) ]
    messages[-1] = insert_image(messages[-1], cast(bytes, await image.read()))
    return StreamingResponse(file_stream(
        request, request.app.multi_modal.model, chat, messages, 'image', image), media_type='text/plain')

@router.websocket('/ws')
@load_model_catch
@websocket_catch
async def create_chat_websocket(websocket: WebSocket):
    await websocket.accept()
    messages: List[ChatMessage] = [ SYSTEM_PROMPT ]
    while True:
        input_content = tw_to_cn.convert(await websocket.receive_text())
        messages.append(ChatMessage(role=ChatMessage.ROLE_USER, content=input_content))
        messages = await create_socket(websocket, websocket.app.chatbot.model, messages)
        await websocket.send_json(dict(content=messages[-1].content, completion=True))
        await asyncio.sleep(1)
        messages = list(remove_tool_calls(messages))
