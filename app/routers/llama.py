import asyncio, datetime, contextlib
from pydantic import Field
from ollama import ChatResponse, Message
from starlette.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, Body, File, Form
from typing import Any, List, Annotated, AsyncIterator, cast
from ..chats import SYSTEM_PROMPT
from ..config import get_settings
from ..utils import block_bad_words, md_to_html
from ..llama import LlamaSettings, LlamaOption, LlamaResponse, LlamaClient

settings = get_settings()
client = LlamaClient()
router = APIRouter(prefix='/llama', tags=[ 'Llama' ], responses={ 404: dict(description='Not found') })

class LlamaChat(LlamaOption):
    uuid: str | None = Field(default='', description='User `UUID` _(empty will be without Redis cache)_')
    prompt: str = Field(..., examples=[ '为什么天空是蓝的?' ], description='Message content')
    messages: List[Any] = Field(default=[], description='Conversation history list for assistant reference')
    system: str = Field(default=SYSTEM_PROMPT.content, description='Role `system` content for declare bot')
    html: bool = Field(default=False, description='Response to `HTML` context directly')

async def create_conversation(chat: LlamaChat) -> ChatResponse:
    options = LlamaOption(**chat.model_dump()).model_dump()
    return cast(ChatResponse, await client.base(messages=chat.messages, options=options))

async def create_streaming(chat: LlamaChat) -> AsyncIterator[str]:
    options = LlamaOption(**chat.model_dump()).model_dump()
    with contextlib.suppress(asyncio.CancelledError):
        streaming = await client.vision(messages=chat.messages, options=options, stream=True)
        async for chunk in cast(AsyncIterator[ChatResponse], streaming):
            yield block_bad_words(chunk["message"]["content"])
            await asyncio.sleep(LlamaSettings.stream_wait_time)

async def create_messages(chat: LlamaChat) -> List[Message]:
    if not chat.messages:
        chat.messages.append(Message(role='system', content=chat.system))
    if chat.messages[0]["role"] != 'system':
        chat.messages = [ m for m in chat.messages if m["role"] != 'system' ]
        chat.messages[0: 0] = [ Message(role='system', content=chat.system) ]
    chat.messages.append(Message(role='user', content=chat.prompt))
    return chat.messages

@router.post('/conversation', response_model=None, response_description='Chat Conversation')
async def create_chat_conversation(chat: Annotated[LlamaChat, Body(...)]) -> LlamaResponse:
    """Create a single round chat conversation with the model
    ``llama3.2:3b`` _(Meta release)_.

    _(More parameters usage please refer to `Schema`)_
    """
    chat.messages = await create_messages(chat)
    conversation = await create_conversation(chat)
    chat.messages.append(conversation["message"])
    response = conversation["message"]["content"]
    return LlamaResponse(
        response=md_to_html(response) if chat.html else response,
        messages=chat.messages,
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=conversation["total_duration"]
    )

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Chat')
async def create_chat_stream(chat: Annotated[LlamaChat, Body(...)]) -> StreamingResponse:
    """Create a streaming chat conversations with the model
    ``llama3.2-vision:11b`` _(Meta release)_.
    """
    chat.messages = await create_messages(chat)
    return StreamingResponse(create_streaming(chat), media_type='text/plain')

@router.post('/image', response_model=None, response_description='Analyze Uploaded Image')
async def analyze_uploaded_image(
    image: Annotated[UploadFile, File(..., description='A image to be analyzed')],
    prompt: Annotated[str, Form(..., description='Describe the requirement')] = '请示著描述这张图片'
) -> LlamaResponse:
    """Analyzing **uploaded images** based on **prompt** with
    the model ``llama3.2-vision:11b`` _(Meta release)_.
    """
    message = dict(role='user', content=prompt, images=[ await image.read() ])
    response = cast(ChatResponse, await client.vision(
        messages=[ message ], options=dict(temperature=.0)))
    return LlamaResponse(
        response=response["message"]["content"],
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=response["total_duration"]
    )
