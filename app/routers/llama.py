import asyncio, colorama, datetime, contextlib
from pydantic import BaseModel, Field
from ollama import ChatResponse, Message
from starlette.responses import StreamingResponse
from typing import Any, List, Annotated, AsyncIterator, cast
from fastapi import APIRouter, Request, UploadFile, Body, File, Form
from ..coder import save_code
from ..chats import SYSTEM_PROMPT
from ..config import get_settings
from ..llama import LlamaSettings, LlamaOption, LlamaResponse, llama_conn
from ..utils import print_process, block_bad_words, md_to_html, md_no_codeblock

colorama.init(autoreset=True)
settings = get_settings()
router = APIRouter(prefix='/llama', tags=[ 'Llama' ], responses={ 404: dict(description='Not found') })

class LlamaChat(LlamaOption):
    uuid: str | None = Field(default='', description='User `UUID` _(empty will be without Redis cache)_')
    prompt: str = Field(..., examples=[ '为什么天空是蓝的?' ], description='Message content')
    messages: List[Any] = Field(default=[], description='Conversation history list for assistant reference')
    system: str = Field(default=SYSTEM_PROMPT.content, description='Role `system` content for declare bot')
    html: bool = Field(default=False, description='Response to `HTML` context directly')
    model: str = Field(default='llama3.2', examples=list(LlamaSettings().model_names), description='Select LLM model name')

class LlamaCode(BaseModel):
    prompt: str = Field(..., examples=[ 'Write a bubble sort function' ], description='Coding indicate')
    lang: str = Field(default='Python', description='Coding language')

class Programmer(BaseModel):
    response_content: str = Field(..., description='Model response content')
    markdown_codeblock: str = Field(..., description='Response markdown codeblock')
    language: str = Field(..., description='Use language')

async def create_conversation(chat: LlamaChat) -> ChatResponse:
    options = LlamaOption(**chat.model_dump()).model_dump()
    return cast(ChatResponse, await llama_conn(chat.model, chat.messages, options=options))

async def create_streaming(chat: LlamaChat) -> AsyncIterator[str]:
    options = LlamaOption(**chat.model_dump()).model_dump()
    with contextlib.suppress(asyncio.CancelledError):
        streaming = await llama_conn(chat.model, chat.messages, options=options, stream=True)
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

@router.post('/conversation', response_model=LlamaResponse, response_description='Chat Conversation')
async def create_chat_conversation(chat: Annotated[LlamaChat, Body(...)]) -> LlamaResponse:
    """Create a single round chat conversation.

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
    """Create a streaming chat conversations."""
    chat.messages = await create_messages(chat)
    return StreamingResponse(create_streaming(chat), media_type='text/plain')

@router.post('/image', response_model=LlamaResponse, response_description='Analyze Uploaded Image')
async def analyze_uploaded_image(
    image: Annotated[UploadFile, File(..., description='A image to be analyzed')],
    prompt: Annotated[str, Form(..., description='Describe the requirement')] = '请示著描述这张图片',
    model: Annotated[str, Form(..., description='Select LLM model name')] = 'llama3.2-vision'
) -> LlamaResponse:
    """Analyzing **uploaded images** based on **prompt**."""
    message = dict(role='user', content=prompt, images=[ await image.read() ])
    response = cast(ChatResponse, await llama_conn(model, [ message ], options=dict(temperature=.0)))
    return LlamaResponse(
        response=response["message"]["content"],
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=response["total_duration"]
    )

@router.post('/code', response_model=LlamaResponse, response_description='Code Content')
async def structured_code_output(
    request: Request, data: Annotated[LlamaCode, Body(...)]
) -> LlamaResponse:
    """Use model `qwen2.5-coder:7b` to generate code more precisely
    according to structured output.
    """
    prompt = f'Write a program using the {data.lang} language that meets the requirements: {data.prompt}'
    print_process(f'programming prompt: {colorama.Fore.MAGENTA}{prompt}')
    messages = [ Message(role='user', content=prompt) ]
    response = cast(ChatResponse, await llama_conn(
        'qwen2.5-coder:7b', messages, format=Programmer.model_json_schema(), options=dict(temperature=.0)))
    result = Programmer.model_validate_json(response["message"]["content"])
    print_process(f'programming response:\n{colorama.Fore.GREEN}{result}')
    code = md_no_codeblock(result.markdown_codeblock)
    return LlamaResponse(
        response=await save_code(settings, str(request.base_url), data.lang, code),
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=response["total_duration"]
    )
