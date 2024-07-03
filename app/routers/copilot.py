from datetime import datetime
from chatglm_cpp import Pipeline
from asyncio import sleep as asleep
from pydantic import BaseModel, Field
from typing import Annotated, Generator
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket, HTTPException, Body
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse

from ..config import get_settings
from ..utils import websocket_catch, copilot_prompt

settings = get_settings()
router = APIRouter(prefix='/copilot', tags=[ 'Copilot' ], responses={ 404: { "description": "Not found" } })

class CodingParam(BaseModel):
    max_length: int = Field(1024, le=2048, description='Response length maximum is `2k`')
    top_k: int = Field(1, description='Lower also concentrates sampling on the highest probability tokens for each step')
    top_p: float = Field(.95, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(.2, description='Higher will make **outputs** more `random` and `diverse`')

class Coding(CodingParam):
    lang: str = Field(..., examples=[ 'Python' ], description=f'Programming `language`: [{", ".join(settings.lang_tags)}]')
    prompt: str = Field(..., examples=[ 'Write a quick sort function' ], description='Describe program details')
    html: bool = Field(False, description='Response to `HTML` context directly')

@router.post('/coding', response_model=None, response_description='Code AI Completion')
def create_coding_task(
    request: Request, task: Annotated[Coding, Body()]
) -> HTMLResponse | JSONResponse:
    """
    Create a **Single Round AI** programming which is like `Github Copilot` assistant.\n
    _(More parameters usage please refer to `Schema`)_
    """
    begin = datetime.now()
    pipeline: Pipeline = request.app.copilot.model
    if not settings.lang_tags.get(task.lang):
        lang_tag = ', '.join(settings.lang_tags)
        raise HTTPException(status_code=404, detail=f'Invalid Language, Available: [{lang_tag}]')
    prompt, _ = copilot_prompt(settings.lang_tags, task.lang, task.prompt)
    response = pipeline.generate(
        prompt, do_sample=task.temperature > 0, **dict(CodingParam(**dict(task))))
    if task.html:
        return HTMLResponse(status_code=200, content=response)
    return JSONResponse(status_code=200, content=dict(
        response     = response,
        lang         = task.lang,
        datetime     = (now := datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time = (now - begin).total_seconds()
    ))

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Code')
async def create_coding_stream(request: Request, task: Annotated[Coding, Body()]) -> StreamingResponse:
    """
    Create a **Single Round AI** programming stream which is like `Github Copilot` assistant.\n
    _(More parameters usage please refer to `Schema`)_
    """
    pipeline: Pipeline = request.app.copilot.model
    prompt, _ = copilot_prompt(settings.lang_tags, task.lang, task.prompt)
    streaming: Generator[str, None, None] = pipeline.generate(
        prompt, do_sample=task.temperature > 0, **dict(CodingParam(**dict(task))), stream=True)
    return StreamingResponse(streaming, media_type='text/plain')

@router.websocket('/ws')
@websocket_catch
async def create_coding_socket(
    websocket : WebSocket,
    lang      : str,
    prompt    : str,
    max_length: int = 512,
    top_k     : int = 1
):
    await websocket.accept()
    pipeline: Pipeline = websocket.app.copilot.model
    prompt, _ = copilot_prompt(settings.lang_tags, lang, prompt)
    streaming: Generator[str, None, None] = pipeline.generate(
        prompt, do_sample=True, max_length=max_length, top_k=top_k, stream=True)
    while True:
        code = ''
        for content in streaming:
            code += content
            await websocket.send_text(code)
            await asleep(.01)
        await asleep(1)
        await websocket.close()
        break

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_coding_demo(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('copilot.html', context=dict(request=request))

@router.get('/langs')
async def get_languages() -> JSONResponse:
    """
    Listing currently **SUPPORT LANGUAGE** for AI programming.
    """
    return JSONResponse(status_code=200, content=settings.lang_tags)
