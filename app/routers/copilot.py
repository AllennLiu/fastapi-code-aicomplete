import datetime
from chatglm_cpp import Pipeline
from asyncio import sleep as asleep
from pydantic import BaseModel, Field
from typing import Annotated, Final, cast
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket, HTTPException, status, Body
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse
from ..coder import save_code
from ..config import get_settings
from ..utils import copilot_prompt
from ..catch import load_model_catch, websocket_catch

settings = get_settings()
router = APIRouter(prefix='/copilot', tags=[ 'Copilot' ], responses={ 404: dict(description='Not found') })
STREAM_WAIT_TIME: Final[float] = .05

class CodingParam(BaseModel):
    max_length: int = Field(default=1024, le=2048, description='Response length maximum is `2k`')
    top_k: int = Field(default=1, description='Lower also concentrates sampling on the highest probability tokens for each step')
    top_p: float = Field(default=.95, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(default=.2, description='Higher will make **outputs** more `random` and `diverse`')

class Coding(CodingParam):
    lang: str = Field(..., examples=[ 'Python' ], description=f'Programming `language`: [{", ".join(settings.lang_tags)}]')
    prompt: str = Field(..., examples=[ 'Write a quick sort function' ], description='Describe program details')
    html: bool = Field(default=False, description='Response to `HTML` context directly')

@router.post('/coding', response_model=None, response_description='Code AI Completion')
@load_model_catch
async def create_coding_task(
    request: Request, task: Annotated[Coding, Body(...)]
) -> HTMLResponse | JSONResponse:
    """
    Create a **Single Round AI** programming which is like `Github Copilot` assistant.\n
    _(More parameters usage please refer to `Schema`)_
    """
    begin = datetime.datetime.now(settings.timezone)
    pipeline: Pipeline = request.app.copilot.model
    if not settings.lang_tags.get(task.lang):
        lang_tag = ', '.join(settings.lang_tags)
        raise HTTPException(status.HTTP_404_NOT_FOUND, f'Invalid Language, Available: [{lang_tag}]')
    prompt, _ = copilot_prompt(settings.lang_tags, task.lang, task.prompt)
    response = cast(str, pipeline.generate(
        prompt, do_sample=task.temperature > 0, **CodingParam(**task.model_dump()).model_dump()))
    if task.html:
        return HTMLResponse(response)
    return JSONResponse(dict(
        response=response,
        lang=task.lang,
        datetime=(now := datetime.datetime.now(settings.timezone)).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time=(now - begin).total_seconds(),
        url=await save_code(settings, str(request.base_url), task.lang, response)
    ))

@router.post('/stream', response_class=StreamingResponse, response_description='Streaming Code')
@load_model_catch
async def create_coding_stream(request: Request, task: Annotated[Coding, Body(...)]) -> StreamingResponse:
    """
    Create a **Single Round AI** programming stream which is like `Github Copilot` assistant.\n
    _(More parameters usage please refer to `Schema`)_
    """
    pipeline: Pipeline = request.app.copilot.model
    prompt, _ = copilot_prompt(settings.lang_tags, task.lang, task.prompt)
    streaming = pipeline.generate(
        prompt, do_sample=task.temperature > 0, **CodingParam(**task.model_dump()).model_dump(), stream=True)
    return StreamingResponse(streaming, media_type='text/plain')

@router.websocket('/ws')
@load_model_catch
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
    streaming = pipeline.generate(prompt, do_sample=True, max_length=max_length, top_k=top_k, stream=True)
    while True:
        code = ''
        for content in streaming:
            code += content
            await websocket.send_text(code)
            await asleep(STREAM_WAIT_TIME)
        await asleep(1)
        await websocket.close()
        break

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_coding_demo(request: Request):
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('copilot.html', context=dict(request=request))

@router.get('/langs')
async def get_languages() -> JSONResponse:
    """
    Listing currently **SUPPORT LANGUAGE** for AI programming.
    """
    return JSONResponse(settings.lang_tags)
