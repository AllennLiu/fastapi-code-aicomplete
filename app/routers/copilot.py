import re
from torch import Tensor
from datetime import datetime
from pydantic import BaseModel
from asyncio import sleep as asleep
from typing import Generator, Iterator
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModel
from starlette.responses import JSONResponse, HTMLResponse
from fastapi import APIRouter, Request, WebSocket, HTTPException

from ..config import get_settings
from ..utils import websocket_catch, copilot_prompt

try:
    import chatglm_cpp
    enable_chatglm_cpp = True
except:
    warning_msg = """[WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance.
    Check out https://github.com/li-plus/chatglm.cpp for more details.
    """
    print(warning_msg)
    enable_chatglm_cpp = False

app_settings = get_settings()

router = APIRouter(
    prefix    = '/copilot',
    tags      = [ 'Copilot' ],
    responses = { 404: { "description": "Not found" } }
)
pretrained_args = { "trust_remote_code": True, "local_files_only": True, "device": app_settings.load_dev }

def device(model_name: str, quantize: int, use_stream: bool = False) -> chatglm_cpp.Pipeline:
    if not use_stream:
        if enable_chatglm_cpp:
            print('Using chatglm-cpp to improve performance')
            dtype = 'f32'
            if quantize in [ 4, 5, 8 ]:
                dtype = f'q{quantize}_0'
            model = chatglm_cpp.Pipeline(model_name, dtype=dtype)
            return model
        print('chatglm-cpp not enabled, falling back to transformers')

    model = AutoModel.from_pretrained(model_name, **pretrained_args)
    return model.eval()

tokenizer = AutoTokenizer.from_pretrained(app_settings.models.copilot_name, **pretrained_args)
model: chatglm_cpp.Pipeline = device(
    app_settings.models.copilot_name, int(app_settings.models.copilot_quantize))
model_stream = device(
    app_settings.models.copilot_name, int(app_settings.models.copilot_quantize), True)

class CodingParam(BaseModel):
    max_length  : int   = 512   # maximum is 2048
    top_k       : int   = 1
    top_p       : float = .95
    temperature : float = .2

class Coding(CodingParam):
    lang   : str
    prompt : str
    html   : bool = False

@router.post('/coding', response_model=None)
async def create_copilot_task(task: Coding) -> HTMLResponse | JSONResponse:
    """
    Create an AI programming process like copilot assistant:

    - **lang**: programming language, such as: [Python, JavaScript, Shell]
    - **prompt**: describe program requirement
    - **html**: response to HTML context
    """
    begin = datetime.now()
    if not app_settings.lang_tags.get(task.lang):
        lang_tag = ', '.join(app_settings.lang_tags)
        raise HTTPException(status_code=404, detail=f'Language Not Found, Try: [{lang_tag}]')
    prompt, _ = copilot_prompt(app_settings.lang_tags, task.lang)
    args = dict(CodingParam(**dict(task)))
    if enable_chatglm_cpp:
        response = model.generate(prompt, do_sample=task.temperature > 0, **args)
    else:
        response: Iterator[chatglm_cpp.DeltaMessage] | chatglm_cpp.ChatMessage = model.chat(tokenizer, prompt, **args)
    resp = {
        "response"    : response,
        "lang"        : task.lang,
        "datetime"    : (now := datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_time": (now - begin).total_seconds()
    }
    if task.html:
        _prompt = prompt.replace('\n', ' ')
        return HTMLResponse(status_code=200, content=f'{_prompt}\n{response}')
    return JSONResponse(status_code=200, content=resp)

@router.websocket('/coding/ws')
@websocket_catch
async def create_copilot_stream(
    websocket : WebSocket,
    lang      : str,
    prompt    : str,
    max_length: int = 512,
    top_k     : int = 1
):
    await websocket.accept()
    prompt, comment_str = copilot_prompt(app_settings.lang_tags, lang, prompt)
    encoding: Tensor = tokenizer.encode(prompt, return_tensors='pt')
    inputs = encoding.to(app_settings.load_dev)
    streaming: Generator[Tensor, None, None] = model_stream.stream_generate(inputs, max_length=max_length, top_k=top_k)
    while True:
        origin_content = ''
        for idx, outputs in enumerate(streaming):
            content = tokenizer.decode(outputs[0])
            if idx == 0:
                origin_content = re.sub(r'\s|\n', '', content)
            await websocket.send_text(content)
            print(content)
            await asleep(0.1)
        if origin_content == re.sub(r'\s|\n', '', content):
            await websocket.send_text(f'{content}\n{comment_str} I have done my best')
        await asleep(1)
        await websocket.close()
        break

@router.get('/coding/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_copilot_demo(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('copilot.html', context={ "request": request })
