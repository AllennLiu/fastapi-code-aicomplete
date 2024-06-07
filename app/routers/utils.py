from opencc import OpenCC
from datetime import datetime
from pydantic import BaseModel
from asyncio import sleep as asleep
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModel
from starlette.responses import JSONResponse, HTMLResponse
from fastapi import APIRouter, Request, WebSocket, HTTPException

from ..config import get_settings

try:
    import chatglm_cpp
    enable_chatglm_cpp = True
except:
    warning_msg = """[WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance.
    Check out https://github.com/li-plus/chatglm.cpp for more details.
    """
    print(warning_msg)
    enable_chatglm_cpp = False

CC = OpenCC('tw2sp')
app_settings = get_settings()

router = APIRouter(
    prefix    = '/utils',
    tags      = [ 'utils' ],
    responses = { 404: { "description": "Not found" } }
)

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

    if app_settings.load_dev == 'gpu':
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
        if quantize in [ 4, 8 ]:
            print(f'Model is quantized to INT{quantize} format.')
            model = model.half().quantize(quantize)
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device='cpu')

    return model.eval()

copilot_tokenizer = AutoTokenizer.from_pretrained(
    app_settings.private('COPILOT_MODEL'), trust_remote_code=True)
copilot_model = device(
    app_settings.private('COPILOT_MODEL'), int(app_settings.private('COPILOT_QUANTIZE')))
copilot_model_stream = device(
    app_settings.private('COPILOT_MODEL'), int(app_settings.private('COPILOT_QUANTIZE')), True)

class TaskArgs(BaseModel):
    max_length  : int = 512   # maximum is 2048
    top_k       : int = 1
    top_p       : float = .95
    temperature : float = .2

class TaskInfo(TaskArgs):
    lang   : str
    prompt : str
    html   : bool = False

@router.post('/copilot', response_model=None)
async def create_copilot_task(task: TaskInfo) -> HTMLResponse | JSONResponse:
    begin = datetime.now()
    if not app_settings.lang_tags.get(task.lang):
        lang_tag = ', '.join(app_settings.lang_tags)
        raise HTTPException(status_code=404, detail=f'Language Not Found, Try: [{lang_tag}]')
    prompt = f'{app_settings.lang_tags[task.lang]}\n{CC.convert(task.prompt)}\n'
    args = dict(TaskArgs(**dict(task)))
    if enable_chatglm_cpp:
        response = copilot_model.generate(prompt, do_sample=task.temperature > 0, **args)
    else:
        response = copilot_model.chat(copilot_tokenizer, prompt, **args)
    dt = (now := datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time = (now - begin).total_seconds()
    resp = { "response": response, "lang": task.lang, "elapsed_time": elapsed_time, "datetime": dt }
    print(f'[{dt}] ", prompt: "{prompt}", response: "{repr(response)}"')
    if task.html:
        _prompt = prompt.replace('\n', ' ')
        return HTMLResponse(status_code=200, content=f'{_prompt}\n{response}')
    return JSONResponse(status_code=200, content=resp)

@router.websocket('/copilot/ws')
async def create_copilot_stream(
    websocket: WebSocket,
    lang: str,
    prompt: str,
    max_length: int = 256,
    top_k: int = 1
):
    await websocket.accept()
    load_device = 'cuda' if app_settings.load_dev == 'gpu' else 'cpu'
    prompt = f'{app_settings.lang_tags[lang]}\n{CC.convert(prompt)}\n'
    inputs = copilot_tokenizer.encode(prompt, return_tensors='pt').to(load_device)
    while True:
        streaming = await copilot_model_stream.stream_generate(inputs, max_length=max_length, top_k=top_k)
        for outputs in streaming:
            print(copilot_tokenizer.decode(outputs[0]))
        await asleep(1)

@router.get('/copilot/coding', response_class=HTMLResponse, include_in_schema=False)
async def render_copilot_coding(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('copilot.html', context={ "request": request })
