import uvicorn
from opencc import OpenCC
from datetime import datetime
from pydantic import BaseModel
from config import get_settings
from fastapi import FastAPI, responses
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware

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
CC = OpenCC('tw2sp')
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*" ],
    allow_methods=[ "*" ],
    allow_headers=[ "*" ],
    allow_credentials=True)

def device():
    if enable_chatglm_cpp and args.chatglm_cpp:
        print('Using chatglm-cpp to improve performance')
        dtype = 'f16' if args.half else 'f32'
        if args.quantize in [ 4, 5, 8 ]:
            dtype = f'q{args.quantize}_0'
        model = chatglm_cpp.Pipeline(args.model_path, dtype=dtype)
        return model

    print('chatglm-cpp not enabled, falling back to transformers')
    if not args.cpu:
        if not args.half:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).cuda()
        else:
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).cuda().half()
        if args.quantize in [ 4, 8 ]:
            print(f'Model is quantized to INT{args.quantize} format.')
            model = model.half().quantize(args.quantize)
    else:
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, device='cpu')

    return model.eval()

class TaskInfo(BaseModel):
    lang        : str
    prompt      : str
    max_length  : int = 512   # maximum is 2048
    top_p       : float = .95
    temperature : float = .2
    top_k       : int = 0

@app.post('/copilot')
async def create_copilot_task(task: TaskInfo) -> responses.JSONResponse:
    global model, tokenizer
    begin = datetime.now()
    prompt = f'{app_settings.lang_tags[task.lang]}\n{CC.convert(task.prompt)}'
    if enable_chatglm_cpp and args.chatglm_cpp:
        response = model.generate(
            prompt,
            max_length=task.max_length,
            do_sample=task.temperature > 0,
            top_p=task.top_p,
            top_k=task.top_k,
            temperature=task.temperature)
    else:
        response = model.chat(
            tokenizer,
            prompt,
            max_length=task.max_length,
            top_p=task.top_p,
            top_k=task.top_k,
            temperature=task.temperature)
    dt = (now := datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time = (now - begin).total_seconds()
    resp = { "response": response, "lang": task.lang, "elapsed_time": elapsed_time, "datetime": dt }
    print(f'[{dt}] ", prompt: "{prompt}", response: "{repr(response)}"')

    return responses.JSONResponse(status_code=200, content=resp)

def args_parser() -> Namespace:
    parser = ArgumentParser(description='Evaluating trained model effect')
    group = parser.add_argument_group(title='CodeGeeX2 API service')
    group.add_argument('-m', '--model-path', type=str, default='THUDM/codegeex2-6b')
    group.add_argument('-H', '--host', type=str, default='0.0.0.0')
    group.add_argument('-p', '--port', type=int, default=7860)
    group.add_argument('-w', '--workers', type=int, default=1)
    group.add_argument('--cpu', action='store_true')
    group.add_argument('--half', action='store_true')
    group.add_argument('--quantize', type=int, default=None)
    group.add_argument('--chatglm-cpp', action='store_true' )
    return parser.parse_args()

if __name__ == '__main__':
    args = args_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = device()
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
