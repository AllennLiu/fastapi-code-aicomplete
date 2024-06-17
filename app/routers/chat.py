import re, json
from datetime import datetime
from pydantic import BaseModel
from asyncio import sleep as asleep
from typing import Dict, List, Generator
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, WebSocket
from transformers import AutoTokenizer, AutoModel
from starlette.responses import JSONResponse, HTMLResponse

from ..toolkit import Tools
from ..config import get_settings
from ..utils import websocket_catch, tw_to_cn

app_settings = get_settings()

router = APIRouter(
    prefix    = '/chat',
    tags      = [ 'Chat' ],
    responses = { 404: { "description": "Not found" } }
)

system_info = {
    "role"   : "system",
    "content": "Answer the following questions as best as you can. You have access to the following tools:",
    "tools"  : Tools.tools
}

pretrained_args = { "trust_remote_code": True, "local_files_only": True, "device": app_settings.load_dev }
tokenizer = AutoTokenizer.from_pretrained(app_settings.models.chatbot_name, **pretrained_args)
model = AutoModel.from_pretrained(app_settings.models.chatbot_name, **pretrained_args).eval()

class ChatParam(BaseModel):
    max_length  : int   = 2048   # maximum is 8192
    top_p       : float = .8
    temperature : float = .01

class ChatInfo(ChatParam):
    query   : str
    history : List[dict] = []
    role    : str = 'user'

class Chatting(ChatInfo):
    html: bool = False

def ignore_bad_words(content: str) -> str:
    return re.sub('chatg[\w|-]+', '', content, flags=re.I)

def start_chat(conversation: Chatting) -> tuple[str, dict]:
    info = ChatInfo(**dict(conversation) | { "history": [ system_info ] })
    info.query = tw_to_cn.convert(info.query)
    response, history = model.chat(tokenizer, **dict(info))
    return run_task(response, ChatInfo(**dict(info) | { "history": history }))

def run_task(response: str | dict, param: ChatInfo) -> tuple[str, dict]:
    if isinstance(response, dict):
        func = getattr(Tools, response.get('name'))
        func_response = func(**response.get('parameters'))
        args = dict(param) | { "query": json.dumps(func_response, ensure_ascii=False) }
        response, history = model.chat(tokenizer, **(args | { "role": "observation" }))
        return run_task(response, ChatInfo(**dict(param) | { "history": history, "role": "user" }))
    else:
        return ( ignore_bad_words(response), param.history )

@router.post('/conversation', response_model=None, response_description='Conversation info')
async def create_chat_conversation(conversation: Chatting) -> HTMLResponse | JSONResponse:
    """
    Create an **AI chat conversation** for single round:

    - **query**: message content by user
    - **history**: conversation history list for assistant reference
    - **role**: chat role could be: `[user, assistant, system, observation]`
    - **html**: response to `HTML` context directly
    """
    begin = datetime.now()
    response, history = start_chat(conversation)
    resp = {
        "response"    : (response := ignore_bad_words(response)),
        "history"     : history,
        "datetime"    : (now := datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_time": (now - begin).total_seconds()
    }
    if conversation.html:
        return HTMLResponse(status_code=200, content=response)
    return JSONResponse(status_code=200, content=resp)

@router.websocket('/ws')
@websocket_catch
async def create_chat_stream(websocket: WebSocket):
    await websocket.accept()
    history: List[Dict[str, str]] = []
    while True:
        content = tw_to_cn.convert(await websocket.receive_text())
        streaming: Generator[tuple[str, List[Dict[str, str]]], None, None] = model.stream_chat(
            tokenizer, content, history, **dict(ChatParam()))
        for response, history in streaming:
            response = ignore_bad_words(response)
            await websocket.send_json({ "content": response, "completion": False })
            print(response)
            await asleep(0.1)
        await websocket.send_json({ "content": response, "completion": True })
        await asleep(1)

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_chat_demo(request: Request) -> HTMLResponse:
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('chat.html', context={ "request": request })

@router.get('/help')
async def show_toolkit() -> JSONResponse:
    resp = [ { tool["description"]: tool["parameters"] } for tool in Tools.tools ]
    return JSONResponse(status_code=200, content=resp)
