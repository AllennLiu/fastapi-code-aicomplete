import json, textwrap
from operator import itemgetter
from pydantic import BaseModel, Field
from typing import Dict, List, Annotated, Final
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Request, Body, Path
from starlette.responses import JSONResponse, HTMLResponse
from ..config import get_settings
from ..db import RedisAsynchronous
from ..chats import REGISTERED_TOOLS

settings = get_settings()
router = APIRouter(prefix='/chat/utils', tags=[ 'Chat Utils' ], responses={ 404: dict(description='Not found') })
PATH_UUID: Final = Path(..., description='User `UUID`')
PATH_DATE: Final = Path(..., description='Tab created **datetime**')

class ChatTab(BaseModel):
    label: str = ''
    datetime: str = ''

class ChatRename(BaseModel):
    uuid: str = Field(..., description='User `UUID`')
    name: str = Field(..., description='The `Tab Name` which is going to change')
    datetime: str = Field(..., description='Tab created **datetime**')

@router.get('/demo', response_class=HTMLResponse, include_in_schema=False)
async def render_chat_demo(request: Request):
    template = Jinja2Templates(directory='./app/templates')
    return template.TemplateResponse('chat.html', context=dict(request=request))

@router.get('/help')
async def show_toolkit() -> JSONResponse:
    resp = list(map(itemgetter('description'), REGISTERED_TOOLS))
    return JSONResponse(resp)

@router.get('/tabs/{uuid}')
async def get_chat_tabs(uuid: Annotated[str, Path(..., description='User `UUID`')]) -> JSONResponse:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        name = f'talk-history-hash-{uuid}'
        tabs = [
            ChatTab(**json.loads(await r.hget(name, k) or '{}')).model_dump(mode='json')
            for k in await r.hkeys(name)
        ]
    resp = sorted(tabs, key=itemgetter('datetime'), reverse=True)
    return JSONResponse(resp)

@router.put('/tab/rename')
async def rename_chat_tab(chat_tab: Annotated[ChatRename, Body(...)]) -> JSONResponse:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        args = ( f'talk-history-hash-{chat_tab.uuid}', chat_tab.datetime )
        data = json.loads(await r.hget(*args) or '{}') | dict(label=chat_tab.name)
        await r.hset(*args, json.dumps(data, ensure_ascii=False))
        resp = ChatTab(**json.loads(await r.hget(*args) or '{}')).model_dump(mode='json')
    return JSONResponse(resp)

@router.get('/history/{uuid}/{datetime}')
async def get_chat_history_by_datetime(
    uuid: Annotated[str, PATH_UUID], datetime: Annotated[str, PATH_DATE]
) -> JSONResponse:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        data = json.loads(await r.hget(f'talk-history-hash-{uuid}', datetime) or '{}')
    return JSONResponse(data.get('history', []))

@router.delete('/history/{uuid}/{datetime}')
async def delete_chat_history_by_datetime(
    uuid: Annotated[str, PATH_UUID], datetime: Annotated[str, PATH_DATE]
) -> JSONResponse:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        await r.hdel(f'talk-history-hash-{uuid}', datetime)
        result = await r.hget(f'talk-history-hash-{uuid}', datetime) is None
    return JSONResponse(dict(ok=result))

@router.delete('/forget/{uuid}/{range}')
async def clean_chat_history(
    uuid : Annotated[str, PATH_UUID],
    range: Annotated[int, Path(..., description=textwrap.dedent("""\
    Chat history **forget range**: the range of `pair entry` is **count from the end**\n
    **Pair entry** means the history which is the role both of user and assistant.
    - `0 = let it go` _(all the history)_
    - `-1 = the last one`
    - `-2 = the last two`
    - and so on ...
    """))]
) -> JSONResponse:
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        history: List[Dict[str, str]] = json.loads(await r.get(f'talk-history-{uuid}') or '[]')
        await r.set(f'talk-history-{uuid}', json.dumps(history[: range * 2], ensure_ascii=False))
    return JSONResponse(dict(ok=True))
