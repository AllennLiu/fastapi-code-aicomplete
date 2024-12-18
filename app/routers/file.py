import json
from redis.asyncio import Redis as AsyncRedis
from typing import Dict, List, Annotated, Final, Generator
from fastapi import APIRouter, HTTPException, status, Path
from starlette.responses import PlainTextResponse, StreamingResponse
from ..config import get_settings

settings = get_settings()
router = APIRouter(prefix='/file', tags=[ 'File' ], responses={ 404: dict(description='Not found') })
PARAM_PATH_UUID: Final = Path(..., description='Generated script `UUID` number')

@router.get('/download/script/{uuid}', response_class=StreamingResponse)
async def get_model_generated_script(uuid: Annotated[str, PARAM_PATH_UUID]) -> StreamingResponse:
    r = AsyncRedis(**settings.redis.model_dump(), decode_responses=True)
    query: Dict[str, str | int] = json.loads(await r.hget('model-generated-scripts', uuid) or '{}')
    uuids = ', '.join(await r.hkeys('model-generated-scripts'))
    if not query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'UUID Not Found, Available: [{uuids}]')
    query |= dict(download_count=int(query.get('download_count', 0)) + 1)
    await r.hset('model-generated-scripts', uuid, json.dumps(query, ensure_ascii=False))
    await r.close()
    filename = query.get('filename', '')
    resp = StreamingResponse(query.get('code', ''), media_type='text/plain')
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}; filename*=utf-8''{filename}"
    return resp

@router.get('/tool/list', response_class=PlainTextResponse)
async def get_tool_list() -> PlainTextResponse:
    def flatten(items: List[dict]) -> Generator[str, None, None]:
        for file in items:
            if isinstance(file.get('children', ''), list):
                yield from flatten(file["children"])
            else:
                if file.get('title'):
                    yield file["title"]
    r = AsyncRedis(**settings.redis.model_dump(), decode_responses=True)
    scripts: List[str] = await r.hkeys('gitlab-script-list')
    collections: dict = eval(await r.hget('script-management-collections', 'Collection') or '{}')
    await r.close()
    tools_text = '\n'.join(scripts + list(flatten(collections.get('children', []))))
    return PlainTextResponse(tools_text)
