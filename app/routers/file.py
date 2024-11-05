import json
from typing import Dict, List, Annotated, Generator
from fastapi import APIRouter, HTTPException, status, Path
from starlette.responses import PlainTextResponse, StreamingResponse

from ..config import get_settings
from ..utils import RedisContextManager

settings = get_settings()
router = APIRouter(prefix='/file', tags=[ 'File' ], responses={ 404: { "description": "Not found" } })
PATH_UUID = Path(description='Generated script `UUID` number')

@router.get('/download/script/{uuid}', response_class=StreamingResponse)
def get_model_generated_script(uuid: Annotated[str, PATH_UUID]) -> StreamingResponse:
    with RedisContextManager(settings.db.redis) as r:
        query: Dict[str, str] = json.loads(r.hget('model-generated-scripts', uuid) or '{}')
        uuids = ', '.join(r.hkeys('model-generated-scripts'))
    if not query:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f'UUID Not Found, Available: [{uuids}]')
    with RedisContextManager(settings.db.redis) as r:
        query |= dict(download_count=query.get('download_count', 0) + 1)
        r.hset('model-generated-scripts', uuid, json.dumps(query, ensure_ascii=False))
    filename = query.get('filename', '')
    resp = StreamingResponse(query.get('code', ''), media_type='text/plain')
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}; filename*=utf-8''{filename}"
    return resp

@router.get('/tool/list', response_class=PlainTextResponse)
def get_tool_list() -> PlainTextResponse:
    def flatten(items: List[dict]) -> Generator[str, None, None]:
        for file in items:
            if isinstance(file.get('children', ''), list):
                yield from flatten(file["children"])
            else:
                if file.get('title'): yield file.get('title')
    with RedisContextManager(settings.db.redis) as r:
        scripts: List[str] = r.hkeys('gitlab-script-list')
        collections: dict = eval(r.hget('script-management-collections', 'Collection') or '{}')
    return '\n'.join(scripts + list(flatten(collections.get('children', []))))
