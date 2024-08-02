import json
from typing import Dict, Annotated
from starlette.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Path

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
        raise HTTPException(status_code=404, detail=f'UUID Not Found, Available: [{uuids}]')
    with RedisContextManager(settings.db.redis) as r:
        query |= dict(download_count=query.get('download_count', 0) + 1)
        r.hset('model-generated-scripts', uuid, json.dumps(query, ensure_ascii=False))
    filename = query.get('filename', '')
    resp = StreamingResponse(query.get('code', ''), media_type='text/plain')
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}; filename*=utf-8''{filename}"
    return resp
