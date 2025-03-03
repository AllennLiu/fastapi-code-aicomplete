import os
import uuid
import datetime
from pydantic import BaseModel
from .config import Settings
from .db import RedisAsynchronous

class Script(BaseModel):
    code: str = ''
    filename: str = ''
    url: str = ''
    created_at: str = '2024-07-29T14:35:04.380'
    download_count: int = 0

async def save_code(settings: Settings, base_url: str, lang: str, code: str) -> str:
    """
    Save the model generated code to `Redis` which is going to be
    converted as the download **URL link** of API response.

    Args:
        settings (Settings): service :class:`~Settings` model.
        base_url (str): service base URL.
        lang (str): model generated language.
        code (str): model generated code.

    Returns:
        str: model generated code download URL.
    """
    data = Script(code=code, created_at=datetime.datetime.now(settings.timezone).strftime('%FT%T'))
    data.filename = (script_uuid := str(uuid.uuid4())) + f'.{settings.lang_tags[lang]["ext"]}'
    data.url = os.path.join(base_url, 'file/download/script', script_uuid)
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        await r.hset('model-generated-scripts', script_uuid, data.model_dump_json())
    return data.url
