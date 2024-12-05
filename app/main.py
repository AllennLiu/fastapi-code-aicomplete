from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

from .utils import ML, load_llm
from .config import get_settings
from .routers import chat, chatutils, copilot, file

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
    for model in (model_dict := settings.models.model_dump()):
        if not model_dict[model].get('enabled'): continue
        setattr(app, model, ML(*load_llm(**model_dict[model], device=settings.load_dev)))
    yield
    for model in model_dict:
        if model_dict[model].get('enabled'):
            delattr(app, model)

app = FastAPI(title=settings.app_name, description=settings.desc, lifespan=lifespan)
app.mount('/static', StaticFiles(directory='app/static'), name='static')
cors_allows = dict(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
app.add_middleware(CORSMiddleware, **cors_allows)
if settings.ssl_active:
    app.add_middleware(HTTPSRedirectMiddleware)
for r in chat, chatutils, copilot, file: app.include_router(r.router)
