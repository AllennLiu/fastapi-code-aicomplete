from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .utils import ML, load_llm
from .config import get_settings
from .routers import chat, chatutils, copilot, file

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
    if settings.models.chatbot.enabled:
        app.chat = ML(*load_llm(**dict(settings.models.chatbot), device=settings.load_dev))
    if settings.models.copilot.enabled:
        app.copilot = ML(*load_llm(**dict(settings.models.copilot), device=settings.load_dev))
    if settings.models.multi_modal.enabled:
        app.multi_modal = ML(*load_llm(**dict(settings.models.multi_modal), device=settings.load_dev))
    yield
    if settings.models.chatbot.enabled: del app.chat
    if settings.models.copilot.enabled: del app.copilot
    if settings.models.multi_modal.enabled: del app.multi_modal

app = FastAPI(title=settings.app_name, description=settings.desc, lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*" ],
    allow_methods=[ "*" ],
    allow_headers=[ "*" ],
    allow_credentials=True)

for r in [ chat.router, chatutils.router, copilot.router, file.router ]:
    app.include_router(r)
