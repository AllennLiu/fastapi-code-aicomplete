from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .utils import ML, device
from .config import get_settings
from .routers import chat, copilot

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
    param = dict(quantize=int(settings.models.quantize), device=settings.load_dev)
    app.chat = ML(*device(settings.models.chatbot_name, quantize=8, device=settings.load_dev))
    # app.copilot = ML(*device(settings.models.copilot_name, dtype=settings.models.dtype, **param))
    yield
    del app.chat
    # del app.copilot

app = FastAPI(title=settings.app_name, description=settings.desc, lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*" ],
    allow_methods=[ "*" ],
    allow_headers=[ "*" ],
    allow_credentials=True)

app.include_router(chat.router)
app.include_router(copilot.router)
