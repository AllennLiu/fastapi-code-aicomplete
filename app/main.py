from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import chat, copilot

app_settings = get_settings()
app = FastAPI(
    title        = app_settings.app_name,
    description  = app_settings.desc,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*" ],
    allow_methods=[ "*" ],
    allow_headers=[ "*" ],
    allow_credentials=True)

app.include_router(chat.router)
app.include_router(copilot.router)
