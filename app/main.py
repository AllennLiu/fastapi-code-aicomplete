from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import ai
from .config import get_settings

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

app.include_router(ai.router)
