
import datetime
from zoneinfo import ZoneInfo
from functools import partial
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from ollama import AsyncClient, ChatResponse, Message
from typing import Any, Dict, List, AsyncIterator, Callable, Coroutine

class LlamaResponse(BaseModel):
    response: str
    messages: List[Dict[str, str] | Message] | None = []
    create_at: str = str(datetime.datetime.now(ZoneInfo('Asia/Shanghai')))
    elapsed_time: int | float | None = 0

    @field_validator('elapsed_time', mode='before')
    def prerequisite_total_duration(cls, value):
        """Prerequisite the ``total_duration`` by model response
        from nanosecond to second."""
        return round(value / 1e9, 4)

class LlamaOption(BaseModel):
    top_k: int | None = Field(default=None, description='Selects `top-k` highest probability tokens for output')
    top_p: float = Field(default=.9, description='Lower values **reduce `diversity`** and focus on more **probable tokens**')
    temperature: float = Field(default=.6, description='Higher will make **outputs** more `random` and `diverse`')
    repeat_penalty: float = Field(default=1., le=1., description='Higher values bot will not be repeating')

async def llama_conn(*args: Any, **keywords: Any) -> ChatResponse | AsyncIterator[ChatResponse]:
    return await AsyncClient(timeout=180.).chat(*args, **keywords)

@dataclass
class LlamaClient:
    base: Callable[..., Coroutine[Any, Any, ChatResponse | AsyncIterator[ChatResponse]]] = field(
        default_factory=lambda: partial(llama_conn, model='llama3.2'))
    vision: Callable[..., Coroutine[Any, Any, ChatResponse | AsyncIterator[ChatResponse]]] = field(
        default_factory=lambda: partial(llama_conn, model='llama3.2-vision'))

@dataclass(frozen=True)
class LlamaSettings:
    stream_wait_time: float = .05
