import re, contextlib
from functools import wraps
from typing import Any, Callable, TypeVar
from fastapi import status, HTTPException, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from .config import get_settings

SETTINGS = get_settings()
NO_MODEL_T = TypeVar('NO_MODEL_T')

def load_model_catch(func: Callable[..., NO_MODEL_T]) -> Callable[..., NO_MODEL_T]:
    """
    **Tracing** wether the API requests has occurred an exception
    of ``Model Not Loaded`` and raise `status.HTTP_404_NOT_FOUND`
    or raise the root cause exception.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> NO_MODEL_T | None:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            key = '|'.join(SETTINGS.models.model_dump())
            if re.search(f'no attribute \'{key}\'', str(e)):
                error = dict(status_code=status.HTTP_404_NOT_FOUND, detail='Model Not Loaded')
                raise HTTPException(**error) from e
            raise e from e
    return wrapper

WEBSOCKET_T = TypeVar('WEBSOCKET_T')

def websocket_catch(func: Callable[..., WEBSOCKET_T]) -> Callable[..., WEBSOCKET_T]:
    """
    The main caveat of this method is that route can't access
    the request object in the wrapper and this primary intention
    of websocket exception purpose.

    由於裝飾器 (`decorator`) 會接收一個函數當參數，然後返 回新的\
    函數，這樣會導致被包裝函數的名子與注釋消失，如此便需要使用\
    :func:`~functools.wraps` 裝飾子修正

    函數的名子與注釋：`func.__name__`、`func.__doc__`
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> WEBSOCKET_T | None:
        exceptions = ( WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK, Exception )
        with contextlib.suppress(*exceptions):
            return await func(*args, **kwargs)
    return wrapper
