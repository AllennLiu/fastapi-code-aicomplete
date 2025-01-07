import re, functools, contextlib
from typing import Any, Callable, Coroutine, Final, TypeVar
from fastapi import status, HTTPException, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from .config import get_settings

SETTINGS: Final = get_settings()
NO_MODEL_T: Final = TypeVar('NO_MODEL_T')

def load_model_catch(
    func: Callable[..., Coroutine[Any, Any, NO_MODEL_T]]
) -> Callable[..., Coroutine[Any, Any, NO_MODEL_T]]:
    """
    **Tracing** wether the API requests has occurred an exception
    of ``Model Not Loaded`` and raise `status.HTTP_404_NOT_FOUND`
    or raise the root cause exception.
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> NO_MODEL_T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            pattern = re.compile(f'no attribute \'{"|".join(SETTINGS.models.model_dump())}\'')
            if pattern.search(str(e)):
                raise HTTPException(status.HTTP_404_NOT_FOUND, 'Model Not Loaded') from e
            raise e from e
    return wrapper

WEBSOCKET_T: Final = TypeVar('WEBSOCKET_T')

def websocket_catch(
    func: Callable[..., Coroutine[Any, Any, WEBSOCKET_T | None]]
) -> Callable[..., Coroutine[Any, Any, WEBSOCKET_T | None]]:
    """
    The main caveat of this method is that route can't access
    the request object in the wrapper and this primary intention
    of websocket exception purpose.

    由於裝飾器 (`decorator`) 會接收一個函數當參數，然後返 回新的\
    函數，這樣會導致被包裝函數的名子與注釋消失，如此便需要使用\
    :func:`~functools.wraps` 裝飾子修正

    函數的名子與注釋：`func.__name__`、`func.__doc__`
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> WEBSOCKET_T | None:
        exceptions = (WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK, Exception)
        with contextlib.suppress(*exceptions):
            return await func(*args, **kwargs)
    return wrapper
