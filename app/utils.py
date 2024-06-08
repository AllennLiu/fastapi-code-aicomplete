from re import sub
from opencc import OpenCC
from functools import wraps
from contextlib import suppress
from fastapi import WebSocketDisconnect
from typing import Any, Dict, Callable, TypeVar
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

tw_to_cn = OpenCC('tw2sp')
cn_to_tw = OpenCC('s2tw')

WEBSOCKET_T = TypeVar('WEBSOCKET_T')

def websocket_catch(func: Callable[..., WEBSOCKET_T]) -> Callable[..., WEBSOCKET_T]:
    """
    The main caveat of this method is that route can't
    access the request object in the wrapper and this
    primary intention of websocket exception purpose.

    由於裝飾器 (`decorator`) 會接收一個函數當參數，然後返
    回新的函數，這樣會導致被包裝函數的名子與注釋消失，如此
    便需要使用 :func:`~functools.wraps` 裝飾子修正

    函數的名子與注釋：`func.__name__`、`func.__doc__`
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> WEBSOCKET_T | None:
        with suppress(WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK, Exception):
            return await func(*args, **kwargs)
    return wrapper

def copilot_prompt(lang_tags: Dict[str, str], lang: str, prompt: str) -> tuple[str, str]:
    """
    依指定代碼語言來擷取注釋符號，連接 prompt 生成代碼描述\n
    最後返回 ``prompt 生成代碼描述`` 與 ``注釋符號`` 的元組
    """
    comment_char = sub('\s+language:.*', '', lang_tags[lang])
    result = f'{lang_tags[lang]}\n{comment_char} {tw_to_cn.convert(prompt)}\n'
    return ( result, comment_char )
