import re, opencc, textwrap, redis, string
from functools import wraps
from itertools import chain
from contextlib import suppress
from fastapi import WebSocketDisconnect
from dataclasses import dataclass, field
from typing import Any, Dict, Callable, TypeVar
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast

tw_to_cn = opencc.OpenCC('tw2sp')
cn_to_tw = opencc.OpenCC('s2tw')

try:
    import chatglm_cpp
    enable_chatglm_cpp = True
except:
    print(textwrap.dedent("""\
    [WARN] chatglm-cpp not found. Install it by `pip install chatglm-cpp` for better performance.
    Check out https://github.com/li-plus/chatglm.cpp for more details.
    """))
    enable_chatglm_cpp = False

@dataclass
class ML:
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast = field()
    model     : chatglm_cpp.Pipeline = field()

class RedisContextManager:
    """A class used to handle Redis cache.

    Attributes
    ----------
    host: str
        The redis server host
    db: int
        The redis server db index
    decode_responses: bool
        Decode the responses of hash data

    Examples
    -------
    ```
    with RedisContextManager() as r:
        r.exists(name)    : Boolean
        r.ttl(name)       : Number
        r.hget(name, key) : String
        r.hgetall(name)   : Object<string>
        r.hkeys(name)     : Array
        r.hset(name, key, value)
        r.hsetnx(name, key, value)
        r.hdel(name, key)
        r.delete(name)
        r.expire(name, expired_time)
    ```
    Get hash keys:
    ```
    with RedisContextManager('127.0.0.1:6379') as r:
        print(r.hkeys('my-data'))
    ```
    >>> ['a', 'b', 'c', 'd']
    """
    def __init__(self, host: str, db: int = 0, decode_responses: bool = True) -> None:
        self.host = host.split(':')[0]
        self.port = int(host.split(':')[1])
        self.db = db
        self.rd = None
        self.decode_responses = decode_responses

    def __enter__(self) -> redis.client.Redis:
        pool: redis.connection.ConnectionPool = redis.connection.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=self.decode_responses)
        self.rd: redis.client.Redis = redis.client.Redis(connection_pool=pool)
        return self.rd

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        if self.rd is not None:
            self.rd.connection_pool.disconnect()
            self.rd.close()
        if any(( type, value, traceback )):
            assert False, value

def device(
    model_name: str,
    quantize  : int,
    dtype     : str  = 'f32',
    use_stream: bool = False,
    device    : str  = 'cpu',
    init_token: bool = True
) -> tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, chatglm_cpp.Pipeline]:
    """
    使用 :module:`~chatglm_cpp` 量化加速推理，實現通過 `CPU+MEM` 也能與
    模型實時交互，取代原先通過 :module:`~transformers` 進行推理的方式

    - 量化精度：``fp32 > fp16 > int8 > int4`` 越小模型推理能力越差，但對於硬體要求就越低
    - 同上，``fp32`` 和 ``fp16`` 需在有 `GPU` 的設備上運行才能擁有正常的推理表現
    """
    pretrained_args = dict(trust_remote_code=True, local_files_only=True, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **pretrained_args) if init_token else None
    if not use_stream:
        if enable_chatglm_cpp:
            print('Using chatglm-cpp to improve performance')
            if quantize in [ 4, 5, 8 ]:
                dtype = f'q{quantize}_0'
            pipeline = chatglm_cpp.Pipeline(model_name, dtype=dtype)
            return ( tokenizer, pipeline )
        print('chatglm-cpp not enabled, falling back to transformers')

    model = AutoModel.from_pretrained(model_name, **pretrained_args)
    return ( tokenizer, model.eval() )

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
        exceptions = ( WebSocketDisconnect, ConnectionClosedError, ConnectionClosedOK, Exception )
        with suppress(*exceptions):
            return await func(*args, **kwargs)
    return wrapper

def copilot_prompt(lang_tags: Dict[str, str], lang: str, prompt: str) -> tuple[str, str]:
    """
    依指定代碼語言來擷取注釋符號，連接 prompt 生成代碼描述\n
    最後返回 ``prompt 生成代碼描述`` 與 ``注釋符號`` 的元組

    >>> ('# language: Python\\n# 幫我寫一個冒泡排序\\n', '#')
    """
    comment_char = re.sub('\s+language:.*', '', lang_tags[lang])
    result = f'{lang_tags[lang]}\n{comment_char} {tw_to_cn.convert(prompt)}\n'
    return ( result, comment_char )

def block_bad_words(content: str) -> str:
    """屏蔽在聊天信息中不允許的字符"""
    regexp = 'chatg[\w|-]+|清华大学.*KEG|智谱.*AI|GLM.*\d+'
    return re.sub(regexp, 'Black.Milan', content, flags=re.I)

def remove_punctuation(content: str) -> str:
    """
    將下列拼接起來使用 :func:`~str.maketrans` 建立翻譯表，將所有的
    標點符號映射成 `None`
    - 英文字符的符號 `string.punctuation`
    - 常用中文字符和符號的 `Unicode ASCII 碼` 範圍 ``0x3000`` 到 ``0x303F``
    - 全角字符的 `Unicode 碼` 範圍 ``0xFF00`` 到 ``0xFFEF``\n
    最後使用 :func:`~str.translate` 刪除文本中所有的**中英文標點符號**
    """
    punctuation = string.punctuation
    punctuation += ''.join(chr(i) for i in chain(range(0x3000, 0x303F), range(0xFF00, 0xFFEF)))
    translator = str.maketrans('', '', punctuation)
    return re.sub(r'\s', '', content.translate(translator))
