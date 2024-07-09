import re, json, textwrap, traceback, requests, inspect, paramiko
from operator import itemgetter
from chatglm_cpp import ChatMessage
from pydantic import BaseModel
from types import GenericAlias
from typing import Any, Dict, List, Annotated, Callable, Generator, get_origin

class ToolBaseParam(BaseModel):
    name        : str
    description : str

class ToolParameter(ToolBaseParam):
    type     : str
    required : bool

class ToolFunction(ToolBaseParam):
    params: List[Dict[str, str | bool]]

class Tools:
    """
    在此类下方添加``类静态方法`` (:class:`~staticmethod`) 可以
    自动 **注册模型调用** 的工具

    静态方法需确保以下注意事项：
    - 函数名称前缀必须使用 `get_xxx_xxx` 开头
    - 需填写函数注释作为工具调用的匹配语句
    - 需使用 :class:`~Annotated` 来定义参数注释 *(type hint)* 语句分词时将自动适配参数
    """
    @staticmethod
    def get_code_help(
        language: Annotated[str, 'language for programming', True],
        prompt: Annotated[str, 'prompt for describe the code', True]
    ) -> str:
        """
        代码小帮手，语言: xxx，提示: xxx
        """
        url = 'http://172.17.1.243:7862/copilot/coding'
        headers = { "Content-Type": "application/json" }
        data = {
            "max_length": 512,
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.2,
            "lang": language.capitalize(),
            "prompt": prompt, "html": False
        }
        print(f'posting: {url}')
        resp = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        resp.raise_for_status()
        print(resp.json())
        return json.dumps(resp.json())

    @staticmethod
    def get_weather(city_name: Annotated[str, 'The name of the city to be queried', True]) -> str:
        """
        根据城市名称获取当前天气
        """
        if not isinstance(city_name, str):
            raise TypeError('City name must be a string')
        keys = { "current_condition": [ 'temp_C', 'FeelsLikeC', 'humidity', 'weatherDesc', 'observation_time' ] }
        resp = requests.get(f'https://wttr.in/{city_name}?format=j1')
        resp: dict = resp.json()
        return json.dumps({ k: { _v: resp[k][0][_v] for _v in keys[k] } for k in keys })

    @staticmethod
    def get_shell(
        host: Annotated[str, 'Access host IP address', True],
        password: Annotated[str, 'SSH password', True],
        query: Annotated[str, 'Run command', True]
    ) -> str:
        """
        连线到 IP 地址，密碼是: xxx，执行命令 xxx
        """
        if not isinstance(query, str):
            raise TypeError('Command must be a string')
        timeout = 10.
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, 22, 'root', password, timeout=timeout)
        transport = client.get_transport()
        transport.set_keepalive(int(timeout))
        _, stdout, stderr = client.exec_command(query)
        print(output := ''.join(stderr.readlines() + stdout.readlines()))
        return output

def register_tool(instance: object) -> List[Dict[str, str | List[Dict[str, str | bool]]]]:
    """Parsing tool function calls by :class:`~staticmethod` \
    of :class:`~Tools` object, it generate function document
    and type hint of parameter to be system prompt definitions."""
    docs: List[Dict[str, str | List[Dict[str, str | bool]]]] = []
    for func in instance.__dict__.values():
        if not isinstance(func, staticmethod): continue
        tool_name = func.__name__
        tool_desc = inspect.getdoc(func).strip()
        python_params = inspect.signature(func).parameters
        tool_params: List[Dict[str, str | bool]] = []
        for name, param in python_params.items():
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                raise TypeError(f'Parameter `{name}` missing type annotation')
            if get_origin(annotation) != Annotated:
                raise TypeError(f'Annotation type for `{name}` must be :class:`~typing.Annotated`')
            _type, (desc, required) = annotation.__origin__, annotation.__metadata__
            _type: str = str(_type) if isinstance(_type, GenericAlias) else _type.__name__
            tool_params.append(dict(ToolParameter(
                name=name, description=desc, type=_type, required=required)))
        docs.append(dict(ToolFunction(name=tool_name, description=tool_desc, params=tool_params)))
    return docs

OBSERVATION_MAX_LENGTH = 1024
REGISTERED_TOOLS = register_tool(Tools)

def build_tool_system_prompt(content: str) -> ChatMessage:
    """Appending all available :class:`~Tools` to system prompt message,
    this method is fitting with the ``GLM-4`` model."""
    content += '\n\n# 可用工具'
    for tool in REGISTERED_TOOLS:
        content += f'\n\n## {tool["name"]}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}'
    return ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=content)

SYSTEM_PROMPT = build_tool_system_prompt(textwrap.dedent("""\
你是人工智能 AI 助手，你叫 Black.Milan，你是由 SIT TA 团队创造的。
你是基于 BU6 SIT TA 团队开发与发明的，你被该团队长期训练而成为优秀的助理。\
"""))

def func_called(message: ChatMessage) -> bool:
    """Determine whether the first line of message content which is
    included in currently available :class:`~Tools`."""
    return message.content.splitlines()[0] in map(itemgetter('name'), REGISTERED_TOOLS)

def run_func(name: str, arguments: str) -> str:
    """Run `observation` mode with assistant which function name or
    arguments were been passed, finally it returns the **stringify**
    :class:`~dict` response to next round of conversation."""
    print(f'Calling tool {name}, args: {arguments}')
    def tool_call(**kwargs: Any) -> Dict[str, Any]:
        return kwargs
    func: Callable[..., Any] = getattr(Tools, name)
    if func is None:
        return f'Function `{name}` is not defined'
    try:
        kwargs = eval(arguments, dict(tool_call=tool_call))
    except Exception:
        return f'Invalid arguments {arguments}'
    try:
        return str(func(**kwargs))
    except Exception:
        return traceback.format_exc()

def remove_tool_calls(messages: List[ChatMessage]) -> Generator[ChatMessage, None, None]:
    """Removing wether the :class:`~ChatMessage` which is a role
    of `observation` or `tool calls by assistant` then pop out of
    the list."""
    for message in messages:
        if message.role == ChatMessage.ROLE_OBSERVATION:
            continue
        elif message.role == ChatMessage.ROLE_ASSISTANT and func_called(message):
            continue
        yield message

def compress_message(messages: List[ChatMessage], max_length: int = 1536) -> List[ChatMessage]:
    """If a message content length is large over than ``512``
    bytes, truncate it to ``128`` bytes before the content,
    just for collecting the characteristic of message content."""
    if (re.search('call tool:', messages[-1].content, re.I)
        or messages[-1].role == ChatMessage.ROLE_OBSERVATION):
        return [ messages[0], messages[-1] ]
    content = ''
    for message in messages:
        if message.role != ChatMessage.ROLE_SYSTEM:
            message.content = re.sub(
                r'^```[^\S\r\n]*[a-z]*(?:\n(?!```$).*)*\n```', '', message.content, 0, re.MULTILINE)
            message.content = re.sub('get_[\w_]+\n+\{\S.*\}\n+', '', message.content)
            if len(message.content) > 512:
                message.content = message.content[:128]
        content += message.content
    if len(content) > max_length and len(messages) > 1:
        return compress_message([ messages[0] ] + messages[3:], max_length)
    return messages
