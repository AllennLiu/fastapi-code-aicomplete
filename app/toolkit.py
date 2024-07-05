import json, textwrap, traceback, requests, subprocess
from operator import itemgetter
from chatglm_cpp import ChatMessage
from typing import Any, Dict, List, Annotated, Callable, Generator

class Tools:
    tools = [
        {
            "name"       : "code_helper",
            "description": "调用 code_helper，程序语言是 xxx，提示是 xxx 功能",
            "parameters" : {
                "type"      : "object",
                "properties": {
                    "language": { "description": "程序语言 e.g. Python，Shell", "type": "string" },
                    "prompt"  : { "description": "提示要实现的功能", "type": "string" }
                },
                "required"  : [ 'language', 'prompt' ]
            }
        },
        {
            "name"       : "get_weather",
            "description": "根据城市获取当前天气",
            "parameters" : {
                "type"      : "object",
                "properties": { "city_name": { "description": "城市名称 e.g. 北京，上海", "type": "string" } },
                "required"  : [ 'city_name' ]
            }
        },
        {
            "name"       : "get_shell",
            "description": "在服务端本地通过 Shell 执行命令",
            "parameters" : {
                "type"      : "object",
                "properties": { "query": { "description": "执行命令 e.g. ls or ping 8.8.8.8", "type": "string" } },
                "required"  : [ 'query' ]
            }
        }
    ]

    @staticmethod
    def code_helper(language: str, prompt: str) -> str:
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
        根据城市获取当前天气
        """
        if not isinstance(city_name, str):
            raise TypeError('City name must be a string')
        keys = { "current_condition": [ 'temp_C', 'FeelsLikeC', 'humidity', 'weatherDesc', 'observation_time' ] }
        resp = requests.get(f'https://wttr.in/{city_name}?format=j1')
        resp.raise_for_status()
        resp: dict = resp.json()
        return json.dumps({ k: { _v: resp[k][0][_v] for _v in keys[k] } for k in keys })

    @staticmethod
    def get_shell(query: Annotated[str, 'The command should run in Linux shell', True]) -> str:
        """
        在服务端本地通过 Shell 执行命令
        """
        if not isinstance(query, str):
            raise TypeError('Command must be a string')
        try:
            result = subprocess.run(
                query, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return e.stderr

OBSERVATION_MAX_LENGTH = 1024
TOOL_SYSTEM_PROMPT = ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=textwrap.dedent(f"""\
Answer the following questions as best as you can. You have access to the following tools:
{json.dumps(Tools.tools, ensure_ascii=False, indent=4)}\
"""))

def build_tool_system_prompt(content: str) -> ChatMessage:
    """Appending all available :class:`~Tools` to system prompt message,
    this method is fitting with the ``GLM-4`` model."""
    content += '\n\n# 可用工具'
    for tool in Tools.tools:
        content += f'\n\n## {tool["name"]}\n\n{json.dumps(tool, ensure_ascii=False, indent=4)}'
        content += '\n在调用上述函数时，请使用 Json 格式表示调用的参数。'
    return ChatMessage(role=ChatMessage.ROLE_SYSTEM, content=content)

def is_function_call(message: ChatMessage) -> bool:
    """Determine whether the first line of message content which is
    included in currently available :class:`~Tools`."""
    return message.content.splitlines()[0] in map(itemgetter('name'), Tools.tools)

def run_function(name: str, arguments: str) -> str:
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

def remove_tool_calls_message(messages: List[ChatMessage]) -> Generator[ChatMessage, None, None]:
    """Removing if the role of :class:`~ChatMessage` which is a
    `observation` or `assistant tool calls` then pop it out."""
    for message in messages:
        if message.role == ChatMessage.ROLE_OBSERVATION:
            continue
        elif message.role == ChatMessage.ROLE_ASSISTANT and is_function_call(message):
            continue
        yield message
