import json, random, textwrap, traceback, requests
from chatglm_cpp import ChatMessage
from typing import Any, Dict, Callable

class Tools:
    tools = [
        {
            "name"       : "code_helper",
            "description": "调用 code_helper 工具，程序语言是 xxx，提示是 xxx 功能",
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
            "name": "random_number_generator",
            "description": "生成随机数字 x, s.t. range[0] <= x < range[1]",
            "parameters": {
                "type"      : "object",
                "properties": {
                    "seed": {
                        "description": "随机数作为对象的以真随机数（种子）为初始条件的随机数",
                        "type"       : "integer"
                    },
                    "range": {
                        "description": "生成数字的范围区间",
                        "type"       : "array",
                        "items"      : [ { "type": "integer" }, { "type": "integer" } ],
                    },
                },
                "required": [ 'seed', 'range' ],
            }
        }
    ]

    @staticmethod
    def code_helper(language: str, prompt: str) -> str:
        print('代码生成接口被调用！')
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
        print(f'posting {url}')
        resp = requests.post(url, headers=headers, data=json.dumps(data), verify=False)
        resp.raise_for_status()
        print(resp.json())
        return json.dumps(resp.json())

    @staticmethod
    def get_weather(city_name: str) -> str:
        """根据城市获取当前天气"""
        keys = { "current_condition": [ 'temp_C', 'FeelsLikeC', 'humidity', 'weatherDesc', 'observation_time' ] }
        resp = requests.get(f'https://wttr.in/{city_name}?format=j1')
        resp.raise_for_status()
        resp: dict = resp.json()
        return json.dumps({ k: { _v: resp[k][0][_v] for _v in keys[k] } for k in keys })

    @staticmethod
    def random_number_generator(seed: int, range: tuple[int, int]) -> int:
        """生成随机数字"""
        return random.Random(seed).randint(*range)

OBSERVATION_MAX_LENGTH = 1024
TOOL_SYSTEM_PROMPT = ChatMessage(role='system', content=textwrap.dedent(f"""\
Answer the following questions as best as you can. You have access to the following tools:
{json.dumps(Tools.tools, ensure_ascii=False, indent=4)}\
"""))

def run_function(name: str, arguments: str) -> str:
    """Run `observation` mode by assistant which is passing function
    name and arguments, finally return the stringify dict response.
    """
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
