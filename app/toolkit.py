import os, json, requests, inspect, paramiko
from types import GenericAlias
from pydantic import BaseModel
from typing import Dict, List, Annotated, Generator, get_origin

from .config import get_settings
from .utils import RedisContextManager

settings = get_settings()

class ToolBaseParam(BaseModel):
    name        : str
    description : str

class ToolParameter(ToolBaseParam):
    type     : str
    required : bool

class ToolFunction(ToolBaseParam):
    params: List[Dict[str, str | bool]]

class ToolDownload(BaseModel):
    filename: str
    link: str

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
        代码生成，语言: xxx，提示: 写一个 xxxxxxxx 的功能
        """
        url = 'https://ares-copilot.sit.ipt.inventec:7860/copilot/coding'
        headers = { "Content-Type": "application/json" }
        payload = json.dumps(dict(
            max_length  = 1024,
            top_k       = 1,
            top_p       = 0.95,
            temperature = 0.2,
            lang        = language.capitalize(),
            prompt      = prompt,
            html        = False
        ))
        print(f'posting: {url}')
        response = requests.post(url, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        print(data := response.json())
        message = f'耗时：{round(data["elapsed_time"])}秒，请点击连结下载脚本'
        return json.dumps(dict(message=message, url=data.get('url')), ensure_ascii=False)

    @staticmethod
    def get_weather(city_name: Annotated[str, 'The name of the city to be queried', True]) -> str:
        """
        帮我查城市的天气
        """
        attrs = [ 'temp_C', 'FeelsLikeC', 'humidity', 'weatherDesc', 'observation_time' ]
        keys = { "current_condition": attrs }
        resp = requests.get(f'https://wttr.in/{city_name}?format=j1', timeout=60)
        resp_json: dict = resp.json()
        data = { k: { _v: resp_json[k][0][_v] for _v in keys[k] } for k in keys }
        data["current_condition"] |= dict(city_name=city_name)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def get_shell(
        host: Annotated[str, 'SSH IP address', True],
        password: Annotated[str, 'SSH password', True],
        query: Annotated[str, 'Run command', True]
    ) -> str:
        """
        连线到 IP 地址，密碼: ******，执行命令 xxx
        """
        assert host.strip().lower() not in ('localhost', '127.0.0.1'), 'Invalid hostname'
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, 22, 'root', password, timeout=(timeout := 10.))
        transport = client.get_transport()
        transport.set_keepalive(int(timeout))
        _, stdout, stderr = client.exec_command(query)
        print(output := ''.join(stderr.readlines() + stdout.readlines()))
        data = dict(output=output, status_code=stdout.channel.recv_exit_status())
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def get_file_available() -> str:
        """
        有哪些工具可以下载
        """
        url = 'http://ares-ai.sit.ipt.inventec:7860/file/tool/list'
        return json.dumps(dict(message='点击连结查看可下载的工具清单', url=url), ensure_ascii=False)

    @staticmethod
    def get_file_download(
        filename: Annotated[str, 'Download filename', True],
        path: Annotated[str, 'File saved path', True],
        host: Annotated[str, 'SSH IP address', True],
        password: Annotated[str, 'SSH password', True]
    ) -> str:
        """
        下载工具 xxx 到路径: /tmp 下，服务器 IP: xxx，密码: ******
        """
        def search(items: List[dict], filename: str) -> Generator[str, None, None]:
            for file in items:
                if isinstance(file.get('children', ''), list):
                    yield from search(file["children"], filename)
                else:
                    if filename.lower() in file.get('title', '').lower():
                        yield file.get('path')
        validate_path = json.loads(Tools.get_shell(host, password, f'ls {path}'))
        if validate_path.get('status_code', -1) != 0:
            return validate_path.get('output', '无法连线或发生例外错误')
        files: List[ToolDownload] = []
        responses: List[Dict[str, str]] = []
        api_url = 'https://ares-g2-fastapi.sit.ipt.inventec/api/v1'
        with RedisContextManager(settings.db.redis) as r:
            collections: dict = eval(r.hget('script-management-collections', 'Collection') or '{}')
            for script in r.hkeys('gitlab-script-list'):
                if filename.lower() in script.lower():
                    script_data: dict = eval(r.hget('gitlab-script-list', script) or '{}')
                    files.append(ToolDownload(
                        filename=f'{script}-{script_data.get("rev", "master")}.zip',
                        link=os.path.join(api_url, 'scripts/download', script, 'master')))
        for tool in search(collections.get('children', []), filename):
            files.append(ToolDownload(
                filename=os.path.basename(tool),
                link=os.path.join(api_url, 'collection/download', tool[1:])))
        for file in files:
            dest = os.path.join(path, file.filename)
            cmd = f'wget --no-check-certificate {file.link} -O {dest}'
            resp = json.loads(Tools.get_shell(host, password, cmd))
            result = '成功' if resp["status_code"] == 0 else '失败'
            responses.append(f'下载工具 {file.filename} {result}')
        if not responses:
            responses = dict(message='找不到该工具，或是您可以问我：【有哪些工具可以下载】')
        return json.dumps(responses, ensure_ascii=False)

def register_tool(instance: object) -> List[Dict[str, str | List[Dict[str, str | bool]]]]:
    """
    Parsing tool function calls by :class:`~staticmethod` of\
    :class:`~Tools` object, it generate function document and
    type hint of parameter to be system prompt definitions.
    """
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
