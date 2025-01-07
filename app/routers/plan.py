import re, httpx, colorama, datetime
from ollama import ChatResponse, Message
from fastapi import APIRouter, Body, Path
from urllib.parse import ParseResult, urlparse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Annotated, Final, cast
from ..utils import print_process
from ..config import get_settings
from ..db import MongoAsynchronous
from ..llama import LlamaOption, LlamaResponse, LlamaClient

PARAM_PATH_PLAN_ID: Final = Path(..., example=15155, description='A number of ARES `Plan ID`')
USER_PROMPT_EXAMPLE: Final[str] = f"""自动化脚本"SIT-BIOS-SUTInfoCheckTest"路径：
https://sms-sit.inventec.com.cn/#
/Scripts/SIT-BIOS-SUTInfoCheckTest"""
SYSTEM_PROMPT: Final[Message] = Message(role='system', content=f"""\
Please find paths or URLs to tools and scripts within the user's input text.""")

colorama.init(autoreset=True)
settings = get_settings()
client = LlamaClient()
router = APIRouter(prefix='/plan', tags=[ 'ARES Plan' ], responses={ 404: dict(description='Not found') })

class Tool(BaseModel):
    path: str = Field(..., description='Reference filepath for tools or scripts')
    url: str = Field(..., description='URL for tools or scripts')

class Tools(BaseModel):
    tools: List[Tool]

class ToolSummary(BaseModel):
    paths: set[str]
    urls: set[str]

class PlanChat(LlamaOption):
    prompt: str = Field(..., examples=[ USER_PROMPT_EXAMPLE ], description='Message content')
    temperature: float = Field(default=.0, description='Default make responses more deterministic')

class Prerequisite(BaseModel):
    bkm_uuid: str = Field(..., description='ARES BKM UUID')
    prerequisite: str = Field(..., description='ARES BKM prerequisite content')

    @field_validator('prerequisite', mode='before')
    def prerequisite_without_newline(cls, value):
        return str(value).replace('\n', '')

async def ai_summary(prerequisite: Prerequisite, maximum_retry: int = 3) -> tuple[Tools, int]:
    """If the AI analysis returns unable to parse data, the system
    prompt should be re-used to attempt analysis again, until the
    maximum retry limit of ``3 times`` is reached, at which point
    the requests should return a `HTTP_500_INTERNAL_SERVER_ERROR`
    with an exception error message.

    Args:
        prerequisite (Prerequisite): model for case's prerequisite
        maximum_retry (int, optional): predict retry. Defaults to 3.

    Returns:
        tuple[Tools, int]: a tuple of tool's list and elapsed time
    """
    error_msg = f'{colorama.Fore.RED}Unable to parse the prompt:\n{"~" * 80}\n{prerequisite.prerequisite}\n{"~" * 80}'
    elapsed_time: int = 0
    chat = PlanChat(prompt=f'{prerequisite.prerequisite}. Return a list of tools in JSON format.')
    for retry in range(maximum_retry):
        try:
            print_process(f'BKM UUID is: {colorama.Fore.YELLOW}{prerequisite.bkm_uuid}')
            print_process(f'predict: {colorama.Fore.MAGENTA}{prerequisite.prerequisite}')
            response = cast(ChatResponse, await client.base(
                messages = [ SYSTEM_PROMPT, Message(role='user', content=chat.prompt) ],
                format   = Tools.model_json_schema(),
                options  = LlamaOption(**chat.model_dump()).model_dump()
            ))
            elapsed_time += cast(int, response.total_duration)
            tools = Tools.model_validate_json(response["message"]["content"])
            print_process(f'response:\n{colorama.Fore.GREEN}{tools}')
            return tools, elapsed_time
        except httpx.ReadTimeout:
            print(f'{error_msg}\nException error: Predict TimeOut')
        except Exception as e:
            print(f'{error_msg}\nException error: {str(e)}')
        print_process(f'{colorama.Fore.YELLOW}retry {retry + 1} times ...')
    return Tools(tools=[]), elapsed_time

async def get_plan_prerequisites(plan_id: int) -> List[Prerequisite]:
    """Asynchronous to fetch all case's prerequisites by
    ``ARES plan ID`` with ``Mongodb``.

    Args:
        plan_id (int): ``ARES plan ID``.

    Returns:
        List[Dict[str, str]]: a list of case UUID + prerequisite.
    """
    pipeline = [
        { "$match": { "case_status": 1, "case_report": "1" } },
        {
            "$lookup": {
                "from"    : "bkms",
                "let"     : { "bkm_uuid": "$case_bkm_uuid" },
                "pipeline": [
                    { "$match": { "$expr": { "$eq": [ '$bkm_uuid', '$$bkm_uuid' ] } } },
                    { "$project": { "bkm_id": 1, "bkm_uuid": 1, "bkm_prerequisite": 1, "_id": 0 } }
                ],
                "as": "case_bkm"
            }
        },
        { "$unwind": { "path": "$case_bkm", "preserveNullAndEmptyArrays": True } }
    ]
    async with MongoAsynchronous(settings.db.mongo).connect() as m:
        db = m.chrysaetos
        collection = db.plans
        plan = cast(Dict[str, Any], await collection.find_one(dict(plan_id=plan_id)))
        cases = db.cases.aggregate(pipeline)
        prerequisites: List[Prerequisite] = []
        async for case in collection.find(dict(case_plan_uuid=plan["plan_uuid"])):
            bkm = case.get('case_bkm') or {}
            if not str(prerequisite := bkm.get('bkm_prerequisite') or '').strip():
                continue
            prerequisites.append(Prerequisite(bkm_uuid=bkm.get('bkm_uuid', ''), prerequisite=prerequisite))
    return prerequisites

def tools_summary(tools: Tools) -> ToolSummary:
    """Processing structured response fulfill customize requirement."""
    pattern_unix_path = re.compile(r'^(/[^/ ]*)+/?$')
    summary = ToolSummary(paths=set(), urls=set())
    for tool in tools.tools:
        parsed_url: ParseResult = urlparse(tool.url)
        if 'inventec.com' in parsed_url.netloc:
            summary.urls.add(tool.url)
        if bool(pattern_unix_path.match(tool.path)):
            summary.paths.add(tool.path)
    return summary

@router.post('/prerequisite', response_model=LlamaResponse, response_description='Prerequisites Summary')
async def prerequisite_summary(data: Annotated[PlanChat, Body(...)]) -> LlamaResponse:
    """Summarize a single prerequisite to makes structured output."""
    tools, elapsed_time = await ai_summary(Prerequisite(bkm_uuid='None', prerequisite=data.prompt))
    return LlamaResponse(
        response=tools.model_dump(),
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=elapsed_time
    )

@router.post(
    '/plan/prerequisite/{plan_id}',
    response_model=LlamaResponse,
    response_description='Plan Prerequisites Summary'
)
async def plan_prerequisite_summary(plan_id: Annotated[int, PARAM_PATH_PLAN_ID]) -> LlamaResponse:
    """Summarize the **prerequisite** in each test cases of a specified
    ``ARES plan``, just be patient it may take a few minutes to analyze
    prerequisite content to satisfy for ``system prompt``.
    """
    tools = Tools(tools=[])
    elapsed_time = 0
    prerequisites = await get_plan_prerequisites(plan_id)
    print_process(f'Total process {colorama.Fore.MAGENTA}case{colorama.Fore.RESET} number: {colorama.Fore.RED}{len(prerequisites)}')
    for prerequisite in prerequisites:
        _tools, _elapsed_time = await ai_summary(prerequisite)
        tools.tools.extend(_tools.tools)
        elapsed_time += _elapsed_time
    return LlamaResponse(
        response=tools_summary(tools).model_dump(mode='json'),
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=elapsed_time
    )
