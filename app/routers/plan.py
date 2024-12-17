import json, httpx, colorama, datetime
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from ollama import ChatResponse, Message
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Any, Dict, List, Annotated, Final, cast
from ..config import get_settings
from ..utils import md_no_codeblock, is_office_file
from ..llama import LlamaOption, LlamaResponse, LlamaClient

PROMPT_FOR_SINGLE_ARRAY: Final[str] = """Just return a single JSON array, the array format must follow: ['A', 'B', 'C', …]."""
USER_PROMPT: Final[str] = f"""I don't need code to teach me how to write;
{PROMPT_FOR_SINGLE_ARRAY}
I just need you to help me find the paths and download links for the tools or scripts here:"""
USER_PROMPT_FOR_JSON: Final[str] = f"""I don't need code to teach me how to write;
{PROMPT_FOR_SINGLE_ARRAY}
Please help me find the JSON array within it:"""
USER_PROMPT_EXAMPLE: Final[str] = f"""自动化脚本"SIT-BIOS-SUTInfoCheckTest"路径：
https://sms-sit.inventec.com.cn/#
/Scripts/SIT-BIOS-SUTInfoCheckTest"""
SYSTEM_PROMPT: Final[Message] = Message(role='system', content=f"""\
Please find all tool or script paths in the input text if they exist.
{PROMPT_FOR_SINGLE_ARRAY}
I just need the compiled answers; I don't need code on how to implement it.
If none are found, return an empty array.""")

colorama.init(autoreset=True)
settings = get_settings()
client = LlamaClient()
router = APIRouter(prefix='/plan', tags=[ 'ARES Plan' ], responses={ 404: dict(description='Not found') })

class LlamaChat(LlamaOption):
    prompt: str = Field(..., examples=[ USER_PROMPT_EXAMPLE ], description='Message content')
    system: str = Field(default=SYSTEM_PROMPT["content"], description='Role `system` content for declare bot')

class Plan(BaseModel):
    plan_id: int = Field(..., examples=[ 15155 ], description='ARES Plan ID')

class Prerequisite(BaseModel):
    bkm_uuid: str = Field(..., description='ARES BKM UUID')
    prerequisite: str = Field(..., description='ARES BKM prerequisite content')

async def model_predict(chat: LlamaChat) -> ChatResponse:
    options = LlamaOption(**chat.model_dump()).model_dump() | dict(temperature=.0)
    messages = [ SYSTEM_PROMPT ]
    if chat.system != SYSTEM_PROMPT.content:
        messages[0]["content"] = chat.system
    messages.append(Message(role='user', content=chat.prompt))
    return cast(ChatResponse, await client.base(messages=messages, options=options))

async def ai_summary(prerequisite: Prerequisite, maximum_retry: int = 3) -> tuple[List[str], int]:
    """If the AI analysis returns unable to parse data, the system
    prompt should be re-used to attempt analysis again, until the
    maximum retry limit of ``3 times`` is reached, at which point
    the requests should return a `HTTP_500_INTERNAL_SERVER_ERROR`
    with an exception error message.
    """
    chat = LlamaChat(prompt=f'{USER_PROMPT} {prerequisite.prerequisite}')
    error_msg = f'Unable to parse the prompt:\n{"~" * 80}\n{prerequisite.prerequisite}\n{"~" * 80}'
    elapsed_time: int = 0
    for _ in range(maximum_retry):
        try:
            print(f'{colorama.Fore.LIGHTCYAN_EX} ➠{colorama.Fore.RESET} BKM UUID is: {colorama.Fore.YELLOW}{prerequisite.bkm_uuid}')
            print(f'{colorama.Fore.LIGHTCYAN_EX} ➠{colorama.Fore.RESET} predict: {colorama.Fore.MAGENTA}{prerequisite.prerequisite}')
            response = await model_predict(chat)
            elapsed_time += response["total_duration"]
            jsonify = md_no_codeblock(response["message"]["content"])
            print(f'{colorama.Fore.LIGHTCYAN_EX} ➠{colorama.Fore.RESET} before: {colorama.Fore.BLUE}{jsonify}')
            tools = [ e for e in cast(List[str], json.loads(jsonify)) if not is_office_file(e) ]
            print(f'{colorama.Fore.LIGHTCYAN_EX} ➠{colorama.Fore.RESET} after: {colorama.Fore.GREEN}{tools}')
            return tools, elapsed_time
        except json.decoder.JSONDecodeError:
            tools = [ f'{error_msg}\nException error: Invalid Jsonify Content' ]
            chat.prompt = f'{USER_PROMPT_FOR_JSON} {jsonify}'
        except httpx.ReadTimeout:
            tools = [ f'{error_msg}\nException error: Model Predict TimeOut' ]
        except Exception as e:
            tools = [ f'{error_msg}\nException error: {str(e)}' ]
        print(f'{colorama.Fore.RED}{tools[0]}')
    return tools, elapsed_time

async def get_plan_prerequisites(plan_id: int) -> List[Prerequisite]:
    """Asynchronous to fetch all case's prerequisites by
    ``ARES plan ID`` with ``Mongodb``.

    Args:
        plan_id (int): ``ARES plan ID``.

    Returns:
        List[Dict[str, str]]: a list of case UUID + prerequisite.
    """
    client = AsyncIOMotorClient(f'mongodb://{settings.db.mongo}')
    db = client["chrysaetos"]
    collection = db["plans"]
    plan = cast(Dict[str, Any], await collection.find_one(dict(plan_id=plan_id)))
    collection = db["cases"]
    prerequisites: List[Prerequisite] = []
    async for case in collection.find(dict(case_plan_uuid=plan["plan_uuid"])):
        bkm = case.get('case_bkm') or {}
        if not str(prerequisite := bkm.get('bkm_prerequisite') or '').strip():
            continue
        prerequisites.append(Prerequisite(bkm_uuid=bkm.get('bkm_uuid', ''), prerequisite=prerequisite))
    return prerequisites

@router.post('/prerequisite', response_model=None, response_description='Prerequisites Summary')
async def prerequisite_summary(data: Annotated[LlamaChat, Body(...)]) -> LlamaResponse:
    tools, elapsed_time = await ai_summary(Prerequisite(bkm_uuid='unknown', prerequisite=data.prompt))
    return LlamaResponse(
        response=json.dumps(list(set(tools)), ensure_ascii=False),
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=elapsed_time
    )

@router.post('/plan/prerequisite', response_model=None, response_description='Plan Prerequisites Summary')
async def plan_prerequisite_summary(data: Annotated[Plan, Body(...)]) -> LlamaResponse:
    """Summarize the **prerequisite** in each test cases of a specified
    ``ARES plan``, just be patient it may take a few minutes to analyze
    prerequisite content to satisfy for ``system prompt``.

    _(More parameters usage please refer to `Schema`)_
    """
    tools, elapsed_time = [], 0
    prerequisites = await get_plan_prerequisites(data.plan_id)
    for prerequisite in prerequisites:
        _tools, _elapsed_time = await ai_summary(prerequisite)
        tools.extend(_tools)
        elapsed_time += _elapsed_time
    return LlamaResponse(
        response=json.dumps(list(set(tools)), ensure_ascii=False),
        create_at=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=elapsed_time
    )
