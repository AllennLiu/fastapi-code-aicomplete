import re, json, httpx, colorama, datetime, operator, itertools, traceback, aiopathlib
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from chatglm_cpp import Pipeline, ChatMessage
from urllib.parse import ParseResult, urlparse
from fastapi import APIRouter, Request, Body, Path, Form
from typing import Any, Dict, List, Annotated, Final, Iterable, cast
from ..config import get_settings
from ..chats import ChatOption, ChatResponse
from ..utils import print_process, md_no_codeblock
from ..db import MongoAsynchronous, RedisAsynchronous

PARAM_PATH_PLAN_ID: Final = Path(..., example=15155, description='A number of ARES `Plan ID`')
FORM_RENEW: Final = Form(..., description='Renew the prediction jobs')
USER_PROMPT_EXAMPLE: Final[str] = f"""自动化脚本"SIT-BIOS-SUTInfoCheckTest"路径：
https://sms-sit.inventec.com.cn/#
/Scripts/SIT-BIOS-SUTInfoCheckTest"""
SYSTEM_PROMPT: Final[ChatMessage] = ChatMessage(role='system', content=f"""\
Please find the URLs or tool/script paths in user's input content if they exist.
Just return the result as a single JSON list, the format must be list[str].
There must be no duplicate items, and the number of items must not exceed 20.
I just need the compiled answers; I don't need code on how to implement it.
""")

class SummaryResult(BaseModel):
    paths: set[str] = Field(default_factory=set, description='parsed paths')
    urls: set[str] = Field(default_factory=set, description='parsed urls')
    others: set[str] = Field(default_factory=set, description='the other parsed contents')

class PrerequisiteTable(BaseModel):
    tools: set[str] | List[str] = Field(default_factory=set, description='predict prerequisite tools')
    elapsed_time: float = Field(default=0., description='predict elapsed time')
    prerequisite: str = Field(default='', description='saved prerequisite content')

class Prerequisite(BaseModel):
    bkm_uuid: str = Field(..., description='ARES BKM UUID')
    prerequisite: str = Field(..., description='ARES BKM prerequisite content')

class PrerequisiteJob(BaseModel):
    bkm_uuid: str = Field(..., description='ARES BKM UUID')
    completed: bool = Field(default=False, description='Whether the prediction is completed')
    rate: float = Field(default=0., description='The partial rate of the prediction progress')

class PlanChat(ChatOption):
    prompt: str = Field(..., examples=[ USER_PROMPT_EXAMPLE ], description='Message content')
    temperature: float = Field(default=.0, description='Default make responses more deterministic')

colorama.init(autoreset=True)
settings = get_settings()
router = APIRouter(prefix='/plan', tags=[ 'ARES Plan' ], responses={ 404: dict(description='Not found') })
job_default = PrerequisiteJob(bkm_uuid='')

async def cleanup_model_core_dump() -> None:
    """
    Cleanup the system core dump files (core.*) which is generated
    by model inference error, it's use to avoid the model's memory
    leak while processing the prediction.
    """
    path = aiopathlib.AsyncPath('/workspace')
    for core_dump in path.iterdir():
        if core_dump.name.startswith('core.'):
            await core_dump.unlink(missing_ok=True)

async def get_plan_uuid_by_id(plan_id: int) -> str:
    """Asynchronous to fetch the ARES plan ``UUID`` by ``ID`` from ``MongoDB``."""
    async with MongoAsynchronous(settings.db.mongo).connect() as m:
        db = m.chrysaetos
        collection = db.plans
        plan = cast(Dict[str, Any], await collection.find_one(dict(plan_id=plan_id)))
    return plan["plan_uuid"]

async def predict_prerequisites(
    pipeline: Pipeline,
    prerequisite: Prerequisite | PrerequisiteJob,
    table: PrerequisiteTable = PrerequisiteTable(),
    maximum_retry: int = 3
) -> PrerequisiteTable:
    """
    If the AI analysis returns unable to parse data, the system
    prompt should be re-used to attempt analysis again, until the
    maximum retry limit of ``3 times`` is reached, at which point
    the requests should return a `HTTP_500_INTERNAL_SERVER_ERROR`
    with an exception error message.

    Args:
        pipeline (Pipeline): model ``ChatGLM`` pipeline.
        prerequisite (Prerequisite | PrerequisiteJob): \
            model for case's prerequisite.
        table (PrerequisiteTable): saved prerequisite table. \
            Defaults to :class:`~PrerequisiteTable`.
        maximum_retry (int, optional): predict retry. Defaults to 3.

    Returns:
        PrerequisiteTable: table model include tools and elapsed time.
    """
    error_msg = f'{colorama.Fore.RED}Unable to parse the prompt:\n{"~" * 80}\n{table.prerequisite}\n{"~" * 80}'
    table.elapsed_time = 0.
    chat = PlanChat(prompt=f'{table.prerequisite}\nReturn a list[str] in JSON format.')
    chat_option = ChatOption(**chat.model_dump())
    messages = [ SYSTEM_PROMPT, ChatMessage(role='user', content=chat.prompt) ]
    for remain in range(maximum_retry - 1, -1, -1):
        try:
            print_process(f'BKM UUID is: {colorama.Fore.YELLOW}{prerequisite.bkm_uuid}')
            print_process(f'predict: {colorama.Fore.MAGENTA}{table.prerequisite}')
            begin = datetime.datetime.now(settings.timezone)
            response = cast(ChatMessage, pipeline.chat(
                messages, **chat_option.model_dump(), do_sample=chat.temperature > 0))
            now = datetime.datetime.now(settings.timezone)
            table.elapsed_time += (now - begin).total_seconds()
            content_without_md = md_no_codeblock(response.content)
            print_process(f'parsed:\n{colorama.Fore.BLUE}{content_without_md}')
            table.tools = cast(set[str], json.loads(content_without_md))
            print_process(f'response:\n{colorama.Fore.GREEN}{table.tools}')
            return table
        except json.decoder.JSONDecodeError:
            print(f'{error_msg}\nException error: Invalid Jsonify Content')
            messages[-1].content = f'{table.prerequisite}\nReturn the format of JSON list[str] only.'
        except httpx.ReadTimeout:
            print(f'{error_msg}\nException error: Predict TimeOut')
        except Exception as e:
            print(f'{error_msg}\nException error: {str(e)}')
            print(f'{colorama.Fore.RED}{traceback.format_exc()}{"~" * 80}')
        print_process(f'{colorama.Fore.YELLOW}retry remain {remain} times ...')
    return table

async def update_prerequisites_model(
    name: str, models: Iterable[Prerequisite | PrerequisiteJob], plan_uuid: str
) -> None:
    """
    Update the model's prerequisites data within an array then save
    it to ``Redis`` with ``hset`` method.

    Args:
        name (str): which :func:`~hset` name to be updated.
        models (Iterable[Prerequisite | PrerequisiteJob]): a list \
            of prerequisite or prerequisite job.
        plan_uuid (str): which key of ``ARES plan UUID`` to be queried.
    """
    async with RedisAsynchronous(**settings.redis.model_dump()).connect() as r:
        await r.hset(name, plan_uuid, json.dumps([ i.model_dump(mode='json') for i in models ]))

async def get_prerequisites_todo(plan_uuid: str) -> List[Prerequisite]:
    """
    Asynchronous to fetch all case's prerequisites by ARES plan UUID
    with ``Mongodb``, use for requests todo list.

    Args:
        plan_uuid (str): which ``ARES plan UUID`` to be queried.

    Returns:
        tuple[List[Prerequisite], str]: a tuple with list of \
            case UUID + prerequisite and plan UUID.
    """
    pipeline = [
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
        match = dict(case_status=1, case_report='1', case_plan_uuid=plan_uuid)
        pipeline[0: 0] = [ { "$match": match } ]
        cases = db.cases.aggregate(pipeline)
        prerequisites: List[Prerequisite] = []
        async for case in cases:
            bkm = case.get('case_bkm') or {}
            if not str(prerequisite := bkm.get('bkm_prerequisite') or '').strip():
                continue
            prerequisites.append(Prerequisite(bkm_uuid=bkm.get('bkm_uuid', ''), prerequisite=prerequisite))
    await update_prerequisites_model('model-prerequisites-todo', prerequisites, plan_uuid)
    return prerequisites

async def create_prerequisites_job(
    todos: List[Prerequisite], plan_uuid: str, renew: bool
) -> List[PrerequisiteJob]:
    """
    Asynchronous to create todo's job list by fetched todo list and
    iterate in todo list to check if each the prerequisite of attrs
    ``model-prerequisites-table`` is not equal as todo, then update
    the todo job's completed status set it to ``False``.

    This also to iterate in each todo's prerequisites, save it for
    updating the newest content if any difference been found.

    Args:
        prerequisites (List[Prerequisite]): fetched prerequisites.
        plan_uuid (str): which ``ARES plan UUID`` to be queried.
        renew (bool): whether to force renew the prediction jobs.

    Returns:
        List[PrerequisiteJob]: a list of prerequisite todo job.
    """
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        query = await r.hget('model-prerequisites-job', plan_uuid)
        jobs = [ PrerequisiteJob(**j) for j in cast(List[Dict[str, Any]], json.loads(query or '[]')) ]
        if renew:
            jobs.clear()
        if jobs and todos:
            print_process(f'{colorama.Fore.YELLOW}Found existing jobs to be updated.')
        for todo in todos:
            if todo.bkm_uuid in map(operator.attrgetter('bkm_uuid'), jobs):
                job = next((e for e in jobs if e.bkm_uuid == todo.bkm_uuid), job_default)
                query_table = await r.hget('model-prerequisites-table', todo.bkm_uuid)
                table = PrerequisiteTable(**json.loads(query_table or '{}'))
                if todo.prerequisite.strip() != table.prerequisite.strip():
                    print_process(f'Detected BKM: {colorama.Fore.YELLOW}{todo.bkm_uuid}{colorama.Fore.RESET} prerequisite changed.')
                    job.completed = False
                    job.rate = 0.
            else:
                jobs.append(job := PrerequisiteJob(**todo.model_dump()))
            if not job.completed:
                table_json = PrerequisiteTable(**todo.model_dump()).model_dump_json()
                await r.hset('model-prerequisites-table', todo.bkm_uuid, table_json)
    await update_prerequisites_model('model-prerequisites-job', jobs, plan_uuid)
    return jobs

async def predict_prerequisites_job(
    pipeline: Pipeline, jobs: List[PrerequisiteJob], plan_id: int, plan_uuid: str
) -> List[PrerequisiteTable]:
    """
    Asynchronous to predict the prerequisites of each test cases by
    ``ARES plan UUID`` with ``Redis`` and ``ChatGLM`` model, then
    update tools and elapsed time to ``model-prerequisites-table``
    in realtime during the prediction loop.

    Finally, update the ``model-prerequisites-job`` to indicate the
    job is completed.

    Args:
        pipeline (Pipeline): model ``ChatGLM`` pipeline.
        jobs (List[PrerequisiteJob]): jobs to be predicted.
        plan_id (int): display the specified ``ARES plan ID``.
        plan_uuid (str): which ``ARES plan UUID`` to be queried.

    Returns:
        List[PrerequisiteTable]: a list in each job's table.
    """
    tables: List[PrerequisiteTable] = []
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        num = sum(1 for _ in itertools.filterfalse(operator.attrgetter('completed'), jobs))
        rate = 100 / len(jobs) if jobs else 0.
        print_process(f'TODO plan: {colorama.Fore.MAGENTA}{plan_uuid}({plan_id}) {colorama.Fore.BLUE}case\'s job{colorama.Fore.RESET} remain: {colorama.Fore.RED}{num}')
        for job in jobs:
            query = await r.hget('model-prerequisites-table', job.bkm_uuid)
            table = PrerequisiteTable(**json.loads(query or '{}'))
            if job.completed:
                tables.append(table)
            else:
                predict_result = await predict_prerequisites(pipeline, job, table)
                await r.hset('model-prerequisites-table', job.bkm_uuid, predict_result.model_dump_json())
                tables.append(predict_result)
                job_save = next((e for e in jobs if e.bkm_uuid == job.bkm_uuid), job_default)
                job_save.completed = True
                job_save.rate = rate
                await update_prerequisites_model('model-prerequisites-job', jobs, plan_uuid)
    return tables

def summary_tools(tables: List[PrerequisiteTable]) -> tuple[SummaryResult, float]:
    """
    Processing structured response fulfill customize requirement
    and summarize each of the prerequisite elapsed time.

    Args:
        table (List[PrerequisiteTable]): a list in each job's table.

    Returns:
        tuple[SummaryResult, float]: a tuple with summary result \
            and total elapsed time.
    """
    pattern_iec = re.compile(r'.*inventec.com|inventeccorp-my.sharepoint.com.*')
    pattern_url = re.compile(r'^(https?|ftp)://([a-zA-Z0-9.-]+)(:\d+)?(/.*)?$')
    pattern_unix_path = re.compile(r'^(/[^/ ]*)+/?$')
    summary = SummaryResult()
    elapsed_time = sum(map(operator.attrgetter('elapsed_time'), tables))
    tools = itertools.chain.from_iterable(map(operator.attrgetter('tools'), tables))
    for tool in cast(Iterable[str], tools):
        if bool(pattern_url.match(tool)):
            url: ParseResult = urlparse(tool)
            condition = re.sub(r'^\/', '', url.path) or url.params or url.query
            if bool(pattern_iec.match(url.netloc)) and condition:
                summary.urls.add(tool)
        elif bool(pattern_unix_path.match(tool.strip())):
            summary.paths.add(tool)
        else:
            summary.others.add(tool)
    return summary, elapsed_time

@router.post('/prerequisite', response_model=ChatResponse, response_description='Prerequisites Summary')
async def prerequisite_summary(request: Request, data: Annotated[PlanChat, Body(...)]) -> ChatResponse:
    """Summarize a single prerequisite to makes structured output."""
    await cleanup_model_core_dump()
    predict_result = await predict_prerequisites(
        request.app.chatbot.model, Prerequisite(bkm_uuid='None', prerequisite=data.prompt))
    return ChatResponse(
        response=list(predict_result.tools),
        datetime=datetime.datetime.now(settings.timezone).strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time=predict_result.elapsed_time
    )

@router.post(
    '/prerequisite/{plan_id}',
    response_model=ChatResponse,
    response_description='Plan Prerequisites Summary'
)
async def plan_prerequisite_summary(
    request: Request,
    plan_id: Annotated[int, PARAM_PATH_PLAN_ID],
    renew: Annotated[bool, FORM_RENEW] = False
) -> ChatResponse:
    """
    Summarize the **prerequisite** in each test cases of a specified
    ``ARES plan``, just be patient it may take a few minutes to
    analyze prerequisite content to satisfy for ``system prompt``.

    Probably occur error when the ``system prompt`` is not able to
    parse the prerequisite content, model inference encountered OOM,
    please try again.
    """
    await cleanup_model_core_dump()
    todos = await get_prerequisites_todo(plan_uuid := await get_plan_uuid_by_id(plan_id))
    jobs = await create_prerequisites_job(todos, plan_uuid, renew)
    tables = await predict_prerequisites_job(request.app.chatbot.model, jobs, plan_id, plan_uuid)
    result, elapsed_time = summary_tools(tables)
    return ChatResponse(
        response=result.model_dump(mode='json'),
        datetime=str(datetime.datetime.now(settings.timezone)),
        elapsed_time=elapsed_time
    )

@router.get('/prerequisite/{plan_id}', response_description='Plan Predict Progress')
async def plan_prerequisite_progress(plan_id: Annotated[int, PARAM_PATH_PLAN_ID]) -> JSONResponse:
    """
    Summarize to calculate the progress of plan's prerequisite jobs
    for monitoring the progress of the prediction.
    """
    plan_uuid = await get_plan_uuid_by_id(plan_id)
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        query = await r.hget('model-prerequisites-job', plan_uuid)
        jobs = cast(List[Dict[str, Any]], json.loads(query or '[]'))
        completed = sum(map(operator.itemgetter('completed'), jobs))
        progress = int(round(sum(map(operator.itemgetter('rate'), jobs))))
    return JSONResponse(dict(plan_uuid=plan_uuid, progress=progress, completed=f'{completed}/{len(jobs)}'))

@router.delete('/prerequisite/{plan_id}')
async def remove_prerequisites_job(plan_id: Annotated[int, PARAM_PATH_PLAN_ID]) -> JSONResponse:
    """Remove the prerequisite job and todo list by plan UUID."""
    plan_uuid = await get_plan_uuid_by_id(plan_id)
    async with RedisAsynchronous(**settings.redis.model_dump(), decode_responses=True).connect() as r:
        for key in 'job', 'todo':
            await r.hdel(f'model-prerequisites-{key}', plan_uuid)
    return JSONResponse(dict(ok=True))
