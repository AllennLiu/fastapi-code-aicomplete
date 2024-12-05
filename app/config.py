import os, textwrap
from functools import lru_cache
from starlette.config import Config
from typing import Dict, Final, cast
from pydantic_settings import BaseSettings

LANGUAGE_TAG: Final[Dict[str, Dict[str, str]]] = {
    "Abap"         : dict(mark='* language: Abap', ext='abap'),
    "ActionScript" : dict(mark='// language: ActionScript', ext='as'),
    "Ada"          : dict(mark='-- language: Ada', ext='ada'),
    "Agda"         : dict(mark='-- language: Agda', ext='agda'),
    "ANTLR"        : dict(mark='// language: ANTLR', ext='g4'),
    "AppleScript"  : dict(mark='-- language: AppleScript', ext='scpt'),
    "Assembly"     : dict(mark='; language: Assembly', ext='asm'),
    "Augeas"       : dict(mark='// language: Augeas', ext='aug'),
    "AWK"          : dict(mark='// language: AWK', ext='awk'),
    "Basic"        : dict(mark='\' language: Basic', ext='bas'),
    "C"            : dict(mark='// language: C', ext='c'),
    "C#"           : dict(mark='// language: C#', ext='cs'),
    "C++"          : dict(mark='// language: C++', ext='cpp'),
    "CMake"        : dict(mark='# language: CMake', ext='cmake'),
    "Cobol"        : dict(mark='// language: Cobol', ext='cbl'),
    "CSS"          : dict(mark='/* language: CSS */', ext='css'),
    "CUDA"         : dict(mark='// language: Cuda', ext='cu'),
    "Dart"         : dict(mark='// language: Dart', ext='dart'),
    "Delphi"       : dict(mark='{language: Delphi}', ext='dpr'),
    "Dockerfile"   : dict(mark='# language: Dockerfile', ext=''),
    "Elixir"       : dict(mark='# language: Elixir', ext='ex'),
    "Erlang"       : dict(mark=f'% language: Erlang', ext='erl'),
    "Excel"        : dict(mark='\' language: Excel', ext='xlsx'),
    "F#"           : dict(mark='// language: F#', ext='fs'),
    "Fortran"      : dict(mark='!language: Fortran', ext='f'),
    "GDScript"     : dict(mark='# language: GDScript', ext='gdscript'),
    "GLSL"         : dict(mark='// language: GLSL', ext='glsl'),
    "Go"           : dict(mark='// language: Go', ext='go'),
    "Groovy"       : dict(mark='// language: Groovy', ext='groovy'),
    "Haskell"      : dict(mark='-- language: Haskell', ext='hs'),
    "HTML"         : dict(mark='<!--language: HTML-->', ext='html'),
    "Isabelle"     : dict(mark='(*language: Isabelle*)', ext='thy'),
    "Java"         : dict(mark='// language: Java', ext='java'),
    "JavaScript"   : dict(mark='// language: JavaScript', ext='js'),
    "Julia"        : dict(mark='# language: Julia', ext='jl'),
    "Kotlin"       : dict(mark='// language: Kotlin', ext='kt'),
    "Lean"         : dict(mark='-- language: Lean', ext='lean'),
    "Lisp"         : dict(mark='; language: Lisp', ext='lisp'),
    "Lua"          : dict(mark='// language: Lua', ext='lua'),
    "Markdown"     : dict(mark='<!--language: Markdown-->', ext='md'),
    "Matlab"       : dict(mark=f'% language: Matlab', ext='m'),
    "Objective-C"  : dict(mark='// language: Objective-C', ext='objs'),
    "Objective-C++": dict(mark='// language: Objective-C++', ext='objs'),
    "Pascal"       : dict(mark='// language: Pascal', ext='pas'),
    "Perl"         : dict(mark='# language: Perl', ext='pl'),
    "PHP"          : dict(mark='// language: PHP', ext='php'),
    "PowerShell"   : dict(mark='# language: PowerShell', ext='ps1'),
    "Prolog"       : dict(mark=f'% language: Prolog', ext='pl'),
    "Python"       : dict(mark='# language: Python', ext='py'),
    "R"            : dict(mark='# language: R', ext='r'),
    "Racket"       : dict(mark='; language: Racket', ext='rkt'),
    "RMarkdown"    : dict(mark='# language: RMarkdown', ext='Rmd'),
    "Ruby"         : dict(mark='# language: Ruby', ext='rb'),
    "Rust"         : dict(mark='// language: Rust', ext='rs'),
    "Scala"        : dict(mark='// language: Scala', ext='scala'),
    "Scheme"       : dict(mark='; language: Scheme', ext='scm'),
    "Shell"        : dict(mark='# language: Shell', ext='sh'),
    "Solidity"     : dict(mark='// language: Solidity', ext='sol'),
    "SPARQL"       : dict(mark='# language: SPARQL', ext='srx'),
    "SQL"          : dict(mark='-- language: SQL', ext='sql'),
    "Swift"        : dict(mark='// language: swift', ext='swift'),
    "TeX"          : dict(mark=f'% language: TeX', ext='tex'),
    "Thrift"       : dict(mark='/* language: Thrift */', ext='thrift'),
    "TypeScript"   : dict(mark='// language: TypeScript', ext='ts'),
    "Vue"          : dict(mark='<!--language: Vue-->', ext='vue'),
    "Verilog"      : dict(mark='// language: Verilog', ext='v'),
    "Visual Basic" : dict(mark='\' language: Visual Basic', ext='vb')
}
DOTENV_CONFIG: Final[str] = os.path.join('app', '.env')
FASTAPI_ENV: Final[str] = os.getenv('FASTAPI_ENV') or 'stag'
HUB_PATH: Final[str] = '/root/.cache/huggingface/hub'

class Database(BaseSettings):
    redis: str = '10.99.104.251:8003' if FASTAPI_ENV == 'stag' else '172.17.1.242:6379'

class ChatModel(BaseSettings):
    name     : str = '/workspace/glm-4-9b-chat'
    quantize : int = 8
    dtype    : str = 'f16'
    enabled  : bool = (os.getenv('MODE_CHATBOT') or '').lower() == 'true'

class MultiModal(BaseSettings):
    name     : str = '/workspace/glm-4v-9b'
    quantize : int = 10
    dtype    : str = 'f16'
    enabled  : bool = (os.getenv('MODE_MULTIMODAL') or '').lower() == 'true'

class CodeModel(BaseSettings):
    name     : str = '/workspace/codegeex2-6b'
    quantize : int = 10
    dtype    : str = 'f16'
    enabled  : bool = (os.getenv('MODE_CODE') or '').lower() == 'true'

class ModelConfig(BaseSettings):
    chatbot     : ChatModel = ChatModel()
    copilot     : CodeModel = CodeModel()
    multi_modal : MultiModal = MultiModal()

class Settings(BaseSettings):
    app_name : str = 'Powerful API with 🇦🇮 🤖'
    desc     : str = textwrap.dedent("""\
        Integrating as following **`LLM`** models to implement **Useful API** to community:

        - **`CodeGeeX2-6B`** 🔗 [https://github.com/THUDM/CodeGeeX2](https://github.com/THUDM/CodeGeeX2)
        - **`GLM-4-9B-Chat`** 🔗 [https://github.com/THUDM/GLM-4](https://github.com/THUDM/GLM-4)
        - **`GLM-4v-9B`** 🔗 [https://github.com/THUDM/GLM-4](https://github.com/THUDM/GLM-4)

        You could download these models from [Hugging Face](https://huggingface.co/THUDM) 🤗

        ## 🚀 Quick Start

        See following our **🎥 DEMO Web 🌐** pages for more experience with `🇦🇮`:

        - 💬 **Chatbot 🇦🇮** 🔗 [/chat/utils/demo](/chat/utils/demo)
        - 👩‍💻 **Coding Assistant** 🔗 [/copilot/coding/demo](/copilot/coding/demo)

        #### 🎮 Just enjoy it !\
        """)
    lang_tags  : Dict[str, Dict[str, str]] = LANGUAGE_TAG
    private    : Config = Config(cast(str, DOTENV_CONFIG if os.path.isfile(DOTENV_CONFIG) else None))
    models     : ModelConfig = ModelConfig()
    db         : Database = Database()
    load_dev   : str = 'cuda' if os.getenv('LOAD_MODEL_DEVICE') == 'gpu' else 'cpu'
    ssl_active : bool = (os.getenv('SERVE_HTTPS') or '').lower() == 'true'

@lru_cache
def get_settings():
    return Settings()
