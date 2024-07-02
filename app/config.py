import os
from typing import Dict
from textwrap import dedent
from functools import lru_cache
from starlette.config import Config
from pydantic_settings import BaseSettings

LANGUAGE_TAG: Dict[str, str] = {
    "Abap"         : "* language: Abap",
    "ActionScript" : "// language: ActionScript",
    "Ada"          : "-- language: Ada",
    "Agda"         : "-- language: Agda",
    "ANTLR"        : "// language: ANTLR",
    "AppleScript"  : "-- language: AppleScript",
    "Assembly"     : "; language: Assembly",
    "Augeas"       : "// language: Augeas",
    "AWK"          : "// language: AWK",
    "Basic"        : "' language: Basic",
    "C"            : "// language: C",
    "C#"           : "// language: C#",
    "C++"          : "// language: C++",
    "CMake"        : "# language: CMake",
    "Cobol"        : "// language: Cobol",
    "CSS"          : "/* language: CSS */",
    "CUDA"         : "// language: Cuda",
    "Dart"         : "// language: Dart",
    "Delphi"       : "{language: Delphi}",
    "Dockerfile"   : "# language: Dockerfile",
    "Elixir"       : "# language: Elixir",
    "Erlang"       : f"% language: Erlang",
    "Excel"        : "' language: Excel",
    "F#"           : "// language: F#",
    "Fortran"      : "!language: Fortran",
    "GDScript"     : "# language: GDScript",
    "GLSL"         : "// language: GLSL",
    "Go"           : "// language: Go",
    "Groovy"       : "// language: Groovy",
    "Haskell"      : "-- language: Haskell",
    "HTML"         : "<!--language: HTML-->",
    "Isabelle"     : "(*language: Isabelle*)",
    "Java"         : "// language: Java",
    "JavaScript"   : "// language: JavaScript",
    "Julia"        : "# language: Julia",
    "Kotlin"       : "// language: Kotlin",
    "Lean"         : "-- language: Lean",
    "Lisp"         : "; language: Lisp",
    "Lua"          : "// language: Lua",
    "Markdown"     : "<!--language: Markdown-->",
    "Matlab"       : f"% language: Matlab",
    "Objective-C"  : "// language: Objective-C",
    "Objective-C++": "// language: Objective-C++",
    "Pascal"       : "// language: Pascal",
    "Perl"         : "# language: Perl",
    "PHP"          : "// language: PHP",
    "PowerShell"   : "# language: PowerShell",
    "Prolog"       : f"% language: Prolog",
    "Python"       : "# language: Python",
    "R"            : "# language: R",
    "Racket"       : "; language: Racket",
    "RMarkdown"    : "# language: RMarkdown",
    "Ruby"         : "# language: Ruby",
    "Rust"         : "// language: Rust",
    "Scala"        : "// language: Scala",
    "Scheme"       : "; language: Scheme",
    "Shell"        : "# language: Shell",
    "Solidity"     : "// language: Solidity",
    "SPARQL"       : "# language: SPARQL",
    "SQL"          : "-- language: SQL",
    "Swift"        : "// language: swift",
    "TeX"          : f"% language: TeX",
    "Thrift"       : "/* language: Thrift */",
    "TypeScript"   : "// language: TypeScript",
    "Vue"          : "<!--language: Vue-->",
    "Verilog"      : "// language: Verilog",
    "Visual Basic" : "' language: Visual Basic"
}
DOTENV_CONFIG = os.path.join('app', '.env')
HUB_PATH = '/root/.cache/huggingface/hub'

class Database(BaseSettings):
    redis: str = '10.99.104.250:6379'

class ModelConfig(BaseSettings):
    chatbot_name : str = '/workspace/glm-4-9b-chat'
    copilot_name : str = '/workspace/codegeex2-6b'
    quantize     : int = 10
    dtype        : str = 'f16'

class Settings(BaseSettings):
    app_name  : str = 'Powerful API with 🇦🇮 🤖'
    desc      : str = dedent("""\
        Integrating as following **`LLM`** models to implement **Useful API** to community:

        - **`CodeGeeX2-6B`** 🔗 [https://github.com/THUDM/CodeGeeX2](https://github.com/THUDM/CodeGeeX2)
        - **`GLM-4-9B-Chat`** 🔗 [https://github.com/THUDM/GLM-4](https://github.com/THUDM/GLM-4)

        You could download these models from [Hugging Face](https://huggingface.co/THUDM) 🤗

        ## 🚀 Quick Start

        See following our **🎥 DEMO Web 🌐** pages for more experience with `🇦🇮`:

        - 💬 **Chatbot 🇦🇮** 🔗 [/chat/demo](/chat/demo)
        - 👩‍💻 **Coding Assistant** 🔗 [/copilot/coding/demo](/copilot/coding/demo)

        #### 🎮 Just enjoy it !\
        """)
    lang_tags : Dict[str, str] = LANGUAGE_TAG
    private   : Config = Config(DOTENV_CONFIG if os.path.isfile(DOTENV_CONFIG) else None)
    models    : ModelConfig = ModelConfig()
    db        : Database = Database()
    load_dev  : str = 'cuda' if os.getenv('LOAD_MODEL_DEVICE') == 'gpu' else 'cpu'

@lru_cache
def get_settings():
    return Settings()
