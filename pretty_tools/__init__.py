from pathlib import Path

from .echo import X_Logging
from .resources import *

PATH_PRETTY = Path(__file__).parent
VERSION = "0.2.0"

# 类似pytorch的操作
from ._C_pretty_tools import *