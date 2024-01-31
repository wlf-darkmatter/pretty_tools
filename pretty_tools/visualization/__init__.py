# * 手动添加一下matplotlib的中文字体，字体路径位于 /workspace/utils/pretty_tools/pretty_tools/resources/NotoColorEmoji-Regular.ttf
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from pretty_tools.resources import *

from .draw.draw import Pretty_Draw, Visiual_Tools
from .draw.draw_base import draw_bboxes
from .draw.draw_graph import Visual_Graph
from .draw.draw_track import Visual_Track

# 由于更改了字体导致显示不出负号，将配署文件中axes.unicode minus : True修改为False 就可以了，当然这而可以在代码中完成
mpl.rcParams["axes.unicode_minus"] = False

# * 开始添加注册中文字体
font_dirs = [str(PATH_RESOURCES)]
font_files = fm.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    fm.fontManager.addfont(font_file)
f"""
自己库里的字体名字叫做
{path_font_emoji.name}: Noto Color Emoji
{path_font_arial.name}: Arial
"""

# 打印所有可用字体族
# font_families = sorted(set([f.name for f in fm.fontManager.ttflist]))
# for font_family in font_families:
#     print(font_family)

__all__ = ["draw_bboxes", "bbox_convert_np", "Visual_Graph", "Visual_Track", "Pretty_Draw"]
classes = __all__
