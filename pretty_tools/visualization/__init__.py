# * 手动添加一下matplotlib的中文字体，字体路径位于 /workspace/utils/pretty_tools/pretty_tools/resources/NotoColorEmoji-Regular.ttf
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
from pretty_tools.resources import *
from PIL import ImageFont


from .draw.draw import Pretty_Draw, Visiual_Tools
from .draw.draw_base import draw_bboxes
from .draw.draw_graph import Visual_Graph
from .draw.draw_track import Visual_Track

# 由于更改了字体导致显示不出负号，将配署文件中 axes.unicode minus : True修改为False 就可以了，当然这而可以在代码中完成
mpl.rcParams["axes.unicode_minus"] = False

# * 开始添加注册中文字体
font_dirs = [str(PATH_RESOURCES)]
font_files = fm.findSystemFonts(fontpaths=font_dirs)
Dict_PIL_font = {}
for font_file in font_files:
    fm.fontManager.addfont(font_file)
    font = ImageFont.truetype(font_file)
    Dict_PIL_font[font.font.family] = font

f"""
自己库里的字体名字叫做
{path_font_arial.name}: Arial
{path_font_time_new_roman.name}: Times New Roman
{path_font_msyh.name}: Microsoft YaHei
"""

plt.rcParams["font.family"] = ["Times New Roman", "Microsoft YaHei"]

# 打印所有可用字体族
# font_families = sorted(set([f.name for f in fm.fontManager.ttflist]))
# for font_family in font_families:
#     print(font_family)
