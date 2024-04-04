import os
from distutils.command.build import build as _build

import numpy as np
from setuptools import Extension, find_packages, setup

if os.name == "nt":
    compile_args = {"gcc": ["/Qstd=c99"]}
else:
    compile_args = ["-Wno-cpp"]

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# * =================================  编译 cython_misc =============================================


def get_cython_misc_extensions():
    if os.name == "nt":
        compile_args = {"gcc": ["/Qstd=c99"]}
    else:
        compile_args = ["-Wno-cpp"]

    os.makedirs("pretty_tools/datastruct/build/", exist_ok=True)
    #! 如果有多个文件，则直接在下面这个列表中添加拓展即可
    ext_modules = [
        Extension(
            name="pretty_tools.datastruct._C_misc",
            sources=["pretty_tools/datastruct/src/_C_misc.pyx"],
            extra_compile_args=compile_args,
            include_dirs=[numpy_include],
        )
    ]
    return ext_modules


# * =================================  编译 cython_bbox =============================================
def get_ext_cythonbbox():
    ext_modules_cythonbbox = [
        Extension(
            name="pretty_tools.datastruct.cython_bbox",
            sources=["pretty_tools/datastruct/src/cython_bbox.pyx"],
            # extra_compile_args=compile_args,
            include_dirs=[numpy_include],
        )
    ]
    return ext_modules_cythonbbox


# * =================================  编译 np_enhance =============================================
def get_ext_numpy_enhance():
    ext_modules_np_enhance = [
        Extension(
            name="pretty_tools.datastruct._C_np_enhance",
            sources=["pretty_tools/datastruct/src/_C_np_enhance.pyx"],
            # extra_compile_args=compile_args,
            include_dirs=[numpy_include],
        )
    ]
    return ext_modules_np_enhance


def get_ext_graph_enhance():
    ext_modules_graph_enhance = [
        Extension(
            name="pretty_tools.datastruct._C_graph_enhance",
            sources=["pretty_tools/datastruct/src/_C_graph_enhance.pyx"],
            # extra_compile_args=compile_args,
            include_dirs=[numpy_include],
        )
    ]
    return ext_modules_graph_enhance


setup(
    name="pretty_tools",
    version="0.2.0",
    author="Lingfeng Wang",
    author_email="wlf_x_contion@sina.com",
    description="A pretty tools",
    url="http://git.x-contion.top/lab_share/pretty_tools",
    packages=find_packages(),
    ext_modules=get_ext_cythonbbox() + get_cython_misc_extensions() + get_ext_numpy_enhance() + get_ext_graph_enhance(),
    setup_requires=["setuptools>=18.0", "Cython", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "Cython",
        "numpy",
        "seaborn",
        "rich",
        "rich[jupyter]",
        "gpuinfo",
        "imagesize",
        "colorlog",
        "vispy",
        "imgaug",  # *图像增强库
        "streamlit",  # 自动生成网页的工具
    ],
    # py_modules=['dataset', 'message', 'multi_works', 'progress', 'video', 'visualization', 'x_logger'],
    extra_compile_args=["-pthread"],  # 启动多线程编译
)
