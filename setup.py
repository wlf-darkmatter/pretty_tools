import os

import numpy as np
from setuptools import Extension, find_packages, setup
from glob import glob


if os.name == "nt":
    compile_args = {"gcc": ["/Qstd=c99"]}
else:
    compile_args = ["-Wno-cpp"]

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def get_all_ext_modules():
    ext_modules = []

    return ext_modules


setup(
    name="pretty_tools",
    version="0.2.1",
    author="Lingfeng Wang",
    author_email="wlf_x_contion@sina.com",
    description="A pretty tools",
    url="http://www.x-contion.top:1214/lab_share/pretty_tools",
    packages=find_packages(),
    # cmdclass={"build_ext": build_ext},
    ext_modules=get_all_ext_modules(),
    setup_requires=["setuptools>=60.0", "Cython", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "Cython",
        "numpy",
        "seaborn",
        "rich",
        "rich[jupyter]",
        "imagesize",
        "iteration_utilities",
        "colorlog",
        "joblib",
        "vispy",
        "imgaug",  # *图像增强库
        "pybind11",  # c++ binding
    ],
    # py_modules=['dataset', 'message', 'multi_works', 'progress', 'video', 'visualization', 'x_logger'],
    # extra_compile_args=["-pthread"],  # 启动多线程编译
)
