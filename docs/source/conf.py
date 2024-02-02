# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys

import pretty_tools
import sphinx

project = "pretty_tools"
copyright = "2023, Lingfeng Wang"
author = "Lingfeng Wang"
release = "0.1.10.0"

sys.path.insert(0, str(pretty_tools.PATH_PRETTY))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#! 对 cython 文件的文档生成需要编译最新版的二进制文件才行

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  #! 增加[source]查看源码功能。
    "sphinx-mathjax-offline",  #! 增加math渲染
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinx_autodoc_typehints",  #! 生成 cython 文档
    "sphinxcontrib.programoutput",
    "sphinxcontrib.mermaid",  # 绘图
]

exclude_patterns = []
autodoc_member_order = "bysource"  # * 根据代码的定义顺序进行展示
language = "zh_CN"
suppress_warnings = ["autodoc.import_object"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "torch-gemetric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_style = "css/x_contion.css"  #! 制定一个自己的附加主题格式
# 隐藏模块名称
add_module_names = False
