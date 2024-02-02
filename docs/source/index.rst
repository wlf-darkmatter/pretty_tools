.. pretty_tools documentation master file, created by
   sphinx-quickstart on Thu Jul 27 13:18:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pretty Tools Documentation
========================================

Start
------------

安装
^^^^^^^^^^^
安装方法 1

.. code-block:: bash

   # 公网安装（如果公网服务器正常运行的话）
   pip install git+http://git.x-contion.top/lab_share/pretty_tools.git

   # or
   pip install git+http://lux4.x-contion.top:1213/lab_share/pretty_tools.git
   pip install git+http://lux.x-contion.top:1213/lab_share/pretty_tools.git


安装方法 2

.. code-block:: bash

   git clone http://lux.x-contion.top:1213/lab_share/pretty_tools.git
   cd pretty_tools
   pip install -e pretty_tools


.. note::

   模块在 Python3.9 版本上开发，后续迁移到了 Python3.11，较低的 Python 版本可能会有问题，但是没有测试过。


.. note::

   安装方法 1 和安装方法 2 的区别
   * 方法一会自动通过 git 克隆代码，然后进行编译安装，安装路径和其他通过 pip 安装的包别无二致；
   * 方法二会通过 git 将代码进行克隆，存放到当前目录下，然后通过 pip 进行安装，安装后的包路径是当前目录，
   而不是系统的包路径，这样做的好处是可以直接修改代码，不需要重新安装，
   但是如果修改了 cython 文件，需要重新编译。

模块说明
^^^^^^^^^^^
大板块分别有 **datastruct** 、 **echo** 、 **visualization**、 **resources**、 **multi_works**

**datastruct**: 有较多自定义数据结构，以及一些增强函数

**echo**: 简单的 echo 模块，可以直接打印输出，也可以打印输出到文件，也可以打印输出到控制台

**visualization**: 可视化函数

**resources**: 资源文件

**multi_works**: 多进程、多线程、多机并行处理相关的内容




.. important::

   这里之后会有更多的使用示例，方便大家查看，当然，得有人一起来开发，否则一个人是没有必要写这些东西的。

   如果你参与开发了，你会学到 cython 编译、更底层的 python 操作。

   师兄毕业之后也会继续开发，但是不会再教了，要学的话抓紧加入开发团队。

.. toctree::
   :maxdepth: 2
   :glob:
   :caption: PACKAGE REFERENCE

   module/datastruct
   module/echo
   module/visualization
   module/resources
   module/multi_works
   module/transform




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
