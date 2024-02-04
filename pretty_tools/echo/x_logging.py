from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime
from logging import Logger
from pathlib import Path

import click
from rich.logging import RichHandler

__dict_logging = {}

default_stream_level = logging.DEBUG
default_fileError_log_level = logging.ERROR
TRACE = 5
default_file_log_level = TRACE


class X_Logging:
    """
    Example
    -------

    >>> x_logger = X_Logging("Calib")
    >>> logger = x_logger.logger
    >>> logger.info("这样就能打印日志了")

    20240120 更新
    1. 如果是多卡训练，会出现日志的重复打印，所以添加初始化参数，说明这个日志是哪个卡的，避免重复打印，默认卡 0 会打印信息

    """

    def __init__(
        self,
        logname="base",
        dir_log=None,
        stream_level=default_stream_level,  # * 终端打印的等级
        file_log_level=default_file_log_level,  # * 日志文件的等级
        fileError_log_level=default_fileError_log_level,  # * 错误日志的等级
        multi_process_mode=False,  # * 是否多进程模式，如果是多进程模式，会通过，且仅会通过一个全局变量来判断当前进程的排序，默认使用最先启动的进程来打印，避免重复打印
    ) -> None:
        """
        Args:
            logname (str, optional): 日志名称. Defaults to "base".
            dir_log (str, optional): 日志文件目录. Defaults to None.
            stream_level (int, optional): 终端打印的等级. Defaults to default_stream_level.
            file_log_level (int, optional): 日志文件的等级. Defaults to default_file_log_level.
            fileError_log_level (int, optional): 错误日志的等级. Defaults to default_fileError_log_level.
            multi_process_mode (bool, optional): 是否多进程模式，如果是多进程模式，会通过，且仅会通过一个全局变量来判断当前进程的排序，默认使用最先启动的进程来打印，避免重复打印. Defaults to False.

        """
        self.logger: Logger = logging.getLogger(logname)
        self.multi_process_mode = multi_process_mode
        setattr(self.logger, "x_logging", self)

        self.__now_time = datetime.now().strftime("%Y-%m-%d")
        self.logger.setLevel(TRACE)

        # * 构造日志处理器
        # handler_Stream = logging.StreamHandler()
        handler_Stream = RichHandler(rich_tracebacks=True, tracebacks_suppress=[click], log_time_format="[%m-%d %H:%M:%S]")  # * 终端打印的就没必要显示年份了
        # * 设定日志处理等级
        handler_Stream.setLevel(stream_level)

        if dir_log is not None:
            self.add_file_logger(file_log_level, fileError_log_level)
        # * 设定日志打印格式
        # console_formatter = colorlog.ColoredFormatter(
        #     fmt=
        #     '%(thin_green)s[%(asctime)s%(thin_black)s.%(msecs)03d%(reset)s%(thin_white)s]%(reset)s%(log_color)s[%(levelname)s] %(filename)s%(thin_yellow)s:%(lineno)d -> %(reset)s%(log_color)s %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S',
        #     log_colors=self.log_colors_config)

        # handler_Stream.setFormatter(console_formatter)

        # * 分配处理器
        self.logger.addHandler(handler_Stream)

    def set_rank(self, world_size: int, local_size: int, rank: int):
        self.world_size = world_size  #! 当前启动的所有的进程的数量 (所有机器进程的和)
        self.rank = rank  #! 这个是全局的排序
        self.local_size = local_size
        # * 修改底层实现，这种方式有点笨拙
        self.multi_process_mode = True
        _log = self.logger._log

        def new_log(self: Logger, level: int, msg: object, args, exc_info=None, extra=None, stack_info: bool = False, stacklevel: int = 1, **kwargs):
            #! 如果rank 不是 0，则不打印，除非设置了 force 为 True
            force = kwargs.pop("force", False)
            if force or rank == 0:
                _log(level, msg, args, exc_info, extra, stack_info, stacklevel)  # todo 在日志文件中，打印的代码位置会一直显示这里的行号，20240204发现

        # * 替换原方法
        self.logger._log = new_log.__get__(self.logger, self.logger.__class__)

    def check(self):
        print("logging 示范")
        self.logger.debug("This is a customer debug message")
        self.logger.info("This is an customer info message")
        self.logger.warning("This is a customer warning message")
        self.logger.error("This is an customer error message")
        self.logger.critical("This is a customer critical message")
        if self.multi_process_mode:
            print(f"启动了多进程模式")

    def add_file_logger(self, dir_log, file_log_level=default_file_log_level, fileError_log_level=default_fileError_log_level):
        self.path = Path(dir_log).resolve().absolute()
        os.makedirs(self.path, exist_ok=True)

        handler_File = logging.handlers.TimedRotatingFileHandler(
            filename=str(self.path.joinpath(self.__now_time + ".log")),
            when="H",
            interval=24,
            backupCount=30,
        )
        handler_File_errorlogger = logging.handlers.TimedRotatingFileHandler(
            filename=str(self.path.joinpath(self.__now_time + "_error.log")),
            when="H",
            interval=24,
            backupCount=30,
        )
        # * 设定日志处理等级

        handler_File.setLevel(file_log_level)
        handler_File_errorlogger.setLevel(fileError_log_level)
        # * 设定日志打印格式
        # * Ns 是用于对齐的 N 表示字符长度, 负号表示左对齐
        """
                            levelname
                           |----------|
        2023-08-21 12:33:04       INFO Enrty to Logging Module!!
        2023-08-21 12:33:04      DEBUG This is Debugging Message!!
        2023-08-21 12:33:04       INFO This is Info Message!!
        2023-08-21 12:33:04    WARNING This is WARNING Message!!
                           |0123456789|
                           | width=10 |
        """
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s %(name)s %(filename)-15s| %(funcName)-20s line:%(lineno)5d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler_File.setFormatter(formatter)
        handler_File_errorlogger.setFormatter(formatter)
        # * 分配处理器
        self.logger.addHandler(handler_File)
        self.logger.addHandler(handler_File_errorlogger)


def build_logging(
    logname="base",
    dir_log=None,
    **kwargs,
) -> X_Logging:
    """
    Args:
        kwargs 参考 `class:X_Logging` 的传参
    """
    if logname not in __dict_logging:
        __dict_logging[logname] = X_Logging(
            logname=logname,
            dir_log=dir_log,
            **kwargs,
        )
    return __dict_logging[logname]
