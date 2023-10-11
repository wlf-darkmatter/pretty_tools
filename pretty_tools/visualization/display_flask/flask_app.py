import abc
import socket
import time
import urllib
from multiprocessing import Process
from typing import List, Tuple

from flask import Flask, Response, render_template, request

from .app_register import dict_id_app, dict_name_app
from .model import Display_Model


class Display_App_Flask:
    @staticmethod
    def get_dict_name_app():
        return dict_name_app

    def __init__(self, name, host=None, port=0) -> None:
        self._id = len(dict_id_app)
        self.name = name
        self.host = host
        if host is None:
            self.host = "127.0.0.1"
        if name in dict_name_app:
            raise NameError(f"Already existed app named '{name}'")
        dict_id_app[self._id] = self
        dict_name_app[name] = self

        self.app = Flask(name)
        self.__sock = socket.socket()
        self.__sock.bind(("", port))  # * 服务端
        self.port = self.__sock.getsockname()[1]

        self.server = Process(target=self.app.run, args=(self.host, self.port))
        self.list_module: List[Display_Model] = []

        self.url = f"http://{self.host}:{self.port}/"

    def shutdown(self, block=True):
        if self.server is not None:
            if self.server.is_alive():
                self.server.terminate()
        if block:
            while self.server.is_alive():
                time.sleep(0.001)

    @property
    def is_running(self):
        return self.server.is_alive()

    def app_init(self):
        for module in self.list_module:
            module.module_init()

        @self.app.route("/ready")
        def check_ready():
            for module in self.list_module:
                if not module.ready:
                    return "Not Ready"
            return "Ready"

        @self.app.route("/test")
        def str_test():
            return "Display_App"

        @self.app.route("/shutdown", methods=["GET"])
        def shutdown():
            self.shutdown()
            return "Server shutting down..."

        # * 批量注册 路由函数

        for module in self.list_module:
            for route, fn in module.dict_route_fn.items():
                self.app.route(route)(fn)

    def add_model(self, model: Display_Model):
        self.list_module.append(model)

    def run(self, block=True, timeout=10):
        self.app_init()
        # 超时时间设置为 timeout 秒
        self.__sock.close()
        self.server.start()
        # * 需要等待到启动完毕
        time_start = time.time()
        if block:
            while True:
                try:
                    req = urllib.request.Request(self.url + "ready")
                    resp = urllib.request.urlopen(req)
                    data = resp.read().decode("utf-8")
                    if data == "Ready":
                        break
                    if time.time() - time_start > timeout:
                        raise TimeoutError(f"Timeout {timeout} seconds")
                except urllib.error.URLError as e:
                    if e.reason.strerror == "Connection refused":
                        time.sleep(0.001)
                        continue
                    else:
                        raise e
                except Exception as e:
                    raise e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, id={self._id}, port={self.port})"

    def kill(self):
        if self.server is not None:
            self.shutdown()
        del dict_id_app[self._id]
        del dict_name_app[self.name]
