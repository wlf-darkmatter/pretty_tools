import abc
from typing import Dict, List, Tuple


class Display_Model:
    """
    是 Display_App 的模块，所有模块都要继承这个类
    """

    def __init__(self, route: str) -> None:
        assert route.startswith("/"), "route must start with '/'"
        self.route = route

        self.ready = False
        self.dict_route_fn: Dict[str, function] = {}  # * 路由函数, key: route, value: function

    @abc.abstractmethod
    def module_init(self):
        pass

    @abc.abstractmethod
    def module_close(self):
        pass
