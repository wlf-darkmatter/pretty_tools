from pretty_tools.visualization.display_flask import Display_Model


class Model_Images(Display_Model):
    def __init__(self, route) -> None:
        super().__init__(route)

        self.dict_route_fn[route + "/check"] = self.check

    def module_init(self):
        self.ready = True

    def index(self):
        # * 首页
        return "default_page"

    def check(self):
        return "Model_Images"
