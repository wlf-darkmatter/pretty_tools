import time
import pytest
import numpy as np
import scipy as sp
import http.client


def assert_rep_str(rep: str, excepted_rep: str):
    assert rep == excepted_rep, f"返回值错误, 期望: {excepted_rep}, 实际: {rep}"


def assert_rep_status(status: int, excepted_status: int):
    assert status == excepted_status, f"期望状态码: {excepted_status}, 实际: {status}"


def print_mean_cost(list_record: list[float]):
    mean, std = np.mean(list_record), np.std(list_record)
    print(f"平均响应时间: {mean*1000:0.2f} ± {std*1000:0.2f} ms")



class HTTP_Test_Base:
    """
    @brief: 用于测试 HTTP 服务的基类, 这类考虑放到 pretty_tools 中去


    """

    def __init__(self, ip, port):
        # 要测试服务端，就要建立客户端
        self.ip = ip
        self.port = port
        self.delay = 1000
        self.conn = http.client.HTTPConnection(self.ip, self.port, self.delay)

    def request_and_return_response(self, method: str, url: str, body=None, headers=None):
        print("\033[33m--> 发送请求: \033[36m* " + url + "\033[0m")

        assert method in ["GET", "POST", "PUT", "DELETE"], "不支持的请求方法"
        if headers is None:
            headers = {}
        self.conn.request(method, url, body, headers)
        response = self.conn.getresponse()
        status = response.status
        if status != 200:
            print(f"\033[35m 获取非 200 状态码: {status}\033[0m")

        str_type = response.headers.get("Content-Type")

        if str_type == "text/html" or str_type == "text/plain":
            resolved_rep = response.read().decode()
        elif str_type == "application/json":
            import json

            resolved_rep = json.loads(response.read())

        elif str_type == "image/jpeg":
            import cv2
            import numpy as np

            rep = response.read()
            img = cv2.imdecode(np.frombuffer(rep, np.uint8), cv2.IMREAD_COLOR)
            print(f"\033[33m<-- 收到回应: ({type(img)})\033[36m* 图片: 长={img.shape[0]}, 宽={img.shape[1]}.\033[0m")
            # 解码图片为 numpy
            return status, img

        else:
            print(f"\033[31m 未知类型: {str_type}\033[0m")
            resolved_rep = None

        print(f"\033[33m<-- 收到回应: ({type(resolved_rep)}) \033[36m* " + resolved_rep + "\033[0m")
        return status, resolved_rep
