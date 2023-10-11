import requests


def message_WeChat_send(title, message):
    url = "http://www.pushplus.plus/send"
    data_json = {
        "token": "c04fad3e0c01471ab58114c69730b1e7",
        "title": title,
        "content": message,
    }
    # 给自己发送消息

    requests.post(url, json=data_json, timeout=10)
    return
