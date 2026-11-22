import time, urllib.request, json

SVC = "http://172.17.0.1:8795"
UPSTREAM = "socks5://sockstest:socks-pass%401@172.17.0.1:40001"


def refresh():
    try:
        req = urllib.request.Request(
            SVC + "/bridge",
            data=json.dumps({"upstream": UPSTREAM}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        # 绝不 DELETE：DELETE 会 release_bridge 按 upstream 减引用计数，
        # 归零就把共享端口关了（502 的根因）。只累加租约即可。
        print("[bridge-keeper] port=%s ok" % d.get("port"), flush=True)
    except Exception as e:
        print("[bridge-keeper] failed: %s" % e, flush=True)


while True:
    refresh()
    time.sleep(120)   # 2 分钟，远小于 900s TTL，保证始终有活租约
