"""Локальная страница весов. Сеть здесь не учится.

Сервер только отдаёт HTML и JSON со текущими весами.
Если train.py уже писал output/checkpoint.pt, читаем его.
Если снимка нет, показываем начальные веса с тем же seed, что у обучения.
Страница опрашивает /api/weights сама, поэтому цифры на экране
меняются, когда обучение подменяет файл снимка.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from model import CHECKPOINT_PATH, ROOT, SEED, DigitNet

HOST = "127.0.0.1"
PORT = 8765
INDEX_PATH = ROOT / "web" / "index.html"

# Одна и та же начальная сеть на всё время процесса.
# Если создавать её заново на каждый запрос, картинки прыгали бы
# от одного случайного шума к другому, и смотреть было бы не на что.
torch.manual_seed(SEED)
INITIAL_MODEL = DigitNet()

# Последний удачно прочитанный снимок, если файл на миг занят заменой.
_cached_mtime = None
_cached_payload = None


def model_payload(model, trained, epoch, batch, test_accuracy):
    """Сложить веса в обычные списки, которые страница нарисует на canvas.

    Порядок слоёв фиксирован и совпадает с подписями в index.html:
    первый скрытый, второй скрытый, выход.
    """
    layers = []
    named = (
        ("hidden1", model.hidden1, "pixels"),
        ("hidden2", model.hidden2, "matrix"),
        ("output", model.output, "matrix"),
    )
    for name, layer, kind in named:
        # detach: это снимок для экрана, не часть графа ошибки.
        weight = layer.weight.detach().cpu()
        bias = layer.bias.detach().cpu()
        layers.append(
            {
                "name": name,
                "kind": kind,
                # weight[нейрон][вход]. Для первого слоя вход — пиксель 0…783.
                "weight": weight.tolist(),
                "bias": bias.tolist(),
            }
        )
    return {
        "trained": trained,
        "epoch": epoch,
        "batch": batch,
        "test_accuracy": test_accuracy,
        "layers": layers,
    }


def current_payload():
    """Прочитать снимок с диска или отдать начальные веса."""
    global _cached_mtime, _cached_payload

    if not CHECKPOINT_PATH.exists():
        return model_payload(INITIAL_MODEL, False, None, None, None)

    mtime = CHECKPOINT_PATH.stat().st_mtime
    if _cached_payload is not None and mtime == _cached_mtime:
        return _cached_payload

    try:
        # Файл пишем мы сами из train.py. weights_only=False, потому что
        # в словаре рядом с тензорами лежат номер эпохи и доля верных ответов.
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = DigitNet()
        model.load_state_dict(blob["state_dict"])
        payload = model_payload(
            model,
            True,
            int(blob["epoch"]),
            int(blob["batch"]),
            blob["test_accuracy"],
        )
    except (OSError, RuntimeError, KeyError, ValueError):
        # Снимок могли поймать ровно в момент замены. Отдаём прошлый удачный.
        if _cached_payload is not None:
            return _cached_payload
        return model_payload(INITIAL_MODEL, False, None, None, None)

    _cached_mtime = mtime
    _cached_payload = payload
    return payload


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = INDEX_PATH.read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/api/weights":
            body = json.dumps(current_payload(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self._send(404, "text/plain; charset=utf-8", "нет такой страницы".encode("utf-8"))

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Снимок не кэшировать: страница как раз ждёт свежие веса.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Опрос два раза в секунду забил бы консоль. Обычные запросы оставляем.
        if self.path.split("?", 1)[0] == "/api/weights":
            return
        super().log_message(fmt, *args)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Страница весов: http://{HOST}:{PORT}", flush=True)
    print("Обучение этот процесс не запускает. Остановить сервер: Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
