"""Локальная страница весов. Сеть здесь не учится.

Сервер только отдаёт HTML и JSON со текущими весами.
Если train.py уже писал output/checkpoint.pt, читаем его.
Если снимка нет, показываем начальные веса с тем же seed, что у обучения.
Страница опрашивает /api/weights сама, поэтому цифры на экране
меняются, когда обучение подменяет файл снимка.
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch

from model import CHECKPOINT_PATH, HISTORY_DIR, ROOT, SEED, DigitNet

HOST = "127.0.0.1"
PORT = 8765
INDEX_PATH = ROOT / "web" / "index.html"

# Одна и та же начальная сеть на всё время процесса.
# Если создавать её заново на каждый запрос, картинки прыгали бы
# от одного случайного шума к другому, и смотреть было бы не на что.
torch.manual_seed(SEED)
INITIAL_MODEL = DigitNet()

# Отдельный кэш на каждый файл. Общий кэш отдал бы веса другого кадра.
# Значение: (mtime, payload). Если файл поймали в момент замены, остаётся прошлый.
_cache = {}
# Имя кадра — ровно пять цифр, как 00010.pt. Иначе это путь мимо каталога истории.
FRAME_NAME = re.compile(r"^\d{5}$")


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
        "step": None,
        "epoch": epoch,
        "batch": batch,
        "test_accuracy": test_accuracy,
        "layers": layers,
    }


def history_names():
    """Имена кадров без загрузки весов. Пустой список, если обучения ещё не было."""
    if not HISTORY_DIR.is_dir():
        return []
    names = []
    for path in HISTORY_DIR.glob("*.pt"):
        if FRAME_NAME.fullmatch(path.stem):
            names.append(path.stem)
    names.sort()
    return names


def payload_from_file(path, step):
    """Прочитать один кадр. step — номер из имени файла, не из чужого кэша."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        cached = _cache.get(key)
        if cached is not None:
            return cached[1]
        return model_payload(INITIAL_MODEL, False, None, None, None)

    cached = _cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        # Файл пишем мы сами из train.py. weights_only=False, потому что
        # в словаре рядом с тензорами лежат номер эпохи и доля верных ответов.
        # Закрыть файл сразу: иначе Windows не даёт train.py подменить снимок.
        with path.open("rb") as handle:
            blob = torch.load(handle, map_location="cpu", weights_only=False)
        model = DigitNet()
        model.load_state_dict(blob["state_dict"])
        payload = model_payload(
            model,
            True,
            int(blob["epoch"]),
            int(blob["batch"]),
            blob["test_accuracy"],
        )
        # Номер шага берём из имени файла: в старом снимке поля step могло не быть.
        payload["step"] = step if step is not None else blob.get("step")
    except (OSError, RuntimeError, KeyError, ValueError):
        if cached is not None:
            return cached[1]
        return model_payload(INITIAL_MODEL, False, None, None, None)

    _cache[key] = (mtime, payload)
    return payload


def current_payload():
    """Последний снимок или начальные веса, если файла ещё нет."""
    if not CHECKPOINT_PATH.exists():
        return model_payload(INITIAL_MODEL, False, None, None, None)
    return payload_from_file(CHECKPOINT_PATH, None)


def frame_payload(name):
    """Один кадр истории. Неверное имя не превращается в путь по диску."""
    if not FRAME_NAME.fullmatch(name or ""):
        return None
    path = HISTORY_DIR / f"{name}.pt"
    if not path.is_file():
        return None
    return payload_from_file(path, int(name))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Query отдельно от пути: иначе ?frame=00010 потеряется до разбора.
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            body = INDEX_PATH.read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if path == "/api/history":
            body = json.dumps(history_names()).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        if path == "/api/weights":
            frame = parse_qs(parsed.query).get("frame", [None])[0]
            if frame is None:
                payload = current_payload()
            else:
                payload = frame_payload(frame)
                if payload is None:
                    self._send(404, "text/plain; charset=utf-8", "нет такого кадра".encode("utf-8"))
                    return
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
        if urlparse(self.path).path in ("/api/weights", "/api/history"):
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
