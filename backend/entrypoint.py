import os
import signal
import subprocess
import sys
import time


children: list[subprocess.Popen] = []
stopping = False


def stop(_signum=None, _frame=None) -> None:
    global stopping
    if stopping:
        return
    stopping = True
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline and any(child.poll() is None for child in children):
        time.sleep(0.2)
    for child in children:
        if child.poll() is None:
            child.kill()


def main() -> int:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    port = os.environ.get("PORT", "8000")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", port]
    )
    worker = subprocess.Popen([sys.executable, "-m", "backend.worker"])
    children.extend([api, worker])
    try:
        while not stopping:
            for child in children:
                code = child.poll()
                if code is not None:
                    stop()
                    return int(code)
            time.sleep(0.5)
    finally:
        stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
