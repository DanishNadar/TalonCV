import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


statePath = Path(os.environ.get("TALONCV_WORKER_STATE_PATH", "/tmp/taloncv-worker-state.json"))


def writeWorkerState(status: str, **values: Any) -> None:
    statePath.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updatedAt": datetime.now(UTC).isoformat(), **values}
    temporary = statePath.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, statePath)


def readWorkerState() -> dict[str, Any]:
    try:
        return json.loads(statePath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unknown", "updatedAt": None}
