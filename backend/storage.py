import hashlib
import shutil
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from backend.settings import Settings
from backend.security import validateStoragePath


class ObjectStorage(Protocol):
    def ping(self) -> bool: ...
    def downloadRecording(self, objectPath: str, destination: Path) -> None: ...
    def uploadArtifact(self, objectPath: str, source: Path, contentType: str) -> dict[str, Any]: ...


class SupabaseObjectStorage:
    def __init__(self, settings: Settings):
        if not settings.supabaseConfigured:
            raise RuntimeError("Supabase service configuration is missing.")
        from supabase import Client, create_client

        self.settings = settings
        self.client: Client = create_client(settings.supabaseUrl, settings.supabaseServiceRoleKey)
        self.recordingsBucket = settings.recordingsBucket
        self.artifactsBucket = settings.artifactsBucket

    def ping(self) -> bool:
        self.client.storage.get_bucket(self.recordingsBucket)
        self.client.storage.get_bucket(self.artifactsBucket)
        return True

    def downloadRecording(self, objectPath: str, destination: Path) -> None:
        path = validateStoragePath(objectPath)
        url = (
            f"{self.settings.supabaseUrl}/storage/v1/object/authenticated/"
            f"{quote(self.recordingsBucket, safe='')}/{quote(path, safe='/')}"
        )
        request = Request(
            url,
            headers={
                "apikey": self.settings.supabaseServiceRoleKey,
                "Authorization": f"Bearer {self.settings.supabaseServiceRoleKey}",
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(request, timeout=120) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)

    def uploadArtifact(self, objectPath: str, source: Path, contentType: str) -> dict[str, Any]:
        path = validateStoragePath(objectPath)
        with source.open("rb") as stream:
            self.client.storage.from_(self.artifactsBucket).upload(
                path,
                stream,
                file_options={"content-type": contentType, "upsert": "true"},
            )
        digestBuilder = hashlib.sha256()
        with source.open("rb") as payload:
            while chunk := payload.read(1024 * 1024):
                digestBuilder.update(chunk)
        return {
            "storage_path": path,
            "content_type": contentType,
            "size_bytes": source.stat().st_size,
            "sha256": digestBuilder.hexdigest(),
        }
