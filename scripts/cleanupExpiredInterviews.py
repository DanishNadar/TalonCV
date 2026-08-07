import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from backend.settings import getSettings


def cleanupExpiredInterviews(dryRun: bool = False) -> dict[str, int]:
    settings = getSettings()
    if not settings.supabaseConfigured:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
    from supabase import create_client

    client = create_client(settings.supabaseUrl, settings.supabaseServiceRoleKey)
    sessions = client.table("interview_sessions").select("id").lt("expires_at", datetime.now(UTC).isoformat()).execute().data or []
    deleted = 0
    objects = 0
    for session in sessions:
        sessionId = session["id"]
        recordingRows = client.table("recordings").select("storage_path").eq("session_id", sessionId).execute().data or []
        artifactRows = client.table("artifacts").select("storage_path").eq("session_id", sessionId).execute().data or []
        if not dryRun:
            recordingPaths = [row["storage_path"] for row in recordingRows]
            artifactPaths = [row["storage_path"] for row in artifactRows]
            if recordingPaths:
                client.storage.from_(settings.recordingsBucket).remove(recordingPaths)
            if artifactPaths:
                client.storage.from_(settings.artifactsBucket).remove(artifactPaths)
            client.table("interview_sessions").delete().eq("id", sessionId).execute()
        objects += len(recordingRows) + len(artifactRows)
        deleted += 1
    return {"expiredSessions": deleted, "storageObjects": objects}


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete expired TalonCV sessions and private objects.")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    print(cleanupExpiredInterviews(options.dry_run))


if __name__ == "__main__":
    main()
