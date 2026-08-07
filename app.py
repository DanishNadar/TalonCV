import base64
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyzeInterviewDemo import (  # noqa: E402
    analyzeVideo,
    createEvents,
    getOutputStem,
    loadMediaPipe,
    saveEvents,
    saveFeatures,
    saveReport,
)
from src.cvPipeline.cueDefinitions import getCueInfo  # noqa: E402
from src.cvPipeline.cueClassifier import defaultCueModelPath  # noqa: E402
from src.cvPipeline.reportUtils import reviewEventTypes, strengthEventTypes  # noqa: E402
from src.cvPipeline.webcamRecorder import RecordingState, WebcamRecorder  # noqa: E402
from scripts.generateLlmReadyReview import buildLlmReadyJson, buildPrompt  # noqa: E402
from src.audioPipeline.mediaUtils import inspectMedia, supportedMediaExtensions  # noqa: E402
from src.multimodalPipeline.artifacts import (  # noqa: E402
    getArtifactPaths,
    loadSessionContext,
    readJson,
    saveSessionContext,
    writeJson,
    writeText,
)
from src.multimodalPipeline.diagnostics import buildDemoDiagnostics  # noqa: E402
from src.multimodalPipeline.pipeline import (  # noqa: E402
    MultimodalAnalysisError,
    loadMultimodalAnalysis,
    runMultimodalAnalysis,
)

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


RECORDING_DIR = PROJECT_ROOT / "data" / "demo" / "recordings"
FEATURE_DIR = PROJECT_ROOT / "data" / "demo" / "features"
EVENT_DIR = PROJECT_ROOT / "data" / "demo" / "events"
REPORT_DIR = PROJECT_ROOT / "reports"
REPLAY_DIR = PROJECT_ROOT / "data" / "demo" / "replays"
SYSTEM_EVENT_TYPES = {
    "faceMissing",
    "poseMissing",
    "lowFaceConfidence",
    "faceMeshMissing",
    "multipleFaces",
    "dimLighting",
    "overexposedLighting",
    "lowContrast",
    "blurryImage",
}
SEEK_BUTTON_LIMIT = 24
IMPORTANT_REVIEW_THRESHOLDS = {
    "tensionLikeInstability": 1.0,
    "possibleFidgeting": 1.0,
    "lookingAway": 1.0,
    "postureShift": 1.0,
    "highHeadMovement": 1.0,
    "lookingDown": 1.0,
    "nodding": 1.0,
    "faceMissing": 2.0,
    "poseMissing": 2.0,
    "offCenterFraming": 1.0,
    "faceTooClose": 2.0,
    "faceTooFar": 2.0,
    "facePartiallyOutOfFrame": 1.0,
    "multipleFaces": 1.0,
    "lowFaceConfidence": 2.0,
    "faceMeshMissing": 2.0,
    "dimLighting": 2.0,
    "overexposedLighting": 2.0,
    "lowContrast": 2.0,
    "blurryImage": 2.0,
    "eyesClosedLike": 1.0,
    "rapidBlinkLikeActivity": 1.0,
    "headTurnedLeft": 1.0,
    "headTurnedRight": 1.0,
    "headTilt": 1.0,
    "lateralHeadMovement": 1.0,
    "shoulderTilt": 1.0,
    "bodyLean": 1.0,
    "bodyOffCenter": 1.0,
    "handGestureActivity": 1.5,
    "handsRaised": 1.0,
}
HIGH_SEVERITY_EVENT_TYPES = {"tensionLikeInstability", "possibleFidgeting", "lookingAway"}
# Localhost WebRTC does not need public ICE infrastructure. Remote deployments
# must supply their own trusted, self-hosted ICE service outside this app.
RTC_CONFIGURATION = {"iceServers": []}
COMMON_INTERVIEW_QUESTIONS = [
    "Custom question",
    "Tell me about yourself.",
    "Tell me about a time you solved a difficult problem.",
    "Describe a time you handled conflict on a team.",
    "Tell me about a project you are proud of.",
    "Describe a mistake you made and what you learned.",
    "Why are you interested in this role?",
    "What is one strength you would bring to this role?",
]
logger = logging.getLogger("taloncv")


st.set_page_config(page_title="TalonCV Demo", page_icon="TCV", layout="wide")


def ensure_demo_folders():
    folders = (
        RECORDING_DIR,
        FEATURE_DIR,
        EVENT_DIR,
        REPLAY_DIR,
        REPORT_DIR,
        PROJECT_ROOT / "data" / "demo" / "audio",
        PROJECT_ROOT / "data" / "demo" / "sessions",
        PROJECT_ROOT / "data" / "demo" / "transcripts",
        PROJECT_ROOT / "data" / "demo" / "audioFeatures",
        PROJECT_ROOT / "data" / "demo" / "audioEvents",
        PROJECT_ROOT / "data" / "demo" / "responseAnalysis",
        PROJECT_ROOT / "data" / "demo" / "semanticAnalysis",
        PROJECT_ROOT / "data" / "demo" / "multimodal",
        PROJECT_ROOT / "data" / "demo" / "scores",
        PROJECT_ROOT / "data" / "demo" / "diagnostics",
        PROJECT_ROOT / "data" / "demo" / "llmReady",
    )
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def list_recordings():
    if not RECORDING_DIR.exists():
        return []

    return sorted(
        (path for path in RECORDING_DIR.iterdir() if path.is_file() and path.suffix.lower() in supportedMediaExtensions),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def save_uploaded_recording(uploaded_file):
    ensure_demo_folders()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in supportedMediaExtensions:
        raise ValueError(f"Unsupported media extension: {extension or '(none)'}")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(uploaded_file.name).stem).strip("_") or "media"
    output_path = RECORDING_DIR / f"uploaded_{safe_name}_{timestamp}{extension}"
    output_path.write_bytes(uploaded_file.getbuffer())
    media_info = inspectMedia(output_path)
    if not media_info.get("valid"):
        output_path.unlink(missing_ok=True)
        raise ValueError((media_info.get("warnings") or ["The uploaded media could not be decoded."])[0])
    return output_path


def get_webcam_recorder():
    if "webcam_recorder" not in st.session_state:
        ensure_demo_folders()
        st.session_state.webcam_recorder = WebcamRecorder(RECORDING_DIR)
    return st.session_state.webcam_recorder


@st.fragment(run_every=1.0)
def render_webcam_recorder(sessionContext=None):
    if not WEBRTC_AVAILABLE:
        st.info(
            "Live webcam recording needs the `streamlit-webrtc` package. Install it with "
            "`pip install streamlit-webrtc`, or record locally with `python scripts/recordInterviewDemo.py`."
        )
        return

    recorder = get_webcam_recorder()
    recorder.mark_requesting_permissions()

    def video_frame_callback(frame):
        recorder.add_video_frame(frame)
        return frame

    def audio_frame_callback(frame):
        recorder.add_audio_frame(frame)
        return frame

    try:
        webrtc_ctx = webrtc_streamer(
            key="talon-cv-webcam-recorder",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": True},
            video_frame_callback=video_frame_callback,
            audio_frame_callback=audio_frame_callback,
            async_processing=True,
            sendback_audio=False,
        )
    except Exception as error:
        recorder.discard()
        st.info(f"The browser webcam session could not initialize ({error}). Use the local recorder instead.")
        return

    recorder.set_connected(bool(webrtc_ctx.state.playing))
    visual_only = st.checkbox(
        "Visual-only recording (explicitly omit microphone audio)",
        value=False,
        help="Use only when no microphone is available. Transcript, answer, and vocal analysis will be unavailable.",
        key="webcam_visual_only",
    )
    status = recorder.status()
    status_columns = st.columns(3)
    status_columns[0].metric("Camera", "Connected" if status["cameraConnected"] else "Waiting")
    status_columns[1].metric("Microphone", "Connected" if status["microphoneConnected"] else "Waiting")
    status_columns[2].metric("Mic signal", "Audible" if status["microphoneAudible"] else "No signal yet")
    st.caption(
        f"State: `{status['state']}` · elapsed {status['elapsedSeconds']:.1f}s · "
        f"video {status['videoFrameCount']} · audio {status['audioFrameCount']} · "
        f"dropped {status['droppedVideoFrames']} video/{status['droppedAudioFrames']} audio · "
        f"bounded queue peak {status['maxQueuedMemoryBytes'] / (1024 * 1024):.1f} MiB"
    )
    if status.get("savedFilename"):
        st.success(f"Saved and validated: {status['savedFilename']}")
    if status.get("error"):
        st.error(status["error"])
    st.caption(
        f"Click WebRTC START above and approve media access. H.264/AAC takes are capped at "
        f"{recorder.max_duration_seconds // 60:.0f} minutes and stream through a bounded writer queue."
    )

    def finish_recording(reason=None):
        saved_path, stats = recorder.stop()
        if saved_path:
            saveSessionContext(saved_path, sessionContext or {})
            st.session_state["last_saved_recording"] = str(saved_path)
            st.session_state["browser_permissions_tested"] = True
            st.success(
                f"Saved {saved_path.name} — {stats['frameCount']} video frames at ~{stats['fps']:.1f} FPS; "
                f"audio {'captured' if stats['audioCaptured'] else 'omitted'}; "
                f"duration {stats['durationSeconds']:.1f}s; start offset "
                f"{stats.get('audioVideoStartOffsetSeconds')}s; drift {stats.get('estimatedDriftSeconds')}s."
            )
            if stats.get("truncated"):
                st.warning("The five-minute duration cap was reached; the streamed recording was saved automatically.")
            if reason:
                st.info(reason)
        else:
            st.error(f"Recording could not be saved: {stats.get('error') or 'The recording was too short.'}")
        return saved_path

    if recorder.reached_duration_cap and recorder.has_pending_recording:
        with st.spinner("Finalizing the duration-capped recording..."):
            finish_recording("Recording stopped at the configured duration cap.")
        st.rerun()

    if not webrtc_ctx.state.playing:
        if recorder.has_pending_recording and recorder.elapsed_seconds >= 0.5:
            with st.spinner("Media disconnected; safely finalizing the buffered queue..."):
                finish_recording("The browser media connection ended, so TalonCV finalized the take safely.")
            st.rerun()
        st.info("Camera and microphone are not connected yet. Click WebRTC START above and approve both permissions.")
        return

    control_columns = st.columns(2)
    if recorder.is_recording:
        control_columns[0].button(
            f"Recording… {recorder.elapsed_seconds:.1f}s",
            disabled=True,
            width="stretch",
            key="webcam_recording_status",
        )
        if control_columns[1].button("Stop & Save", type="primary", width="stretch", key="stop_webcam_recording"):
            with st.spinner("Flushing encoders, validating streams, and saving atomically..."):
                finish_recording()
            st.rerun()
    else:
        ready = status["cameraConnected"] and (status["microphoneConnected"] or visual_only)
        if control_columns[0].button(
            "Start Recording",
            type="primary",
            width="stretch",
            key="start_webcam_recording",
            disabled=not ready or status["state"] == RecordingState.finalizing,
        ):
            if not recorder.start(allow_visual_only=visual_only):
                st.error(recorder.last_error or "The recorder was not ready.")
            st.rerun()
        control_columns[1].button("Stop & Save", disabled=True, width="stretch", key="stop_webcam_recording_disabled")
        st.caption(
            "Required media callbacks are active. Start Recording when ready."
            if ready
            else "Start Recording remains disabled until the required callbacks receive data."
        )


def get_artifact_paths(video_path):
    paths = getArtifactPaths(video_path)
    return {
        "media": paths.media,
        "features": paths.visualFeatures,
        "events": paths.visualEvents,
        "report": paths.report,
        "audio": paths.audio,
        "audioMeta": paths.audioMeta,
        "transcriptText": paths.transcriptText,
        "transcriptJson": paths.transcriptJson,
        "audioFeatures": paths.audioFeatures,
        "audioEvents": paths.audioEvents,
        "responseAnalysis": paths.responseAnalysis,
        "semanticAnalysis": paths.semanticAnalysis,
        "session": paths.session,
        "multimodalMoments": paths.multimodalMoments,
        "multimodal": paths.multimodal,
        "scores": paths.scores,
        "deterministicReport": paths.deterministicReport,
        "localCoaching": paths.localCoaching,
        "localCoachMeta": paths.localCoachMeta,
        "diagnostics": paths.diagnostics,
        "llmReadyJson": paths.llmReadyJson,
        "llmReadyPrompt": paths.llmReadyPrompt,
    }


def event_category(event_type):
    if event_type in strengthEventTypes:
        return "strength"
    if event_type in SYSTEM_EVENT_TYPES:
        return "system/visibility"
    if event_type in reviewEventTypes:
        return "review"
    return "review"


def event_category_badge(category):
    if category == "strength":
        return "Strength"
    if category == "system/visibility":
        return "System / visibility"
    return "Review"


def read_events(event_path):
    if not event_path.exists():
        return []

    try:
        return json.loads(event_path.read_text())
    except json.JSONDecodeError:
        st.error(f"Could not read events JSON: {event_path}")
        return []


def read_features(feature_path):
    if not feature_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(feature_path)
    except Exception as error:
        st.error(f"Could not read feature CSV: {error}")
        return pd.DataFrame()


def features_have_labels(features):
    if features.empty or "frameLabels" not in features.columns:
        return False

    labels = features["frameLabels"].dropna().astype(str).str.strip()
    labels = labels[(labels != "") & (labels.str.lower() != "nan")]
    return not labels.empty


def repair_empty_events_from_features(video_path, features, events, artifact_paths):
    if events or not features_have_labels(features):
        return events

    rows = features.where(pd.notna(features), None).to_dict("records")
    duration_seconds = calculate_duration(video_path) or float(features["timestampSeconds"].max() or 0.0)
    rebuilt_events = createEvents(rows, duration_seconds)
    if not rebuilt_events:
        return events

    event_path = saveEvents(rebuilt_events, getOutputStem(video_path))
    report_path = saveReport(video_path, rows, rebuilt_events, duration_seconds, getOutputStem(video_path))
    artifact_paths["events"] = event_path
    artifact_paths["report"] = report_path
    st.success("Recovered cue events from the existing feature CSV. No re-analysis was needed.")
    return rebuilt_events


def build_events_table(events):
    rows = []

    for event in events:
        cue_info = getCueInfo(event["eventType"])
        category = event_category(event["eventType"])
        priority = review_priority(event)
        rows.append(
            {
                "category": event_category_badge(category),
                "review priority": priority,
                "cue": event.get("cue", cue_info["cue"]),
                "eventType": event["eventType"],
                "start": f"{event['startTime']:.2f}s",
                "end": f"{event['endTime']:.2f}s",
                "duration": f"{event['durationSeconds']:.2f}s",
                "detected by": "+".join(event.get("detectionSources", ["rule"])),
                "ML confidence": event.get("mlConfidenceMean"),
                "meaning": cue_info["detectionMeaning"],
                "coaching note": event.get("description", cue_info["safeInterpretation"]),
            }
        )

    return pd.DataFrame(rows)


COACHING_GUIDANCE = {
    "lookingAway": (
        "Were you thinking, checking notes, or losing camera engagement?",
        "Try returning your gaze to the camera before delivering your main point.",
    ),
    "tensionLikeInstability": (
        "Did the movement distract from the substance of your answer?",
        "Try taking a short pause, grounding your posture, and continuing at a steady pace.",
    ),
    "possibleFidgeting": (
        "Did the movement look natural, or did it make the answer feel less settled?",
        "Try keeping your hands and posture settled while delivering important points.",
    ),
    "postureShift": (
        "Did this posture change pull attention away from your answer?",
        "Reset into a comfortable, supported posture before starting a key answer.",
    ),
    "highHeadMovement": (
        "Did the head movement support emphasis, or compete with the answer?",
        "Use smaller, intentional movements around your most important points.",
    ),
    "lookingDown": (
        "Were you checking notes or collecting your thoughts during this moment?",
        "Place notes closer to the camera and return your gaze before the main point.",
    ),
    "nodding": (
        "Did the repeated nodding add useful emphasis or become visually repetitive?",
        "Reserve nods for agreement or emphasis so they remain intentional.",
    ),
    "cameraFacing": (
        "Notice how your delivery looks when your attention is centered.",
        "Use this strength moment as a reference for camera engagement.",
    ),
    "stablePosture": (
        "What about this position made your delivery look visually steady?",
        "Use this strength moment as a posture reference in the next practice take.",
    ),
    "neutralExpression": (
        "Does this composed moment support the tone of your answer?",
        "Keep this steadiness while adding expression where it supports your message.",
    ),
    "positiveExpression": (
        "How did this expression affect the warmth of your delivery?",
        "Use this moment as a reference for natural, camera-facing warmth.",
    ),
    "faceMissing": (
        "Did you move out of frame, become obscured, or encounter a detection limitation?",
        "Keep your face evenly lit and inside the frame; review this as a visibility issue.",
    ),
    "poseMissing": (
        "Was your upper body outside the frame or difficult for the system to detect?",
        "Frame your shoulders and upper body clearly if you want posture feedback.",
    ),
}


def coaching_guidance(event_type):
    return COACHING_GUIDANCE.get(
        event_type,
        (
            "What was happening in your answer during this visual cue?",
            "Review the moment in context and choose one small delivery adjustment to practice.",
        ),
    )


def coach_priority(event, same_type_count):
    if event_category(event["eventType"]) == "strength":
        return "Low"

    duration = float(event.get("durationSeconds", 0.0))
    if duration >= 3.0 or same_type_count >= 3:
        return "High"
    if duration >= 1.0 or same_type_count >= 2:
        return "Medium"
    return "Low"


def event_label(event, index):
    return (
        f"{index + 1:02d} · {event['startTime']:.2f}s–{event['endTime']:.2f}s · "
        f"{event['eventType']} · {event_category_badge(event_category(event['eventType']))}"
    )


def video_metadata(video_path):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": frames / fps if fps else 0.0,
    }


def format_timestamp(seconds):
    return f"{seconds:.2f}s"


def is_company_review_moment(event):
    event_type = event["eventType"]
    threshold = IMPORTANT_REVIEW_THRESHOLDS.get(event_type)
    if threshold is None:
        return False

    return event.get("durationSeconds", 0.0) >= threshold


def review_priority(event):
    if not is_company_review_moment(event):
        return "Cue log only"

    event_type = event["eventType"]
    duration = event.get("durationSeconds", 0.0)

    if event_type in HIGH_SEVERITY_EVENT_TYPES and duration >= 2.0:
        return "High"
    if event_type in SYSTEM_EVENT_TYPES:
        return "Visibility review"
    return "Review"


def review_reason(event):
    event_type = event["eventType"]

    if event_type == "tensionLikeInstability":
        return "Sustained visual instability that a reviewer would likely want to inspect."
    if event_type == "possibleFidgeting":
        return "Repeated motion long enough to matter in interview playback."
    if event_type == "lookingAway":
        return "Camera-facing attention dropped for a sustained segment."
    if event_type in {"postureShift", "highHeadMovement", "lookingDown", "nodding"}:
        return "Body or head movement persisted long enough to review in context."
    if event_type in SYSTEM_EVENT_TYPES:
        return "Visibility was missing long enough to affect cue reliability."
    return "Important enough to review in the original video."


def get_important_review_events(events):
    review_events = [event for event in events if is_company_review_moment(event)]
    return sorted(review_events, key=lambda item: (item["startTime"], item["eventType"]))


def get_seek_events(events):
    seen = set()
    unique_events = []
    for event in get_important_review_events(events):
        key = (event["eventType"], event["startTime"], event["endTime"])
        if key in seen:
            continue
        seen.add(key)
        unique_events.append(event)

    return unique_events[:SEEK_BUTTON_LIMIT]


def get_video_frame(video_path, timestamp_seconds):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_seconds, 0.0) * 1000.0)
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def zoom_crop_from_features(features, start_time, end_time, width, height):
    if features.empty or "timestampSeconds" not in features:
        return None

    window = features[
        (features["timestampSeconds"] >= start_time) & (features["timestampSeconds"] <= end_time)
    ]
    required = {"faceCenterX", "faceCenterY", "faceWidth", "faceHeight"}
    if window.empty or not required.issubset(window.columns):
        return None

    face_rows = window.dropna(subset=list(required))
    if face_rows.empty:
        return None

    center_x = float(face_rows["faceCenterX"].median())
    center_y = float(face_rows["faceCenterY"].median())
    face_width = float(face_rows["faceWidth"].median())
    face_height = float(face_rows["faceHeight"].median())

    crop_width = min(max(face_width * 2.4, 0.38), 1.0)
    crop_height = min(max(face_height * 3.2, 0.55), 1.0)
    center_y = min(max(center_y + face_height * 0.45, crop_height / 2), 1 - crop_height / 2)
    center_x = min(max(center_x, crop_width / 2), 1 - crop_width / 2)

    x1 = int((center_x - crop_width / 2) * width)
    y1 = int((center_y - crop_height / 2) * height)
    x2 = int((center_x + crop_width / 2) * width)
    y2 = int((center_y + crop_height / 2) * height)
    return max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)


def create_replay_clip(video_path, event, features, buffer_seconds, zoomed):
    ensure_demo_folders()
    metadata = video_metadata(video_path)
    clip_start = max(float(event["startTime"]) - buffer_seconds, 0.0)
    clip_end = min(float(event["endTime"]) + buffer_seconds, metadata["duration"])
    mode = "zoom" if zoomed else "context"
    filename = (
        f"{video_path.stem}_{event['eventType']}_{clip_start:.2f}_{clip_end:.2f}_{mode}.mp4"
        .replace(".", "p")
        .replace("pmp4", ".mp4")
    )
    output_path = REPLAY_DIR / filename

    crop = None
    if zoomed:
        crop = zoom_crop_from_features(
            features,
            float(event["startTime"]),
            float(event["endTime"]),
            metadata["width"],
            metadata["height"],
        )
        if crop is None:
            return None, clip_start, clip_end, False

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path, clip_start, clip_end, bool(crop)

    capture = cv2.VideoCapture(str(video_path))
    fps = metadata["fps"] or 30.0
    capture.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000.0)

    if crop:
        x1, y1, x2, y2 = crop
        output_size = (x2 - x1, y2 - y1)
    else:
        output_size = (metadata["width"], metadata["height"])

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        return None, clip_start, clip_end, False

    while capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 <= clip_end:
        ok, frame = capture.read()
        if not ok:
            break
        if crop:
            frame = frame[y1:y2, x1:x2]
        writer.write(frame)

    capture.release()
    writer.release()
    return output_path, clip_start, clip_end, bool(crop)


@st.cache_data(show_spinner=False)
def build_review_clip(video_path_text, start_time, duration_seconds, target_fps=8, max_width=960):
    capture = cv2.VideoCapture(video_path_text)
    if not capture.isOpened():
        return {"frames": [], "width": 0, "height": 0, "fps": target_fps}

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    video_duration = frame_count / source_fps if source_fps and frame_count else duration_seconds
    safe_start = max(float(start_time), 0.0)
    safe_duration = min(max(float(duration_seconds), 1.0), 20.0)
    safe_end = min(safe_start + safe_duration, video_duration)

    frames = []
    width = 0
    height = 0
    timestamp = safe_start
    frame_interval = 1.0 / target_fps

    while timestamp <= safe_end and len(frames) < 200:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        original_height, original_width = frame.shape[:2]
        if original_width > max_width:
            scale = max_width / original_width
            frame = cv2.resize(frame, (max_width, int(original_height * scale)))

        height, width = frame.shape[:2]
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            frames.append("data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"))

        timestamp += frame_interval

    capture.release()
    return {"frames": frames, "width": width, "height": height, "fps": target_fps}


def render_review_clip(video_path, event, playback_speed=1.0, timeline_start=None):
    clip_duration = calculate_duration(video_path) or max(event.get("durationSeconds", 0.0), 2.0)
    clip = build_review_clip(str(video_path), 0.0, clip_duration)
    frames = clip["frames"]

    if not frames:
        st.warning("The selected review clip could not be decoded.")
        return

    frame_json = json.dumps(frames)
    event_label = (
        f"{event['eventType']} | {event['startTime']:.2f}s - {event['endTime']:.2f}s | "
        f"{review_priority(event)}"
    )
    html = f"""
    <div class="tcv-player">
      <div class="tcv-title">{event_label}</div>
      <canvas id="tcvCanvas" width="{clip['width']}" height="{clip['height']}"></canvas>
      <div class="tcv-controls">
        <button id="tcvRestart" type="button">Replay</button>
        <button id="tcvToggle" type="button">Pause</button>
        <span id="tcvTime">0.00s</span>
      </div>
    </div>
    <style>
      .tcv-player {{
        width: 100%;
        color: #f4f7fb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      .tcv-title {{
        font-size: 14px;
        font-weight: 650;
        margin: 0 0 8px;
        color: #d8e6ff;
      }}
      #tcvCanvas {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 8px;
        background: #05070d;
      }}
      .tcv-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
      }}
      .tcv-controls button {{
        border: 1px solid #536172;
        border-radius: 7px;
        background: #111827;
        color: #f4f7fb;
        font-size: 14px;
        font-weight: 650;
        padding: 7px 12px;
        cursor: pointer;
      }}
      .tcv-controls button:hover {{
        background: #1f2937;
      }}
      #tcvTime {{
        color: #bac7d5;
        font-size: 14px;
      }}
    </style>
    <script>
      const frames = {frame_json};
      const fps = {clip['fps']};
      const startTime = {event['startTime'] if timeline_start is None else timeline_start};
      const playbackSpeed = {playback_speed};
      const canvas = document.getElementById("tcvCanvas");
      const context = canvas.getContext("2d");
      const restartButton = document.getElementById("tcvRestart");
      const toggleButton = document.getElementById("tcvToggle");
      const timeLabel = document.getElementById("tcvTime");
      const images = frames.map((src) => {{
        const image = new Image();
        image.src = src;
        return image;
      }});
      let frameIndex = 0;
      let playing = true;
      let timer = null;

      function drawFrame() {{
        const image = images[frameIndex];
        if (image && image.complete) {{
          context.drawImage(image, 0, 0, canvas.width, canvas.height);
        }}
        timeLabel.textContent = (startTime + frameIndex / fps).toFixed(2) + "s";
      }}

      function tick() {{
        drawFrame();
        frameIndex += 1;
        if (frameIndex >= images.length) {{
          frameIndex = 0;
        }}
      }}

      function play() {{
        playing = true;
        toggleButton.textContent = "Pause";
        if (!timer) {{
          timer = setInterval(tick, 1000 / (fps * playbackSpeed));
        }}
      }}

      function pause() {{
        playing = false;
        toggleButton.textContent = "Play";
        clearInterval(timer);
        timer = null;
      }}

      restartButton.addEventListener("click", () => {{
        frameIndex = 0;
        drawFrame();
        play();
      }});
      toggleButton.addEventListener("click", () => {{
        if (playing) {{
          pause();
        }} else {{
          play();
        }}
      }});

      images[0].onload = () => {{
        drawFrame();
        play();
      }};
      play();
    </script>
    """
    player_height = min(max(int((clip["height"] / max(clip["width"], 1)) * 1000) + 92, 320), 720)
    encoded_html = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{encoded_html}", height=player_height)


def render_seekable_video(video_path, events):
    if not video_path.exists():
        st.error(f"Video does not exist: {video_path}")
        return

    seek_events = get_seek_events(events)
    state_key = f"video_start_time_{video_path.stem}"
    event_key = f"selected_review_event_{video_path.stem}"

    if state_key not in st.session_state:
        st.session_state[state_key] = seek_events[0]["startTime"] if seek_events else 0.0
    if seek_events and event_key not in st.session_state:
        st.session_state[event_key] = 0

    selected_time = st.session_state[state_key]
    selected_event = None
    if seek_events:
        selected_index = min(st.session_state.get(event_key, 0), len(seek_events) - 1)
        selected_event = seek_events[selected_index]

    if selected_event:
        st.subheader("Selected Review Clip")
        render_review_clip(video_path, selected_event)

    st.subheader("Original Video")
    st.video(str(video_path), start_time=selected_time, width="stretch")
    st.caption(f"Current selected jump point: {selected_time:.2f}s")

    frame = get_video_frame(video_path, selected_time)
    if frame is not None:
        st.image(frame, caption=f"Decoded review frame at {selected_time:.2f}s", width="stretch")
    else:
        st.warning("The selected frame could not be decoded. Try re-recording the demo video.")

    if seek_events:
        st.write("Company-review-worthy moments")
        st.caption(
            "Only sustained or important review cues are timestamped here. Short cue blips stay in the full event log."
        )
        for index, event in enumerate(seek_events):
            label = (
                f"{format_timestamp(event['startTime'])} - {format_timestamp(event['endTime'])}: "
                f"{event['eventType']} ({review_priority(event)})"
            )
            if st.button(
                label,
                key=f"seek_{video_path.stem}_{index}_{event['eventType']}_{event['startTime']}",
                help=review_reason(event),
            ):
                st.session_state[state_key] = float(event["startTime"])
                st.session_state[event_key] = index
                st.rerun()
    else:
        st.info("No sustained review-worthy moments were found yet. The full cue log is still available below.")


def render_main_interview_player(video_path):
    metadata = video_metadata(video_path)
    st.header("Main Interview Player")
    st.caption("Full-context player · cue replay controls below do not change this player")
    st.video(str(video_path), width="stretch")

    columns = st.columns(4)
    columns[0].metric("Recording", video_path.name)
    columns[1].metric("Duration", f"{metadata['duration']:.2f}s")
    columns[2].metric("Resolution", f"{metadata['width']} × {metadata['height']}")
    columns[3].metric("Source FPS", f"{metadata['fps']:.1f}")


def render_cue_timeline(events):
    if not events:
        return

    timeline_rows = []
    ordered_types = list(dict.fromkeys(event["eventType"] for event in events))
    for event in events:
        timeline_rows.append(
            {
                "cue": event["eventType"],
                "start": event["startTime"],
                "end": event["endTime"],
                "category": event_category_badge(event_category(event["eventType"])),
            }
        )

    spec = {
        "mark": {"type": "bar", "cornerRadius": 3, "height": 13},
        "encoding": {
            "x": {"field": "start", "type": "quantitative", "title": "Interview time (seconds)"},
            "x2": {"field": "end"},
            "y": {"field": "cue", "type": "nominal", "sort": ordered_types, "title": None},
            "color": {
                "field": "category",
                "type": "nominal",
                "scale": {
                    "domain": ["Strength", "Review", "System / visibility"],
                    "range": ["#2ca02c", "#ff9f1c", "#7f8c8d"],
                },
            },
            "tooltip": ["cue", "category", "start", "end"],
        },
    }
    st.header("Cue Timeline")
    st.caption("Every detected event in full-interview timestamp order")
    st.vega_lite_chart(pd.DataFrame(timeline_rows), spec, width="stretch")


def render_ranked_moments(events):
    review_events = [event for event in events if event_category(event["eventType"]) == "review"]
    strengths = [event for event in events if event_category(event["eventType"]) == "strength"]

    left, right = st.columns(2)
    with left:
        st.subheader("Most Important Review Moments")
        ranked = sorted(review_events, key=lambda item: item.get("durationSeconds", 0), reverse=True)[:5]
        if not ranked:
            st.info("No review cues were detected.")
        for event in ranked:
            st.markdown(
                f"**{event['eventType']}** · {event['startTime']:.2f}s–{event['endTime']:.2f}s "
                f"· {event['durationSeconds']:.2f}s"
            )
    with right:
        st.subheader("Strength Moments")
        ranked = sorted(strengths, key=lambda item: item.get("durationSeconds", 0), reverse=True)[:5]
        if not ranked:
            st.info("No strength cues were detected.")
        for event in ranked:
            st.markdown(
                f"**{event['eventType']}** · {event['startTime']:.2f}s–{event['endTime']:.2f}s "
                f"· {event['durationSeconds']:.2f}s"
            )


def render_coach_mode(event, all_events):
    cue_info = getCueInfo(event["eventType"])
    question, suggestion = coaching_guidance(event["eventType"])
    same_type_count = sum(item["eventType"] == event["eventType"] for item in all_events)
    priority = coach_priority(event, same_type_count)
    category = event_category_badge(event_category(event["eventType"]))

    st.subheader("Coach Mode")
    st.caption("Supportive guidance from structured visual cues—not a psychological assessment")
    label_columns = st.columns(2)
    label_columns[0].metric("Cue category", category)
    label_columns[1].metric("Review priority", priority, help="Based on cue duration and repetition; not a grade.")

    st.markdown("**What happened**")
    st.write(
        f"From {event['startTime']:.2f}s to {event['endTime']:.2f}s, TalonCV detected "
        f"`{event['eventType']}`. {event.get('description', cue_info['safeInterpretation'])}"
    )
    st.markdown("**Why it was flagged**")
    st.write(
        f"The rule uses this visual signal: {cue_info['detectionMeaning']} "
        "This is a visual proxy and should be reviewed in context."
    )
    st.markdown("**What to review**")
    st.write(question)
    st.markdown("**Coaching suggestion**")
    st.write(suggestion)


def notes_exports(video_path, event, note):
    payload = {
        "recording": video_path.name,
        "eventType": event["eventType"],
        "startTime": event["startTime"],
        "endTime": event["endTime"],
        "category": event_category_badge(event_category(event["eventType"])),
        "note": note,
    }
    markdown = (
        f"# Cue Review Note\n\n- Recording: `{video_path.name}`\n"
        f"- Cue: `{event['eventType']}`\n- Timestamp: {event['startTime']:.2f}s–{event['endTime']:.2f}s\n"
        f"- Category: {payload['category']}\n\n## Notes\n\n{note or '_No notes entered._'}\n"
    )
    return payload, markdown


@st.fragment
def render_cue_review_workspace(video_path, events, features, artifact_paths):
    st.header("Cue Replay Player")
    st.caption("Select any cue to loop a buffered replay without changing the full-context player above.")

    if not events:
        st.info("Run analysis to populate cue replay and Coach Mode.")
        return

    filter_options = {
        "All cues": None,
        "Review cues": "review",
        "Strength cues": "strength",
        "Visibility/system cues": "system/visibility",
    }
    selected_filter = st.selectbox("Filter cue events", list(filter_options))
    category_filter = filter_options[selected_filter]
    filtered_events = [
        event for event in events if category_filter is None or event_category(event["eventType"]) == category_filter
    ]
    if not filtered_events:
        st.info(f"No {selected_filter.lower()} were detected in this recording.")
        return

    event_options = [event_label(event, index) for index, event in enumerate(filtered_events)]
    selection_key = f"cue_selection_{video_path.stem}_{selected_filter}"
    selected_label = st.selectbox(
        "Select a cue event / jump to replay",
        event_options,
        key=selection_key,
    )
    selected_index = event_options.index(selected_label)
    event = filtered_events[selected_index]

    table = build_events_table(filtered_events)
    table.insert(0, "selected", [index == selected_index for index in range(len(table))])
    st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("Cue navigation / copy timestamps"):
        for index, candidate in enumerate(filtered_events):
            timestamp_column, cue_column, jump_column = st.columns([1, 2, 1])
            timestamp_column.code(
                f"{candidate['startTime']:.2f}s–{candidate['endTime']:.2f}s", language=None
            )
            cue_column.write(
                f"**{candidate['eventType']}**  \n"
                f"{event_category_badge(event_category(candidate['eventType']))}"
            )
            jump_column.button(
                "Jump to Replay",
                key=f"jump_{video_path.stem}_{selected_filter}_{index}",
                on_click=st.session_state.__setitem__,
                args=(selection_key, event_options[index]),
                width="stretch",
            )

    control_columns = st.columns(3)
    with control_columns[0]:
        playback_speed = st.select_slider(
            "Replay speed", options=[0.25, 0.5, 0.75, 1.0], value=1.0, format_func=lambda value: f"{value:.2g}x"
        )
    with control_columns[1]:
        buffer_seconds = st.select_slider("Context buffer", options=[1.0, 1.5, 2.0], value=1.5, format_func=lambda value: f"{value:g}s")
    with control_columns[2]:
        zoomed = st.toggle("Zoom to face / upper body", value=False)

    replay_path, clip_start, clip_end, used_zoom = create_replay_clip(
        video_path, event, features, buffer_seconds, zoomed
    )
    if zoomed and not used_zoom:
        st.warning("Zoomed replay is unavailable because face coordinates were not detected for this cue. Showing the normal replay.")
        replay_path, clip_start, clip_end, used_zoom = create_replay_clip(
            video_path, event, features, buffer_seconds, False
        )

    player_column, coach_column = st.columns([1.15, 1])
    with player_column:
        cue_info = getCueInfo(event["eventType"])
        st.markdown(
            f"**{cue_info['cue']}** · {event['startTime']:.2f}s–{event['endTime']:.2f}s "
            f"· {event_category_badge(event_category(event['eventType']))}"
        )
        st.caption(
            f"Buffered replay: {clip_start:.2f}s–{clip_end:.2f}s · looping at {playback_speed:g}x"
            + (" · zoomed crop" if used_zoom else "")
        )
        if replay_path:
            render_review_clip(replay_path, event, playback_speed, clip_start)
        else:
            st.error("The replay clip could not be generated from this recording.")
        st.code(f"{event['startTime']:.2f}s–{event['endTime']:.2f}s", language=None)

    with coach_column:
        render_coach_mode(event, events)

    note_key = f"review_note_{video_path.stem}_{event['eventType']}_{event['startTime']}"
    note = st.text_area("Review Notes", key=note_key, placeholder="What did you notice? What will you try next time?")
    payload, markdown = notes_exports(video_path, event, note)
    note_columns = st.columns(3)
    note_columns[0].download_button(
        "Export note (Markdown)", markdown, file_name=f"{video_path.stem}_{event['eventType']}_note.md"
    )
    note_columns[1].download_button(
        "Export note (JSON)", json.dumps(payload, indent=2), file_name=f"{video_path.stem}_{event['eventType']}_note.json"
    )
    prompt_text = buildPrompt([event])
    note_columns[2].download_button(
        "Create LLM Coaching Prompt",
        prompt_text,
        file_name=f"{video_path.stem}_{event['eventType']}_coachingPrompt.txt",
        help="Creates a local structured prompt. It does not call an external model or send video frames.",
    )

    st.caption("The coaching prompt contains structured cue data only. No raw video frames are sent anywhere.")

    with st.expander("Download analysis artifacts"):
        download_columns = st.columns(3)
        if artifact_paths["report"].exists():
            download_columns[0].download_button(
                "Markdown report", artifact_paths["report"].read_bytes(), artifact_paths["report"].name
            )
        if artifact_paths["events"].exists():
            download_columns[1].download_button(
                "Event JSON", artifact_paths["events"].read_bytes(), artifact_paths["events"].name
            )
        if artifact_paths["features"].exists():
            download_columns[2].download_button(
                "Feature CSV", artifact_paths["features"].read_bytes(), artifact_paths["features"].name
            )


def calculate_duration(video_path):
    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()

        if fps and frame_count:
            return frame_count / fps
    except Exception:
        return None

    return None


def recommendation_from_events(events):
    review_events = [event["eventType"] for event in get_important_review_events(events)]

    if not review_events:
        return "No sustained review-worthy moments stood out. Review one strength cue and keep the same delivery pattern in the next practice take."

    most_common = Counter(review_events).most_common(1)[0][0]
    cue_info = getCueInfo(most_common)
    return f"Start by reviewing `{most_common}` moments. {cue_info['safeInterpretation']}"


def render_summary(video_path, events, features):
    event_counts = Counter(event["eventType"] for event in events)
    important_events = get_important_review_events(events)
    review_counts = Counter(event["eventType"] for event in important_events)
    strength_counts = Counter(
        event["eventType"] for event in events if event_category(event["eventType"]) == "strength"
    )
    duration = calculate_duration(video_path)

    total_frames = len(features) if not features.empty else 0
    strongest_positive = strength_counts.most_common(1)[0][0] if strength_counts else "No strength cue yet"
    most_frequent_review = review_counts.most_common(1)[0][0] if review_counts else "No review cue yet"

    metric_cols = st.columns(5)
    metric_cols[0].metric("Duration", f"{duration:.2f}s" if duration else "Unknown")
    metric_cols[1].metric("Analyzed frames", total_frames)
    metric_cols[2].metric("Cue events", len(events))
    metric_cols[2].caption(f"{len(important_events)} important review moments")
    metric_cols[3].metric("Top strength", strongest_positive)
    metric_cols[4].metric("Top review", most_frequent_review)

    st.subheader("Cue Counts")
    if event_counts:
        counts_df = pd.DataFrame(
            [
                {
                    "eventType": event_type,
                    "category": event_category_badge(event_category(event_type)),
                    "count": count,
                    "learned evidence events": sum(
                        "ml" in event.get("detectionSources", [])
                        for event in events
                        if event["eventType"] == event_type
                    ),
                }
                for event_type, count in sorted(event_counts.items())
            ]
        )
        st.dataframe(counts_df, width="stretch", hide_index=True)
    else:
        st.info("No cue events have been detected for this recording yet.")

    st.subheader("What To Review Next")
    st.write(recommendation_from_events(events))


def render_timeline(events):
    if not events:
        st.info("No timestamped cue events are available yet. Run analysis first.")
        return

    for event in sorted(events, key=lambda item: (item["startTime"], item["eventType"])):
        category = event_category(event["eventType"])
        cue_info = getCueInfo(event["eventType"])
        st.markdown(
            f"**{event['startTime']:.2f}s - {event['endTime']:.2f}s:** "
            f"`{event['eventType']}` - {event_category_badge(category)}"
        )
        st.caption(cue_info["safeInterpretation"])
        if event.get("detectionSources"):
            sourceText = "+".join(event["detectionSources"])
            confidenceText = (
                f" at mean confidence {event['mlConfidenceMean']:.2f}"
                if event.get("mlConfidenceMean") is not None
                else ""
            )
            st.caption(f"Decision evidence: {sourceText}{confidenceText}")


def render_event_sections(events):
    strength_events = [event for event in events if event_category(event["eventType"]) == "strength"]
    review_events = [
        event for event in get_important_review_events(events) if event_category(event["eventType"]) == "review"
    ]
    system_events = [
        event
        for event in get_important_review_events(events)
        if event_category(event["eventType"]) == "system/visibility"
    ]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Strength Moments")
        render_timeline(strength_events)

    with col_b:
        st.subheader("Review Moments")
        render_timeline(review_events)

    with st.expander("System / visibility moments", expanded=bool(system_events)):
        render_timeline(system_events)


def run_analysis(video_path, session_context=None, force=False, progress_callback=None):
    try:
        return runMultimodalAnalysis(
            video_path,
            sessionContext=session_context,
            force=force,
            progressCallback=progress_callback,
        )
    except (MultimodalAnalysisError, SystemExit) as error:
        detail = str(error).strip()
        message = detail if detail else "Analysis stopped because the selected media could not be processed."
        raise RuntimeError(message) from error
    except Exception as error:
        logger.exception("Multimodal analysis failed for %s", video_path)
        raise RuntimeError(f"Multimodal analysis failed: {error}") from error


def render_report(report_path):
    st.subheader("Generated Markdown Report")
    if not report_path.exists():
        st.info("No Markdown report exists yet. Run analysis to generate one.")
        return

    st.markdown(report_path.read_text())


def render_interview_setup():
    st.header("Step 1 · Interview Setup")
    selected_sample = st.selectbox("Common interview question", COMMON_INTERVIEW_QUESTIONS, key="question_sample")
    previous_sample = st.session_state.get("previous_question_sample")
    if selected_sample != "Custom question" and selected_sample != previous_sample:
        st.session_state["interview_question_draft"] = selected_sample
    st.session_state["previous_question_sample"] = selected_sample
    question = st.text_area(
        "Question you are answering",
        key="interview_question_draft",
        placeholder="Enter the exact interview question. Relevance is marked unavailable when this is blank.",
    )
    target_role = st.text_input("Target role (optional)", key="target_role_draft")
    job_description = st.text_area(
        "Job description or role context (optional)", key="job_description_draft", height=90
    )
    competencies = st.text_input(
        "Desired competencies (optional)", key="desired_competencies_draft", placeholder="e.g. ownership, collaboration"
    )
    return {
        "interviewQuestion": question,
        "targetRole": target_role,
        "jobDescription": job_description,
        "desiredCompetencies": competencies,
    }


def render_demo_diagnostics(media_path=None):
    with st.expander("Offline readiness / diagnostics"):
        load_models = st.button(
            "Re-run readiness checks and load every local model",
            width="stretch",
            key=f"diagnostics_{Path(media_path).stem if media_path else 'none'}",
        )
        with st.spinner("Loading configured local models..." if load_models else "Checking local files and codecs..."):
            diagnostics = buildDemoDiagnostics(
                media_path,
                loadModels=load_models,
                browserPermissionsTested=bool(st.session_state.get("browser_permissions_tested")),
            )
        if media_path:
            diagnostic_path = getArtifactPaths(media_path).diagnostics
            writeJson(diagnostic_path, diagnostics)
        status = diagnostics["offlineStatus"]
        if diagnostics["offlineReady"]:
            st.success(status)
        else:
            st.warning(status)
        policy = diagnostics["runtimePolicy"]
        st.code(
            "\n".join(
                [
                    f"External inference APIs: {policy['externalInferenceApis']}",
                    f"Runtime model downloads: {policy['runtimeModelDownloads']}",
                    f"Local model loading: {policy['localModelLoading']}",
                    f"Offline operation after setup: {policy['offlineOperationAfterSetup']}",
                ]
            ),
            language="text",
        )
        for model_name, model in diagnostics["models"].items():
            if not model.get("requiredFilesPresent"):
                st.error(f"{model_name}: {model.get('setupCommand') or model.get('error')}")
            elif load_models and model.get("loadError"):
                st.error(f"{model_name} could not load: {model['loadError']}")
        if not diagnostics["disk"]["ready"]:
            st.warning(
                f"Only {diagnostics['disk']['freeGb']:.1f} GiB is free; "
                f"{diagnostics['disk']['minimumFreeGb']:.1f} GiB is required for the offline demo."
            )
        st.json(diagnostics, expanded=False)


def render_media_player(media_path, media_info):
    st.subheader("Selected interview media")
    if media_info.get("hasVideo"):
        st.video(str(media_path), width="stretch")
    elif media_info.get("hasAudio"):
        st.audio(str(media_path), width="stretch")
        st.info("Audio-only mode: transcript, vocal delivery, and answer quality are available; visual cues are excluded.")
    for warning in media_info.get("warnings", []):
        st.warning(warning)
    if media_info.get("hasVideo") and not media_info.get("hasAudio"):
        st.warning("This video has no audio stream. TalonCV will preserve visual analysis and skip speech/audio analysis.")


def render_multimodal_overview(analysis):
    media_info = analysis.get("mediaInfo", {})
    transcript = analysis.get("transcript", {})
    response = analysis.get("responseAnalysis", {})
    audio = analysis.get("audioFeatures", {})
    moments = analysis.get("moments", [])
    scores = analysis.get("scores", {}).get("scores", {})
    overall = scores.get("overallInterviewPracticeDelivery", {})
    metrics = response.get("metrics", {})
    speaking_time = None
    if audio.get("speechRatio") is not None and audio.get("durationSeconds") is not None:
        speaking_time = float(audio["speechRatio"]) * float(audio["durationSeconds"])
    columns = st.columns(6)
    columns[0].metric("Duration", f"{float(media_info.get('durationSeconds') or 0):.1f}s")
    columns[1].metric("Spoken words", metrics.get("wordCount", 0))
    columns[2].metric("Speech rate", f"{audio['speechRateWpm']:.0f} WPM" if audio.get("speechRateWpm") else "Unavailable")
    columns[3].metric("Speaking time", f"{speaking_time:.1f}s" if speaking_time is not None else "Unavailable")
    columns[4].metric("Overall practice", f"{overall['score']:.0f}/100" if overall.get("score") is not None else "Unavailable")
    columns[5].metric(
        "Transcript confidence",
        f"{float(transcript['averageConfidence']):.0%}" if transcript.get("averageConfidence") is not None else "Unavailable",
    )
    columns = st.columns(5)
    columns[0].metric("Fillers", metrics.get("fillerCount", 0))
    columns[1].metric("Filler rate", f"{float(metrics.get('fillerRatePer100Words') or 0):.1f}/100")
    columns[2].metric("Long pauses", audio.get("longPauseCount", 0))
    columns[3].metric("Aligned moments", len(moments))
    available_modalities = [
        name
        for name, available in (
            ("transcript", bool(transcript.get("text"))),
            ("audio", bool(audio.get("available"))),
            ("visual", bool(media_info.get("hasVideo"))),
        )
        if available
    ]
    columns[4].metric("Data coverage", "+".join(available_modalities) or "None")

    st.subheader("Component ratings")
    rating_keys = [
        ("Audio quality", "audioRecordingQuality"),
        ("Answer quality", "verbalResponseQuality"),
        ("Vocal delivery", "vocalDelivery"),
        ("Visual delivery", "visualDelivery"),
        ("Multimodal alignment", "multimodalAlignment"),
    ]
    for column, (label, key) in zip(st.columns(5), rating_keys):
        item = scores.get(key, {})
        value = f"{item['score']:.0f}/100" if item.get("score") is not None else "Unavailable"
        column.metric(label, value, help=f"{item.get('rating', 'Unavailable')} · confidence {item.get('confidence', 'unavailable')}")

    strong_phrases = response.get("strongPhrases", [])
    vocal_strengths = scores.get("vocalDelivery", {}).get("positiveObservations", [])
    visual_strengths = scores.get("visualDelivery", {}).get("positiveObservations", [])
    priority_moments = [moment for moment in moments if moment.get("classification") == "review"]
    strongest_moments = [moment for moment in moments if moment.get("classification") == "strength"]
    if not priority_moments:
        priority_moments = [
            {"coachingRecommendation": item}
            for item in scores.get("overallInterviewPracticeDelivery", {}).get("practiceAreas", [])
        ]
    st.subheader("At-a-glance coaching evidence")
    moment_columns = st.columns(2)
    moment_columns[0].write(
        "**Strongest observed moment**\n\n"
        + (
            f"{strongest_moments[0]['startTime']:.2f}s — {strongest_moments[0]['explanation']}"
            if strongest_moments
            else "No aligned strength moment was available."
        )
    )
    moment_columns[1].write(
        "**Highest-priority review moment**\n\n"
        + (
            f"{priority_moments[0].get('startTime', 0):.2f}s — "
            f"{priority_moments[0].get('coachingRecommendation', 'Review this moment in context.')}"
            if priority_moments
            else "No aligned review moment was available."
        )
    )
    summary_columns = st.columns(4)
    summary_columns[0].write(
        f"**Top verbal strength**\n\n{strong_phrases[0]['text'] if strong_phrases else 'No timestamped content strength was available.'}"
    )
    summary_columns[1].write(
        f"**Top vocal strength**\n\n{vocal_strengths[0] if vocal_strengths else 'No evidence-backed vocal strength was available.'}"
    )
    summary_columns[2].write(
        f"**Top visual strength**\n\n{visual_strengths[0] if visual_strengths else 'No evidence-backed visual strength was available.'}"
    )
    summary_columns[3].write(
        "**Highest-priority review**\n\n"
        + (
            priority_moments[0].get("coachingRecommendation", "Review the first evidence-backed practice area.")
            if priority_moments
            else "No evidence-backed review priority was available."
        )
    )

    st.subheader("Explainable coaching scores")
    score_rows = []
    for name, score in scores.items():
        score_rows.append(
            {
                "component": name,
                "score": score.get("score"),
                "rating": score.get("rating"),
                "confidence": score.get("confidence"),
                "evidence": " · ".join(score.get("evidence", [])[:3]),
                "formula": score.get("formula"),
            }
        )
    st.dataframe(pd.DataFrame(score_rows), width="stretch", hide_index=True)
    st.caption(analysis.get("scores", {}).get("safetyNote", ""))

    positives = []
    practices = []
    for score in scores.values():
        positives.extend(item for item in score.get("positiveObservations", []) if not item.startswith("No specific"))
        practices.extend(item for item in score.get("practiceAreas", []) if not item.startswith("No specific"))
    left, right = st.columns(2)
    with left:
        st.subheader("Top evidence-backed strengths")
        for item in list(dict.fromkeys(positives))[:5]:
            st.write(f"- {item}")
    with right:
        st.subheader("Highest-priority practice")
        for item in list(dict.fromkeys(practices))[:5]:
            st.write(f"- {item}")
    st.subheader("Three next-step recommendations")
    recommendations = list(dict.fromkeys(practices))[:3]
    for index, item in enumerate(recommendations, start=1):
        st.write(f"{index}. {item}")
    for warning in analysis.get("warnings", []):
        st.warning(warning)


def render_transcript_view(media_path, analysis):
    transcript = analysis.get("transcript", {})
    if not transcript.get("text"):
        st.info("No transcript is available. Check the audio stream and local transcription diagnostics.")
        return
    st.write(transcript["text"])
    fillers = analysis.get("responseAnalysis", {}).get("fillerOccurrences", [])
    strong_phrases = analysis.get("responseAnalysis", {}).get("strongPhrases", [])
    audio_events = analysis.get("audioEvents", [])
    semantic_segments = analysis.get("semanticAnalysis", {}).get("segmentAssessments", [])
    rows = []
    for segment in transcript.get("segments", []):
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        segment_fillers = [item["phrase"] for item in fillers if min(end, item["endTime"]) > max(start, item["startTime"])]
        segment_audio = [event["eventType"] for event in audio_events if min(end, event["endTime"]) > max(start, event["startTime"])]
        segment_strengths = [item["text"] for item in strong_phrases if min(end, item["endTime"]) > max(start, item["startTime"])]
        semantic = next(
            (
                item
                for item in semantic_segments
                if min(end, float(item.get("endTime", 0))) > max(start, float(item.get("startTime", 0)))
            ),
            {},
        )
        content_flags = []
        confidence = segment.get("confidence")
        if segment_strengths:
            content_flags.append("strong content")
        if confidence is not None and float(confidence) < 0.55:
            content_flags.append("low transcript confidence")
        if segment_fillers:
            content_flags.append("contains contextual filler")
        if set(segment_audio) & {"longPause", "rapidSpeech", "speechFragmentation"}:
            content_flags.append("delivery review")
        if semantic.get("marker"):
            content_flags.append(str(semantic["marker"]))
        rows.append(
            {
                "start": f"{start:.2f}s",
                "end": f"{end:.2f}s",
                "text": segment.get("text", ""),
                "confidence": confidence,
                "fillers": ", ".join(segment_fillers),
                "audio cues": ", ".join(sorted(set(segment_audio))),
                "semantic relevance": semantic.get("questionRelevanceScore"),
                "content flags": ", ".join(content_flags),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    segments = transcript.get("segments", [])
    labels = [
        f"{index + 1:02d} · {float(segment.get('start', 0)):.2f}s · {str(segment.get('text', ''))[:70]}"
        for index, segment in enumerate(segments)
    ]
    if labels:
        selected = st.selectbox("Replay a transcript segment", labels, key="transcript_replay")
        segment = segments[labels.index(selected)]
        start = max(0.0, float(segment.get("start", 0)) - 1.0)
        end = float(segment.get("end", start)) + 1.0
        if analysis.get("mediaInfo", {}).get("hasVideo"):
            st.video(str(media_path), start_time=start, end_time=end, width="stretch")
        else:
            st.audio(str(media_path), start_time=start, end_time=end, width="stretch")
        highlighted = str(segment.get("text", ""))
        for filler in sorted(set(item["phrase"] for item in fillers), key=len, reverse=True):
            highlighted = re.sub(
                rf"\b({re.escape(filler)})\b", r"**\1**", highlighted, flags=re.IGNORECASE
            )
        st.markdown(f"Segment text (fillers highlighted): {highlighted}")
    st.caption("Low-confidence transcript wording should be verified against playback before treating it as a communication issue.")


def render_answer_quality(analysis):
    response = analysis.get("responseAnalysis", {})
    if not response.get("available"):
        st.info("Answer-quality analysis requires a usable transcript.")
        return
    metrics = response.get("metrics", {})
    metric_columns = st.columns(6)
    metric_columns[0].metric("Words", metrics.get("wordCount", 0))
    metric_columns[1].metric("Fillers / 100", metrics.get("fillerRatePer100Words", 0))
    metric_columns[2].metric("STAR components", f"{response.get('starAnalysis', {}).get('componentsPresent', 0)}/4")
    relevance = response.get("relevanceAnalysis", {})
    metric_columns[3].metric("Question relevance", relevance.get("score") if relevance.get("available") else "Unavailable")
    role = response.get("roleAlignmentAnalysis", {})
    metric_columns[4].metric("Role alignment", role.get("score") if role.get("available") else "Unavailable")
    metric_columns[5].metric("Repeated points", metrics.get("repeatedPhraseCount", 0))
    st.subheader("Rubric evidence")
    st.dataframe(
        pd.DataFrame(
            [
                {"component": name, **item}
                for name, item in response.get("rubric", {}).items()
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Strong phrases worth preserving")
    if response.get("strongPhrases"):
        st.dataframe(pd.DataFrame(response["strongPhrases"]), width="stretch", hide_index=True)
    else:
        st.info("No timestamped phrase met the deterministic action/result criteria.")
    st.subheader("Suggested answer improvements")
    for item in response.get("practiceAreas", []):
        st.write(f"- {item}")
    with st.expander("Suggested answer structure", expanded=True):
        for index, item in enumerate(response.get("suggestedAnswerStructure", []), start=1):
            st.write(f"{index}. {item}")
    development = response.get("answerDevelopment", {})
    st.subheader("Answer development evidence")
    development_rows = [
        ("Opening", development.get("openingNote", "Unavailable")),
        ("Length", development.get("lengthAssessment", "Unavailable")),
        ("Example quality", development.get("exampleQuality", "Unavailable")),
        ("Result quality", development.get("resultQuality", "Unavailable")),
        (
            "Responsibility vs. action",
            "Separated" if development.get("responsibilityAndActionSeparated") else "Needs clearer separation",
        ),
        (
            "Strongest sentence placement",
            "Move it earlier" if development.get("strongestSentenceBuried") else "Not detected as buried",
        ),
    ]
    st.dataframe(pd.DataFrame(development_rows, columns=["area", "finding"]), width="stretch", hide_index=True)
    missing = development.get("missingElements", [])
    st.write("**Missing or underdeveloped elements:** " + (", ".join(missing) if missing else "none detected"))
    if development.get("passagesToRevise"):
        st.write("**Exact passages to revise**")
        st.dataframe(pd.DataFrame(development["passagesToRevise"]), width="stretch", hide_index=True)
    semantic = analysis.get("semanticAnalysis", {})
    with st.expander("Local MiniLM semantic evidence"):
        st.write(semantic.get("questionRelevance", {}).get("explanation", "Semantic evidence unavailable."))
        if semantic.get("mostRelevantSegments"):
            st.write("Most relevant segments")
            st.dataframe(pd.DataFrame(semantic["mostRelevantSegments"]), width="stretch", hide_index=True)
        if semantic.get("semanticRedundancy"):
            st.write("Potential semantic repetition")
            st.dataframe(pd.DataFrame(semantic["semanticRedundancy"]), width="stretch", hide_index=True)
    coaching = analysis.get("localCoaching", {})
    with st.expander("Optional local coaching example", expanded=False):
        if coaching.get("available"):
            st.warning("Coaching example only: it must use facts already present in your transcript and must not be treated as a factual rewrite if wording is uncertain.")
            st.markdown(coaching.get("text", ""))
        else:
            st.info("The deterministic outline above remains available; local Qwen coaching was not generated for this take.")
    st.caption(response.get("fairnessNote", ""))


def render_vocal_delivery(analysis):
    features = analysis.get("audioFeatures", {})
    events = analysis.get("audioEvents", [])
    if not features.get("available"):
        st.info("Audible decoded audio is required for vocal-delivery and recording-quality analysis.")
        return
    recording = st.columns(5)
    recording[0].metric("Audibility", f"{features.get('overallRmsDb', 0):.1f} dBFS")
    recording[1].metric("Noise floor", f"{features.get('noiseFloorDb', 0):.1f} dBFS")
    recording[2].metric("Clipping", f"{features.get('clippingPercentage', 0):.3f}%")
    recording[3].metric("Dropout ratio", f"{features.get('dropoutRatio', 0):.2%}")
    recording[4].metric("SNR proxy", f"{features['snrProxyDb']:.1f} dB" if features.get("snrProxyDb") is not None else "Unavailable")
    st.caption("Recording-quality measurements describe the microphone and decoded signal; they are not delivery faults.")

    sections = {
        "Pace": {"rapidSpeech", "slowSpeech"},
        "Pauses": {"longPause", "speechFragmentation"},
        "Fillers": set(),
        "Volume": {"lowVolume", "highVolume", "audioClipping", "audioDropout", "recordingQualityIssue"},
        "Variation": {"limitedVocalVariation"},
        "Emphasis": {"strongVocalEmphasis"},
    }
    filler_items = analysis.get("responseAnalysis", {}).get("fillerOccurrences", [])
    for title, event_types in sections.items():
        with st.expander(title, expanded=title in {"Pace", "Pauses", "Fillers"}):
            if title == "Pace":
                st.write(
                    f"Overall rate: {features.get('speechRateWpm', 'unavailable')} WPM; "
                    f"local variation: {features.get('speechRateVariationWpm', 'unavailable')} WPM."
                )
            elif title == "Pauses":
                st.write(
                    f"Long pauses: {features.get('longPauseCount', 0)}; "
                    f"short fragments: {features.get('fragmentedSpeechSegmentCount', 0)}."
                )
            elif title == "Fillers":
                st.write(
                    f"Contextual fillers: {len(filler_items)}; "
                    f"{analysis.get('responseAnalysis', {}).get('metrics', {}).get('fillerRatePer100Words', 0)} per 100 words."
                )
                if filler_items:
                    st.dataframe(pd.DataFrame(filler_items), width="stretch", hide_index=True)
                continue
            elif title == "Volume":
                st.write(f"Volume consistency: {features.get('volumeConsistencyStdDb', 'unavailable')} dB standard deviation.")
            elif title == "Variation":
                st.write(
                    f"Energy range: {features.get('energyVariationDb', 'unavailable')} dB; "
                    f"pitch variation: {features.get('pitchVariationSemitones', 'unavailable')} semitones "
                    "when reliable."
                )
            elif title == "Emphasis":
                st.write("Review whether measured energy peaks supported an important action or result phrase.")
            matches = [event for event in events if event.get("eventType") in event_types]
            if matches:
                st.dataframe(pd.DataFrame(matches), width="stretch", hide_index=True)
            else:
                st.info(f"No timestamped {title.lower()} event exceeded the evidence threshold.")
    with st.expander("All timestamped vocal and recording events"):
        if events:
            st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)
        else:
            st.info("No timestamped audio events exceeded the configured evidence thresholds.")
    st.caption("Pitch, energy, and pace values are signal-based coaching proxies and are omitted when evidence is insufficient.")


def render_multimodal_moments(media_path, analysis):
    moments = analysis.get("moments", [])
    if not moments:
        st.info("No combined moments were available. Both aligned timestamps and at least two usable modalities are required.")
        return
    labels = [
        f"{index + 1:02d} · {moment['startTime']:.2f}s–{moment['endTime']:.2f}s · {moment['alignmentCategory']}"
        for index, moment in enumerate(moments)
    ]
    selected_label = st.selectbox("Select a combined moment", labels)
    moment = moments[labels.index(selected_label)]
    buffer_seconds = st.select_slider("Playback context", options=[1.0, 1.5, 2.0], value=1.5)
    start = max(0.0, float(moment["startTime"]) - buffer_seconds)
    end = float(moment["endTime"]) + buffer_seconds
    media_info = analysis.get("mediaInfo", {})
    if media_info.get("hasVideo"):
        st.video(str(media_path), start_time=start, end_time=end, loop=True, autoplay=False, width="stretch")
    elif media_info.get("hasAudio"):
        st.audio(str(media_path), start_time=start, end_time=end, loop=True, autoplay=False, width="stretch")
    st.markdown(f"**{moment['classification'].title()} · {moment['alignmentCategory']}**")
    st.write(moment["explanation"])
    st.write(f"Coaching recommendation: {moment['coachingRecommendation']}")
    st.write(f"Transcript: {moment.get('transcriptExcerpt') or 'No overlapping transcript text'}")
    st.write(f"Audio events: {', '.join(moment.get('audioEvents', [])) or 'none'}")
    st.write("Audio measurements:")
    st.json(moment.get("audioMeasurements", {}), expanded=False)
    st.write(f"Visual events: {', '.join(moment.get('visualEvents', [])) or 'none'}")
    st.caption(f"Evidence: {', '.join(moment.get('evidenceSources', []))}; confidence: {moment.get('confidence')}")
    st.dataframe(pd.DataFrame(moments), width="stretch", hide_index=True)


def render_downloads(artifact_paths, analysis):
    st.subheader("Analysis artifacts")
    events = analysis.get("visualEvents", [])
    prompt = buildPrompt(events, analysis)
    artifact_paths["llmReadyPrompt"].parent.mkdir(parents=True, exist_ok=True)
    writeText(artifact_paths["llmReadyPrompt"], prompt)
    llm_payload = buildLlmReadyJson(events, analysis)
    writeJson(artifact_paths["llmReadyJson"], llm_payload)
    downloadable = [
        ("Original recording", artifact_paths["media"]),
        ("Extracted WAV", artifact_paths["audio"]),
        ("Audio extraction metadata", artifact_paths["audioMeta"]),
        ("Full combined report", artifact_paths["report"]),
        ("Deterministic report", artifact_paths["deterministicReport"]),
        ("Local enhanced coaching", artifact_paths["localCoaching"]),
        ("Local coach metadata", artifact_paths["localCoachMeta"]),
        ("Session context", artifact_paths["session"]),
        ("Offline diagnostics", artifact_paths["diagnostics"]),
        ("Transcript text", artifact_paths["transcriptText"]),
        ("Transcript JSON", artifact_paths["transcriptJson"]),
        ("Audio features", artifact_paths["audioFeatures"]),
        ("Audio events", artifact_paths["audioEvents"]),
        ("Response analysis", artifact_paths["responseAnalysis"]),
        ("Semantic analysis", artifact_paths["semanticAnalysis"]),
        ("Visual features", artifact_paths["features"]),
        ("Visual events", artifact_paths["events"]),
        ("Multimodal moments", artifact_paths["multimodalMoments"]),
        ("Multimodal analysis", artifact_paths["multimodal"]),
        ("Scores", artifact_paths["scores"]),
        ("LLM-ready JSON", artifact_paths["llmReadyJson"]),
        ("LLM-ready prompt", artifact_paths["llmReadyPrompt"]),
    ]
    for row_start in range(0, len(downloadable), 3):
        columns = st.columns(3)
        for column, (label, path) in zip(columns, downloadable[row_start : row_start + 3]):
            if path.exists():
                column.download_button(label, path.read_bytes(), path.name, key=f"download_{path.name}")


def render_visual_review(media_path, events, features, artifact_paths):
    if not inspectMedia(media_path).get("hasVideo"):
        st.info("Visual analysis is unavailable for audio-only media.")
        return
    render_summary(media_path, events, features)
    render_cue_timeline(events)
    render_ranked_moments(events)
    render_cue_review_workspace(media_path, events, features, artifact_paths)
    important_events = get_important_review_events(events)
    if important_events:
        st.subheader("Important visual review moments")
        st.dataframe(build_events_table(important_events), width="stretch", hide_index=True)
    with st.expander("Full detected visual cue log"):
        st.dataframe(build_events_table(events), width="stretch", hide_index=True)


def context_has_values(context):
    return any(
        str(context.get(key, "")).strip()
        for key in ("interviewQuestion", "targetRole", "jobDescription", "desiredCompetencies")
    )


def queue_saved_context(context):
    st.session_state["pending_session_context"] = context


def apply_pending_context():
    context = st.session_state.pop("pending_session_context", None)
    if not context:
        return
    question = context.get("interviewQuestion", "")
    st.session_state["interview_question_draft"] = question
    st.session_state["target_role_draft"] = context.get("targetRole", "")
    st.session_state["job_description_draft"] = context.get("jobDescription", "")
    st.session_state["desired_competencies_draft"] = context.get("desiredCompetencies", "")
    st.session_state["question_sample"] = question if question in COMMON_INTERVIEW_QUESTIONS else "Custom question"
    st.session_state["previous_question_sample"] = st.session_state["question_sample"]


def main():
    ensure_demo_folders()
    apply_pending_context()

    st.title("TalonCV Multimodal Interview Practice")
    st.caption(
        "Fully local video, voice, semantic, and response coaching with YOLOv11, MediaPipe, "
        "faster-whisper, MiniLM, and Qwen. Runtime networking and model downloads are disabled. "
        "All cues are practice proxies, not emotional truth, diagnosis, or hiring evidence."
    )
    setup_context = render_interview_setup()

    with st.sidebar:
        st.header("Step 2 · Capture or Upload")
        st.write("Record synchronized camera and microphone streams, upload media, or choose an existing take.")

        with st.expander("Record from webcam", expanded=not list_recordings()):
            render_webcam_recorder(setup_context)

        upload_types = sorted(extension.lstrip(".") for extension in supportedMediaExtensions)
        uploaded_file = st.file_uploader("Upload interview media", type=upload_types)
        if uploaded_file is not None:
            upload_identifier = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("processed_upload_identifier") != upload_identifier:
                try:
                    saved_path = save_uploaded_recording(uploaded_file)
                    saveSessionContext(saved_path, setup_context)
                    st.session_state["processed_upload_identifier"] = upload_identifier
                    st.session_state["media_selector"] = str(saved_path.relative_to(PROJECT_ROOT))
                    st.success(f"Uploaded to {saved_path.relative_to(PROJECT_ROOT)}")
                except (OSError, ValueError) as error:
                    st.error(str(error))

        recordings = list_recordings()
        if not recordings:
            st.warning("No supported recordings found yet. Record above, upload media, or run:")
            st.code(".venv/Scripts/python scripts/recordInterviewDemo.py --seconds 30", language="bash")
            render_demo_diagnostics()
            return

        labels = [str(path.relative_to(PROJECT_ROOT)) for path in recordings]
        last_saved = st.session_state.pop("last_saved_recording", None)
        if last_saved:
            last_saved_path = Path(last_saved)
            if last_saved_path.exists():
                st.session_state["media_selector"] = str(last_saved_path.relative_to(PROJECT_ROOT))
        if st.session_state.get("media_selector") not in labels:
            st.session_state["media_selector"] = labels[0]
        selected_label = st.selectbox("Select recording", labels, key="media_selector")
        selected_media = PROJECT_ROOT / selected_label

        saved_context = loadSessionContext(selected_media)
        if context_has_values(saved_context):
            st.caption("This recording has saved interview setup metadata.")
            if st.button("Load its saved setup", width="stretch"):
                queue_saved_context(saved_context)
                st.rerun()
        if st.button("Save current setup to this take", width="stretch"):
            saveSessionContext(selected_media, setup_context)
            st.success("Interview setup saved for this take.")

        st.divider()
        if defaultCueModelPath.exists():
            st.success("Learned cue layer active")
            st.caption(str(defaultCueModelPath.relative_to(PROJECT_ROOT)))
        else:
            st.caption("Rule-only mode: no trained cue classifier is installed yet.")
            st.code("python scripts/recordCueSamples.py --list-cues", language="bash")
        render_demo_diagnostics(selected_media)

    media_info = inspectMedia(selected_media)
    if not media_info.get("valid"):
        st.error((media_info.get("warnings") or ["The selected media could not be decoded."])[0])
        return

    saved_context = loadSessionContext(selected_media)
    analysis_context = setup_context if context_has_values(setup_context) else saved_context
    artifact_paths = get_artifact_paths(selected_media)
    render_media_player(selected_media, media_info)

    st.divider()
    st.header("Step 3 · Run Local Multimodal Analysis")
    st.caption(
        "The pipeline extracts audio, transcribes locally, analyzes the response and vocal delivery, "
        "runs visual cue detection when video exists, aligns timestamps, and builds the report."
    )
    run_col, option_col = st.columns([2, 1])
    force_analysis = option_col.checkbox("Ignore valid cache", value=False)
    with run_col:
        run_clicked = st.button("Run Multimodal Analysis", type="primary", width="stretch")

    analysis = loadMultimodalAnalysis(selected_media)
    if run_clicked:
        progress_bar = st.progress(0.0)
        progress_message = st.empty()

        def update_progress(stage, fraction, message):
            progress_bar.progress(min(1.0, max(0.0, float(fraction))))
            progress_message.caption(f"{stage}: {message}")

        try:
            analysis = run_analysis(
                selected_media,
                session_context=analysis_context,
                force=force_analysis,
                progress_callback=update_progress,
            )
            st.success("Analysis complete. All completed local artifacts were saved.")
        except RuntimeError as error:
            st.error(str(error))
            analysis = loadMultimodalAnalysis(selected_media)

    existing = [name for name, path in artifact_paths.items() if path.exists()]
    with st.expander(f"Artifact status · {len(existing)}/{len(artifact_paths)} files currently available"):
        st.json(
            {
                name: {"exists": path.exists(), "path": str(path.relative_to(PROJECT_ROOT))}
                for name, path in artifact_paths.items()
            },
            expanded=False,
        )

    if not analysis.get("complete"):
        legacy_events = read_events(artifact_paths["events"])
        legacy_features = read_features(artifact_paths["features"])
        if legacy_events or not legacy_features.empty:
            st.info(
                "Legacy visual artifacts are available below. Run multimodal analysis to add transcript, "
                "voice, scores, and alignment."
            )
            render_visual_review(selected_media, legacy_events, legacy_features, artifact_paths)
        else:
            st.info("Run analysis to create the review dashboard and report.")
        return

    st.divider()
    st.header("Step 4 · Review Evidence and Practice")
    tabs = st.tabs(
        [
            "Overview",
            "Transcript",
            "Answer Quality",
            "Vocal Delivery",
            "Visual Cues",
            "Multimodal Moments",
            "Full Report",
            "Downloads",
        ]
    )
    events = analysis["visualEvents"] if "visualEvents" in analysis else read_events(artifact_paths["events"])
    features = read_features(artifact_paths["features"])
    with tabs[0]:
        render_multimodal_overview(analysis)
    with tabs[1]:
        render_transcript_view(selected_media, analysis)
    with tabs[2]:
        render_answer_quality(analysis)
    with tabs[3]:
        render_vocal_delivery(analysis)
    with tabs[4]:
        render_visual_review(selected_media, events, features, artifact_paths)
    with tabs[5]:
        render_multimodal_moments(selected_media, analysis)
    with tabs[6]:
        render_report(artifact_paths["report"])
    with tabs[7]:
        render_downloads(artifact_paths, analysis)


if __name__ == "__main__":
    main()
