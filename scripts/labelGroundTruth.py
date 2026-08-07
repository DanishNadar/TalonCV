import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
from src.cvPipeline.cueDefinitions import eventCueMap  # noqa: E402
from src.cvPipeline.reportUtils import reviewEventTypes, strengthEventTypes  # noqa: E402
from src.cvPipeline.yoloFaceDetector import loadYoloFaceDetector  # noqa: E402


RECORDING_DIR = PROJECT_ROOT / "data" / "demo" / "recordings"
FEATURE_DIR = PROJECT_ROOT / "data" / "demo" / "features"
EVENT_DIR = PROJECT_ROOT / "data" / "demo" / "events"
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "demo" / "groundTruth"
AVAILABLE_INDICATORS = sorted(eventCueMap.keys())


def indicator_category(indicator):
    if indicator in strengthEventTypes:
        return "Strength"
    if indicator in reviewEventTypes:
        return "Review"
    return "Other"


def indicator_option_label(indicator):
    return f"{indicator} ({indicator_category(indicator)})"


st.set_page_config(page_title="TalonCV Ground Truth Labeling", page_icon="TCV", layout="wide")


def ensure_folders():
    for folder in (RECORDING_DIR, FEATURE_DIR, EVENT_DIR, GROUND_TRUTH_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def list_recordings():
    if not RECORDING_DIR.exists():
        return []
    return sorted(RECORDING_DIR.glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)


def get_artifact_paths(video_path):
    output_stem = getOutputStem(video_path)
    return {
        "features": FEATURE_DIR / f"{output_stem}_features.csv",
        "events": EVENT_DIR / f"{output_stem}_events.json",
        "groundTruth": GROUND_TRUTH_DIR / f"{output_stem}_groundTruth.json",
    }


def calculate_duration(video_path):
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if fps and frame_count:
        return frame_count / fps
    return None


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


def read_events(event_path):
    if not event_path.exists():
        return []
    try:
        return json.loads(event_path.read_text())
    except json.JSONDecodeError:
        st.error(f"Could not read events JSON: {event_path}")
        return []


def load_ground_truth(ground_truth_path):
    if not ground_truth_path.exists():
        return []
    try:
        return json.loads(ground_truth_path.read_text())
    except json.JSONDecodeError:
        st.error(f"Could not read ground truth JSON: {ground_truth_path}")
        return []


def save_ground_truth(ground_truth_path, segments):
    ensure_folders()
    ordered = sorted(segments, key=lambda item: (item["startTime"], item["indicator"]))
    ground_truth_path.write_text(json.dumps(ordered, indent=2))
    return ordered


@st.cache_resource(show_spinner="Loading YOLOv11 face model...")
def get_yolo_face_detector():
    return loadYoloFaceDetector()


def run_analysis(video_path):
    mp = loadMediaPipe()
    rows, duration_seconds = analyzeVideo(video_path, mp, get_yolo_face_detector())
    events = createEvents(rows, duration_seconds)
    output_stem = getOutputStem(video_path)
    feature_path = saveFeatures(rows, output_stem)
    event_path = saveEvents(events, output_stem)
    saveReport(video_path, rows, events, duration_seconds, output_stem)
    return feature_path, event_path


def prefill_from_event(stem, event):
    st.session_state[f"indicator_{stem}"] = indicator_option_label(event["eventType"])
    st.session_state[f"start_{stem}"] = float(event["startTime"])
    st.session_state[f"end_{stem}"] = float(event["endTime"])


def remove_segment(ground_truth_path, segments, index):
    remaining = [segment for position, segment in enumerate(segments) if position != index]
    save_ground_truth(ground_truth_path, remaining)


def render_reference_events(stem, events):
    st.subheader("Auto-detected cues (reference candidates)")
    st.caption(
        "These come from the current rule-based detector. Use them as a starting point, then correct "
        "the timestamps below to match what you actually did -- that correction is exactly the signal "
        "a future ML model needs."
    )

    if not events:
        st.info("No auto-detected events yet for this recording.")
        return

    for index, event in enumerate(events):
        columns = st.columns([2, 2, 2, 2, 2])
        columns[0].write(f"**{event['eventType']}**")
        columns[1].write(indicator_category(event["eventType"]))
        columns[2].write(f"{event['startTime']:.2f}s")
        columns[3].write(f"{event['endTime']:.2f}s")
        columns[4].button(
            "Use as ground truth",
            key=f"use_event_{stem}_{index}",
            on_click=prefill_from_event,
            args=(stem, event),
        )


def render_add_segment_form(stem, video_path, ground_truth_path, segments, duration_seconds):
    st.subheader("Add a ground truth segment")

    max_seconds = float(duration_seconds) if duration_seconds else 3600.0

    default_indicator = st.session_state.get(f"indicator_{stem}", indicator_option_label(AVAILABLE_INDICATORS[0]))
    default_start = st.session_state.get(f"start_{stem}", 0.0)
    default_end = st.session_state.get(f"end_{stem}", min(1.0, max_seconds))

    option_labels = [indicator_option_label(indicator) for indicator in AVAILABLE_INDICATORS]
    if default_indicator not in option_labels:
        default_indicator = option_labels[0]

    selected_label = st.selectbox(
        "Indicator you were expressing",
        option_labels,
        index=option_labels.index(default_indicator),
        key=f"indicator_{stem}",
    )
    selected_indicator = AVAILABLE_INDICATORS[option_labels.index(selected_label)]

    time_columns = st.columns(3)
    start_time = time_columns[0].number_input(
        "Start (s)", min_value=0.0, max_value=max_seconds, value=float(default_start), step=0.1, key=f"start_{stem}"
    )
    end_time = time_columns[1].number_input(
        "End (s)", min_value=0.0, max_value=max_seconds, value=float(default_end), step=0.1, key=f"end_{stem}"
    )
    preview_time = time_columns[2].number_input(
        "Preview frame at (s)", min_value=0.0, max_value=max_seconds, value=float(start_time), step=0.1, key=f"preview_{stem}"
    )

    frame = get_video_frame(video_path, preview_time)
    if frame is not None:
        st.image(frame, caption=f"Decoded frame at {preview_time:.2f}s", width=360)

    note = st.text_input("Note (optional)", key=f"note_{stem}")

    if st.button("Add segment", type="primary", key=f"add_{stem}"):
        if end_time <= start_time:
            st.error("End time must be after start time.")
        else:
            segments.append(
                {
                    "indicator": selected_indicator,
                    "startTime": round(float(start_time), 3),
                    "endTime": round(float(end_time), 3),
                    "durationSeconds": round(float(end_time) - float(start_time), 3),
                    "note": note,
                }
            )
            save_ground_truth(ground_truth_path, segments)
            st.success(f"Added {selected_indicator}: {start_time:.2f}s-{end_time:.2f}s")
            st.rerun()


def render_segment_table(stem, ground_truth_path, segments):
    st.subheader("Labeled ground truth for this recording")

    if not segments:
        st.info("No ground truth segments recorded yet for this recording.")
        return

    for index, segment in enumerate(segments):
        columns = st.columns([2, 2, 2, 2, 3, 1])
        columns[0].write(f"**{segment['indicator']}**")
        columns[1].write(f"{segment['startTime']:.2f}s")
        columns[2].write(f"{segment['endTime']:.2f}s")
        columns[3].write(f"{segment['durationSeconds']:.2f}s")
        columns[4].write(segment.get("note") or "")
        columns[5].button(
            "Remove",
            key=f"remove_{stem}_{index}",
            on_click=remove_segment,
            args=(ground_truth_path, segments, index),
        )

    coverage = pd.DataFrame(segments).groupby("indicator")["durationSeconds"].sum().reset_index()
    coverage = coverage.rename(columns={"durationSeconds": "labeledSeconds"})
    st.caption("Labeled coverage per indicator for this recording")
    st.dataframe(coverage, width="stretch", hide_index=True)


def main():
    ensure_folders()

    st.title("TalonCV Ground Truth Labeling")
    st.caption(
        "Mark which indicators you were actually expressing, with start/end timestamps, so a future "
        "ML model can be trained and evaluated against real ground truth instead of only the rule-based detector."
    )

    with st.sidebar:
        st.header("Recording")
        recordings = list_recordings()
        if not recordings:
            st.warning("No `.mp4` recordings found yet.")
            st.code("python scripts/recordLabelingSession.py", language="bash")
            return

        labels = [str(path.relative_to(PROJECT_ROOT)) for path in recordings]
        selected_label = st.selectbox("Select recording", labels)
        selected_video = PROJECT_ROOT / selected_label

        st.divider()
        st.write("Need more takes?")
        st.code("python scripts/recordLabelingSession.py", language="bash")

        st.divider()
        st.write("Once several recordings are labeled:")
        st.code("python scripts/buildGroundTruthDataset.py", language="bash")

    stem = getOutputStem(selected_video)
    artifact_paths = get_artifact_paths(selected_video)
    duration_seconds = calculate_duration(selected_video)

    st.header(selected_video.name)
    st.caption(f"Duration: {duration_seconds:.2f}s" if duration_seconds else "Duration unknown")
    st.video(str(selected_video), width="stretch")

    if not artifact_paths["events"].exists():
        st.info("No auto-detected events yet. Running detection first gives you reference candidates to correct.")
        if st.button("Run detection pipeline (MediaPipe + YOLOv11)"):
            with st.spinner("Running MediaPipe + YOLOv11 analysis..."):
                try:
                    run_analysis(selected_video)
                    st.success("Analysis complete.")
                    st.rerun()
                except SystemExit as error:
                    detail = str(error).strip()
                    st.error(detail if detail else "Analysis stopped. Check that Python 3.12 and requirements are installed.")
                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    events = read_events(artifact_paths["events"])
    segments = load_ground_truth(artifact_paths["groundTruth"])

    st.divider()
    render_reference_events(stem, events)

    st.divider()
    render_add_segment_form(stem, selected_video, artifact_paths["groundTruth"], segments, duration_seconds)

    st.divider()
    render_segment_table(stem, artifact_paths["groundTruth"], segments)


if __name__ == "__main__":
    main()
