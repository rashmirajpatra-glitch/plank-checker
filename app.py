import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import tempfile
import os
import urllib.request

SKIP_FRAMES = 3

def calculate_angle_three_points(a, b, c):
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax * cx + ay * cy
    mag = (math.sqrt(ax**2 + ay**2)) * (math.sqrt(cx**2 + cy**2))
    if mag == 0:
        return 0
    angle = math.degrees(math.acos(max(-1, min(1, dot / mag))))
    return angle

def check_plank_form(landmarks):
    shoulder = landmarks[11]
    hip      = landmarks[23]
    ankle    = landmarks[27]

    body_angle = calculate_angle_three_points(shoulder, hip, ankle)

    feedback  = []
    good_form = True

    if body_angle < 140:
        if body_angle < 110:
            feedback.append(f"Hips too high — bring them down (body angle: {body_angle:.1f}°)")
        else:
            feedback.append(f"Body not straight — adjust your hips (body angle: {body_angle:.1f}°)")
        good_form = False
    else:
        feedback.append(f"Great form! Body angle: {body_angle:.1f}° — Hold it!")

    return good_form, feedback

def draw_landmarks_on_frame(frame, landmarks):
    h, w = frame.shape[:2]
    connections = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
    ]
    for start, end in connections:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    return frame

st.title("Plank Timer + Form Checker")
st.write("Upload a short plank video (under 15 seconds for best speed).")

model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    with st.spinner("Downloading pose model (one time only)..."):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            model_path
        )
    st.success("Model downloaded!")

uploaded_file = st.file_uploader("Upload your plank video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Video: {total_video_frames} frames at {fps:.0f} fps — {total_video_frames/fps:.1f} seconds")

    stframe      = st.empty()
    feedback_box = st.empty()
    timer_box    = st.empty()
    progress     = st.progress(0, text="Starting...")

    plank_start     = None
    total_hold_time = 0
    good_frames     = 0
    total_frames    = 0

    base_options = python.BaseOptions(model_asset_path=model_path)
    options      = vision.PoseLandmarkerOptions(
                     base_options=base_options,
                     output_segmentation_masks=False)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            pct = min(int(total_frames / total_video_frames * 100), 100)
            progress.progress(pct, text=f"Processing frame {total_frames} of {total_video_frames}...")

            if total_frames % SKIP_FRAMES != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect(mp_image)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                frame     = draw_landmarks_on_frame(frame, landmarks)
                is_good, feedback = check_plank_form(landmarks)

                if is_good:
                    good_frames += 1
                    if plank_start is None:
                        plank_start = total_frames / fps
                    hold_so_far = (total_frames / fps) - plank_start
                    timer_box.metric("Hold time", f"{hold_so_far:.1f} sec")
                else:
                    if plank_start is not None:
                        total_hold_time += (total_frames / fps) - plank_start
                        plank_start = None
                    timer_box.metric("Hold time", f"{total_hold_time:.1f} sec")

                feedback_box.info("\n".join(feedback))
            else:
                feedback_box.warning("No pose detected — make sure your full body is visible")

            stframe.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width=600
            )

    cap.release()
    os.unlink(tfile.name)
    progress.progress(100, text="Done!")

    if plank_start is not None:
        total_hold_time += (total_frames / fps) - plank_start

    accuracy = (good_frames / total_frames * 100) if total_frames > 0 else 0

    st.success("Video processed!")
    st.subheader("Summary")
    col1, col2 = st.columns(2)
    col1.metric("Total plank hold", f"{total_hold_time:.1f} sec")
    col2.metric("Good form", f"{accuracy:.1f}%")
