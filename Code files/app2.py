import streamlit as st
import cv2
import os
import tempfile
import random
import asyncio
import time
from ultralytics import YOLO
from telegram import Bot

# -------------------------------
# Configurations
# -------------------------------
LOW_THRESHOLD = 10000  # Area < 10,000 px: Low severity
MEDIUM_THRESHOLD = 40000  # Area between 10,000 and 40,000 px: Medium severity
TOKEN = ""
CHAT_ID = ""

# Load YOLO model
model = YOLO("my_model.pt")

# Streamlit UI
st.title("🚧 Pothole Detection System")
st.write("Upload an image or video for pothole detection and classification.")

uploaded_file = st.file_uploader(
    "Upload a video or image", type=["mp4", "avi", "mov", "jpg", "png"]
)


def generate_coordinates():
    return round(random.uniform(-90, 90), 6), round(random.uniform(-180, 180), 6)


def process_results(results, is_video=False):
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
    annotated_path = os.path.join(OUTPUT_DIR, "output.mp4")

    reports = []

    if is_video:
        first_frame = results[0].orig_img
        height, width = first_frame.shape[:2]

        writer = cv2.VideoWriter(
            annotated_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
        )

        for result in results:
            frame = result.orig_img.copy()
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                area = (x2 - x1) * (y2 - y1)
                severity = (
                    "High"
                    if area >= MEDIUM_THRESHOLD
                    else "Medium" if area >= LOW_THRESHOLD else "Low"
                )
                lat, lon = generate_coordinates()
                reports.append(f"Detected {severity} pothole at ({lat}, {lon})")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    severity,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            writer.write(frame)
        writer.release()
    else:
        result = results[0]
        img = result.orig_img.copy()
        annotated_path = os.path.join(OUTPUT_DIR, "output.jpg")

        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            severity = (
                "High"
                if area >= MEDIUM_THRESHOLD
                else "Medium" if area >= LOW_THRESHOLD else "Low"
            )
            lat, lon = generate_coordinates()
            reports.append(f"Detected {severity} pothole at ({lat}, {lon})")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                severity,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        cv2.imwrite(annotated_path, img)

    return annotated_path, reports


async def send_telegram_report(report_text, media_path, is_video):
    async with Bot(token=TOKEN) as bot:
        with open(media_path, "rb") as media:
            if is_video:
                await bot.send_video(chat_id=CHAT_ID, video=media)
            else:
                await bot.send_photo(chat_id=CHAT_ID, photo=media)
        for i in range(0, len(report_text), 4096):
            await bot.send_message(chat_id=CHAT_ID, text=report_text[i : i + 4096])


if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov"]

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_input.write(uploaded_file.read())
    temp_input.close()

    st.write("🔍 Processing... Please wait.")
    results = model.predict(source=temp_input.name, conf=0.2, save=False)
    annotated_path, reports = process_results(results, is_video=is_video)

    if is_video:
        time.sleep(2)
        if os.path.exists(annotated_path):
            st.video(annotated_path)
        else:
            st.error(f"Error: Processed video file {annotated_path} not found!")
    else:
        st.image(annotated_path)

    if reports:
        final_report = "\n".join(reports)
        st.text_area("Detection Report:", final_report, height=150)

        if st.button("📤 Send Report to Telegram"):
            asyncio.run(send_telegram_report(final_report, annotated_path, is_video))
            st.success("Report sent to Telegram!")
