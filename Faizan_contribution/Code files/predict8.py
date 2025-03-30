from ultralytics import YOLO
import cv2
import os
import random
import asyncio
from telegram import Bot

# -------------------------------
# Configuration and thresholds
# -------------------------------
LOW_THRESHOLD = 10000      # Area < 10,000 px: Low severity
MEDIUM_THRESHOLD = 40000   # Area between 10,000 and 40,000 px: Medium severity
# Above MEDIUM_THRESHOLD -> High severity

source = '5.jpg'  # Change this to your input file (video or image)
base_name = os.path.splitext(os.path.basename(source))[0]
output_folder = base_name + "_output"
os.makedirs(output_folder, exist_ok=True)

model = YOLO('my_model.pt')

video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
ext = os.path.splitext(source)[1].lower()
is_video = ext in video_extensions

results = model.predict(source=source, conf=0.2, show=False, line_width=2, save_txt=True)

def generate_synthetic_coordinates():
    """Generate random latitude and longitude coordinates."""
    lat = round(random.uniform(-90, 90), 6)
    lon = round(random.uniform(-180, 180), 6)
    return lat, lon

# -------------------------------
# Telegram Bot configuration
# -------------------------------
TOKEN = ""
CHAT_ID = ""

async def send_telegram_message_with_photo(report_text, photo_path=None):
    """
    Send a Telegram message with the given report text.
    If photo_path is provided, send the image with the message as a caption.
    Splits the message into chunks if necessary.
    """
    MAX_LENGTH = 4096  # Telegram message length limit
    async with Bot(token=TOKEN) as bot:
        for i in range(0, len(report_text), MAX_LENGTH):
            chunk = report_text[i:i+MAX_LENGTH]
            if photo_path:
                # Open the image file in binary mode and send it as a photo.
                with open(photo_path, 'rb') as photo:
                    await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=chunk)
            else:
                await bot.send_message(chat_id=CHAT_ID, text=chunk)
    print("Telegram message sent successfully!")

# -------------------------------
# Process Video vs. Image
# -------------------------------
if is_video:
    # -------- Video Processing --------
    first_frame = results[0].orig_img
    height, width = first_frame.shape[:2]
    fps = 30  
    output_video_path = os.path.join(output_folder, base_name + "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    aggregated_reports = []  # List to accumulate reports from qualifying frames
    representative_image_path = None  # Will store the first detected frame as image

    for result in results:
        frame = result.orig_img.copy()
        report_this_frame = None  # For storing report for current frame

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                area = (x2 - x1) * (y2 - y1)
                lat, lon = generate_synthetic_coordinates()

                # Determine severity
                if area < LOW_THRESHOLD:
                    severity = "Low"
                elif area < MEDIUM_THRESHOLD:
                    severity = "Medium"
                else:
                    severity = "High"
                
                label = f"{severity} ({int(area)} px)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # If detection qualifies and not yet reported for this frame,
                # create a report entry and, if not already set, save a representative image.
                if severity in ["High", "Medium"] and report_this_frame is None:
                    report_this_frame = (f"Frame {frame_count}: Detected pothole with "
                                         f"{severity} severity at coordinates ({lat}, {lon})")
                    if representative_image_path is None:
                        # Save the first frame that has a qualifying detection.
                        representative_image_path = os.path.join(output_folder, base_name + "_detected.jpg")
                        cv2.imwrite(representative_image_path, frame)
                    break  # Process only one qualifying detection per frame

        writer.write(frame)

        if report_this_frame:
            aggregated_reports.append(report_this_frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {output_video_path}")

    # Send one aggregated Telegram message with the representative detected image.
    if aggregated_reports:
        final_report = "\n".join(aggregated_reports)
        asyncio.run(send_telegram_message_with_photo(final_report, photo_path=representative_image_path))
else:
    # -------- Image Processing --------
    # For image, we expect only one result.
    result = results[0]
    img = result.orig_img.copy()
    report_generated = None

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            area = (x2 - x1) * (y2 - y1)
            if area < LOW_THRESHOLD:
                severity = "Low"
            elif area < MEDIUM_THRESHOLD:
                severity = "Medium"
            else:
                severity = "High"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{severity} ({int(area)} px)"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Create a report only for the first qualifying detection.
            if severity in ["High", "Medium"] and report_generated is None:
                lat, lon = generate_synthetic_coordinates()
                report_generated = (f"Image: Detected pothole with {severity} severity at "
                                    f"coordinates ({lat}, {lon}).")
                break

    output_image_path = os.path.join(output_folder, base_name + "_annotated.jpg")
    cv2.imwrite(output_image_path, img)
    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Annotated image saved to: {output_image_path}")

    # Send the Telegram message along with the annotated image.
    if report_generated:
        asyncio.run(send_telegram_message_with_photo(report_generated, photo_path=output_image_path))
