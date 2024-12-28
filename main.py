import time

import cv2
import easyocr
import torch
from PIL import Image
from ultralytics import YOLO

from utils.ocr_helpers import ocr_image

MODEL_PATH = "models/best_180_epochs.pt"
model = YOLO(MODEL_PATH)

IMAGE_PATH = "inputs/Dieu_0052.png"
IMAGE_OUTPUT_PATH = "outputs/Dieu_0052.png"

VIDEO_PATH = "inputs/ss.mp4"
VIDEO_OUTPUT_PATH = "outputs/ss.avi"

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def process_image(image_path: str):
    try:
        start_time = time.time()

        img = cv2.imread(image_path)

        results = model.predict(source=image_path)

        inference_time = round(time.time() - start_time, 3)

        detected_texts = []
        confidences = []
        output_img = img.copy()

        for m_result in results:
            for box in m_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                text_ocr = ocr_image(img, [x1, y1, x2, y2])
                detected_texts.append(text_ocr)
                confidences.append(conf)

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, f"{text_ocr} ({conf: .2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imwrite(IMAGE_OUTPUT_PATH, output_img)

        output_image_pil = Image.open(IMAGE_OUTPUT_PATH)
        output_image_pil.show()

        return {
            "statusCode": 200,
            "ocrTexts": detected_texts,
            "confidenceScores": confidences,
            "takeTime": inference_time,
            "outputImagePath": IMAGE_OUTPUT_PATH
        }

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {
            "statusCode": 500,
            "message": f"Processing error: {str(e)}"
        }


def process_video(video_path: str, output_video_path: str):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return {"statusCode": 500, "message": "Cannot open video file"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        detected_texts = []
        confidences = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame)

            for m_result in results:
                for box in m_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    text_ocr = ocr_image(frame, [x1, y1, x2, y2])
                    detected_texts.append(text_ocr)
                    confidences.append(conf)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text_ocr} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        return {
            "statusCode": 200,
            "ocrTexts": detected_texts,
            "confidenceScores": confidences,
            "outputVideoPath": output_video_path
        }

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {
            "statusCode": 500,
            "message": f"Processing error: {str(e)}"
        }


if __name__ == "__main__":
    image_result = process_image(IMAGE_PATH)
    print(image_result)

    # video_result = process_video(VIDEO_PATH, VIDEO_OUTPUT_PATH)
    # print(video_result)

