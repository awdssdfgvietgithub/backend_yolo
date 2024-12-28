import time
import os
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
from utils.image_helpers import decode_base64_to_image, encode_image_to_base64
from utils.ocr_helpers import ocr_image

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_180_epochs.pt")
model = YOLO(MODEL_PATH)


@app.route("/process-image/", methods=["POST"])
def process_image():
    try:
        start_time = time.time()

        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"statusCode": 400, "message": "Invalid input data"}), 400

        img = decode_base64_to_image(data["image_base64"])

        results = model.predict(source=img)

        inference_time = round(time.time() - start_time, 3)

        detected_texts = []
        confidences = []
        output_img = img.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                text_ocr = ocr_image(img, [x1, y1, x2, y2])
                detected_texts.append(text_ocr)
                confidences.append(conf)

                cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_img, f"{text_ocr} ({conf: .2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        output_base64 = encode_image_to_base64(output_img)

        return jsonify({
            "statusCode": 200,
            "ocrTexts": detected_texts,
            "confidenceScores": confidences,
            "takeTime": inference_time,
            "imageBase64OCR": output_base64
        })

    except Exception as e:
        return jsonify({"statusCode": 500, "message": f"Processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)