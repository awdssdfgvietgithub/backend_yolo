import base64
import numpy as np
from io import BytesIO
import cv2
from fastapi import HTTPException
from PIL import Image


def decode_base64_to_image(base64_string: str) -> np.ndarray:
    try:
        img_data = base64.b64decode(base64_string)
        img_pil = Image.open(BytesIO(img_data))
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 input: {str(e)}")


def encode_image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64
