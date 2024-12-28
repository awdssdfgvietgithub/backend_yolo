import cv2
import easyocr
import numpy as np
import torch

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def ocr_image(img: np.ndarray, coordinates: list) -> str:
    x, y, w, h = map(int, coordinates)
    cropped_img = img[y:h, x:w]
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text = ""

    if len(result) == 1:
        text = result[0][1]
    else:
        for res in result:
            if len(res[1]) > 6 and res[2] > 0.2:
                text = res[1]
                break
    return text
