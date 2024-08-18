import cv2
import numpy as np

def drawing(img, kondisi):
    img_ = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]

    konturs = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for kontur in konturs:
        x, y, w, h = cv2.boundingRect(kontur)
        if kondisi == 1:    # segi 4
            cv2.rectangle(img_, (x, y), (x + w, y + h), (0, 255, 0), 5)
        if kondisi == 2:    # lingkaran
            cv2.circle(img_, (int((w / 2) + x), int((h / 2) + y)), int(w / 2), (0, 255, 0), 5)
        if kondisi == 3:    # segi 3
            # Mengapproximate kontur menjadi segitiga
            epsilon = 0.05 * cv2.arcLength(kontur, True)
            approx = cv2.approxPolyDP(kontur, epsilon, True)

            # Pastikan segitiga
            if len(approx) == 3:
                cv2.drawContours(img_, [approx], 0, (0, 255, 0), 5)

    return img_