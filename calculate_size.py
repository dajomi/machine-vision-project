import cv2 as cv
import numpy as np
import obb_detection
# import perspective_img


# 실제 : perspective_img에서 계산한 값 사용
# ratio = perspective_img.ratio

# test : 임의로 계산된 객체
ratio = 0.0093415

def calculate_real_size(cx, cy, w, h, ratio):
    x_cm = cx * ratio
    y_cm = cy * ratio
    w_cm = w * ratio
    h_cm = h * ratio

    return x_cm, y_cm, w_cm, h_cm


if __name__ == "__main__":
    
    image = cv.imread("./data/test.jpg")
    result = obb_detection.get_obb(image)

    if result is None:
        print("계산할 객체가 없습니다.")
    else:
        cx, cy, w, h, angle, box, mask = result

        x_cm, y_cm, w_cm, h_cm = calculate_real_size(cx, cy, w, h, ratio)

        print("center(cm):", round(x_cm, 2), round(y_cm, 2))
        print("size(cm):", round(w_cm, 2), round(h_cm, 2))
        print("angle:", round(angle, 2))