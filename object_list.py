import cv2
import numpy as np
from PIL import Image

# 제가 임의로 작성한 mask 추출 코드 입니다
# 리턴 값 : object의 mask (numpy array)

def show(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(image).show()

def get_mask(img):
    # BGR → HSV                                                                                 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 파란색 범위
    lower = np.array([90, 60, 60])
    upper = np.array([130, 255, 255])

    # 1차 mask
    mask = cv2.inRange(hsv, lower, upper)

    # contour 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 큰 객체 선택
    cnt = max(contours, key=cv2.contourArea)

    # 최종 tissue mask
    tissue_mask = np.zeros_like(mask)
    cv2.drawContours(tissue_mask, [cnt], -1, 255, -1)

    return tissue_mask


# 테스트 코드
# image = cv2.imread("./data/tissue.jpg")
# mask = get_object_mask(image)

# tissue_only = cv2.bitwise_and(image, image, mask=mask)

# show(mask)