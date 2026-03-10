import cv2 as cv
import numpy as np
import object_list

# mask 값으로 obb detection

def get_obb(image):
    """
    input:
        image (BGR image)

    return:
        cx, cy : 중심좌표
        w, h   : width, height
        angle  : rotation angle
        box    : 4개 꼭짓점
        mask   : 객체 mask
    """

    mask = object_list.get_mask(image)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv.contourArea)

    rect = cv.minAreaRect(cnt)

    (cx, cy), (w, h), angle = rect

    box = cv.boxPoints(rect)
    box = np.int32(box)

    return cx, cy, w, h, angle, box, mask


# 테스트 실행용
if __name__ == "__main__":

    image = cv.imread("./data/test.jpg")

    result = get_obb(image)

    if result is None:
        print("no object for OBB")
    else:
        cx, cy, w, h, angle, box, mask = result

        vis = image.copy()

        cv.drawContours(vis, [box], 0, (0,255,0), 5)
        cv.circle(vis, (int(cx), int(cy)), 10, (0,0,255), -1)

        vis_small = cv.resize(vis, None, fx=0.25, fy=0.25)

        cv.imshow("obb result", vis_small)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imwrite("./test/obb_result.png", vis)