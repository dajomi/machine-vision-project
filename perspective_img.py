import cv2
import numpy as np
def warp_full_view(srcImg, H):
    h, w = srcImg.shape[:2]

    # 1) 원본 이미지 4꼭짓점 (x,y)
    corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # 2) H로 투영
    warped_corners = cv2.perspectiveTransform(corners, H)  # (4,1,2)

    xs = warped_corners[:, 0, 0]
    ys = warped_corners[:, 0, 1]

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # 3) output 사이즈 (넉넉하게 int로 올림)
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    # 4) 음수 영역을 없애기 위해 평행이동 homography
    #    (min_x, min_y)를 (0,0)으로 옮김
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float64)

    H_shifted = T @ H  # 먼저 H로 warp, 그다음 평행이동

    # 5) 최종 warp
    retImg = cv2.warpPerspective(
        srcImg,
        H_shifted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255,0,255)
    )

    return retImg, H_shifted


def perspective_img(srcImg, ARUCO_SIZE=6.1, debug=False):
    aruco = cv2.aruco

    gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    squareLength = 200
    markerLength = 120
    diamond_ids_for_board = np.array([1, 2, 3, 4], dtype=np.int32)

    board = aruco.CharucoBoard(
        (3, 3),
        squareLength,
        markerLength,
        dictionary,
        diamond_ids_for_board
    )

    charuco_detector = aruco.CharucoDetector(board)

    diamond_corners, diamond_ids, marker_corners, marker_ids = \
        charuco_detector.detectDiamonds(srcImg)

    if diamond_corners is None or len(diamond_corners) == 0:
        raise RuntimeError("ChArUco Diamond 마커를 찾지 못했습니다.")
    if marker_corners is None or len(marker_corners) < 4:
        raise RuntimeError("ArUco 마커가 4개 미만입니다. 총 20점 매칭에 필요합니다.")

    if debug:
        debug_img = srcImg.copy()
        if marker_ids is not None and len(marker_ids) > 0:
            aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
        aruco.drawDetectedDiamonds(debug_img, diamond_corners, diamond_ids)
        cv2.imwrite("./test/detected_aruco.png", debug_img)

    # 1) src_pts 구성 (20점)
    src_pts_list = []
    diamond_pts = diamond_corners[0].reshape(4, 2)
    src_pts_list.append(diamond_pts)
    for mc in marker_corners[:4]:
        src_pts_list.append(mc.reshape(4, 2))
    src_pts = np.vstack(src_pts_list).astype(np.float32)  # (20,2)

    if debug:
        debug_img2 = srcImg.copy()
        for (x, y) in diamond_pts:
            cv2.circle(debug_img2, (int(x), int(y)), 5, (0, 0, 255), -1)
        for mc in marker_corners[:4]:
            for (x, y) in mc.reshape(-1, 2):
                cv2.circle(debug_img2, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.imwrite("./test/detected_20points.png", debug_img2)

    # 2) dst_pts (이론 좌표 20점)
    dst_pts = np.array(
         [[200., 100.],
         [200., 200.],
         [100., 200.],
         [100., 100.],

         [280., 120.],
         [280., 180.],
         [220., 180.],
         [220., 120.],

         [ 80., 120.],
         [ 80., 180.],
         [ 20., 180.],
         [ 20., 120.],
         
         [180.,  20.],
         [180.,  80.],
         [120.,  80.],
         [120.,  20.],


         [180., 220.],
         [180., 280.],
         [120., 280.],
         [120., 220.]
], np.float64)

    # 4) 호모그래피 계산
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    if H is None:
        raise RuntimeError("호모그래피 계산 실패")

    retImg, H_shifted = warp_full_view(srcImg, H)

    # 5) 투시변환
    # retImg = cv2.warpPerspective(
    #     srcImg,
    #     H,
    #     (2048, 1080),
    #     flags=cv2.INTER_LINEAR,
    #     borderMode=cv2.BORDER_CONSTANT,
    #     borderValue=(0, 0, 0)
    # )
    pixel_scale = ARUCO_SIZE / 300.0  # cm/pixel

    if debug:
        cv2.imshow("Origin Img (debug)", srcImg[::8, ::8])
        cv2.imshow("Top-View (debug)", retImg[::8, ::8])
        cv2.imwrite("test/result.png", retImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return retImg, pixel_scale


if __name__ == "__main__":
    SRC_IMG_PATH = "./data/pre_persepective.jpg"
    img = cv2.imread(SRC_IMG_PATH)
    if img is None:
        import sys
        print("Err: image not found!!")
        sys.exit()

    perspective_img(img, debug=True)
