import cv2
import numpy as np


class DeskQuadInteractor:
    def __init__(self, img_small, desk_w_mm=1200.0, desk_h_mm=600.0):
        self.img_small = img_small
        self.h, self.w = img_small.shape[:2]
        self.desk_w_mm = float(desk_w_mm)
        self.desk_h_mm = float(desk_h_mm)

        # TL, TR, BR(RD), BL(LD)
        self.quad = self._init_center_quad().astype(np.float32)

        # 드래그 상태
        self.dragging_corner = None   # 0~3 or None
        self.dragging_quad = False
        self.last_mouse = None
        self.corner_radius = 10  # 픽셀

        # 숫자 키로 선택된 코너 (0~3, None)
        # 0: LT, 1: RT, 2: RD, 3: LD
        self.selected_corner_by_key = None

        self.window_name = "Desk Selector"

    def _init_center_quad(self):
        ratio_img = self.w / self.h
        ratio_desk = self.desk_w_mm / self.desk_h_mm

        if ratio_desk > ratio_img:
            quad_w = int(self.w * 0.6)
            quad_h = int(quad_w / ratio_desk)
        else:
            quad_h = int(self.h * 0.6)
            quad_w = int(quad_h * ratio_desk)

        cx, cy = self.w // 2, self.h // 2
        x0 = cx - quad_w // 2
        x1 = cx + quad_w // 2
        y0 = cy - quad_h // 2
        y1 = cy + quad_h // 2

        quad = np.array(
            [
                [x0, y0],  # TL (1번 키)
                [x1, y0],  # TR (2번 키)
                [x1, y1],  # BR/RD (3번 키)
                [x0, y1],  # BL/LD (4번 키)
            ],
            dtype=np.float32,
        )
        return quad

    def _draw_overlay(self):
        base = self.img_small.copy()
        overlay = base.copy()

        pts_int = self.quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts_int], color=(255, 255, 255))
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, base)

        cv2.polylines(
            base,
            [pts_int],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # 코너 포인트
        for i, (x, y) in enumerate(self.quad):
            color = (0, 0, 255)
            if self.selected_corner_by_key == i:
                color = (0, 255, 0)  # 선택된 코너를 초록색으로 표시
            cv2.circle(base, (int(x), int(y)), self.corner_radius,
                       color, 2, cv2.LINE_AA)

        # 중앙 십자
        cx = int(self.quad[:, 0].mean())
        cy = int(self.quad[:, 1].mean())
        cv2.drawMarker(
            base,
            (cx, cy),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=2,
            line_type=cv2.LINE_AA,
        )

        # 안내 텍스트
        text = "1:LT 2:RT 3:RD 4:LD | Drag corners/inside | ENTER=OK, ESC=cancel"
        cv2.putText(
            base, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            base, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        return base

    def _hit_test_corner(self, x, y):
        for i, (cx, cy) in enumerate(self.quad):
            if np.hypot(x - cx, y - cy) <= self.corner_radius + 3:
                return i
        return None

    def _point_in_quad(self, x, y):
        cnt = self.quad.astype(np.float32).reshape(-1, 1, 2)
        res = cv2.pointPolygonTest(cnt, (float(x), float(y)), False)
        return res >= 0

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 키로 코너를 선택해 둔 경우: 그 코너만 드래그
            if self.selected_corner_by_key is not None:
                self.dragging_corner = self.selected_corner_by_key
                self.dragging_quad = False
            else:
                corner_idx = self._hit_test_corner(x, y)
                if corner_idx is not None:
                    self.dragging_corner = corner_idx
                    self.dragging_quad = False
                elif self._point_in_quad(x, y):
                    self.dragging_corner = None
                    self.dragging_quad = True
                else:
                    self.dragging_corner = None
                    self.dragging_quad = False
            self.last_mouse = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.last_mouse is None:
                return
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]

            if self.dragging_corner is not None:
                # 선택된 코너만 이동 (클램프 없음: 화면 밖 허용)
                self.quad[self.dragging_corner, 0] += dx
                self.quad[self.dragging_corner, 1] += dy
            elif self.dragging_quad:
                # 전체 사각형 이동 (이미지 안으로만)
                self.quad[:, 0] += dx
                self.quad[:, 1] += dy
                self.quad[:, 0] = np.clip(self.quad[:, 0], 0, self.w - 1)
                self.quad[:, 1] = np.clip(self.quad[:, 1], 0, self.h - 1)

            self.last_mouse = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_corner = None
            self.dragging_quad = False
            self.last_mouse = None

    def interact(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        accepted = False
        while True:
            vis = self._draw_overlay()
            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                accepted = False
                break
            if key in (13, 10):  # ENTER
                accepted = True
                break

            # 숫자 키 1,2,3,4로 코너 선택 (LT, RT, RD, LD)
            if key == ord('1'):
                self.selected_corner_by_key = 0
            elif key == ord('2'):
                self.selected_corner_by_key = 1
            elif key == ord('3'):
                self.selected_corner_by_key = 2
            elif key == ord('4'):
                self.selected_corner_by_key = 3

        cv2.destroyWindow(self.window_name)

        if not accepted:
            raise RuntimeError("User cancelled desk selection.")

        return self.quad.copy()  # TL,TR,RD,LD 순서 (투시 변환 시 그대로 사용 가능)


# ---- 간단 테스트 코드 ----

def compute_disparity_and_depth(
    undistorted_img_L,
    undistorted_img_R,
    desk_w_mm=1200.0,
    desk_h_mm=500.0,
    baseline_mm=1400.0,  # 두 촬영 위치 간 기준 거리 (대략)
):
    # 1. 그레이 변환
    left_gray  = cv2.cvtColor(undistorted_img_L, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(undistorted_img_R, cv2.COLOR_BGR2GRAY)

    h, w = left_gray.shape[:2]
    right_gray = cv2.resize(right_gray, (w, h), interpolation=cv2.INTER_AREA)

    # 2. StereoSGBM 파라미터 설정 (필요시 튜닝)[web:62][web:68]
    window_size = 5
    num_disp = 16 * 8  # 16의 배수
    min_disp = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # 3. disparity 계산 (CV_16S, 16배 스케일)[web:62]
    disp_raw = stereo.compute(left_gray, right_gray).astype(np.int16)
    disp = disp_raw.astype(np.float32) / 16.0  # 실제 disparity

    # 4. 시각화용 disparity (0~255)
    disp_vis = cv2.normalize(disp, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # 5. depth(mm) 계산
    #  (1) focal length(px)를 desk 물리 크기와 warp 결과 폭으로 근사
    #      f ≈ (W_pix / W_mm) * some_scale
    #      여기선 단순히 f = W_pix / W_mm 로 둠 (상대값)
    desk_w_pix = w
    f_pixel = desk_w_pix / float(desk_w_mm)

    baseline = float(baseline_mm)  # mm
    # Z = f * B / d
    depth_mm = np.zeros_like(disp, dtype=np.float32)
    mask = disp > 0.5  # 유효 disparity
    depth_mm[mask] = (f_pixel * baseline) / disp[mask]

    return disp, disp_vis, depth_mm


def test_desk_interactor(image):

    scale = 0.2
    img_small = cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    interactor = DeskQuadInteractor(img_small, desk_w_mm=1200.0, desk_h_mm=600.0)
    quad_small = interactor.interact()

    quad_full = quad_small / scale
    print("Selected quad (full LT,RT,RD,LD):\n", quad_full)

    # 투시 변환 예시: 사각형 그대로 LT,RT,RD,LD → 목적지 LT,RT,RD,LD
    from math import sqrt

    src_quad_full = quad_full.astype(np.float32)

    area = cv2.contourArea(src_quad_full)
    ratio = interactor.desk_w_mm / interactor.desk_h_mm
    W_pix = int(round(sqrt(area * ratio)))
    H_pix = int(round(sqrt(area / ratio)))
    W_pix = max(W_pix, 32)
    H_pix = max(H_pix, 32)

    dst_quad = np.array(
        [
            [0, 0],             # LT
            [W_pix - 1, 0],     # RT
            [W_pix - 1, H_pix - 1],  # RD
            [0, H_pix - 1],     # LD
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src_quad_full, dst_quad)
    warped = cv2.warpPerspective(image, H, (W_pix, H_pix))

    return warped, H



if __name__ == "__main__":
    SRC_IMG_PATH_1 = "data\\set_2\\KakaoTalk_20260311_112717712_05.jpg"
    SRC_IMG_PATH_2 = "data\\set_2\\KakaoTalk_20260311_112717712_06.jpg"

    img_L = cv2.imread(SRC_IMG_PATH_1)
    img_R = cv2.imread(SRC_IMG_PATH_2)

    undist_L, H_L = test_desk_interactor(img_L)
    undist_R, H_R = test_desk_interactor(img_R)

    undist_R = cv2.resize(undist_R, (undist_L.shape[1], undist_L.shape[0]), interpolation=cv2.INTER_AREA)
    diff_img = cv2.absdiff(cv2.cvtColor(undist_L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(undist_R, cv2.COLOR_BGR2GRAY))

    cv2.imshow("undist_L", cv2.resize(undist_L, None, fx=0.3, fy=0.3))
    cv2.imshow("undist_R", cv2.resize(undist_R, None, fx=0.3, fy=0.3))
    cv2.imshow("disparity_vis", cv2.resize(diff_img, None, fx=0.3, fy=0.3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()