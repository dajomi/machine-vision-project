
import cv2
from get_perspective_image import get_perspective_img

# 0. 한 카메라 기준 Charuco 기반 top-view


warped, pixel_scale, H_img2world = get_perspective_img(imgL, aruco_size_cm=6.1)

# 1. stereo rectification, disparity
rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoSGBM_create(...)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

points3D = cv2.reprojectImageTo3D(disp, Q)  # shape (H,W,3)

# 2. 바닥/테이블 평면 피팅 (RANSAC으로 plane ax+by+cz+d=0 추정) 후
#    각 점을 평면 좌표계 (Xw,Yw,Zw)로 변환, 여기서 Zw는 평면에서의 높이.

# 3. top-view grid에 rasterization
top_h, top_w = warped.shape[:2]
depth_top = np.full((top_h, top_w), np.nan, np.float32)

for v in range(h):
    for u in range(w):
        Xw, Yw, Zw = world_pts[v, u]  # 월드 평면 좌표계
        ix = int(Xw / pixel_scale)
        iy = int(Yw / pixel_scale)
        if 0 <= ix < top_w and 0 <= iy < top_h:
            z = Zw  # 또는 카메라로부터 거리
            if np.isnan(depth_top[iy, ix]) or z < depth_top[iy, ix]:
                depth_top[iy, ix] = z
