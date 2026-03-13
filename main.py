# main.py

import numpy as np
import cv2
import os
import glob

from core.get_perspective_image import get_perspective_img, _get_charuco_pts
from core.crop import detect_table_and_crop


def iter_images_skip_temp(root, exts=(".jpg", ".jpeg", ".png"),
                          skip_dirs={"temp", "Temp", "TEMP"}):
    exts = tuple(e.lower() for e in exts)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for name in filenames:
            if name.lower().endswith(exts):
                full_path = os.path.join(dirpath, name)
                stem, ext = os.path.splitext(name)
                yield full_path, stem, ext


class MaskEditor:
    def __init__(self, img, crop_size=1000):
        self.full_img = img
        self.crop_size = crop_size

        # 1) ROI 선택 및 Crop (초기 ROI는 사용자가 대략적으로 선택)
        roi = cv2.selectROI("Select Rough ROI of Object", img, False)
        cv2.destroyWindow("Select Rough ROI of Object")

        self.offset_x, self.offset_y, w, h = roi
        self.crop_img = cv2.resize(
            img[self.offset_y:self.offset_y + h,
                self.offset_x:self.offset_x + w],
            (crop_size, crop_size)
        )

        # 좌표 스케일 계산 (Crop 좌표 -> 원본 좌표 복원용)
        self.scale_x = w / crop_size
        self.scale_y = h / crop_size

        # 다각형 데이터: [[외곽점들], [내부점들1], [내부점들2]...]
        self.polygons = [[]]
        self.current_poly_idx = 0
        self.selected_node = None  # (poly_idx, node_idx)

        self.window_name = "Masking Tool: L-Click(Move/Add), Shift+L-Click(New Poly), Space(Done)"

    def draw(self):
        img_draw = self.crop_img.copy()
        for p_idx, pts in enumerate(self.polygons):
            color = (0, 255, 0) if p_idx == 0 else (0, 255, 255)  # 외곽 초록, 내부 노랑
            for i, pt in enumerate(pts):
                cv2.circle(img_draw, tuple(pt), 5, color, -1)
                next_pt = pts[(i + 1) % len(pts)]
                cv2.line(img_draw, tuple(pt), tuple(next_pt), color, 2)
                cv2.putText(img_draw, str(i), (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img_draw

    def mouse_callback(self, event, x, y, flags, param):
        pos = np.array([x, y])

        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. 기존 점 선택 확인 (이동)
            for p_idx, pts in enumerate(self.polygons):
                for n_idx, pt in enumerate(pts):
                    if np.linalg.norm(pt - pos) < 10:
                        self.selected_node = (p_idx, n_idx)
                        return

            # 2. Shift 누르고 클릭 시 새로운 다각형 시작 (도넛 구멍용)
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.polygons.append([pos])
                self.current_poly_idx = len(self.polygons) - 1
                self.selected_node = (self.current_poly_idx, 0)
            else:
                # 3. 현재 polygon에 점 추가
                self.polygons[self.current_poly_idx].append(pos)
                self.selected_node = (self.current_poly_idx,
                                      len(self.polygons[self.current_poly_idx]) - 1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selected_node:
                p_idx, n_idx = self.selected_node
                self.polygons[p_idx][n_idx] = pos

        elif event == cv2.EVENT_LBUTTONUP:
            self.selected_node = None

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 기본 사각형 초기화 (중앙에)
        cx, cy = self.crop_size // 2, self.crop_size // 2
        self.polygons[0] = [np.array([cx - 50, cy - 50]), np.array([cx + 50, cy - 50]),
                            np.array([cx + 50, cy + 50]), np.array([cx - 50, cy + 50])]

        while True:
            display = self.draw()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(' '):
                break
            elif key == 27:
                return None, None

        cv2.destroyWindow(self.window_name)
        return self.generate_results()

    def generate_results(self):
        # 1. 원본 해상도의 빈 마스크 생성
        full_mask = np.zeros(self.full_img.shape[:2], dtype=np.uint8)

        # 2. 좌표 역변환 및 마스크 그리기
        for i, pts in enumerate(self.polygons):
            if len(pts) < 3:
                continue
            pts_orig = []
            for pt in pts:
                ox = int(pt[0] * self.scale_x + self.offset_x)
                oy = int(pt[1] * self.scale_y + self.offset_y)
                pts_orig.append([ox, oy])

            pts_np = np.array(pts_orig, dtype=np.int32)

            color = 255 if i == 0 else 0
            cv2.fillPoly(full_mask, [pts_np], color)

        # 3. 마스크가 있는 영역만 타이트하게 Crop (Min-Max)
        coords = np.column_stack(np.where(full_mask > 0))
        if len(coords) == 0:
            return None, None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        final_img = self.full_img[y_min:y_max + 1, x_min:x_max + 1]
        final_mask = full_mask[y_min:y_max + 1, x_min:x_max + 1]

        return final_img, final_mask


def save_target_data(img, mask, name, base_dir="./data/target"):
    save_path = os.path.join(base_dir, name)
    os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(os.path.join(save_path, f"{name}.png"), img)
    cv2.imwrite(os.path.join(save_path, f"{name}_mask.png"), mask)
    print(f"Saved to {save_path}")


# ------------------ 다운샘플 + 가우시안 + SIFT/ORB ------------------ #

def resize_with_short_side(img, short_side=480):
    h, w = img.shape[:2]
    m = min(h, w)
    if m == short_side:
        return img.copy(), 1.0
    if h < w:
        scale = short_side / float(h)
    else:
        scale = short_side / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def draw_feature_matches(src_small, tgt_small, kp2, kp1, good, max_draw=50):
    """
    src_small: source (scene, downsampled)
    tgt_small: target (template, downsampled)
    kp1: target keypoints (tgt_small 기준)
    kp2: source keypoints (src_small 기준)
    good: 매칭 리스트 (DMatch)
    max_draw: 그릴 매칭 개수 제한
    """
    # drawMatches는 img1, kp1, img2, kp2, matches, ...
    matches_to_draw = sorted(good, key=lambda m: m.distance)[:max_draw]
    match_vis = cv2.drawMatches(
        tgt_small, kp1,
        src_small, kp2,
        matches_to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow("Feature Matches (Target | Source)", match_vis)
    cv2.waitKey(0)
    cv2.destroyWindow("Feature Matches (Target | Source)")

def preprocess_for_feature(img, ksize=5, sigma=1.0):
    
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    # gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    # # 2) magnitude = sqrt(gx^2 + gy^2)
    # mag = cv2.magnitude(gx, gy)

    # # 3) 0–255로 정규화 후 uint8 변환
    # mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # mag_u8 = mag_norm.astype(np.uint8)
    return gray


def create_feature_detector(use_orb=False):
    if use_orb:
        return cv2.ORB_create(
            nfeatures=1000,
            scaleFactor=1.2,
            nlevels=8
        )
    else:
        return cv2.SIFT_create(
                                nfeatures=2000,
                                contrastThreshold=0.02,   # 기본 0.04 근처 → 더 낮게
                                edgeThreshold=10,
                                sigma=1.6)


def get_initial_h_feature(src_gray, tgt_gray, use_orb=False,
                          ratio_thresh=0.75, min_good=8, ransac_reproj=5.0):
    detector = create_feature_detector(use_orb=use_orb)

    kp1, des1 = detector.detectAndCompute(tgt_gray, None)
    kp2, des2 = detector.detectAndCompute(src_gray, None)

    if des1 is None or des2 is None:
        return None, kp1, kp2, []

    if use_orb:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < min_good:
        return None, kp1, kp2, good

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj)
    return H, kp1, kp2, good


def align_and_overlay_downsampled(src_img, tgt_img, tgt_mask,
                                  use_orb=False, short_side=480):
    # 1) 다운샘플
    src_small, scale_src = resize_with_short_side(src_img, short_side=short_side)
    tgt_small = cv2.resize(tgt_img, None, fx=scale_src, fy=scale_src, interpolation=cv2.INTER_AREA)
    # tgt_small, scale_tgt = resize_with_short_side(tgt_img, short_side=short_side)
    mask_small = cv2.resize(tgt_mask, (tgt_small.shape[1], tgt_small.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    # 2) 전처리 (Gaussian + gray)
    src_gray = preprocess_for_feature(src_small)
    tgt_gray = preprocess_for_feature(tgt_small)

    # 3) feature 기반 H 추정
    H_small, kp1, kp2, good = get_initial_h_feature(
        src_gray, tgt_gray,
        use_orb=use_orb,
        ratio_thresh=0.9,
        min_good=4,
        ransac_reproj=6
    )
    if len(good) > 0:
        draw_feature_matches(src_small, tgt_small, kp2, kp1, good, max_draw=50)

    method_used = "ORB" if use_orb else "SIFT"

    if H_small is None:
        print(f"{method_used} failed, fallback to Template Matching on small images...")
        method_used = "Template Matching"
        res = cv2.matchTemplate(src_gray, tgt_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        H_small = np.array([[1, 0, max_loc[0]],
                            [0, 1, max_loc[1]],
                            [0, 0, 1]], dtype=np.float32)

    # 4) 저해상도에서 warp + overlay + diff
    h_t, w_t = tgt_small.shape[:2]
    aligned_small = cv2.warpPerspective(
        src_small,
        H_small,
        (w_t, h_t),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    tgt_masked_small = cv2.bitwise_and(tgt_small, tgt_small, mask=mask_small)
    overlay_small = cv2.addWeighted(tgt_masked_small, 0.5, aligned_small, 0.5, 0)

    diff_small = cv2.absdiff(tgt_masked_small, aligned_small)
    diff_small_gray = cv2.cvtColor(diff_small, cv2.COLOR_BGR2GRAY)
    diff_small_color = cv2.cvtColor(diff_small_gray, cv2.COLOR_GRAY2BGR)

    vis = np.hstack([tgt_masked_small, aligned_small, overlay_small, diff_small_color])

    print(f"[{method_used}] H (small):\n{H_small}")
    cv2.imshow("Small: Target | Aligned | Overlay | Diff", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H_small, scale_src, scale_src


def lift_homography_to_full(H_small, scale_src, scale_tgt):
    """
    H_small: 다운샘플된 src_small -> tgt_small 공간에서의 homography
    scale_src: src_small = resize_with_short_side(src)에서의 scale
    scale_tgt: tgt_small = resize_with_short_side(tgt)에서의 scale

    H_full: 원본 source -> 원본 target
    """
    S_src = np.array([[scale_src, 0, 0],
                      [0, scale_src, 0],
                      [0, 0, 1]], dtype=np.float32)
    S_tgt_inv = np.array([[1.0 / scale_tgt, 0, 0],
                          [0, 1.0 / scale_tgt, 0],
                          [0, 0, 1]], dtype=np.float32)
    H_full = S_tgt_inv @ H_small @ S_src
    return H_full


# ------------------ overlay + NCC (원본 해상도) ------------------ #

def overlay_target_on_source(source_img, target_img, target_mask, H):
    """
    H를 이용해 target_img를 source_img 공간에 Warp하고
    마스크 영역만 합성합니다.
    """
    h_s, w_s = source_img.shape[:2]

    warped_target = cv2.warpPerspective(target_img, H, (w_s, h_s))
    warped_mask = cv2.warpPerspective(target_mask, H, (w_s, h_s))

    _, binary_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)

    overlay_img = source_img.copy()

    mask_inv = cv2.bitwise_not(binary_mask)
    img_bg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
    img_fg = cv2.bitwise_and(warped_target, warped_target, mask=binary_mask)

    result = cv2.add(img_bg, img_fg)
    return result


def calculate_masked_ncc(source_img, target_img, target_mask, H):
    """
    정렬된 상태(H)에서 마스크 영역의 NCC를 계산합니다.
    """
    h_s, w_s = source_img.shape[:2]
    warped_target = cv2.warpPerspective(target_img, H, (w_s, h_s), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(target_mask, H, (w_s, h_s), flags=cv2.INTER_NEAREST)

    mask = (warped_mask > 127).astype(np.float32) / 255.0

    src_f = source_img.astype(np.float32)
    tgt_f = warped_target.astype(np.float32)

    def get_masked_centered(img, mask):
        mask_float = mask.astype(np.float32) / 255.0
        if mask_float.ndim == 2:
            mask_float = mask_float[..., np.newaxis]

        masked_data = img * mask_float

        sum_mask = np.sum(mask_float, axis=(0, 1)) + 1e-6
        mean = np.sum(masked_data, axis=(0, 1)) / sum_mask

        return (img - mean) * mask_float

    src_centered = get_masked_centered(src_f, mask)
    tgt_centered = get_masked_centered(tgt_f, mask)

    numerator = np.sum(src_centered * tgt_centered)
    denominator = np.sqrt(np.sum(src_centered ** 2) * np.sum(tgt_centered ** 2) + 1e-9)

    ncc_score = numerator / denominator
    return ncc_score


# ------------------ main ------------------ #

def main():
    PIXEL = 100


    for path, stem, ext in iter_images_skip_temp("data", skip_dirs = {"temp", "Temp", "TEMP"}):
        origin_img = cv2.imread(path)
        if origin_img is None:
            raise SystemExit("Error: image not found.")
        try:
            undistorted_img, pix_scale, H_mat, dst_pts = get_perspective_img(origin_img, ref_marker_size_px=PIXEL, debug=False)
            result_path = path.split('\\')
            result_directory = os.path.join(".undistort",*result_path[1:-1])
            result_path = os.path.join(".undistort",*result_path[1:])
            
            if not os.path.exists(result_directory):
                os.makedirs(result_directory)
            cv2.imwrite(result_path, undistorted_img)
            print(path, "->",  result_path)

        except Exception as e:
            # print(path, stem, ext)
            # print(f"[ERROR] {type(e).__name__}: {e}")
            pass

# #     import sys
# #     sys.exit()
#     #=================================================================#
    
    temp_path_dict = {}

    for path, stem, ext in iter_images_skip_temp(".undistort"):
        keyId = path.split('\\')[1]
        filename = stem + ext

        temp_path_dict.setdefault(keyId, []).append(filename)

    print(temp_path_dict.keys())
    for dictName in temp_path_dict.keys():
        if len(temp_path_dict[dictName])!=2: continue
        croppedImgList = []
        croppedMarkerPts = []
        for filename in temp_path_dict[dictName]:
            
            try:
                print(os.path.join(os.path.join(".undistort", dictName),filename))
                undistort_img = cv2.imread(os.path.join(os.path.join(".undistort", dictName),filename))
                cropped, (x, y, w, h) = detect_table_and_crop(undistort_img)
                pts =_get_charuco_pts(cropped)
                croppedImgList.append(cropped)
                croppedMarkerPts.append(pts)
            except:
                pass

            result_path = f".crop\\crop{dictName[-2:]}"
            print(result_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            cv2.imwrite(os.path.join(result_path,filename), cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
    base_root = "data\\target"
    dirs = sorted(
        [p for p in glob.glob(os.path.join(base_root, "*"))
         if os.path.isdir(p)]
    )

    source_img_path = ".crop\\crop_4\\KakaoTalk_20260311_120335889_07.jpg"
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        raise FileNotFoundError(f"Source image not found: {source_img_path}")

    results = []

    for target_dir in dirs:
        target_name = os.path.basename(target_dir)

        target_img_path = os.path.join(target_dir, f"{target_name}.png")
        mask_img_path = os.path.join(target_dir, f"{target_name}_mask.png")

        if not (os.path.exists(target_img_path) and os.path.exists(mask_img_path)):
            print(f"Skipping {target_name}: Files not found.")
            continue

        target_img = cv2.imread(target_img_path)
        target_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

        print(f"Processing: {target_name}...")

        try:
            # 1) 저해상도 정렬 + overlay + diff (확인용)
            H_small, s_src, s_tgt = align_and_overlay_downsampled(
                source_img, target_img, target_mask,
                use_orb=False,      # ORB 쓰고 싶으면 True
                short_side=640
            )

            # 2) homography를 원본 해상도로 lift
            H_full = lift_homography_to_full(H_small, s_src, s_tgt)

            # 3) NCC 계산 (원본 해상도)
            score = calculate_masked_ncc(source_img, target_img, target_mask, H_full)
            results.append({'name': target_name, 'score': score})
            print(f"-> {target_name} NCC Score: {score:.4f}")

            overlay = overlay_target_on_source(source_img, target_img, target_mask, H_full)
            cv2.imshow("Overlay Result (Full Res)", cv2.resize(overlay, None, fx=0.3, fy=0.3))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing {target_name}: {e}")

    if results:
        best_match = max(results, key=lambda x: x['score'])
        print("\n" + "=" * 30)
        print(f"Best Match: {best_match['name']} (Score: {best_match['score']:.4f})")
        print("=" * 30)
            
        

#     #=================================================================#



#     #=================================================================#
if __name__ == "__main__":
    main()