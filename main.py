# main.py

import numpy as np
import cv2
import os
import glob

from core.get_perspective_image import get_perspective_img, _get_charuco_pts
from core.crop import detect_table_and_crop

def iter_images_skip_temp(root, exts=(".jpg", ".jpeg", ".png"), skip_dirs = {"temp", "Temp", "TEMP"}):
    exts = tuple(e.lower() for e in exts)

    for dirpath, dirnames, filenames in os.walk(root):
        # 여기서 탐색 제외할 폴더 제거 → 그 하위는 안 내려감
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
        self.crop_img = cv2.resize(img[self.offset_y:self.offset_y+h, self.offset_x:self.offset_x+w], 
                                   (crop_size, crop_size))
        
        # 좌표 스케일 계산 (Crop 좌표 -> 원본 좌표 복원용)
        self.scale_x = w / crop_size
        self.scale_y = h / crop_size

        # 다각형 데이터: [[외곽점들], [내부점들1], [내부점들2]...]
        self.polygons = [[]] 
        self.current_poly_idx = 0
        self.selected_node = None # (poly_idx, node_idx)
        
        self.window_name = "Masking Tool: L-Click(Move/Add), Shift+L-Click(New Poly), Space(Done)"

    def draw(self):
        img_draw = self.crop_img.copy()
        for p_idx, pts in enumerate(self.polygons):
            color = (0, 255, 0) if p_idx == 0 else (0, 255, 255) # 외곽 초록, 내부 노랑
            for i, pt in enumerate(pts):
                cv2.circle(img_draw, tuple(pt), 5, color, -1)
                # 점 연결
                next_pt = pts[(i + 1) % len(pts)]
                cv2.line(img_draw, tuple(pt), tuple(next_pt), color, 2)
                cv2.putText(img_draw, str(i), (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
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
                # 3. 선분 위 클릭 시 점 추가 혹은 마지막에 추가
                # (단순화를 위해 현재 레이어의 마지막에 추가)
                self.polygons[self.current_poly_idx].append(pos)
                self.selected_node = (self.current_poly_idx, len(self.polygons[self.current_poly_idx]) - 1)

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
        self.polygons[0] = [np.array([cx-50, cy-50]), np.array([cx+50, cy-50]), 
                            np.array([cx+50, cy+50]), np.array([cx-50, cy+50])]

        while True:
            display = self.draw()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if key == ord(' '): break
            elif key == 27: return None, None
            
        cv2.destroyWindow(self.window_name)
        return self.generate_results()

    def generate_results(self):
        # 1. 원본 해상도의 빈 마스크 생성
        full_mask = np.zeros(self.full_img.shape[:2], dtype=np.uint8)
        
        # 2. 좌표 역변환 및 마스크 그리기
        orig_polygons = []
        for i, pts in enumerate(self.polygons):
            if len(pts) < 3: continue
            # 좌표 복원: (crop_pos * scale) + offset
            pts_orig = []
            for pt in pts:
                ox = int(pt[0] * self.scale_x + self.offset_x)
                oy = int(pt[1] * self.scale_y + self.offset_y)
                pts_orig.append([ox, oy])
            
            pts_np = np.array(pts_orig, dtype=np.int32)
            orig_polygons.append(pts_np)
            
            # 첫 번째는 흰색(255), 이후는 검은색(0)으로 채워 도넛 구현
            color = 255 if i == 0 else 0
            cv2.fillPoly(full_mask, [pts_np], color)

        # 3. 마스크가 있는 영역만 타이트하게 Crop (Min-Max)
        coords = np.column_stack(np.where(full_mask > 0))
        if len(coords) == 0: return None, None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        final_img = self.full_img[y_min:y_max+1, x_min:x_max+1]
        final_mask = full_mask[y_min:y_max+1, x_min:x_max+1]
        
        return final_img, final_mask
# detect_table_and_crop
def save_target_data(img, mask, name, base_dir="./data/target"):
    save_path = os.path.join(base_dir, name)
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), img)
    cv2.imwrite(os.path.join(save_path, f"{name}_mask.png"), mask)
    print(f"Saved to {save_path}")


import cv2
import numpy as np


def ecc_global_with_mask(
    source_img,
    target_img,
    target_mask,
    max_iter=200,
    eps=1e-4,
    max_target_scale_ratio=0.5,  # target이 source의 0.5 이상으로는 안 키움
):
    """
    source_img: 큰 책상 탑뷰 (BGR or gray)
    target_img: 마우스 탑뷰 (BGR or gray)
    target_mask: target_img와 같은 크기의 0/255 mask
    return:
        H (3x3 homography, source->target 정렬),
        ecc_score (float),
        aligned_source (source를 H로 warp한 결과),
        overlap_rect (x, y, w, h) in source/target 좌표계
    """
    # 1. gray 변환
    if source_img.ndim == 3:
        src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = source_img.copy()

    if target_img.ndim == 3:
        tgt_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    else:
        tgt_gray = target_img.copy()

    if target_mask.ndim == 3:
        tgt_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    else:
        tgt_mask = target_mask.copy()

    tgt_mask = (tgt_mask > 0).astype(np.uint8)  # 0/1

    h_s, w_s = src_gray.shape[:2]
    h_t, w_t = tgt_gray.shape[:2]

    # 2. target이 source의 일정 비율 이상으로 커지지 않도록 스케일 제한
    scale_y = (h_s * max_target_scale_ratio) / h_t
    scale_x = (w_s * max_target_scale_ratio) / w_t
    scale = min(scale_x, scale_y, 1.0)  # 1.0 넘지 않도록 (키우지 않음)

    if scale != 1.0:
        new_w = int(round(w_t * scale))
        new_h = int(round(h_t * scale))
        tgt_gray = cv2.resize(tgt_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        tgt_mask = cv2.resize(tgt_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h_t, w_t = new_h, new_w

    # 3. target/마스크를 source 크기로 패딩 (중앙 배치)
    template = np.zeros_like(src_gray, dtype=np.uint8)
    mask_full = np.zeros_like(src_gray, dtype=np.uint8)

    y0 = (h_s - h_t) // 2
    x0 = (w_s - w_t) // 2
    y1 = y0 + h_t
    x1 = x0 + w_t

    template[y0:y1, x0:x1] = tgt_gray
    mask_full[y0:y1, x0:x1] = tgt_mask

    # 4. ECC (homography, 전체 스케일/회전/시어 허용)[web:79][web:82]
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max_iter,
        eps,
    )

    try:
        ecc_score, warp_matrix = cv2.findTransformECC(
            template,        # templateImage (target패딩)
            src_gray,        # inputImage  (source)
            warp_matrix,
            warp_mode,
            criteria,
            mask_full,       # target 마스크만 사용
            5,
        )
    except cv2.error:
        return None, -1.0, None, None

    # 5. source를 template 크기(=source 크기)에 맞게 warp
    aligned_src = cv2.warpPerspective(
        source_img,
        warp_matrix,
        (w_s, h_s),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 6. overlap 사각형 (마우스 영역 기준)
    # 패딩된 target 영역 좌표 (template에서 마우스가 있는 박스)
    # 이 영역을 H로 역변환해서, source/warped space에서의 실제 위치를 얻을 수도 있지만,
    # 간단히 aligned_src vs template에서 "둘 다 0이 아닌" 영역의 AABB를 취함.
    aligned_gray = cv2.cvtColor(aligned_src, cv2.COLOR_BGR2GRAY) \
        if aligned_src.ndim == 3 else aligned_src

    mask_overlap = (aligned_gray != 0).astype(np.uint8) & (template != 0).astype(np.uint8)
    ys, xs = np.where(mask_overlap > 0)
    if len(xs) == 0 or len(ys) == 0:
        overlap_rect = None
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        overlap_rect = (
            int(x_min),
            int(y_min),
            int(x_max - x_min + 1),
            int(y_max - y_min + 1),
        )

    return warp_matrix, float(ecc_score), aligned_src, overlap_rect


def main():
    PIXEL = 100


    for path, stem, ext in iter_images_skip_temp("data", skip_dirs = {"temp", "Temp", "TEMP"}):
        origin_img = cv2.imread(path)
        if origin_img is None:
            raise SystemExit("Error: image not found.")
        try:
            undistorted_img, pix_scale, H_mat, dst_pts = get_perspective_img(origin_img, ref_marker_size_px=PIXEL, debug=False)
            result_path = path.split('\\')
            result_directory = os.path.join(".temp",*result_path[1:-1])
            result_path = os.path.join(".temp",*result_path[1:])
            
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

    for path, stem, ext in iter_images_skip_temp(".temp"):
        keyId = path.split('\\')[1]
        filename = stem + ext

        temp_path_dict.setdefault(keyId, []).append(filename)

    for dictName in temp_path_dict.keys():
        if len(temp_path_dict[dictName])!=2: continue
        croppedImgList = []
        croppedMarkerPts = []
        for filename in temp_path_dict[dictName]:
            
            try:
                undistort_img = cv2.imread(os.path.join(os.path.join(".temp", dictName),filename))
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

            # aaa, bbb, ccc 디렉토리 자동 검색 예시
            dirs = sorted(
                [p for p in glob.glob(os.path.join(base_root, "*"))
                if os.path.isdir(p)]
            )

            source_path = ".temp\\set_6\\KakaoTalk_20260311_120335889_02.jpg"  # 기준 책상 이미지 경로로 교체

            source_img = cv2.imread(".temp\\set_6\\KakaoTalk_20260311_120335889_02.jpg")
            target_img = cv2.imread("data\\target\\book_machinevision\\book_machinevision.png")
            target_mask = cv2.imread("data\\target\\book_machinevision\\book_machinevision_mask.png", cv2.IMREAD_GRAYSCALE)

            H, score, aligned, overlap_rect = ecc_global_with_mask(
                source_img, target_img, target_mask,
                max_iter=300,
                eps=1e-5,
                max_target_scale_ratio=0.5,
            )

            print("ECC score:", score)
            print("H:\n", H)

            vis = aligned.copy()
            if overlap_rect is not None:
                x, y, w, h = overlap_rect
                cv2.rectangle(vis, (x, y), (x + w - 1, y + h - 1),
                            (0, 0, 255), 2)

            cv2.imshow("aligned_src_with_overlap", cv2.resize(vis, None, fx=0.3, fy=0.3))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # test_batch_match_with_ecc_mask(source_path, dirs, pix_scale=PIXEL)
                    
        

#     #=================================================================#



#     #=================================================================#
if __name__ == "__main__":
    main()