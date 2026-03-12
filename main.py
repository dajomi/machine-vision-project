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

def get_initial_h_sift(src_gray, tgt_gray):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(tgt_gray, None)
    kp2, des2 = sift.detectAndCompute(src_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None

def align_sift_only_check(source_img, target_img, target_mask):
    """
    ECC를 실행하지 않고, SIFT 매칭(또는 템플릿 매칭)으로 잡은 
    초기 위치까지만 시각화하여 확인하는 함수입니다.
    """
    # 1. 전처리
    src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY) if source_img.ndim==3 else source_img
    tgt_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if target_img.ndim==3 else target_img
    h_t, w_t = target_img.shape[:2]

    # 2. SIFT 초기화 (이전 단계에서 작성한 get_initial_h_sift 호출)
    H = get_initial_h_sift(src_gray, tgt_gray)
    
    method_used = "SIFT"
    
    if H is None:
        print("SIFT failed, showing Template Matching result instead...")
        method_used = "Template Matching"
        res = cv2.matchTemplate(src_gray, tgt_gray, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        H = np.array([[1, 0, max_loc[0]], [0, 1, max_loc[1]], [0, 0, 1]], dtype=np.float32)

    # 3. ECC를 건너뛰고 SIFT 결과만으로 Warp 수행
    # WARP_INVERSE_MAP을 사용하는 이유는 H가 target -> source 방향이기 때문입니다.
    aligned_sift = cv2.warpPerspective(
        source_img, 
        H, 
        (w_t, h_t), 
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    # 4. 시각화
    print(f"Showing results for: {method_used}")
    print(f"Matrix H:\n{H}")
    
    # 두 이미지를 나란히 비교하거나 겹쳐서 보기 위해 Target 이미지 준비
    # 검정색 배경에 마스크가 적용된 Target 이미지를 만들어 비교하면 더 정확합니다.
    tgt_masked = cv2.bitwise_and(target_img, target_img, mask=target_mask)
    
    # 화면 크기에 맞춰 리사이즈 (고해상도 대비)
    display_res = np.hstack([tgt_masked, aligned_sift])
    if display_res.shape[0] > 1000:
        scale = 1000.0 / display_res.shape[0]
        display_res = cv2.resize(display_res, None, fx=scale, fy=scale)

    cv2.imshow(f"Comparison (Left: Target / Right: {method_used} Result)", display_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return H, aligned_sift
def overlay_target_on_source(source_img, target_img, target_mask, H):
    """
    H를 이용해 target_img를 source_img 공간에 Warp하고 
    마스크 영역만 합성합니다.
    """
    h_s, w_s = source_img.shape[:2]
    
    # 1. Target 이미지와 마스크를 Source 공간으로 Warp
    # 주의: cv2.warpPerspective는 H 행렬을 그대로 사용합니다.
    # (이전에 WARP_INVERSE_MAP을 썼다면 H를 역행렬(cv2.invert(H))로 바꿔야 할 수 있음)
    warped_target = cv2.warpPerspective(target_img, H, (w_s, h_s))
    warped_mask = cv2.warpPerspective(target_mask, H, (w_s, h_s))
    
    # 마스크 이진화 (보간법으로 인해 0~255 사이값이 생길 수 있음)
    _, binary_mask = cv2.threshold(warped_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. 오버레이 합성
    # 마스크 영역은 Target으로, 나머지는 Source로 구성
    overlay_img = source_img.copy()
    
    # 마스크 영역을 복사 (비트마스크 연산)
    # 1) 소스 이미지에서 마스크 영역을 0으로 만듦
    mask_inv = cv2.bitwise_not(binary_mask)
    img_bg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
    
    # 2) 워프된 타겟에서 마스크 영역만 가져옴
    img_fg = cv2.bitwise_and(warped_target, warped_target, mask=binary_mask)
    
    # 3) 두 이미지 합치기
    result = cv2.add(img_bg, img_fg)
    
    return result

def calculate_masked_ncc(source_img, target_img, target_mask, H):
    """
    정렬된 상태(H)에서 마스크 영역의 NCC를 계산합니다.
    """
    # 1. Target 이미지를 Source 이미지 공간으로 정렬 (Warping)
    h_s, w_s = source_img.shape[:2]
    warped_target = cv2.warpPerspective(target_img, H, (w_s, h_s), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(target_mask, H, (w_s, h_s), flags=cv2.INTER_NEAREST)
    
    # 마스크 이진화 (0 혹은 1)
    mask = (warped_mask > 127).astype(np.float32) / 255.0
    
    # 2. 이미지를 float32로 변환 및 정규화
    src_f = source_img.astype(np.float32)
    tgt_f = warped_target.astype(np.float32)
    
    # 3. 마스크 영역 추출 및 평균 중심화 (Mean Centering)
    # 마스크 내부 픽셀들의 평균을 빼서 밝기 변화(조명 차이)에 강건하게 만듭니다.
    def get_masked_centered(img, mask):
        # 1. 마스크를 0~1 사이의 float으로 변환
        mask_float = mask.astype(np.float32) / 255.0
        
        # 2. 3채널 브로드캐스팅을 위해 차원 확장 (h, w, 1)
        if mask_float.ndim == 2:
            mask_float = mask_float[..., np.newaxis]
                
        # 3. 채널별 평균 계산 (axis=(0, 1)은 높이와 너비 방향으로 합산)
        # 이미지(H, W, 3) * 마스크(H, W, 1) = (H, W, 3)
        masked_data = img * mask_float
        
        # 각 채널별 유효 픽셀 수 (마스크 합계)
        sum_mask = np.sum(mask_float, axis=(0, 1)) + 1e-6
        
        # 채널별 평균값 (결과: 3개의 값)
        mean = np.sum(masked_data, axis=(0, 1)) / sum_mask
        
        # 4. 리턴 시에도 확장된 mask_float 사용
        return (img - mean) * mask_float
    
    src_centered = get_masked_centered(src_f, mask)
    tgt_centered = get_masked_centered(tgt_f, mask)
    
    # 4. NCC 계산 (정규화된 코사인 유사도)
    numerator = np.sum(src_centered * tgt_centered)
    denominator = np.sqrt(np.sum(src_centered**2) * np.sum(tgt_centered**2) + 1e-9)
    
    ncc_score = numerator / denominator
    
    return ncc_score

def main():
    PIXEL = 100


#     for path, stem, ext in iter_images_skip_temp("data", skip_dirs = {"temp", "Temp", "TEMP"}):
#         origin_img = cv2.imread(path)
#         if origin_img is None:
#             raise SystemExit("Error: image not found.")
#         try:
#             undistorted_img, pix_scale, H_mat, dst_pts = get_perspective_img(origin_img, ref_marker_size_px=PIXEL, debug=False)
#             result_path = path.split('\\')
#             result_directory = os.path.join(".temp",*result_path[1:-1])
#             result_path = os.path.join(".temp",*result_path[1:])
            
#             if not os.path.exists(result_directory):
#                 os.makedirs(result_directory)
#             cv2.imwrite(result_path, undistorted_img)
#             print(path, "->",  result_path)

#         except Exception as e:
#             # print(path, stem, ext)
#             # print(f"[ERROR] {type(e).__name__}: {e}")
#             pass

# # #     import sys
# # #     sys.exit()
# #     #=================================================================#
    
#     temp_path_dict = {}

#     for path, stem, ext in iter_images_skip_temp(".temp"):
#         keyId = path.split('\\')[1]
#         filename = stem + ext

#         temp_path_dict.setdefault(keyId, []).append(filename)

#     print(temp_path_dict.keys())
#     for dictName in temp_path_dict.keys():
#         if len(temp_path_dict[dictName])!=2: continue
#         croppedImgList = []
#         croppedMarkerPts = []
#         for filename in temp_path_dict[dictName]:
            
#             try:
#                 print(os.path.join(os.path.join(".temp", dictName),filename))
#                 undistort_img = cv2.imread(os.path.join(os.path.join(".temp", dictName),filename))
#                 cropped, (x, y, w, h) = detect_table_and_crop(undistort_img)
#                 pts =_get_charuco_pts(cropped)
#                 croppedImgList.append(cropped)
#                 croppedMarkerPts.append(pts)
#             except:
#                 pass

#             result_path = f".crop\\crop{dictName[-2:]}"
#             print(result_path)
#             if not os.path.exists(result_path):
#                 os.makedirs(result_path)
#             cv2.imwrite(os.path.join(result_path,filename), cropped)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
            
    base_root = "data\\target"

    # aaa, bbb, ccc 디렉토리 자동 검색 예시
    dirs = sorted(
        [p for p in glob.glob(os.path.join(base_root, "*"))
        if os.path.isdir(p)]
    )


    source_img_path = ".crop\\crop_7\\KakaoTalk_20260311_120335889_01.jpg"
    source_img = cv2.imread(source_img_path)

    if source_img is None:
        raise FileNotFoundError(f"Source image not found: {source_img_path}")

    # 2. 결과 저장용 리스트
    results = []

    # 3. 디렉토리 순회
    for target_dir in dirs:
        # 폴더명 가져오기
        target_name = os.path.basename(target_dir)
        
        # 이미지 및 마스크 경로 설정 (파일 이름은 폴더명과 동일하다고 가정)
        target_img_path = os.path.join(target_dir, f"{target_name}.png")
        mask_img_path = os.path.join(target_dir, f"{target_name}_mask.png")
        
        # 파일 존재 확인
        if not os.path.exists(target_img_path) or not os.path.exists(mask_img_path):
            print(f"Skipping {target_name}: Files not found.")
            continue
            
        target_img = cv2.imread(target_img_path)
        target_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        
        print(f"Processing: {target_name}...")
        
        # 4. 정렬 및 NCC 계산
        try:
            # 정렬 (H 행렬 확보)
            H, _ = align_sift_only_check(source_img, target_img, target_mask)
            
            # NCC 점수 계산
            score = calculate_masked_ncc(source_img, target_img, target_mask, H)
            
            # 결과 기록
            results.append({'name': target_name, 'score': score})
            print(f"-> {target_name} NCC Score: {score:.4f}")

            overlay = overlay_target_on_source(source_img, target_img, target_mask, H)
            cv2.imshow("Overlay Result", cv2.resize(overlay, None, fx=0.3, fy=0.3))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing {target_name}: {e}")

    # 5. 최종 결과 요약 (가장 유사도가 높은 물체 찾기)
    if results:

        best_match = max(results, key=lambda x: x['score'])
        print("\n" + "="*30)
        print(f"Best Match: {best_match['name']} (Score: {best_match['score']:.4f})")
        print("="*30)
            
        

#     #=================================================================#



#     #=================================================================#
if __name__ == "__main__":
    main()