# main.py

import numpy as np
import cv2
import os

from core.get_perspective_image import get_perspective_img, _get_charuco_pts
from core.crop import detect_table_and_crop

def iter_images_skip_temp(root, exts=(".jpg", ".jpeg", ".png")):
    exts = tuple(e.lower() for e in exts)
    skip_dirs = {"temp", "Temp", "TEMP"}  # 필요하면 이름 추가

    for dirpath, dirnames, filenames in os.walk(root):
        # 여기서 탐색 제외할 폴더 제거 → 그 하위는 안 내려감
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for name in filenames:
            if name.lower().endswith(exts):
                full_path = os.path.join(dirpath, name)
                stem, ext = os.path.splitext(name)
                yield full_path, stem, ext


# detect_table_and_crop
def main():

    # for path, stem, ext in iter_images_skip_temp("data"):
    #     origin_img = cv2.imread(path)
    #     if origin_img is None:
    #         raise SystemExit("Error: image not found.")
    #     try:
    #         undistorted_img, pix_scale, H_mat, dst_pts = get_perspective_img(origin_img, ref_marker_size_px=100, debug=False)
    #         result_path = path.split('\\')
    #         result_path = os.path.join(".temp",*result_path[1:])
    #         cv2.imwrite(result_path, undistorted_img)

    #     except Exception as e:
    #         print(path, stem, ext)
    #         print(f"[ERROR] {type(e).__name__}: {e}")
    
    #=================================================================#
    
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

            # result_path = f".crop\\crop{dictName[-2:]}"
            # if not os.path.exists(result_path):
            #     os.makedirs(result_path)
            # cv2.imwrite(os.path.join(result_path,filename), cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        H_1to2, mask = cv2.findHomography(croppedMarkerPts[0], croppedMarkerPts[1], cv2.RANSAC, 3.0)
        img1_warp = cv2.warpPerspective(croppedImgList[0], H_1to2, (croppedImgList[1].shape[1], croppedImgList[1].shape[0]))
        diff_img = cv2.absdiff(img1_warp, croppedImgList[1])
        cv2.imshow("diff_img", diff_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    #=================================================================#



    #=================================================================#
if __name__ == "__main__":
    main()