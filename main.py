# main.py

import numpy as np
import cv2

from core.get_perspective_image import get_perspective_img



def main():
    SRC_IMG_PATH = "./data/pre_persepective.jpg"
    img = cv2.imread(SRC_IMG_PATH)
    if img is None:
        raise SystemExit("Error: image not found.")

    get_perspective_img(img, ref_marker_size_px=100, debug=True)



if __name__ == "__main__":
    main()