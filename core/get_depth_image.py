# core/get_depth_image.py

import cv2
import numpy as np
from typing import List, Tuple

from .get_perspective_image import _warp_view


# -------------------------
# 0. Load pre-calibrated intrinsics (K0, dist0)
# -------------------------
def load_intrinsics(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-calibrated camera intrinsics from .npz file.

    The .npz file is expected to contain:
        - 'K'   : 3x3 camera matrix
        - 'dist': distortion coefficients
    """
    data = np.load(npz_path)
    K = data["K"]
    dist = data["dist"]
    return K, dist


# -------------------------
# 1. Pose estimation from ChArUco per frame
# -------------------------
def estimate_charuco_pose(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    square_length: float,
    marker_length: float,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
    min_corners: int = 4,
) -> tuple[np.ndarray, np.ndarray, cv2.aruco_CharucoBoard]:
    """
    Estimate pose (R, t) of ChArUco board in a single image.

    Returns
    -------
    R : (3,3) rotation matrix
    t : (3,1) translation vector
    board : CharucoBoard (for reuse)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (3, 3),
        square_length,
        marker_length,
        aruco_dict,
        np.array([1, 2, 3, 4], dtype=np.int32),
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect aruco markers
    detector = cv2.aruco.CharucoDetector(board)
    diamond_corners, diamond_ids, marker_corners, marker_ids = detector.detectDiamonds(
        img
    )

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers detected for pose estimation.")

    # Interpolate Charuco corners
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )

    if retval is None or retval < min_corners:
        raise RuntimeError("Not enough ChArUco corners for pose estimation.")

    # Pose estimation wrt board coordinate system
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K, dist
    )
    if not success:
        raise RuntimeError("Failed to estimate CharUco board pose.")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    return R, t, board


# -------------------------
# 2. Simple per-frame homography using Charuco corners (for top-view)
# -------------------------
def homography_from_charuco(
    img: np.ndarray,
    board: cv2.aruco_CharucoBoard,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """
    Compute homography that maps image -> board plane (Z=0) using CharUco corners.

    This is used only to build a consistent top-view warp for the reference view.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = board.dictionary

    # detect markers & interpolate charuco corners again (you can optimize by reusing)
    detector = cv2.aruco.CharucoDetector(board)
    diamond_corners, diamond_ids, marker_corners, marker_ids = detector.detectDiamonds(
        img
    )
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers detected for homography.")

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if retval is None or retval < 4:
        raise RuntimeError("Not enough CharUco corners for homography.")

    # image points
    src_pts = charuco_corners.reshape(-1, 2).astype(np.float32)

    # corresponding board coordinates (X, Y, 0) -> we take only X,Y
    # board.chessboardCorners: (N,3)
    all_obj_corners = board.chessboardCorners  # in board coordinate system
    # charuco_ids: indices into board corners
    obj_pts = []
    for cid in charuco_ids.flatten():
        obj_pts.append(all_obj_corners[cid, :2])  # X,Y
    obj_pts = np.array(obj_pts, dtype=np.float32)

    # map: image -> board-plane 2D
    H, mask = cv2.findHomography(src_pts, obj_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise RuntimeError("Failed to compute board homography.")
    return H


# -------------------------
# 3. Triangulation for sparse 3D points
# -------------------------
def triangulate_two_views(
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points from two views using given intrinsics and extrinsics.
    """
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # (3, N)
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T  # (3, N)

    X_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])  # (4, N)
    X = (X_h[:3] / X_h[3]).T  # (N, 3)
    return X


# -------------------------
# 4. RANSAC-based match filtering (optional but recommended)
# -------------------------
def filter_matches_ransac(
    pts1: np.ndarray, pts2: np.ndarray, ransac_thresh: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter matches using fundamental matrix RANSAC.

    Returns inlier subsets of pts1, pts2.
    """
    if pts1.shape[0] < 8:
        return pts1, pts2

    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=ransac_thresh, confidence=0.99
    )
    if F is None:
        return pts1, pts2

    mask = mask.ravel().astype(bool)
    return pts1[mask], pts2[mask]


# -------------------------
# 5. Depth rasterization onto top-view grid
# -------------------------
def rasterize_depth_on_top_view(
    points_3d: np.ndarray,
    pixel_scale: float,
    top_view_shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize sparse 3D points onto a top-view depth map.
    """
    H, W = top_view_shape[:2]
    depth_map = np.full((H, W), np.nan, dtype=np.float32)

    # Assuming board plane is Z=0, and world X->col, Y->row
    xs = points_3d[:, 0] / pixel_scale
    ys = points_3d[:, 1] / pixel_scale
    zs = points_3d[:, 2]

    cols = np.round(xs).astype(int)
    rows = np.round(ys).astype(int)

    for r, c, z in zip(rows, cols, zs):
        if 0 <= r < H and 0 <= c < W:
            if np.isnan(depth_map[r, c]) or z < depth_map[r, c]:
                depth_map[r, c] = z

    return depth_map


# -------------------------
# 6. High-level: get_depth_image
# -------------------------
def get_depth_image_from_three_views(
    imgs: List[np.ndarray],
    intrinsics_npz_path: str,
    square_length: float,
    marker_length: float,
    aruco_size_cm: float,
    ref_marker_size_px: int,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sparse 3D point cloud & depth map from 3 views.

    Strategy:
    - K0, dist0는 별도 캘리브레이션 세트로 추정해 저장된 것을 로드 (intrinsics_npz_path).
    - 각 view마다 Charuco로 pose(R_i, t_i)를 다시 추정 (board 좌표계 기준).
    - view0를 기준으로 top-view를 생성 (board plane rectification).
    - view0-1, view0-2 간 특징 매칭 + RANSAC + triangulation으로 3D point 획득.
    - board 좌표계에서의 (X,Y,Z)로 해석하고, top-view에 depth 래스터.

    Returns
    -------
    top_view_rgb : np.ndarray
    depth_map : np.ndarray (float32, 상대 깊이 위주)
    """
    if len(imgs) != 3:
        raise ValueError("This function expects exactly 3 images.")

    # 1) Load pre-calibrated intrinsics
    K, dist = load_intrinsics(intrinsics_npz_path)

    # 2) Undistort all images for consistency
    undistorted_imgs = []
    for im in imgs:
        h, w = im.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
        und = cv2.undistort(im, K, dist, None, newK)
        undistorted_imgs.append(und)
    # from now on, use newK as intrinsic
    K = newK
    dist = np.zeros_like(dist)  # effectively undistorted

    # 3) Estimate pose (R_i, t_i) of Charuco board in each undistorted view
    extrinsics: List[Tuple[np.ndarray, np.ndarray]] = []
    board_ref = None
    for i, im in enumerate(undistorted_imgs):
        R, t, board = estimate_charuco_pose(
            im,
            K=K,
            dist=dist,
            square_length=square_length,
            marker_length=marker_length,
        )
        extrinsics.append((R, t))
        if board_ref is None:
            board_ref = board

    # 4) Reference view (0) top-view using homography (image0 -> board plane)
    H_img0_to_plane = homography_from_charuco(
        undistorted_imgs[0], board_ref, K=K, dist=dist
    )
    # warp to plane coordinates; we keep "full" for now
    top_view_rgb, H_shifted = _warp_view(undistorted_imgs[0], H_img0_to_plane, full=True)

    pixel_scale = aruco_size_cm / float(ref_marker_size_px)
    H_top, W_top = top_view_rgb.shape[:2]

    # 5) Feature matching between view 0 and view 1,2
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    points_3d_all = []

    for i in [1, 2]:
        img1 = undistorted_imgs[0]
        img2 = undistorted_imgs[i]

        kps1, des1 = orb.detectAndCompute(img1, None)
        kps2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kps1) == 0 or len(kps2) == 0:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:300]

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        # RANSAC filter (기본 fundamental matrix 기반)
        pts1_in, pts2_in = filter_matches_ransac(pts1, pts2, ransac_thresh=1.0)
        if pts1_in.shape[0] < 8:
            continue

        R1, t1 = extrinsics[0]
        R2, t2 = extrinsics[i]

        pts_3d = triangulate_two_views(K, R1, t1, R2, t2, pts1_in, pts2_in)

        # (선택) Z<0 제거, 너무 먼 점 제거 등 간단한 필터링
        z = pts_3d[:, 2]
        valid = (z > -1000) & (z < 1000)  # 대충 클리핑, 사용 용도에 따라 조정
        pts_3d = pts_3d[valid]

        if pts_3d.size > 0:
            points_3d_all.append(pts_3d)

    if not points_3d_all:
        raise RuntimeError("Failed to obtain any 3D points from matched features.")

    points_3d_all = np.vstack(points_3d_all)  # (N, 3)

    if debug:
        print(f"[depth] Triangulated {points_3d_all.shape[0]} 3D points.")

    # 6) Depth rasterization on top-view grid
    depth_map = rasterize_depth_on_top_view(
        points_3d_all,
        pixel_scale=pixel_scale,
        top_view_shape=top_view_rgb.shape[:2],
    )

    return top_view_rgb, depth_map


# -------------------------
# 7. Test main
# -------------------------
if __name__ == "__main__":
    # Example paths (you should adjust these)
    intrinsics_path = "./data/intrinsics_charuco.npz"
    img_paths = [
        "./data/view1.jpg",
        "./data/view2.jpg",
        "./data/view3.jpg",
    ]

    imgs: List[np.ndarray] = []
    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            raise SystemExit(f"Error: image not found: {p}")
        imgs.append(im)

    top_view_rgb, depth_map = get_depth_image_from_three_views(
        imgs=imgs,
        intrinsics_npz_path=intrinsics_path,
        square_length=200.0,   # [same units as board design]
        marker_length=120.0,
        aruco_size_cm=6.1,
        ref_marker_size_px=300,
        debug=True,
    )

    # Simple visualization: show depth as heatmap (ignoring NaN)
    depth_vis = depth_map.copy()
    nan_mask = np.isnan(depth_vis)
    depth_vis[nan_mask] = 0.0
    depth_vis_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis_norm = depth_vis_norm.astype(np.uint8)
    depth_vis_color = cv2.applyColorMap(depth_vis_norm, cv2.COLORMAP_JET)

    cv2.imshow("Top-view RGB", top_view_rgb)
    cv2.imshow("Depth (sparse rasterized)", depth_vis_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
