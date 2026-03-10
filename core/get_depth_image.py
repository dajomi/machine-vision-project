import cv2
import numpy as np
from typing import List, Tuple


# -------------------------
# 0. Warp utility (그대로 재사용)
# -------------------------
def _warp_view(
    src_img: np.ndarray,
    H: np.ndarray,
    full: bool = True,
    border_value: tuple[int, int, int] = (255, 0, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp an image using a homography, either to a full bounding view
    or to an approximate inner cropped view.
    """
    h, w = src_img.shape[:2]

    corners = np.array(
        [[0, 0],
         [w, 0],
         [w, h],
         [0, h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, H)
    xs = warped_corners[:, 0, 0]
    ys = warped_corners[:, 0, 1]

    if full:
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
    else:
        xs_sort = np.sort(xs)
        ys_sort = np.sort(ys)
        min_x, max_x = xs_sort[1], xs_sort[2]
        min_y, max_y = ys_sort[1], ys_sort[2]

    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    T = np.array(
        [[1.0, 0.0, -float(min_x)],
         [0.0, 1.0, -float(min_y)],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_shifted = T @ H

    warped = cv2.warpPerspective(
        src_img,
        H_shifted,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return warped, H_shifted


# -------------------------
# 1. Intrinsic calibration (rough, using ChArUco)
# -------------------------
def calibrate_intrinsic_charuco(
    image_list: List[np.ndarray],
    square_length: float,
    marker_length: float,
    dictionary_id: int = cv2.aruco.DICT_6X6_250,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roughly calibrate camera intrinsics using multiple ChArUco images.

    Returns
    -------
    K : np.ndarray
        Camera matrix (3x3).
    dist : np.ndarray
        Distortion coefficients.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (3, 3),
        square_length,
        marker_length,
        aruco_dict,
        np.array([1, 2, 3, 4], dtype=np.int32),
    )

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    detector = cv2.aruco.CharucoDetector(board)

    for img in image_list:
        if image_size is None:
            image_size = img.shape[1], img.shape[0]

        diamond_corners, diamond_ids, marker_corners, marker_ids = \
            detector.detectDiamonds(img)

        if marker_ids is None or len(marker_ids) == 0:
            continue

        # ArUco 코너를 이용해 Charuco 코너/ID 추출
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            img,
            board,
        )
        if retval is None or retval < 4:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    if not all_charuco_corners:
        raise RuntimeError("Failed to detect enough ChArUco corners for calibration.")

    rms, K, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None,
    )
    print(f"[calibrate] RMS reprojection error: {rms:.4f}")
    return K, dist


# -------------------------
# 2. Extrinsic from homography + K
# -------------------------
def extrinsic_from_homography(
    H: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover extrinsic parameters (R, t) from homography H and camera matrix K.

    Assumes H maps plane (Z=0 in world) to image.

    Returns
    -------
    R : np.ndarray
        Rotation matrix (3x3).
    t : np.ndarray
        Translation vector (3,1).
    """
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    # Normalize
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)

    t = lam * (K_inv @ h3)

    # Orthonormalize R using SVD
    R = np.stack((r1, r2, r3), axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    return R, t.reshape(3, 1)


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
    Triangulate 3D points from two views.

    Parameters
    ----------
    K : np.ndarray
        Camera matrix (3x3).
    R1, t1, R2, t2 : np.ndarray
        Extrinsic parameters of view1 and view2.
    pts1, pts2 : np.ndarray
        Matched points in image1 and image2, shape (N, 2).

    Returns
    -------
    points_3d : np.ndarray
        Triangulated 3D points in world (plane) coordinates, shape (N, 3).
    """
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # (3, N)
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T  # (3, N)

    X_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])  # (4, N)
    X = (X_h[:3] / X_h[3]).T  # (N, 3)
    return X


# -------------------------
# 4. Depth rasterization on top-view
# -------------------------
def rasterize_depth_on_top_view(
    points_3d: np.ndarray,
    pixel_scale: float,
    top_view_shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize sparse 3D points onto a top-view depth map.

    Parameters
    ----------
    points_3d : np.ndarray
        3D points in world coordinates, shape (N, 3).
        Assumes Z is "height" or depth from plane.
    pixel_scale : float
        Centimeters per pixel (same as top-view scale).
    top_view_shape : tuple[int, int]
        (H, W) of the top-view image.

    Returns
    -------
    depth_map : np.ndarray
        Depth map in same resolution as top-view, float32.
        0 or NaN indicates no data.
    """
    H, W = top_view_shape[:2]

    depth_map = np.full((H, W), np.nan, dtype=np.float32)

    # World XY -> pixel indices (assuming world (0,0) maps to (0,0) pixel,
    # and X→col, Y→row with pixel_scale)
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
# 5. High-level: get_depth_image
# -------------------------
def get_depth_image_from_three_views(
    imgs: List[np.ndarray],
    Hs: List[np.ndarray],
    square_length: float,
    marker_length: float,
    aruco_size_cm: float,
    ref_marker_size_px: int,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sparse 3D point cloud & depth map from 3 views and their homographies.

    Parameters
    ----------
    imgs : list of np.ndarray
        List of 3 images.
    Hs : list of np.ndarray
        List of 3 homography matrices (each 3x3) for the ChArUco plane.
    square_length : float
        ChArUco square length (same as used in perspective_img).
    marker_length : float
        ChArUco marker length.
    aruco_size_cm : float
        Physical size of reference marker (cm).
    ref_marker_size_px : int
        Reference marker size (pixels).
    debug : bool
        If True, print some debug info.

    Returns
    -------
    top_view_rgb : np.ndarray
        Top-view RGB image (from first view).
    depth_map : np.ndarray
        Depth map (float32) aligned with top_view_rgb.
    """
    # 1) Rough intrinsic calibration from the 3 images
    K, dist = calibrate_intrinsic_charuco(
        image_list=imgs,
        square_length=square_length,
        marker_length=marker_length,
    )

    # 2) Get extrinsic for each view
    extrinsics: List[Tuple[np.ndarray, np.ndarray]] = []
    for H in Hs:
        R, t = extrinsic_from_homography(H, K)
        extrinsics.append((R, t))

    # 3) Choose reference view (index 0) and build top-view RGB using your perspective_img logic
    #    여기서는 이미 Hs[0]이 "plane -> image" 기준으로 들어왔다고 가정하고,
    #    _warp_view로 top-view를 만든다.
    top_view_rgb, H_shifted = _warp_view(imgs[0], Hs[0], full=True)

    pixel_scale = aruco_size_cm / float(ref_marker_size_px)
    H_top, W_top = top_view_rgb.shape[:2]

    # 4) Feature matching between view 0 and view 1, 2 (간단 ORB 예시)
    orb = cv2.ORB_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    points_3d_all = []

    for i in [1, 2]:
        img1 = imgs[0]
        img2 = imgs[i]

        kps1, des1 = orb.detectAndCompute(img1, None)
        kps2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:200]

        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        R1, t1 = extrinsics[0]
        R2, t2 = extrinsics[i]

        pts_3d = triangulate_two_views(K, R1, t1, R2, t2, pts1, pts2)
        points_3d_all.append(pts_3d)

    if not points_3d_all:
        raise RuntimeError("Failed to obtain any 3D points from matched features.")

    points_3d_all = np.vstack(points_3d_all)  # (N, 3)

    if debug:
        print(f"[depth] Triangulated {points_3d_all.shape[0]} 3D points.")

    # 5) Rasterize depth on top-view grid
    depth_map = rasterize_depth_on_top_view(
        points_3d_all,
        pixel_scale=pixel_scale,
        top_view_shape=top_view_rgb.shape[:2],
    )

    return top_view_rgb, depth_map


# -------------------------
# 6. Test main
# -------------------------
if __name__ == "__main__":
    # Example: you will need to load 3 images and their 3 homographies (Hs)
    # For now, placeholders
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

    # TODO: load or compute these homographies (3x3 each)
    # e.g., from your existing perspective_img / findHomography step
    Hs = [
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
    ]

    top_view_rgb, depth_map = get_depth_image_from_three_views(
        imgs=imgs,
        Hs=Hs,
        square_length=200.0,
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
