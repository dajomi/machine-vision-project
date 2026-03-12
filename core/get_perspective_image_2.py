# core/get_perspective_image_2.py

import cv2
import numpy as np


# =========================
# 외부에서 재사용 가능한 static 유틸
# =========================
class SegmentEditor:
    def __init__(self, img, segments):
        self.img = img
        self.segments = segments
        self.active = np.ones(len(segments), dtype=bool)
        self.window_name = "segment_editor"

        # 드래그 박스 상태
        self.dragging = False
        self.box_start = None   # (x0, y0)
        self.box_end = None     # (x1, y1)

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

        while True:
            vis = self._draw()
            cv2.imshow(self.window_name, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q') or k == 27:  # q or ESC: 취소
                self.active[:] = True
                break
            if k == ord('s'):  # s: 선택 확정
                break

        cv2.destroyWindow(self.window_name)
        filtered = [s for s, a in zip(self.segments, self.active) if a]
        return filtered

    def _draw(self):
        out = self.img.copy()

        # 선분 그리기
        for (idx, seg) in enumerate(self.segments):
            if not self.active[idx]:
                # 비활성화된 선분은 연하게 표시하거나 안 그려도 됨
                color = (0, 0, 255)  # 빨간색
                thickness = 1
            else:
                color = (0, 255, 0)  # 초록색
                thickness = 2

            (p1, p2) = seg
            cv2.line(out,
                     (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])),
                     color, thickness, cv2.LINE_AA)

        # 드래그 중인 박스 시각화
        if self.dragging and self.box_start is not None and self.box_end is not None:
            x0, y0 = self.box_start
            x1, y1 = self.box_end
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 1)

        return out

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 드래그 시작
            self.dragging = True
            self.box_start = (x, y)
            self.box_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                # 드래그 중: 박스 끝점 업데이트
                self.box_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # 드래그 종료: 박스 영역 내 선분 비활성화
            if self.dragging and self.box_start is not None:
                self.box_end = (x, y)
                self._deactivate_segments_in_box(self.box_start, self.box_end)

            self.dragging = False
            self.box_start = None
            self.box_end = None

    def _deactivate_segments_in_box(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1

        # 좌상단/우하단으로 정규화
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        for i, (p1, p2) in enumerate(self.segments):
            if not self.active[i]:
                continue

            # 중점 기반으로 단순 체크 (필요하면 양 끝점 둘 다 체크로 변경)
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])

            if (x_min <= mx <= x_max) and (y_min <= my <= y_max):
                self.active[i] = False

def line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    l = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=float)
    n = np.linalg.norm(l[:2]) + 1e-12
    return l / n


def segment_length(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def normalize_points_2d(pts):
    """
    Hartley 스타일 2D 포인트 정규화: 중심 0, 평균 거리 sqrt(2).
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    d = np.linalg.norm(pts - mean, axis=1).mean() + 1e-12
    s = np.sqrt(2.0) / d

    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1.0]], dtype=float)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=float)])
    pts_n = (T @ pts_h.T).T
    return pts_n[:, :2], T


def warp_view(
    src_img: np.ndarray,
    H: np.ndarray,
    full: bool = True,
    border_value: tuple[int, int, int] = (255, 0, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """
    외부에서도 쓸 수 있는 warp 함수 (static).
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

    # 출력 크기 클램프 (원본의 3배 이내, 최소 64x64)
    max_scale = 3.0
    min_size = 64
    h0, w0 = src_img.shape[:2]
    max_w = int(w0 * max_scale)
    max_h = int(h0 * max_scale)

    out_w = max(min(out_w, max_w), min_size)
    out_h = max(min(out_h, max_h), min_size)

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


def detect_line_segments_lsd(gray_img, min_length=30):
    """
    LSD로 선분 검출. OpenCV 4.x에서 detect 반환이 튜플/단일 모두 대응.
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    res = lsd.detect(gray_img)
    if res is None:
        return []

    if isinstance(res, tuple):
        lines = res[0]
    else:
        lines = res

    segments = []
    if lines is None:
        return segments

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if np.hypot(x2 - x1, y2 - y1) >= min_length:
            segments.append(((float(x1), float(y1)),
                             (float(x2), float(y2))))
    return segments


# =========================
# 내부용: VP / Rectification 유틸
# =========================

def _vp_hypothesis_from_two_segments(seg1, seg2):
    p1, p2 = seg1
    q1, q2 = seg2
    l1 = line_from_points(p1, p2)
    l2 = line_from_points(q1, q2)
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-9:
        return None
    return v / v[2]


def _segment_direction(seg):
    p1, p2 = seg
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    n = np.linalg.norm(v) + 1e-12
    return v / n


def _vp_inlier_score(vp, segments, ang_thr_deg=5.0):
    if vp is None:
        return [], 0.0

    vp_xy = np.array([vp[0], vp[1]], dtype=float)
    cos_thr = np.cos(np.deg2rad(ang_thr_deg))

    inliers = []
    score = 0.0

    for seg in segments:
        p1, p2 = seg
        mid = np.array([(p1[0] + p2[0]) * 0.5,
                        (p1[1] + p2[1]) * 0.5], dtype=float)

        dir_seg = _segment_direction(seg)
        dir_vp = vp_xy - mid
        n = np.linalg.norm(dir_vp) + 1e-12
        dir_vp /= n

        cosang = abs(np.dot(dir_seg, dir_vp))
        if cosang >= cos_thr:
            inliers.append(seg)
            score += segment_length(p1, p2)

    return inliers, score


def _ransac_vanishing_point(segments, num_iter=2000, ang_thr_deg=5.0):
    if len(segments) < 2:
        raise RuntimeError("Need at least 2 segments for VP RANSAC.")

    best_vp = None
    best_score = -1.0
    best_inliers = []

    seg_arr = segments

    for _ in range(num_iter):
        idx = np.random.choice(len(seg_arr), size=2, replace=False)
        seg1 = seg_arr[idx[0]]
        seg2 = seg_arr[idx[1]]

        vp = _vp_hypothesis_from_two_segments(seg1, seg2)
        if vp is None:
            continue

        inliers, score = _vp_inlier_score(vp, seg_arr, ang_thr_deg=ang_thr_deg)
        if score > best_score:
            best_score = score
            best_vp = vp
            best_inliers = inliers

    if best_vp is None:
        raise RuntimeError("Failed to estimate vanishing point by RANSAC.")

    # refine
    if len(best_inliers) >= 2:
        v_list = []
        w_list = []
        for i in range(len(best_inliers) - 1):
            seg1 = best_inliers[i]
            seg2 = best_inliers[i + 1]
            vp2 = _vp_hypothesis_from_two_segments(seg1, seg2)
            if vp2 is None:
                continue
            w = segment_length(*seg1) + segment_length(*seg2)
            v_list.append(vp2)
            w_list.append(w)
        if v_list:
            v_arr = np.stack(v_list, axis=0)
            w_arr = np.array(w_list).reshape(-1, 1)
            vp_ref = (v_arr * w_arr).sum(axis=0) / w_arr.sum()
            best_vp = vp_ref / vp_ref[2]

    return best_vp, best_inliers


def _detect_multiple_vps(segments, num_vps=2, num_iter=2000,
                         ang_thr_deg=5.0, min_inlier_ratio=0.1):
    remaining = segments.copy()
    vp_list = []
    group_list = []

    total_len = sum(segment_length(*s) for s in segments) + 1e-12

    for _ in range(num_vps):
        if len(remaining) < 2:
            break

        vp, inliers = _ransac_vanishing_point(
            remaining, num_iter=num_iter, ang_thr_deg=ang_thr_deg
        )
        if not inliers:
            break

        inlier_len = sum(segment_length(*s) for s in inliers)
        if inlier_len / total_len < min_inlier_ratio:
            break

        vp_list.append(vp)
        group_list.append(inliers)
        remaining = [s for s in remaining if s not in inliers]

    return vp_list, group_list


def _make_orthogonal_pairs_from_vp_groups(vp_list, group_list,
                                          max_pairs_per_group=5):
    if len(vp_list) < 2:
        return []

    dirs = []
    for vp in vp_list:
        d = np.array([vp[0], vp[1]], dtype=float)
        n = np.linalg.norm(d) + 1e-12
        dirs.append(d / n)
    dirs = np.stack(dirs, axis=0)

    best_pair = None
    best_orth = 0.0
    for i in range(len(vp_list)):
        for j in range(i + 1, len(vp_list)):
            cosang = abs(np.dot(dirs[i], dirs[j]))
            orth_score = 1.0 - cosang
            if orth_score > best_orth:
                best_orth = orth_score
                best_pair = (i, j)

    if best_pair is None:
        return []

    i, j = best_pair
    group1 = sorted(group_list[i], key=lambda s: -segment_length(*s))
    group2 = sorted(group_list[j], key=lambda s: -segment_length(*s))

    group1 = group1[:max_pairs_per_group]
    group2 = group2[:max_pairs_per_group]

    orthogonal_pairs = []
    for s1 in group1:
        for s2 in group2:
            orthogonal_pairs.append((s1, s2))

    return orthogonal_pairs


def _affine_rectification_from_two_vps(v1, v2, img_shape):
    """
    v1, v2: 이미지 좌표계에서의 VP (homogeneous)
    img_shape: src_img.shape
    정규화 좌표계에서 line at infinity를 만들고, 다시 되돌린다.
    """
    h, w = img_shape[:2]

    # 이미지 코너를 사용해 정규화 행렬 T 계산
    corners = np.array([[0, 0],
                        [w, 0],
                        [w, h],
                        [0, h]], dtype=float)
    _, T = normalize_points_2d(corners)

    def norm_vp(v):
        v = v / (v[2] + 1e-12)
        hv = np.array([v[0], v[1], 1.0], dtype=float)
        vn = T @ hv
        return vn / (vn[2] + 1e-12)

    v1n = norm_vp(v1)
    v2n = norm_vp(v2)

    # 정규화 좌표계에서 line at infinity
    l_inf_n = np.cross(v1n, v2n)
    l_inf_n = l_inf_n / (np.linalg.norm(l_inf_n[:2]) + 1e-12)
    l1, l2, l3 = l_inf_n

    H_a_n = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [l1,  l2,  l3 + 1.0]], dtype=float)

    T_inv = np.linalg.inv(T)
    H_a = T_inv @ H_a_n @ T
    return H_a


def _metric_rectification_from_orthogonal_pairs(orthogonal_pairs, H_affine,
                                                alpha=1.0):
    """
    STEP1/2/3 공통 metric 단계.
    STEP1: 이 함수를 호출하되 평가에는 쓰지 않음.
    STEP2: 이 함수 + orthogonal term 평가에 weight 작게.
    STEP3: max_pairs_per_group/alpha/weight를 점진적으로 키움.
    """
    A_rows = []

    for seg1, seg2 in orthogonal_pairs:
        def to_affine_point(p):
            x, y = p
            hp = np.array([x, y, 1.0], dtype=float)
            ap = H_affine @ hp
            return ap[:2] / ap[2]

        a1 = to_affine_point(seg1[0])
        a2 = to_affine_point(seg1[1])
        b1 = to_affine_point(seg2[0])
        b2 = to_affine_point(seg2[1])

        l = line_from_points(a1, a2)
        m = line_from_points(b1, b2)

        w = (segment_length(a1, a2) * segment_length(b1, b2)) ** alpha

        l1, l2, _ = l
        m1, m2, _ = m

        A_rows.append(
            w * np.array(
                [l1 * m1,
                 l1 * m2 + l2 * m1,
                 l2 * m2],
                dtype=float,
            )
        )

    if len(A_rows) < 2:
        raise RuntimeError("Need at least 2 orthogonal pairs.")

    A = np.stack(A_rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    s11, s12, s22 = Vt[-1, :]

    S_mat = np.array([[s11, s12],
                      [s12, s22]], dtype=float)

    eigvals, eigvecs = np.linalg.eigh(S_mat)
    # STEP1/2/3: eigenvalue regularization 강화
    lam_min = 1e-4
    eigvals = np.clip(eigvals, lam_min, None)
    K = np.diag(np.sqrt(eigvals)) @ eigvecs.T

    Hm = np.eye(3, dtype=float)
    Hm[0:2, 0:2] = K

    return Hm


def _estimate_vp_from_group(segments):
    if len(segments) < 2:
        raise RuntimeError("Need >=2 segments in group.")

    v_list = []
    w_list = []
    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i + 1]
        vp = _vp_hypothesis_from_two_segments(seg1, seg2)
        if vp is None:
            continue
        w = segment_length(*seg1) + segment_length(*seg2)
        v_list.append(vp)
        w_list.append(w)
    if not v_list:
        raise RuntimeError("Failed to estimate VP from group.")
    v_arr = np.stack(v_list, axis=0)
    w_arr = np.array(w_list).reshape(-1, 1)
    vp = (v_arr * w_arr).sum(axis=0) / w_arr.sum()
    return vp / vp[2]


def evaluate_homography_energy(H, segments, parallel_groups, orthogonal_pairs,
                               ang_thr_deg_parallel=5.0,
                               ang_thr_deg_ortho=5.0,
                               lambda_ortho=0.0):
    """
    STEP0  : lambda_ortho=0.0 (parallel only)
    STEP1  : H_metric은 쓰지만, 여기서는 lambda_ortho=0.0 유지
    STEP2/3: lambda_ortho > 0 으로 직각 term 점진적 반영
    """
    cos_thr_par = np.cos(np.deg2rad(ang_thr_deg_parallel))
    sin_thr_ortho = np.sin(np.deg2rad(ang_thr_deg_ortho))

    def warp_point(p):
        x, y = p
        hp = np.array([x, y, 1.0], dtype=float)
        wp = H @ hp
        return wp[:2] / wp[2]

    # parallel term
    score_par = 0.0
    for group in parallel_groups:
        if len(group) < 2:
            continue
        dirs = []
        lens = []
        for seg in group:
            p1, p2 = seg
            wp1 = warp_point(p1)
            wp2 = warp_point(p2)
            d = wp2 - wp1
            n = np.linalg.norm(d) + 1e-12
            d /= n
            dirs.append(d)
            lens.append(np.linalg.norm(wp2 - wp1))
        dirs = np.stack(dirs, axis=0)
        lens = np.array(lens)

        ref = dirs[0]
        cosang = np.abs(dirs @ ref)
        mask = cosang >= cos_thr_par
        score_par += (lens[mask] * cosang[mask]).sum()

    # orthogonal term
    score_ortho = 0.0
    if lambda_ortho > 0.0:
        for seg1, seg2 in orthogonal_pairs:
            p1, p2 = seg1
            q1, q2 = seg2
            wp1 = warp_point(p1)
            wp2 = warp_point(p2)
            wq1 = warp_point(q1)
            wq2 = warp_point(q2)
            d1 = wp2 - wp1
            d2 = wq2 - wq1
            n1 = np.linalg.norm(d1) + 1e-12
            n2 = np.linalg.norm(d2) + 1e-12
            d1 /= n1
            d2 /= n2
            cosang = np.dot(d1, d2)
            sinang = np.sqrt(max(0.0, 1.0 - cosang * cosang))

            if sinang >= sin_thr_ortho:  # 충분히 직각에 가까울 때만
                score_ortho += (segment_length(p1, p2) *
                                segment_length(q1, q2)) * sinang

    return score_par + lambda_ortho * score_ortho


# =========================
# 메인 클래스
# =========================

class LineBasedRectifier:
    def __init__(self,
                 num_vps=2,
                 min_seg_length=30,
                 vp_ransac_iter_list=(800, 1600),
                 vp_ang_thr_list=(4.0, 6.0),
                 alpha_list=(0.8, 1.0, 1.3),
                 eval_ang_thr_parallel=5.0,
                 eval_ang_thr_ortho=7.0,
                 step="step2"):
        """
        step:
          "step0": affine only, metric off, orthogonal off
          "step1": affine + metric(H_metric 적용), 평가에는 orthogonal off
          "step2": affine + metric, 평가에서 orthogonal term 약하게
          "step3": affine + metric, orthogonal term 더 강하게/쌍 더 많이
        """
        self.num_vps = num_vps
        self.min_seg_length = min_seg_length
        self.vp_ransac_iter_list = vp_ransac_iter_list
        self.vp_ang_thr_list = vp_ang_thr_list
        self.alpha_list = alpha_list
        self.eval_ang_thr_parallel = eval_ang_thr_parallel
        self.eval_ang_thr_ortho = eval_ang_thr_ortho
        self.step = step

    # ----- public API -----
    def rectify(self, src_img, segments=None):
        """
        input: src_img (BGR or gray)
               segments: 사용자가 편집한 선분 리스트(optional)
        output: rectified_img, H, meta
        """
        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if src_img.ndim == 3 else src_img

        if segments is None:
            segments = self._detect_segments(gray)
        if len(segments) < 10:
            raise RuntimeError("Not enough line segments detected.")

        candidates = self._generate_candidate_homographies(src_img, segments)
        if not candidates:
            raise RuntimeError("No candidate homographies generated.")

        best_score = -1.0
        best_H = None
        best_meta = None

        for cand in candidates:
            H = cand["H"]
            score = self._evaluate_energy(
                H,
                segments,
                cand["parallel_groups"],
                cand["orthogonal_pairs"],
            )
            if score > best_score:
                best_score = score
                best_H = H
                best_meta = cand

        rectified_img, _ = warp_view(src_img, best_H, full=True,
                                     border_value=(255, 255, 255))
        return rectified_img, best_H, best_meta

    # ----- internal methods -----

    def _detect_segments(self, gray_img):
        return detect_line_segments_lsd(gray_img, min_length=self.min_seg_length)

    def _generate_candidate_homographies(self, src_img, segments):
        candidates = []

        # step별 orthogonal pair 개수 설정
        if self.step in ("step0", "step1"):
            max_pairs_per_group = 1
        elif self.step == "step2":
            max_pairs_per_group = 1
        else:  # step3
            max_pairs_per_group = 2

        for vp_iters in self.vp_ransac_iter_list:
            for vp_ang_thr in self.vp_ang_thr_list:
                vp_list, group_list = _detect_multiple_vps(
                    segments,
                    num_vps=self.num_vps,
                    num_iter=vp_iters,
                    ang_thr_deg=vp_ang_thr,
                )
                if len(vp_list) < 2:
                    continue

                parallel_groups = group_list[:2]
                orthogonal_pairs = _make_orthogonal_pairs_from_vp_groups(
                    vp_list, group_list, max_pairs_per_group=max_pairs_per_group
                )

                for alpha in self.alpha_list:
                    try:
                        v1 = _estimate_vp_from_group(parallel_groups[0])
                        v2 = _estimate_vp_from_group(parallel_groups[1])
                        H_affine = _affine_rectification_from_two_vps(
                            v1, v2, src_img.shape
                        )

                        if self.step == "step0":
                            H_total = H_affine
                        else:
                            if len(orthogonal_pairs) < 1:
                                H_total = H_affine
                            else:
                                try:
                                    H_metric = _metric_rectification_from_orthogonal_pairs(
                                        orthogonal_pairs, H_affine, alpha=alpha
                                    )
                                    H_total = H_metric @ H_affine
                                except Exception:
                                    H_total = H_affine  # metric 실패 시 fallback

                        candidates.append(
                            {
                                "H": H_total,
                                "parallel_groups": parallel_groups,
                                "orthogonal_pairs": orthogonal_pairs,
                                "vp_iters": vp_iters,
                                "vp_ang_thr": vp_ang_thr,
                                "alpha": alpha,
                            }
                        )
                    except Exception:
                        # VP 추정/affine 자체가 터질 때만 스킵
                        continue

        return candidates

    def _evaluate_energy(self, H, segments, parallel_groups, orthogonal_pairs):
        # step별 orthogonal weight 설정
        if self.step in ("step0", "step1"):
            lambda_ortho = 0.0      # STEP0/1: 평가에서는 직각 무시
        elif self.step == "step2":
            lambda_ortho = 0.3      # STEP2: 약하게 반영
        else:
            lambda_ortho = 0.7      # STEP3: 더 강하게 반영

        return evaluate_homography_energy(
            H,
            segments,
            parallel_groups,
            orthogonal_pairs,
            ang_thr_deg_parallel=self.eval_ang_thr_parallel,
            ang_thr_deg_ortho=self.eval_ang_thr_ortho,
            lambda_ortho=lambda_ortho,
        )


# =========================
# 테스트 / 디버그 코드
# =========================

def draw_segments(img, segments, color, thickness=2):
    out = img.copy()
    for (p1, p2) in segments:
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out


def warp_point(H, p):
    x, y = p
    hp = np.array([x, y, 1.0], dtype=float)
    wp = H @ hp
    return wp[:2] / wp[2]


def warp_segments(H, segments):
    warped = []
    for (p1, p2) in segments:
        wp1 = warp_point(H, p1)
        wp2 = warp_point(H, p2)
        warped.append((wp1, wp2))
    return warped

if __name__ == "__main__":
    SRC_IMG_PATH = "data\\set_2\\KakaoTalk_20260311_112717712_05.jpg"
    img = cv2.imread(SRC_IMG_PATH)
    img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

    rectifier = LineBasedRectifier(
        num_vps=2,
        min_seg_length=30,
        step="step1"
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segments_raw = rectifier._detect_segments(gray)

    editor = SegmentEditor(img, segments_raw)
    segments_edited = editor.run()  # 사용자가 의자 다리 등 끄기

    rectified, H, meta = rectifier.rectify(img, segments=segments_edited)

    print("Best H:\n", H)
    print("Meta keys:", meta.keys())

    parallel_groups = meta["parallel_groups"]
    orthogonal_pairs = meta["orthogonal_pairs"]

    # 1) 원본 이미지에 parallel_groups 색깔별로 그리기
    color_list = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    drawn_parallel_src = img.copy()
    for gi, group in enumerate(parallel_groups):
        color = color_list[gi % len(color_list)]
        drawn_parallel_src = draw_segments(drawn_parallel_src, group, color, 2)

    # 2) rectified 이미지에 parallel_groups warp해서 그리기
    drawn_parallel_rect = rectified.copy()
    for gi, group in enumerate(parallel_groups):
        color = color_list[gi % len(color_list)]
        warped_group = warp_segments(H, group)
        drawn_parallel_rect = draw_segments(drawn_parallel_rect, warped_group, color, 2)

    # 3) 직각 쌍은 다른 스타일(굵기/색)로 overlay
    drawn_ortho_src = drawn_parallel_src.copy()
    drawn_ortho_rect = drawn_parallel_rect.copy()
    for (seg1, seg2) in orthogonal_pairs:
        color = (0, 255, 255)
        drawn_ortho_src = draw_segments(drawn_ortho_src, [seg1, seg2], color, 3)

        warped_pair = warp_segments(H, [seg1, seg2])
        drawn_ortho_rect = draw_segments(drawn_ortho_rect, warped_pair, color, 3)

    # 4) 표시
    cv2.imshow("distorted", img)
    cv2.imshow("parallel_src", drawn_parallel_src)
    cv2.imshow("parallel+ortho_src", drawn_ortho_src)
    cv2.imshow("rectified", rectified)
    cv2.imshow("parallel_rectified", drawn_parallel_rect)
    cv2.imshow("parallel+ortho_rectified", drawn_ortho_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()