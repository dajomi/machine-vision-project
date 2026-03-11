import cv2
import numpy as np
import torch
from PIL import Image

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation


MODEL_NAME = "facebook/mask2former-swin-small-coco-instance"


def get_mask2former_instance_results(image_path, score_threshold=0.5, device=None):
    """
    입력:
        image_path: 이미지 경로
        score_threshold: score threshold
        device: "cuda" or "cpu"

    출력:
        image_bgr: 원본 이미지 (OpenCV BGR)
        object_list: 객체별 정보 리스트
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 / processor 로드
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 이미지 로드
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 전처리
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 원본 해상도 기준 후처리
    target_sizes = [(pil_image.height, pil_image.width)]
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=target_sizes,
        threshold=score_threshold,
        return_binary_maps=True
    )[0]

    # 결과가 없을 수도 있음
    if results is None or "segments_info" not in results:
        return image_bgr, []

    segmentation = results["segmentation"]          # (num_instances, H, W) binary maps
    segments_info = results["segments_info"]        # 객체 메타정보

    out = []
    for idx, seg_info in enumerate(segments_info):
        mask = segmentation[idx].detach().cpu().numpy() > 0

        class_id = int(seg_info["label_id"])
        label_name = model.config.id2label.get(class_id, str(class_id))

        out.append({
            "label": label_name,
            "confidence": float(seg_info["score"]),
            "mask": mask
        })

    return out


def visualize_instance_masks(image_bgr, object_list, alpha=0.45):
    """
    객체별 mask를 원본 이미지 위에 덮어쓰기
    """
    vis = image_bgr.copy()
    rng = np.random.default_rng(42)

    for obj in object_list:
        mask = obj["mask"]
        color = rng.integers(0, 255, size=3, dtype=np.uint8).tolist()

        color_layer = np.zeros_like(vis, dtype=np.uint8)
        color_layer[mask > 0] = color

        vis = np.where(
            color_layer > 0,
            (vis * (1 - alpha) + color_layer * alpha).astype(np.uint8),
            vis
        )

        # 객체 중심 근처에 id 표시
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            cv2.putText(
                vis,
                f'id={obj["object_id"]}, cls={obj["class_id"]}',
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

    return vis


if __name__ == "__main__":
    image, object_list = get_mask2former_instance_results(
        "./output/img001.jpg",
        score_threshold=0.5
    )

    print("검출 객체 수:", len(object_list))
    for obj in object_list:
        print(
            f'object_id={obj["object_id"]}, '
            f'class_id={obj["class_id"]}, '
            f'score={obj["confidence"]:.3f}, '
            f'mask_shape={obj["mask"].shape}'
        )

    vis = visualize_instance_masks(image, object_list)
    vis_small = cv2.resize(vis, None, fx=0.25, fy=0.25)

    cv2.imshow("mask2former instance segmentation", vis_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()