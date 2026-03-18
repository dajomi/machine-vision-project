"""
03_export_ssd_mobilenet_v2_onnx_post.py
후처리 포함 ONNX 내보내기

사용법:
    python 03_export_ssd_mobilenet_v2_onnx_post.py \
        --checkpoint checkpoints_ssd_mbv2/best.pth \
        --output     ssd_mobilenetv2_320_post.onnx
"""

# ============================================================
# 표준 라이브러리
# ============================================================
import argparse
from pathlib import Path

# ============================================================
# 서드파티 라이브러리
# ============================================================
import torch

# ============================================================
# 내부 모듈
# ============================================================
from ssd_mobilenet_v2_common import load_checkpoint_model
from ssd_mobilenet_v2_common_post import SSDPostprocessExportWrapper


# ============================================================
# 인수 파싱
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SSD-MobileNetV2 후처리 포함 ONNX 내보내기")
    p.add_argument("--checkpoint",        type=str,   required=True, help="best.pth 경로")
    p.add_argument("--output",            type=str,   default="ssd_mobilenetv2_320_post.onnx")
    p.add_argument("--opset",             type=int,   default=13)
    p.add_argument("--score-thresh",      type=float, default=None,  help="덮어쓸 score threshold")
    p.add_argument("--nms-thresh",        type=float, default=None,  help="덮어쓸 NMS IoU threshold")
    p.add_argument("--detections-per-img", type=int,  default=None,  help="최대 검출 수")
    p.add_argument("--topk-candidates",   type=int,   default=None,  help="NMS 전 top-k 후보 수")
    return p.parse_args()


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()

    # ── 모델 로드 ─────────────────────────────────────────────
    model, classes, img_size, _ = load_checkpoint_model(args.checkpoint, device="cpu")

    wrapper = SSDPostprocessExportWrapper(
        model,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        detections_per_img=args.detections_per_img,
        topk_candidates=args.topk_candidates,
    ).eval()

    # ── ONNX 내보내기 ─────────────────────────────────────────
    dummy    = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["num_detections", "boxes", "scores", "labels"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # ── 내보내기 요약 출력 ────────────────────────────────────
    max_det = wrapper.detections_per_img
    print(f"\nONNX 내보내기 완료: {out_path}")
    print(f"  Classes    : {classes}")
    print(f"  Input size : {img_size}×{img_size}")
    print(f"  Opset      : {args.opset}")
    print("\nOutput tensors:")
    print(f"  num_detections : [1]")
    print(f"  boxes          : [{max_det}, 4]   (xyxy, padding = -1)")
    print(f"  scores         : [{max_det}]      (padding = 0)")
    print(f"  labels         : [{max_det}]      (padding = -1)")
    print("\nNotes:")
    print("  - 배치 크기는 1로 고정됩니다.")
    print("  - 후처리 하이퍼파라미터는 ONNX 그래프에 bake-in됩니다.")


if __name__ == "__main__":
    main()
