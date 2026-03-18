import argparse
from pathlib import Path

import torch

from ssd_mobilenet_v2_common import SSDRawExportWrapper, load_checkpoint_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="best.pth path")
    p.add_argument("--output", type=str, default="ssd_mobilenetv2_320_raw.onnx")
    p.add_argument("--opset", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    model, classes, img_size, _ = load_checkpoint_model(args.checkpoint, device="cpu")
    wrapper = SSDRawExportWrapper(model).eval()

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["class_logits", "bbox_regression", "anchors"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "class_logits": {0: "batch"},
            "bbox_regression": {0: "batch"},
        },
    )

    print(f"ONNX exported: {out_path}")
    print(f"Classes: {classes}")
    print(f"Input size: {img_size}x{img_size}")
    print("Output tensors:")
    print("  class_logits   : [B, num_anchors, num_classes]")
    print("  bbox_regression: [B, num_anchors, 4]")
    print("  anchors        : [num_anchors, 4]")


if __name__ == "__main__":
    main()
