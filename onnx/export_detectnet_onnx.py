from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import DetectNetONNXExportWrapper, load_checkpoint_model


def parse_args():
    p = argparse.ArgumentParser(description="Export detectNet-compatible ONNX from best.pth")
    p.add_argument("--checkpoint", type=str, required=True, help="path to best.pth or latest.pth")
    p.add_argument("--output", type=str, default="detectNet_01.onnx")
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--input-name", type=str, default="input_0")
    p.add_argument("--scores-name", type=str, default="scores")
    p.add_argument("--boxes-name", type=str, default="boxes")
    p.add_argument("--no-softmax", action="store_true", help="export logits instead of probabilities")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, classes, img_size, ckpt = load_checkpoint_model(args.checkpoint, device=str(device))
    wrapper = DetectNetONNXExportWrapper(model, apply_softmax=not args.no_softmax).to(device).eval()

    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32, device=device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        scores, boxes = wrapper(dummy)
        print(f"checkpoint   : {args.checkpoint}")
        print(f"classes      : {classes}")
        print(f"img_size     : {img_size}")
        print(f"input name   : {args.input_name}")
        print(f"outputs      : {args.scores_name}, {args.boxes_name}")
        print(f"scores shape : {tuple(scores.shape)}")
        print(f"boxes shape  : {tuple(boxes.shape)}")
        print("note         : boxes are normalized xyxy, no NMS embedded")

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[args.input_name],
        output_names=[args.scores_name, args.boxes_name],
        dynamic_axes=None,
    )

    print(f"saved        : {out_path}")
    print("detectNet load example:")
    print(
        f"detectNet(model='{out_path.name}', labels='labels.txt', "
        f"input_blob='{args.input_name}', output_cvg='{args.scores_name}', "
        f"output_bbox='{args.boxes_name}', threshold=0.5)"
    )


if __name__ == "__main__":
    main()
