"""
ssd_mobilenet_v2_common_post.py
후처리 포함 ONNX 내보내기 래퍼
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms


class SSDPostprocessExportWrapper(nn.Module):
    """
    후처리(softmax → decode → NMS → padding)까지 포함한 ONNX 내보내기 래퍼

    Assumptions
    -----------
    - 입력은 model img_size로 리사이즈 + 정규화 완료 상태
    - 배치 크기 == 1 (Jetson 단일 이미지 추론 권장)

    Outputs
    -------
    num_detections : [1]               실제 검출 수
    boxes          : [max_det, 4]      xyxy 좌표, 미사용 슬롯 = -1 padding
    scores         : [max_det]         신뢰도 점수, 미사용 슬롯 = 0 padding
    labels         : [max_det]         클래스 인덱스, 미사용 슬롯 = -1 padding
    """

    def __init__(
        self,
        model,
        score_thresh: float | None = None,
        nms_thresh: float | None = None,
        detections_per_img: int | None = None,
        topk_candidates: int | None = None,
    ):
        super().__init__()
        self.model              = model
        self.score_thresh       = float(score_thresh       if score_thresh       is not None else model.score_thresh)
        self.nms_thresh         = float(nms_thresh         if nms_thresh         is not None else model.nms_thresh)
        self.detections_per_img = int(detections_per_img   if detections_per_img is not None else model.detections_per_img)
        self.topk_candidates    = int(topk_candidates      if topk_candidates    is not None else model.topk_candidates)

        # 상수를 버퍼로 등록 → ONNX 그래프에 임베드됨
        self.register_buffer("_score_thresh", torch.tensor(self.score_thresh, dtype=torch.float32))
        self.register_buffer("_pad_box",      torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32))
        self.register_buffer("_pad_score",    torch.tensor(0.0,  dtype=torch.float32))
        self.register_buffer("_pad_label",    torch.tensor(-1,   dtype=torch.int64))

    # ----------------------------------------------------------
    # 내부 헬퍼
    # ----------------------------------------------------------

    def _decode_boxes(self, rel_codes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """박스 회귀 델타 → 절대 xyxy 좌표 변환"""
        boxes = anchors.to(rel_codes.dtype)

        widths  = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x   = boxes[:, 0] + 0.5 * widths
        ctr_y   = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.model.box_coder.weights
        dx = rel_codes[:, 0] / wx
        dy = rel_codes[:, 1] / wy
        dw = torch.clamp(rel_codes[:, 2] / ww, max=self.model.box_coder.bbox_xform_clip)
        dh = torch.clamp(rel_codes[:, 3] / wh, max=self.model.box_coder.bbox_xform_clip)

        pred_ctr_x = dx * widths  + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        x1 = pred_ctr_x - pred_w * 0.5
        y1 = pred_ctr_y - pred_h * 0.5
        x2 = pred_ctr_x + pred_w * 0.5
        y2 = pred_ctr_y + pred_h * 0.5

        return torch.stack((x1, y1, x2, y2), dim=1)

    def _clip_boxes(self, boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """박스 좌표를 이미지 경계 [0, W] × [0, H] 내로 클리핑"""
        boxes_x = boxes[:, [0, 2]].clamp(min=0.0, max=float(width))
        boxes_y = boxes[:, [1, 3]].clamp(min=0.0, max=float(height))
        return torch.stack(
            (boxes_x[:, 0], boxes_y[:, 0], boxes_x[:, 1], boxes_y[:, 1]),
            dim=1,
        )

    def _pad_to_fixed(
        self,
        boxes:  torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        검출 결과를 detections_per_img 크기로 패딩 (또는 truncate)

        Safety
        ------
        - 검출 0개(빈 텐서)도 정상 처리
        - pad 텐서를 항상 boxes.device 기준으로 생성 → device 불일치 방지
        """
        max_det = self.detections_per_img
        dev     = boxes.device
        n       = boxes.shape[0]

        if n > max_det:
            boxes  = boxes[:max_det]
            scores = scores[:max_det]
            labels = labels[:max_det]
            n      = max_det

        pad_n = max_det - n
        if pad_n > 0:
            pad_boxes  = self._pad_box.to(dev).unsqueeze(0).expand(pad_n, 4).clone().to(boxes.dtype)
            pad_scores = self._pad_score.to(dev).expand(pad_n).clone().to(scores.dtype)
            pad_labels = self._pad_label.to(dev).expand(pad_n).clone()
            boxes  = torch.cat([boxes,  pad_boxes],  dim=0)
            scores = torch.cat([scores, pad_scores], dim=0)
            labels = torch.cat([labels, pad_labels], dim=0)

        num_det = torch.tensor([n], dtype=torch.int64, device=dev)
        return boxes, scores, labels, num_det

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        if x.shape[0] != 1:
            raise ValueError("SSDPostprocessExportWrapper는 배치 크기 1만 지원합니다.")

        # ── 특징 추출 ────────────────────────────────────────
        features = self.model.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())

        # ── 헤드 출력 / 앵커 생성 ────────────────────────────
        head_outputs = self.model.head(feature_list)
        image_h, image_w = int(x.shape[-2]), int(x.shape[-1])
        anchors = self.model.anchor_generator(
            ImageList(x, [(image_h, image_w)]),
            feature_list,
        )[0]

        cls_logits = head_outputs["cls_logits"][0]    # [N, C]
        bbox_reg   = head_outputs["bbox_regression"][0]  # [N, 4]

        # ── 디코딩 ───────────────────────────────────────────
        probs = torch.softmax(cls_logits, dim=-1)
        boxes = self._decode_boxes(bbox_reg, anchors)
        boxes = self._clip_boxes(boxes, image_h, image_w)

        num_classes = int(probs.shape[1])
        num_anchors = int(probs.shape[0])
        k = min(self.topk_candidates, num_anchors)

        per_class_boxes, per_class_scores, per_class_labels = [], [], []

        # ── 클래스별 topk + NMS ──────────────────────────────
        for label in range(1, num_classes):
            cls_scores = probs[:, label]
            topk_scores, topk_idx = torch.topk(cls_scores, k=k, dim=0, largest=True, sorted=True)
            topk_boxes = boxes[topk_idx]

            keep = nms(topk_boxes, topk_scores, self.nms_thresh)
            kept_boxes  = topk_boxes[keep]
            kept_scores = topk_scores[keep]
            kept_labels = torch.full(
                (keep.shape[0],), label,
                dtype=torch.int64, device=kept_scores.device,
            )
            per_class_boxes.append(kept_boxes)
            per_class_scores.append(kept_scores)
            per_class_labels.append(kept_labels)

        all_boxes  = torch.cat(per_class_boxes,  dim=0)
        all_scores = torch.cat(per_class_scores, dim=0)
        all_labels = torch.cat(per_class_labels, dim=0)

        # ── score 필터링 ─────────────────────────────────────
        mask       = all_scores > self._score_thresh
        all_boxes  = all_boxes[mask]
        all_scores = all_scores[mask]
        all_labels = all_labels[mask]

        # ── 최종 topk 선택 ───────────────────────────────────
        final_k = min(self.detections_per_img, int(all_scores.shape[0]))
        if final_k > 0:
            final_scores, order = torch.topk(all_scores, k=final_k, dim=0, largest=True, sorted=True)
            final_boxes  = all_boxes[order]
            final_labels = all_labels[order]
        else:
            final_boxes  = all_boxes[:0]
            final_scores = all_scores[:0]
            final_labels = all_labels[:0]

        return self._pad_to_fixed(final_boxes, final_scores, final_labels)
