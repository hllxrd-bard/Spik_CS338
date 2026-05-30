
import torch

from evcivil_dataset import EVCivilDetectionDataset


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N, 4] in xyxy pixel format.
    return: [N, 4] in cxcywh pixel format.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))

    out = boxes.clone()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
    out[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
    out[:, 0] = boxes[:, 0] + out[:, 2] * 0.5  # cx
    out[:, 1] = boxes[:, 1] + out[:, 3] * 0.5  # cy
    return out


def detection_collate_yolox_fn(batch):
    """
    Return:
        images: [B, T, 2, H, W]
        targets: list[dict]
        yolox_labels: [B, max_boxes, 5]

    YOLOX label format:
        [class_id, cx, cy, w, h]
    in pixel coordinates after resize.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = list(targets)

    max_boxes = 0
    per_sample_labels = []

    for target in targets:
        boxes = target["boxes"]
        labels = target["labels"]

        if boxes.numel() == 0:
            y = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes_cxcywh = xyxy_to_cxcywh(boxes).float()
            cls = labels.float().view(-1, 1)
            y = torch.cat([cls, boxes_cxcywh], dim=1)

        per_sample_labels.append(y)
        max_boxes = max(max_boxes, y.shape[0])

    if max_boxes == 0:
        yolox_labels = torch.zeros((len(targets), 1, 5), dtype=torch.float32)
    else:
        yolox_labels = torch.zeros((len(targets), max_boxes, 5), dtype=torch.float32)
        for i, y in enumerate(per_sample_labels):
            n = y.shape[0]
            if n > 0:
                yolox_labels[i, :n] = y

    return images, targets, yolox_labels
