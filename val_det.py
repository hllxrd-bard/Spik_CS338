import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from spikingjelly.clock_driven import functional

from evcivil_dataset import EVCivilDetectionDataset, detection_collate_fn


# -----------------------------
# Logging
# -----------------------------
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Efficiency + SNN monitors
# -----------------------------
def sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_from_output(output):
    if torch.is_tensor(output):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors = []
        for x in output:
            tensors.extend(tensor_from_output(x))
        return tensors
    if isinstance(output, dict):
        tensors = []
        for x in output.values():
            tensors.extend(tensor_from_output(x))
        return tensors
    return []


class SpikeActivityMonitor:
    """
    Records firing-rate statistics from spiking neuron modules.

    This is enabled only for Spikformer modes. For ViT, these metrics are N/A.
    The firing-rate estimate assumes spike outputs are binary or near-binary.
    """
    def __init__(self, model, enabled=True, spike_thr=0.0):
        self.enabled = enabled
        self.spike_thr = spike_thr
        self.handles = []
        self.layers = {}

        if not enabled:
            return

        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            cls_lower = cls_name.lower()
            is_spike_module = (
                "lifnode" in cls_lower
                or "ifnode" in cls_lower
                or "plif" in cls_lower
                or "spiking" in cls_lower and "neuron" in cls_lower
            )

            if is_spike_module:
                self.layers[name] = {
                    "class": cls_name,
                    "spike_count": 0.0,
                    "neuron_events": 0,
                    "num_forwards": 0,
                }
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            tensors = tensor_from_output(output)
            for t in tensors:
                if not torch.is_tensor(t):
                    continue
                td = t.detach()
                spike_count = (td > self.spike_thr).float().sum().item()
                self.layers[name]["spike_count"] += float(spike_count)
                self.layers[name]["neuron_events"] += int(td.numel())
                self.layers[name]["num_forwards"] += 1
        return hook

    def summary(self, topk=20):
        if not self.enabled:
            return {
                "enabled": False,
                "avg_firing_rate": None,
                "total_spike_count": None,
                "total_neuron_events": None,
                "layers": {},
            }

        total_spikes = sum(v["spike_count"] for v in self.layers.values())
        total_events = sum(v["neuron_events"] for v in self.layers.values())
        avg_rate = total_spikes / max(total_events, 1)

        layer_items = []
        layer_dict = {}
        for name, v in self.layers.items():
            rate = v["spike_count"] / max(v["neuron_events"], 1)
            item = {
                "class": v["class"],
                "firing_rate": rate,
                "spike_count": v["spike_count"],
                "neuron_events": v["neuron_events"],
                "num_forwards": v["num_forwards"],
            }
            layer_dict[name] = item
            layer_items.append((name, item))

        layer_items.sort(key=lambda x: x[1]["spike_count"], reverse=True)
        top_layers = {name: item for name, item in layer_items[:topk]}

        return {
            "enabled": True,
            "avg_firing_rate": avg_rate,
            "total_spike_count": total_spikes,
            "total_neuron_events": total_events,
            "num_spike_layers": len(self.layers),
            "top_layers_by_spike_count": top_layers,
            "layers": layer_dict,
        }

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class SOPSEstimator:
    """
    Approximate SNN synaptic operations for backbone Conv/Linear layers.

    Estimate:
        SOPs ~= number of non-zero input spikes/features * fanout

    This is only a proxy. It is reported for Spikformer modes only and should not
    be directly compared as identical to ANN FLOPs.
    """
    def __init__(self, model, enabled=True, spike_thr=0.0, include_prefix="backbone"):
        self.enabled = enabled
        self.spike_thr = spike_thr
        self.include_prefix = include_prefix
        self.handles = []
        self.layers = {}

        if not enabled:
            return

        for name, module in model.named_modules():
            if include_prefix and not name.startswith(include_prefix):
                continue
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                self.layers[name] = {
                    "class": module.__class__.__name__,
                    "sops_estimate": 0.0,
                    "dense_macs_estimate": 0.0,
                    "input_nonzero": 0.0,
                    "input_numel": 0,
                    "num_forwards": 0,
                }
                self.handles.append(module.register_forward_hook(self._make_hook(name, module)))

    @staticmethod
    def _dense_macs(module, output):
        if not torch.is_tensor(output):
            return 0.0
        if isinstance(module, nn.Conv2d):
            cin = module.in_channels
            groups = module.groups
            kh, kw = module.kernel_size
            return float(output.numel() * (cin // groups) * kh * kw)
        if isinstance(module, nn.Conv1d):
            cin = module.in_channels
            groups = module.groups
            k = module.kernel_size[0]
            return float(output.numel() * (cin // groups) * k)
        if isinstance(module, nn.Linear):
            return float(output.numel() * module.in_features)
        return 0.0

    @staticmethod
    def _fanout(module):
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            return float((module.out_channels // module.groups) * kh * kw)
        if isinstance(module, nn.Conv1d):
            k = module.kernel_size[0]
            return float((module.out_channels // module.groups) * k)
        if isinstance(module, nn.Linear):
            return float(module.out_features)
        return 0.0

    def _make_hook(self, name, module):
        def hook(module, inputs, output):
            if len(inputs) == 0 or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0].detach()
            input_nonzero = (x.abs() > self.spike_thr).float().sum().item()
            input_numel = int(x.numel())
            fanout = self._fanout(module)
            sops = float(input_nonzero * fanout)
            dense = self._dense_macs(module, output)

            self.layers[name]["sops_estimate"] += sops
            self.layers[name]["dense_macs_estimate"] += dense
            self.layers[name]["input_nonzero"] += float(input_nonzero)
            self.layers[name]["input_numel"] += input_numel
            self.layers[name]["num_forwards"] += 1
        return hook

    def summary(self, n_images=1, topk=20):
        if not self.enabled:
            return {
                "enabled": False,
                "sops_estimate_total": None,
                "sops_estimate_per_image": None,
                "dense_macs_backbone_estimate_total": None,
                "dense_macs_backbone_estimate_per_image": None,
                "sops_to_dense_ratio": None,
                "layers": {},
            }

        total_sops = sum(v["sops_estimate"] for v in self.layers.values())
        total_dense = sum(v["dense_macs_estimate"] for v in self.layers.values())
        ratio = total_sops / max(total_dense, 1.0)

        layer_items = []
        layer_dict = {}
        for name, v in self.layers.items():
            input_rate = v["input_nonzero"] / max(v["input_numel"], 1)
            item = {
                "class": v["class"],
                "sops_estimate": v["sops_estimate"],
                "dense_macs_estimate": v["dense_macs_estimate"],
                "sops_to_dense_ratio": v["sops_estimate"] / max(v["dense_macs_estimate"], 1.0),
                "input_activity_rate": input_rate,
                "input_nonzero": v["input_nonzero"],
                "input_numel": v["input_numel"],
                "num_forwards": v["num_forwards"],
            }
            layer_dict[name] = item
            layer_items.append((name, item))

        layer_items.sort(key=lambda x: x[1]["sops_estimate"], reverse=True)
        top_layers = {name: item for name, item in layer_items[:topk]}

        return {
            "enabled": True,
            "note": "Approximate backbone SOPs: non-zero input activations multiplied by fanout. Not directly equivalent to ANN FLOPs.",
            "sops_estimate_total": total_sops,
            "sops_estimate_per_image": total_sops / max(n_images, 1),
            "dense_macs_backbone_estimate_total": total_dense,
            "dense_macs_backbone_estimate_per_image": total_dense / max(n_images, 1),
            "sops_to_dense_ratio": ratio,
            "num_counted_layers": len(self.layers),
            "top_layers_by_sops": top_layers,
            "layers": layer_dict,
        }

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def build_efficiency_metrics(
    n_eval,
    runtime_sec,
    data_time_sec,
    transfer_time_sec,
    forward_time_sec,
    postprocess_time_sec,
    device,
    n_params,
):
    metrics = {
        "n_params": n_params,
        "runtime_sec": runtime_sec,
        "throughput_img_per_sec": n_eval / max(runtime_sec, 1e-9),
        "latency_ms_per_image_end_to_end": 1000.0 * runtime_sec / max(n_eval, 1),
        "data_wait_sec_total": data_time_sec,
        "transfer_sec_total": transfer_time_sec,
        "forward_sec_total": forward_time_sec,
        "postprocess_sec_total": postprocess_time_sec,
        "data_wait_ms_per_image": 1000.0 * data_time_sec / max(n_eval, 1),
        "transfer_ms_per_image": 1000.0 * transfer_time_sec / max(n_eval, 1),
        "forward_ms_per_image": 1000.0 * forward_time_sec / max(n_eval, 1),
        "postprocess_ms_per_image": 1000.0 * postprocess_time_sec / max(n_eval, 1),
    }

    if isinstance(device, torch.device) and device.type == "cuda":
        metrics["peak_gpu_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics["peak_gpu_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        metrics["peak_gpu_memory_allocated_mb"] = None
        metrics["peak_gpu_memory_reserved_mb"] = None

    return metrics


# -----------------------------
# Box utils
# -----------------------------
def box_iou_xyxy(boxes1, boxes2, eps=1e-7):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def nms_pure_torch(boxes, scores, iou_thr=0.5, max_det=300):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        cur_box = boxes[i].view(1, 4)
        rest = boxes[order[1:]]
        ious = box_iou_xyxy(cur_box, rest).view(-1)

        order = order[1:][ious <= iou_thr]

    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def clip_boxes(boxes, image_size):
    h, w = image_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes


def xywh_to_xyxy(boxes):
    out = boxes.clone()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


# -----------------------------
# Naive Option A decode
# -----------------------------
def decode_optiona_boxes(raw_box, grid_h, grid_w):
    """
    raw_box: [B, 4, Gh, Gw]
    returns normalized xyxy: [B, 4, Gh, Gw]
    """
    B, _, Gh, Gw = raw_box.shape
    device = raw_box.device

    yy, xx = torch.meshgrid(
        torch.arange(Gh, device=device),
        torch.arange(Gw, device=device),
        indexing="ij",
    )

    xx = xx.float()[None, None, :, :]
    yy = yy.float()[None, None, :, :]

    tx = torch.sigmoid(raw_box[:, 0:1])
    ty = torch.sigmoid(raw_box[:, 1:2])
    tw = torch.sigmoid(raw_box[:, 2:3])
    th = torch.sigmoid(raw_box[:, 3:4])

    cx = (xx + tx) / Gw
    cy = (yy + ty) / Gh

    x1 = (cx - tw / 2).clamp(0, 1)
    y1 = (cy - th / 2).clamp(0, 1)
    x2 = (cx + tw / 2).clamp(0, 1)
    y2 = (cy + th / 2).clamp(0, 1)

    return torch.cat([x1, y1, x2, y2], dim=1)


def decode_naive_predictions(pred, image_size, score_thr, nms_thr, max_det, num_classes):
    """
    pred: [B, 5 + C, Gh, Gw]
    returns list[dict]: boxes xyxy pixel, scores, labels
    """
    img_h, img_w = image_size
    B, _, Gh, Gw = pred.shape

    raw_box = pred[:, 0:4]
    obj = torch.sigmoid(pred[:, 4:5])
    cls_prob = torch.softmax(pred[:, 5:5 + num_classes], dim=1)

    boxes_norm = decode_optiona_boxes(raw_box, Gh, Gw)
    boxes = boxes_norm.permute(0, 2, 3, 1).reshape(B, Gh * Gw, 4)

    boxes[:, :, [0, 2]] *= img_w
    boxes[:, :, [1, 3]] *= img_h

    obj = obj.permute(0, 2, 3, 1).reshape(B, Gh * Gw, 1)
    cls_prob = cls_prob.permute(0, 2, 3, 1).reshape(B, Gh * Gw, num_classes)

    results = []

    for b in range(B):
        all_boxes = []
        all_scores = []
        all_labels = []

        for c in range(num_classes):
            scores_c = obj[b, :, 0] * cls_prob[b, :, c]
            keep = scores_c >= score_thr

            if keep.sum() == 0:
                continue

            boxes_c = boxes[b, keep]
            scores_keep = scores_c[keep]
            labels_c = torch.full((boxes_c.shape[0],), c, dtype=torch.long, device=pred.device)

            nms_keep = nms_pure_torch(boxes_c, scores_keep, iou_thr=nms_thr, max_det=max_det)

            all_boxes.append(boxes_c[nms_keep])
            all_scores.append(scores_keep[nms_keep])
            all_labels.append(labels_c[nms_keep])

        if all_boxes:
            boxes_b = torch.cat(all_boxes, dim=0)
            scores_b = torch.cat(all_scores, dim=0)
            labels_b = torch.cat(all_labels, dim=0)

            order = scores_b.argsort(descending=True)[:max_det]
            boxes_b = clip_boxes(boxes_b[order], image_size)
            scores_b = scores_b[order]
            labels_b = labels_b[order]
        else:
            boxes_b = torch.zeros((0, 4), device=pred.device)
            scores_b = torch.zeros((0,), device=pred.device)
            labels_b = torch.zeros((0,), dtype=torch.long, device=pred.device)

        results.append({
            "boxes": boxes_b.detach().cpu(),
            "scores": scores_b.detach().cpu(),
            "labels": labels_b.detach().cpu(),
        })

    return results


# -----------------------------
# YOLOX decode
# -----------------------------
def decode_yolox_predictions(outputs, image_size, score_thr, nms_thr, max_det, num_classes):
    """
    outputs: [B, N, 5 + C] = [cx, cy, w, h, obj_logit, cls_logits...]
    returns list[dict]: boxes xyxy pixel, scores, labels
    """
    results = []

    for b in range(outputs.shape[0]):
        out = outputs[b]

        boxes_xywh = out[:, 0:4]
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        boxes_xyxy = clip_boxes(boxes_xyxy, image_size)

        obj = torch.sigmoid(out[:, 4])
        cls_prob = torch.sigmoid(out[:, 5:5 + num_classes])

        all_boxes = []
        all_scores = []
        all_labels = []

        for c in range(num_classes):
            scores_c = obj * cls_prob[:, c]
            keep = scores_c >= score_thr

            if keep.sum() == 0:
                continue

            boxes_c = boxes_xyxy[keep]
            scores_keep = scores_c[keep]
            labels_c = torch.full((boxes_c.shape[0],), c, dtype=torch.long, device=outputs.device)

            nms_keep = nms_pure_torch(boxes_c, scores_keep, iou_thr=nms_thr, max_det=max_det)

            all_boxes.append(boxes_c[nms_keep])
            all_scores.append(scores_keep[nms_keep])
            all_labels.append(labels_c[nms_keep])

        if all_boxes:
            boxes_b = torch.cat(all_boxes, dim=0)
            scores_b = torch.cat(all_scores, dim=0)
            labels_b = torch.cat(all_labels, dim=0)

            order = scores_b.argsort(descending=True)[:max_det]
            boxes_b = boxes_b[order]
            scores_b = scores_b[order]
            labels_b = labels_b[order]
        else:
            boxes_b = torch.zeros((0, 4), device=outputs.device)
            scores_b = torch.zeros((0,), device=outputs.device)
            labels_b = torch.zeros((0,), dtype=torch.long, device=outputs.device)

        results.append({
            "boxes": boxes_b.detach().cpu(),
            "scores": scores_b.detach().cpu(),
            "labels": labels_b.detach().cpu(),
        })

    return results


# -----------------------------
# Metrics
# -----------------------------
def voc_ap(recalls, precisions):
    """
    Continuous/interpolated AP.
    """
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap


def compute_ap_metrics(predictions, targets, num_classes, iou_thr=0.5):
    """
    predictions: list[dict] with boxes, scores, labels
    targets: list[dict] with boxes, labels
    """
    class_metrics = {}
    aps = []

    total_tp = 0
    total_fp = 0
    total_gt = 0

    for cls_id in range(num_classes):
        gt_by_image = {}
        npos = 0

        for img_id, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["labels"]

            cls_mask = labels == cls_id
            cls_boxes = boxes[cls_mask]

            gt_by_image[img_id] = {
                "boxes": cls_boxes,
                "matched": torch.zeros((cls_boxes.shape[0],), dtype=torch.bool),
            }

            npos += cls_boxes.shape[0]

        pred_items = []
        for img_id, pred in enumerate(predictions):
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]

            cls_mask = labels == cls_id
            for box, score in zip(boxes[cls_mask], scores[cls_mask]):
                pred_items.append((img_id, float(score), box))

        pred_items.sort(key=lambda x: x[1], reverse=True)

        tp = []
        fp = []

        for img_id, score, box in pred_items:
            gt_entry = gt_by_image[img_id]
            gt_boxes = gt_entry["boxes"]

            if gt_boxes.numel() == 0:
                tp.append(0.0)
                fp.append(1.0)
                continue

            ious = box_iou_xyxy(box.view(1, 4), gt_boxes).view(-1)
            best_iou, best_idx = ious.max(dim=0)

            if best_iou >= iou_thr and not gt_entry["matched"][best_idx]:
                tp.append(1.0)
                fp.append(0.0)
                gt_entry["matched"][best_idx] = True
            else:
                tp.append(0.0)
                fp.append(1.0)

        if len(tp) == 0:
            ap = 0.0
            precision = 0.0
            recall = 0.0
            tp_sum = 0
            fp_sum = 0
        else:
            tp_t = torch.tensor(tp).cumsum(dim=0)
            fp_t = torch.tensor(fp).cumsum(dim=0)

            recalls = (tp_t / max(npos, 1)).tolist()
            precisions = (tp_t / torch.clamp(tp_t + fp_t, min=1e-9)).tolist()

            ap = voc_ap(recalls, precisions)
            tp_sum = int(tp_t[-1].item())
            fp_sum = int(fp_t[-1].item())
            recall = tp_sum / max(npos, 1)
            precision = tp_sum / max(tp_sum + fp_sum, 1)

        if npos > 0:
            aps.append(ap)

        total_tp += tp_sum
        total_fp += fp_sum
        total_gt += npos

        class_metrics[str(cls_id)] = {
            "ap50": ap,
            "precision": precision,
            "recall": recall,
            "num_gt": npos,
            "num_pred": len(pred_items),
            "tp": tp_sum,
            "fp": fp_sum,
        }

    map50 = sum(aps) / len(aps) if aps else 0.0
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_gt, 1)
    micro_f1 = (
        2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-9)
    )

    return {
        "map50": map50,
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "num_gt": total_gt,
        "num_pred": total_tp + total_fp,
        "tp": total_tp,
        "fp": total_fp,
        "classes": class_metrics,
    }


def parse_score_thresholds(value):
    """
    Accept:
        "0.001,0.01,0.05,0.1,0.2,0.3,0.5"
        [0.001, 0.01, 0.05, 0.1]
        (0.001, 0.01, 0.05, 0.1)
    """
    if value is None:
        return [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return []
        return [float(x.strip()) for x in value.split(",") if x.strip()]

    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]

    raise TypeError(f"Unsupported metric_score_thrs type: {type(value)}")


def filter_predictions_by_score(predictions, score_thr):
    """
    Filter already-decoded predictions by confidence score.
    """
    filtered = []

    for pred in predictions:
        scores = pred["scores"]
        keep = scores >= score_thr

        filtered.append({
            "boxes": pred["boxes"][keep],
            "scores": pred["scores"][keep],
            "labels": pred["labels"][keep],
        })

    return filtered


def compute_threshold_sweep(predictions, targets, num_classes, iou_thr, score_thrs):
    """
    Compute detection metrics at multiple confidence thresholds.
    """
    sweep = {}

    for thr in score_thrs:
        preds_thr = filter_predictions_by_score(
            predictions=predictions,
            score_thr=thr,
        )

        metrics_thr = compute_ap_metrics(
            predictions=preds_thr,
            targets=targets,
            num_classes=num_classes,
            iou_thr=iou_thr,
        )

        n_images = max(len(targets), 1)

        sweep[str(thr)] = {
            "map50": metrics_thr["map50"],
            "precision_micro": metrics_thr["precision_micro"],
            "recall_micro": metrics_thr["recall_micro"],
            "f1_micro": metrics_thr["f1_micro"],
            "num_gt": metrics_thr["num_gt"],
            "num_pred": metrics_thr["num_pred"],
            "pred_per_image": metrics_thr["num_pred"] / n_images,
            "tp": metrics_thr["tp"],
            "fp": metrics_thr["fp"],
        }

    return sweep


# -----------------------------
# Model loading
# -----------------------------
def build_model(mode, args, image_size):
    if mode == "spikformer_baseline":
        from model_det import SpikformerDetectorOptionA

        return SpikformerDetectorOptionA(
            num_classes=args.num_classes,
            in_channels=2,
            embed_dims=256,
        )

    if mode == "spikformer_yolox":
        from model_det_yolox_old import SpikformerYOLOXDetector

        return SpikformerYOLOXDetector(
            num_classes=args.num_classes,
            in_channels=2,
            embed_dims=256,
            num_heads=16,
            depths=2,
            drop_path_rate=0.1,
        )

    if mode == "vit_yolox":
        from model_det_vit_yolox_old import ViTYOLOXDetector

        return ViTYOLOXDetector(
            num_classes=args.num_classes,
            T=args.T,
            img_size=image_size,
            patch_size=args.vit_patch_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
        )

    raise ValueError(f"Unknown mode: {mode}")


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return ckpt


# -----------------------------
# Main eval
# -----------------------------
@torch.no_grad()
def run_eval(args):
    if args.score_thrs is not None:
        score_thrs = [float(x) for x in args.score_thrs.split(",")]
    else:
        score_thrs = [args.score_thr]
    device = torch.device(args.device)
    image_size = (args.input_height, args.input_width)

    val_ratio = args.val_ratio
    if val_ratio > 1.0:
        val_ratio = val_ratio / 100.0
    val_ratio = max(0.0, min(1.0, val_ratio))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = Path(args.checkpoint).stem

    run_dir = Path(args.output_dir) / f"eval_{args.mode}_{ckpt_name}_val{val_ratio:.3f}_seed{args.seed}_{timestamp}"
    mkdir(run_dir)

    log_f = open(run_dir / "eval.log", "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    spike_monitor = None
    sops_monitor = None

    try:
        save_json(vars(args), run_dir / "args.json")

        print("=" * 80)
        print("Evaluation config")
        print("=" * 80)
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        print("Resolved val_ratio:", val_ratio)
        print("Run dir:", str(run_dir))
        print()

        print("Building validation dataset")
        dataset_val = EVCivilDetectionDataset(
            root=args.data_path,
            split="val",
            T=args.T,
            input_size=image_size,
            window_ms=args.window_ms,
            max_samples=None,
            max_sequences=args.max_val_sequences,
            verbose=True,
        )

        n_total = len(dataset_val)
        n_eval = max(1, int(round(n_total * val_ratio)))

        rng = random.Random(args.seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        indices = sorted(indices[:n_eval])

        dataset_eval = Subset(dataset_val, indices)

        print(f"Validation total samples: {n_total}")
        print(f"Evaluation samples: {n_eval}")
        print(f"First 20 selected indices: {indices[:20]}")
        print()

        save_json(
            {
                "n_total_val": n_total,
                "n_eval": n_eval,
                "val_ratio": val_ratio,
                "seed": args.seed,
                "indices": indices,
            },
            run_dir / "selected_indices.json",
        )

        loader = DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=detection_collate_fn,
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )

        print("Building model:", args.mode)
        model = build_model(args.mode, args, image_size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of params:", n_params)

        ckpt = load_checkpoint(model, args.checkpoint, device)
        print("Loaded checkpoint:", args.checkpoint)
        if isinstance(ckpt, dict):
            print("checkpoint epoch:", ckpt.get("epoch"))
            print("checkpoint global_step:", ckpt.get("global_step"))
            print("checkpoint best_val_loss:", ckpt.get("best_val_loss"))
        print()

        is_snn_model = args.mode in {"spikformer_baseline", "spikformer_yolox"}
        snn_metrics_enabled = is_snn_model and not args.disable_snn_metrics

        if snn_metrics_enabled:
            print("SNN-specific metrics: enabled")
            spike_monitor = SpikeActivityMonitor(
                model,
                enabled=True,
                spike_thr=args.spike_thr,
            )
            sops_monitor = SOPSEstimator(
                model,
                enabled=True,
                spike_thr=args.sops_spike_thr,
                include_prefix=args.sops_include_prefix,
            )
            print(f"  spike layers monitored: {len(spike_monitor.layers)}")
            print(f"  SOP layers monitored: {len(sops_monitor.layers)}")
            print(f"  SOP include prefix: {args.sops_include_prefix}")
        else:
            print("SNN-specific metrics: disabled / N/A for this mode")
        print()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        min_score_thr = min(score_thrs)

        all_predictions = []
        all_targets = []

        print("Score thresholds:", score_thrs)
        print("Decode/NMS score threshold:", min_score_thr)

        per_image_path = run_dir / "per_image_predictions.jsonl"
        per_image_f = open(per_image_path, "w", encoding="utf-8")

        start = time.time()
        data_timer_start = time.time()
        seen = 0

        data_time_total = 0.0
        transfer_time_total = 0.0
        forward_time_total = 0.0
        postprocess_time_total = 0.0

        for batch_idx, (images, targets) in enumerate(loader):
            data_time = time.time() - data_timer_start
            data_time_total += data_time

            transfer_start = time.time()
            images = images.to(device, non_blocking=True).float()
            sync_if_cuda(device)
            transfer_time = time.time() - transfer_start
            transfer_time_total += transfer_time

            forward_start = time.time()
            if args.mode == "spikformer_baseline":
                pred_raw = model(images)
                sync_if_cuda(device)
                forward_time = time.time() - forward_start
                forward_time_total += forward_time

                post_start = time.time()

                batch_preds = decode_naive_predictions(
                    pred=pred_raw,
                    image_size=image_size,
                    score_thr=min_score_thr,
                    nms_thr=args.nms_thr,
                    max_det=args.max_det,
                    num_classes=args.num_classes,
                )

                functional.reset_net(model)
                sync_if_cuda(device)
                post_time = time.time() - post_start
                postprocess_time_total += post_time

            else:
                outputs = model(images)
                sync_if_cuda(device)
                forward_time = time.time() - forward_start
                forward_time_total += forward_time

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                post_start = time.time()

                batch_preds = decode_yolox_predictions(
                    outputs=outputs,
                    image_size=image_size,
                    score_thr=min_score_thr,
                    nms_thr=args.nms_thr,
                    max_det=args.max_det,
                    num_classes=args.num_classes,
                )

                if args.mode.startswith("spikformer"):
                    functional.reset_net(model)

                sync_if_cuda(device)
                post_time = time.time() - post_start
                postprocess_time_total += post_time

            for i in range(len(batch_preds)):
                pred_i = batch_preds[i]

                tgt_i = {
                    "boxes": targets[i]["boxes"].detach().cpu(),
                    "labels": targets[i]["labels"].detach().cpu(),
                }

                all_predictions.append(pred_i)
                all_targets.append(tgt_i)

                global_sample_index = indices[seen]

                # Log predictions decoded at the minimum threshold.
                log_thr = min_score_thr
                record = {
                    "eval_id": seen,
                    "dataset_index": global_sample_index,
                    "score_thr": log_thr,
                    "num_gt": int(tgt_i["boxes"].shape[0]),
                    "num_pred": int(pred_i["boxes"].shape[0]),
                    "gt_boxes": tgt_i["boxes"].tolist(),
                    "gt_labels": tgt_i["labels"].tolist(),
                    "pred_boxes": pred_i["boxes"].tolist(),
                    "pred_scores": pred_i["scores"].tolist(),
                    "pred_labels": pred_i["labels"].tolist(),
                }
                per_image_f.write(json.dumps(record) + "\n")

                seen += 1

            if batch_idx % args.print_freq == 0:
                elapsed = time.time() - start
                speed = seen / max(elapsed, 1e-9)
                remaining = max(n_eval - seen, 0)
                eta_sec = remaining / max(speed, 1e-9)
                progress = 100.0 * seen / max(n_eval, 1)
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                avg_forward_ms = 1000.0 * forward_time_total / max(seen, 1)
                avg_post_ms = 1000.0 * postprocess_time_total / max(seen, 1)
                avg_data_ms = 1000.0 * data_time_total / max(seen, 1)

                if device.type == "cuda":
                    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    mem_str = f"peak_mem={peak_mem:.1f}MB"
                else:
                    mem_str = "peak_mem=N/A"

                print(
                    f"Eval [{seen}/{n_eval}] "
                    f"progress={progress:.2f}% "
                    f"batch={batch_idx}/{len(loader)} "
                    f"speed={speed:.2f} img/s "
                    f"forward={avg_forward_ms:.2f} ms/img "
                    f"post={avg_post_ms:.2f} ms/img "
                    f"data={avg_data_ms:.2f} ms/img "
                    f"elapsed={elapsed_str} "
                    f"eta={eta_str} "
                    f"{mem_str}"
                )

            data_timer_start = time.time()

        per_image_f.close()

        runtime_sec = time.time() - start

        threshold_metrics = compute_threshold_sweep(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=args.num_classes,
        iou_thr=args.iou_thr,
        score_thrs=score_thrs,
    )

        for thr, det_m in threshold_metrics.items():
            print(
                f"thr={float(thr):.4f} | "
                f"mAP50={det_m['map50']:.4f} | "
                f"P={det_m['precision_micro']:.4f} | "
                f"R={det_m['recall_micro']:.4f} | "
                f"F1={det_m['f1_micro']:.4f} | "
                f"pred/img={det_m['pred_per_image']:.2f} | "
                f"TP={det_m['tp']} FP={det_m['fp']}"
            )

        main_thr = str(score_thrs[0])
        detection_metrics = threshold_metrics[main_thr]

        

        efficiency_metrics = build_efficiency_metrics(
            n_eval=n_eval,
            runtime_sec=runtime_sec,
            data_time_sec=data_time_total,
            transfer_time_sec=transfer_time_total,
            forward_time_sec=forward_time_total,
            postprocess_time_sec=postprocess_time_total,
            device=device,
            n_params=n_params,
        )

        if snn_metrics_enabled:
            firing_metrics = spike_monitor.summary(topk=args.snn_layer_topk)
            sops_metrics = sops_monitor.summary(n_images=n_eval, topk=args.snn_layer_topk)
        else:
            firing_metrics = {
                "enabled": False,
                "avg_firing_rate": None,
                "total_spike_count": None,
                "total_neuron_events": None,
                "layers": {},
                "reason": "N/A for ViT or disabled by flag",
            }
            sops_metrics = {
                "enabled": False,
                "sops_estimate_total": None,
                "sops_estimate_per_image": None,
                "dense_macs_backbone_estimate_total": None,
                "dense_macs_backbone_estimate_per_image": None,
                "sops_to_dense_ratio": None,
                "layers": {},
                "reason": "N/A for ViT or disabled by flag",
            }

        metrics = {
    "mode": args.mode,
    "checkpoint": args.checkpoint,
    "score_thrs": score_thrs,
    "nms_thr": args.nms_thr,
    "iou_thr": args.iou_thr,
    "val_ratio": val_ratio,
    "seed": args.seed,
    "n_eval": n_eval,
    "n_total_val": n_total,
    "threshold_sweep": threshold_metrics,
    "efficiency": efficiency_metrics,
    "snn": {
        "enabled": snn_metrics_enabled,
        "firing_rate": firing_metrics,
        "sops": sops_metrics,
    },
    "n_params": n_params,
    "runtime_sec": runtime_sec,
}

        save_json(metrics, run_dir / "metrics.json")

        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("Evaluation summary")
        summary_lines.append("=" * 80)
        summary_lines.append(f"mode: {args.mode}")
        summary_lines.append(f"checkpoint: {args.checkpoint}")
        summary_lines.append(f"n_eval: {n_eval}/{n_total}")
        summary_lines.append("")

        summary_lines.append("[1] Detection performance")
        summary_lines.append(f"mAP50: {detection_metrics['map50']:.6f}")
        summary_lines.append(f"precision_micro: {detection_metrics['precision_micro']:.6f}")
        summary_lines.append(f"recall_micro: {detection_metrics['recall_micro']:.6f}")
        summary_lines.append(f"f1_micro: {detection_metrics['f1_micro']:.6f}")
        summary_lines.append(f"num_gt: {detection_metrics['num_gt']}")
        summary_lines.append(f"num_pred: {detection_metrics['num_pred']}")
        summary_lines.append(f"tp: {detection_metrics['tp']}")
        summary_lines.append(f"fp: {detection_metrics['fp']}")
        summary_lines.append("")
        summary_lines.append("[Threshold sweep]")
        for thr, m in threshold_metrics.items():
            summary_lines.append(
                f"thr={thr:>5} | "
                f"mAP50={m['map50']:.4f} | "
                f"P={m['precision_micro']:.4f} | "
                f"R={m['recall_micro']:.4f} | "
                f"F1={m['f1_micro']:.4f} | "
                f"pred/img={m['pred_per_image']:.2f} | "
                f"TP={m['tp']} FP={m['fp']}"
            )
        summary_lines.append("Per-class:")
        for cls_id, cls_m in detection_metrics["classes"].items():
            name = args.class_names.split(",")[int(cls_id)] if args.class_names else cls_id
            summary_lines.append(
                f"  class {cls_id} ({name}): "
                f"AP50={cls_m['ap50']:.6f}, "
                f"P={cls_m['precision']:.6f}, "
                f"R={cls_m['recall']:.6f}, "
                f"GT={cls_m['num_gt']}, "
                f"Pred={cls_m['num_pred']}, "
                f"TP={cls_m['tp']}, FP={cls_m['fp']}"
            )
        summary_lines.append("")

        summary_lines.append("[2] Practical efficiency")
        summary_lines.append(f"params: {efficiency_metrics['n_params']}")
        summary_lines.append(f"runtime_sec: {efficiency_metrics['runtime_sec']:.3f}")
        summary_lines.append(f"throughput_img_per_sec: {efficiency_metrics['throughput_img_per_sec']:.4f}")
        summary_lines.append(f"latency_ms_per_image_end_to_end: {efficiency_metrics['latency_ms_per_image_end_to_end']:.4f}")
        summary_lines.append(f"forward_ms_per_image: {efficiency_metrics['forward_ms_per_image']:.4f}")
        summary_lines.append(f"postprocess_ms_per_image: {efficiency_metrics['postprocess_ms_per_image']:.4f}")
        summary_lines.append(f"data_wait_ms_per_image: {efficiency_metrics['data_wait_ms_per_image']:.4f}")
        summary_lines.append(f"peak_gpu_memory_allocated_mb: {efficiency_metrics['peak_gpu_memory_allocated_mb']}")
        summary_lines.append(f"peak_gpu_memory_reserved_mb: {efficiency_metrics['peak_gpu_memory_reserved_mb']}")
        summary_lines.append("")

        summary_lines.append("[3] SNN-specific behavior")
        if snn_metrics_enabled:
            summary_lines.append(f"avg_firing_rate: {firing_metrics['avg_firing_rate']:.8f}")
            summary_lines.append(f"total_spike_count: {firing_metrics['total_spike_count']:.2f}")
            summary_lines.append(f"total_neuron_events: {firing_metrics['total_neuron_events']}")
            summary_lines.append(f"num_spike_layers: {firing_metrics.get('num_spike_layers')}")
            summary_lines.append(f"sops_estimate_total: {sops_metrics['sops_estimate_total']:.2f}")
            summary_lines.append(f"sops_estimate_per_image: {sops_metrics['sops_estimate_per_image']:.2f}")
            summary_lines.append(f"dense_macs_backbone_estimate_per_image: {sops_metrics['dense_macs_backbone_estimate_per_image']:.2f}")
            summary_lines.append(f"sops_to_dense_ratio: {sops_metrics['sops_to_dense_ratio']:.8f}")
            summary_lines.append("Note: SOPs are approximate backbone SynOps proxy, not directly equivalent to ANN FLOPs.")
        else:
            summary_lines.append("N/A for this mode. ViT does not have spike activity, firing rate, or SOPs.")

        summary = "\n".join(summary_lines)
        print()
        print(summary)

        with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary + "\n")

        print()
        print("Saved:")
        print(" ", run_dir / "eval.log")
        print(" ", run_dir / "args.json")
        print(" ", run_dir / "selected_indices.json")
        print(" ", run_dir / "metrics.json")
        print(" ", run_dir / "summary.txt")
        print(" ", run_dir / "per_image_predictions.jsonl")

    finally:
        if spike_monitor is not None:
            spike_monitor.close()
        if sops_monitor is not None:
            sops_monitor.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_f.close()

def parse_args():
    parser = argparse.ArgumentParser("Evaluate ev-CIVIL detection models on validation set")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["spikformer_baseline", "spikformer_yolox", "vit_yolox"],
    )

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="spikformer/evcivil/logs_eval_det")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--window-ms", type=float, default=30.0)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-names", type=str, default="crack,spalling")

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=1.0,
        help="Fraction of validation set. Use 0.1 for 10%, or 10 for 10%.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-val-sequences", type=int, default=None)

    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)

    # ViT config, only used by mode=vit_yolox.
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--vit-embed-dim", type=int, default=256)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=8)


    # SNN-specific metrics. Enabled only for spikformer_* modes unless disabled.
    parser.add_argument("--disable-snn-metrics", action="store_true")
    parser.add_argument("--spike-thr", type=float, default=0.0)
    parser.add_argument("--sops-spike-thr", type=float, default=0.0)
    parser.add_argument("--sops-include-prefix", type=str, default="backbone")
    parser.add_argument("--snn-layer-topk", type=int, default=20)

    parser.add_argument("--print-freq", type=int, default=20)


    parser.add_argument(
    "--score-thrs",
    type=str,
    default=None,
    help="Comma-separated score thresholds, e.g. 0.001,0.005,0.01,0.05",
)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)


# python val_det.py \
#   --mode spikformer_yolox \
#   --data-path /AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset \
#   --checkpoint /AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/spikformer/evcivil/logs_evcivil_det_yolox/spikformer_yolox_T16_256x256_lr0.0001/checkpoint_best_val_loss.pth \
#   --output-dir logs_eval_evcivil_det \
#   --device cuda:0 \
#   --batch-size 8 \
#   --workers 2 \
#   --T 16 \
#   --input-height 256 \
#   --input-width 256 \
#   --window-ms 30.0 \
#   --num-classes 2 \
#   --class-names crack,spalling \
#   --val-ratio 1.0 \
#   --score-thr 0.001 \
#   --metric-score-thrs 0.001,0.01,0.05,0.1,0.2,0.3,0.5 \
#   --nms-thr 0.5 \
#   --iou-thr 0.5 \
#   --max-det 300