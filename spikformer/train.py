
import argparse
import datetime
import os
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.clock_driven import functional

import utils
from evcivil_dataset_loader import EVCivilDetectionDataset, detection_collate_yolox_fn
from model import SpikformerYOLOXDetector

import json

from val_det import (
    decode_yolox_predictions,
    compute_ap_metrics,
    box_iou_xyxy,
    SpikeActivityMonitor,
    SOPSEstimator,
)


def load_py_config(config_path: str):
    if config_path is None or config_path == "":
        return {}

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("train_config", str(config_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "CONFIG"):
        cfg = module.CONFIG
    elif hasattr(module, "config"):
        cfg = module.config
    elif hasattr(module, "get_config"):
        cfg = module.get_config()
    else:
        raise ValueError(
            f"Config file {config_path} must define CONFIG, config, or get_config()."
        )

    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a dict, got {type(cfg)}")

    return cfg


def parse_args():
    # First pass: only read --config.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="")
    config_args, _ = config_parser.parse_known_args()

    cfg = load_py_config(config_args.config)

    parser = argparse.ArgumentParser(
        "ev-CIVIL / DETRAC Spikformer + YOLOX detection",
        parents=[config_parser],
    )

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./logs_evcivil_det_yolox")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--window-ms", type=float, default=30.0)

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)

    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--log-every-iters", type=int, default=50)
    parser.add_argument("--save-every-iters", type=int, default=1000)
    parser.add_argument("--keep-last-k-iters", type=int, default=3)

    parser.add_argument("--resume", type=str, default="")

    parser.add_argument(
        "--dataset",
        type=str,
        default="evcivil",
        choices=["evcivil", "detrac"],
    )

    parser.add_argument(
        "--representation",
        type=str,
        default="grayscale_dup",
        choices=["grayscale_dup", "grayscale_zero", "frame_diff"],
    )

    parser.add_argument("--frame-stride", type=int, default=1)

    parser.add_argument("--eval-metrics-every", type=int, default=1)
    parser.add_argument("--disable-eval-metrics", action="store_true", default=False)

    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--class-names", type=str, default="vehicle")

    # SNN metrics
    parser.add_argument("--disable-snn-metrics", action="store_true", default=False)
    parser.add_argument("--spike-thr", type=float, default=0.0)
    parser.add_argument("--sops-spike-thr", type=float, default=0.0)
    parser.add_argument("--sops-include-prefix", type=str, default="backbone")
    parser.add_argument("--snn-layer-topk", type=int, default=20)

    # Model config
    parser.add_argument("--model-in-channels", type=int, default=2)
    parser.add_argument("--model-embed-dims", type=int, default=256)
    parser.add_argument("--model-num-heads", type=int, default=16)
    parser.add_argument("--model-depths", type=str, default="2")
    parser.add_argument("--model-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--model-drop-path-rate", type=float, default=0.1)

    # Spiking neuron config
    parser.add_argument(
        "--neuron-type",
        type=str,
        default="lif",
        choices=["lif", "plif"],
    )
    parser.add_argument("--neuron-tau", type=float, default=2.0)
    parser.add_argument("--neuron-v-threshold", type=float, default=1.0)
    parser.add_argument("--neuron-attn-v-threshold", type=float, default=0.5)
    parser.add_argument("--neuron-v-reset", type=str, default="0.0")
    parser.add_argument("--neuron-detach-reset", action="store_true", default=True)
    parser.add_argument("--neuron-backend", type=str, default="cupy", choices=["torch", "cupy"])

    # Validate config keys before applying.
    # valid_keys = {action.dest for action in parser._actions}
    # unknown_keys = sorted(set(cfg.keys()) - valid_keys)

    # if len(unknown_keys) > 0:
    #     raise ValueError(
    #         "Unknown config keys:\n"
    #         + "\n".join(f"  - {k}" for k in unknown_keys)
    #         + "\n\nNote: use argparse dest names, e.g. data_path, batch_size, input_height."
    #     )
    # Apply all config keys, including config-only keys.
    # Argparse Namespace can store keys even if they do not have parser.add_argument().

    # Apply config defaults.
    # Command-line args still override these.
    parser.set_defaults(**cfg)

    args = parser.parse_args()

    if args.data_path is None or args.data_path == "":
        parser.error("--data-path is required unless provided in --config")

    return args




def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def atomic_save(obj, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def cleanup_iter_checkpoints(output_dir, keep_last_k):
    if keep_last_k is None or keep_last_k <= 0:
        return

    output_dir = Path(output_dir)
    files = list(output_dir.glob("checkpoint_iter_*.pth"))

    def get_step(p):
        try:
            return int(p.stem.split("_")[-1])
        except Exception:
            return -1

    files = sorted(files, key=get_step)
    for p in files[:-keep_last_k]:
        try:
            p.unlink()
        except OSError:
            pass


def build_checkpoint(
    model,
    optimizer,
    epoch,
    global_step,
    args,
    best_val_loss,
    scaler=None,
    iter_in_epoch=None,
    finished_epoch=False,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iter_in_epoch": iter_in_epoch,
        "global_step": global_step,
        "args": args,
        "best_val_loss": best_val_loss,
        "finished_epoch": finished_epoch,
    }

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    return ckpt


def log_loss_dict(metric_logger, loss_dict, lr=None):
    metric_logger.update(
        loss=to_float(loss_dict["loss"]),
        giou_loss=to_float(loss_dict["giou_loss"]),
        conf_loss=to_float(loss_dict["conf_loss"]),
        cls_loss=to_float(loss_dict["cls_loss"]),
        l1_loss=to_float(loss_dict["l1_loss"]),
        num_fg=to_float(loss_dict["num_fg"]),
    )
    if lr is not None:
        metric_logger.update(lr=lr)


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler,
    output_dir,
    args,
    global_step,
    best_val_loss,
    writer=None,
):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    for iter_idx, (images, targets, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with amp.autocast():
                outputs, loss_dict = model(images, labels=labels)
                if loss_dict is None:
                    raise RuntimeError("YOLOXHead returned loss_dict=None. Check yolox/head.py patch.")
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, loss_dict = model(images, labels=labels)
            if loss_dict is None:
                raise RuntimeError("YOLOXHead returned loss_dict=None. Check yolox/head.py patch.")
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        global_step += 1
        batch_size = images.shape[0]

        log_loss_dict(
            metric_logger,
            loss_dict,
            lr=optimizer.param_groups[0]["lr"],
        )
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        if writer is not None and global_step % args.log_every_iters == 0:
            writer.add_scalar("train_iter/loss", to_float(loss_dict["loss"]), global_step)
            writer.add_scalar("train_iter/giou_loss", to_float(loss_dict["giou_loss"]), global_step)
            writer.add_scalar("train_iter/conf_loss", to_float(loss_dict["conf_loss"]), global_step)
            writer.add_scalar("train_iter/cls_loss", to_float(loss_dict["cls_loss"]), global_step)
            writer.add_scalar("train_iter/l1_loss", to_float(loss_dict["l1_loss"]), global_step)
            writer.add_scalar("train_iter/num_fg", to_float(loss_dict["num_fg"]), global_step)
            writer.add_scalar("train_iter/lr", optimizer.param_groups[0]["lr"], global_step)

        if args.save_every_iters > 0 and global_step % args.save_every_iters == 0:
            ckpt = build_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                best_val_loss=best_val_loss,
                scaler=scaler,
                iter_in_epoch=iter_idx,
                finished_epoch=False,
            )

            atomic_save(ckpt, os.path.join(output_dir, "checkpoint_latest.pth"))
            atomic_save(ckpt, os.path.join(output_dir, f"checkpoint_iter_{global_step}.pth"))
            cleanup_iter_checkpoints(output_dir, args.keep_last_k_iters)

            print(
                f"[checkpoint] saved latest at epoch={epoch}, "
                f"iter={iter_idx}, global_step={global_step}"
            )

    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, global_step


@torch.no_grad()
def evaluate_loss(model, data_loader, device, print_freq):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    for images, targets, labels in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        outputs, loss_dict = model(images, labels=labels)
        if loss_dict is None:
            raise RuntimeError("YOLOXHead returned loss_dict=None in eval. Check yolox/head.py patch.")

        functional.reset_net(model)

        log_loss_dict(metric_logger, loss_dict)

    metric_logger.synchronize_between_processes()

    print(
        f"* Val loss={metric_logger.loss.global_avg:.4f}, "
        f"giou={metric_logger.giou_loss.global_avg:.4f}, "
        f"conf={metric_logger.conf_loss.global_avg:.4f}, "
        f"cls={metric_logger.cls_loss.global_avg:.4f}, "
        f"num_fg={metric_logger.num_fg.global_avg:.4f}"
    )

    return metric_logger.loss.global_avg


def compute_mean_best_iou(predictions, targets, num_classes):
    """
    Debug-friendly localization metric.

    For each GT box, find the best predicted box of the same class.
    If no prediction exists for that class/image, best IoU = 0.

    This is not COCO mIoU. It is mainly useful for checking whether
    predicted boxes are spatially close to ground truth.
    """
    class_stats = {}
    all_ious = []

    for cls_id in range(num_classes):
        cls_ious = []

        for pred, target in zip(predictions, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]

            gt_mask = gt_labels == cls_id
            pred_mask = pred_labels == cls_id

            gt_cls = gt_boxes[gt_mask]
            pred_cls = pred_boxes[pred_mask]

            if gt_cls.numel() == 0:
                continue

            if pred_cls.numel() == 0:
                cls_ious.extend([0.0] * gt_cls.shape[0])
                continue

            ious = box_iou_xyxy(gt_cls, pred_cls)
            best_ious = ious.max(dim=1).values
            cls_ious.extend(best_ious.tolist())

        if len(cls_ious) == 0:
            mean_iou = 0.0
        else:
            mean_iou = float(sum(cls_ious) / len(cls_ious))

        class_stats[str(cls_id)] = {
            "mean_best_iou": mean_iou,
            "num_gt": len(cls_ious),
        }

        all_ious.extend(cls_ious)

    overall = float(sum(all_ious) / len(all_ious)) if len(all_ious) > 0 else 0.0

    return {
        "mean_best_iou": overall,
        "classes": class_stats,
    }

def parse_score_thresholds(value):
    """
    Read score thresholds from config.

    Accepted formats:
        "0.001,0.01,0.05,0.1"
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
    Filter decoded predictions by score threshold.

    predictions: list[dict]
        each dict has:
            boxes:  Tensor [N, 4]
            scores: Tensor [N]
            labels: Tensor [N]
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
    Compute detection metrics under multiple score thresholds.

    This is mainly for debugging:
        low threshold  -> high recall, many false positives
        high threshold -> lower recall, fewer false positives
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


@torch.no_grad()
def evaluate_detection_metrics(
    model,
    data_loader,
    device,
    image_size,
    args,
    epoch,
    output_dir,
    writer=None,
):
    model.eval()

    is_snn_model = isinstance(model, SpikformerYOLOXDetector)
    snn_enabled = is_snn_model and not args.disable_snn_metrics

    spike_monitor = None
    sops_monitor = None

    if snn_enabled:
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

    all_predictions = []
    all_targets = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Val metrics epoch [{epoch}]:"

    for batch in metric_logger.log_every(data_loader, args.print_freq, header):
        # Compatible with both:
        #   detection_collate_fn       -> images, targets
        #   detection_collate_yolox_fn -> images, targets, labels
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch

        images = images.to(device, non_blocking=True).float()

        outputs, _ = model(images, labels=None)

        batch_preds = decode_yolox_predictions(
            outputs=outputs,
            image_size=image_size,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            max_det=args.max_det,
            num_classes=args.num_classes,
        )

        if is_snn_model:
            functional.reset_net(model)

        for pred_i, tgt_i in zip(batch_preds, targets):
            all_predictions.append(pred_i)
            all_targets.append(
                {
                    "boxes": tgt_i["boxes"].detach().cpu(),
                    "labels": tgt_i["labels"].detach().cpu(),
                }
            )

    detection_metrics = compute_ap_metrics(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=args.num_classes,
        iou_thr=args.iou_thr,
    )

    iou_metrics = compute_mean_best_iou(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=args.num_classes,
    )

    score_thrs = parse_score_thresholds(
        getattr(args, "metric_score_thrs", [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    )

    threshold_sweep = compute_threshold_sweep(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=args.num_classes,
        iou_thr=args.iou_thr,
        score_thrs=score_thrs,
    )

    if snn_enabled:
        firing_metrics = spike_monitor.summary(topk=args.snn_layer_topk)
        sops_metrics = sops_monitor.summary(
            n_images=len(all_targets),
            topk=args.snn_layer_topk,
        )
    else:
        firing_metrics = {
            "enabled": False,
            "avg_firing_rate": None,
            "total_spike_count": None,
            "total_neuron_events": None,
        }
        sops_metrics = {
            "enabled": False,
            "sops_estimate_total": None,
            "sops_estimate_per_image": None,
            "dense_macs_backbone_estimate_per_image": None,
            "sops_to_dense_ratio": None,
        }

    if spike_monitor is not None:
        spike_monitor.close()
    if sops_monitor is not None:
        sops_monitor.close()

    metrics = {
        "epoch": epoch,
        "score_thr": args.score_thr,
        "nms_thr": args.nms_thr,
        "iou_thr": args.iou_thr,
        "num_classes": args.num_classes,
        "class_names": args.class_names,
        "detection": detection_metrics,
        "iou": iou_metrics,
        "snn": {
            "enabled": snn_enabled,
            "firing_rate": firing_metrics,
            "sops": sops_metrics,
        },
        # convenient top-level fields
        "map50": detection_metrics["map50"],
        "precision_micro": detection_metrics["precision_micro"],
        "recall_micro": detection_metrics["recall_micro"],
        "f1_micro": detection_metrics["f1_micro"],
        "mean_best_iou": iou_metrics["mean_best_iou"],
        "threshold_sweep": threshold_sweep,
    }

    output_dir = Path(output_dir)
    metrics_path = output_dir / f"eval_metrics_epoch_{epoch}.json"
    history_path = output_dir / "eval_metrics_history.jsonl"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    print()
    print("=" * 80)
    print(f"Detection eval epoch {epoch}")
    print("=" * 80)
    print(f"mAP50:          {metrics['map50']:.6f}")
    print(f"Precision:      {metrics['precision_micro']:.6f}")
    print(f"Recall:         {metrics['recall_micro']:.6f}")
    print(f"F1:             {metrics['f1_micro']:.6f}")
    print(f"Mean best IoU:  {metrics['mean_best_iou']:.6f}")
    print(f"GT boxes:       {detection_metrics['num_gt']}")
    print(f"Pred boxes:     {detection_metrics['num_pred']}")
    print(f"TP / FP:        {detection_metrics['tp']} / {detection_metrics['fp']}")


    print()
    print("[Threshold sweep]")
    for thr, m in threshold_sweep.items():
        print(
            f"thr={thr:>5} | "
            f"mAP50={m['map50']:.4f} | "
            f"P={m['precision_micro']:.4f} | "
            f"R={m['recall_micro']:.4f} | "
            f"F1={m['f1_micro']:.4f} | "
            f"pred/img={m['pred_per_image']:.2f} | "
            f"TP={m['tp']} FP={m['fp']}"
        )

    if snn_enabled:
        print()
        print("[SNN]")
        print(f"Avg firing rate: {firing_metrics['avg_firing_rate']:.8f}")
        print(f"SOPs / image:    {sops_metrics['sops_estimate_per_image']:.2f}")
        print(f"SOP/Dense ratio: {sops_metrics['sops_to_dense_ratio']:.8f}")

    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {history_path}")
    print()

    if writer is not None:
        writer.add_scalar("val_det/map50", metrics["map50"], epoch)
        writer.add_scalar("val_det/precision_micro", metrics["precision_micro"], epoch)
        writer.add_scalar("val_det/recall_micro", metrics["recall_micro"], epoch)
        writer.add_scalar("val_det/f1_micro", metrics["f1_micro"], epoch)
        writer.add_scalar("val_det/mean_best_iou", metrics["mean_best_iou"], epoch)
        writer.add_scalar("val_det/num_gt", detection_metrics["num_gt"], epoch)
        writer.add_scalar("val_det/num_pred", detection_metrics["num_pred"], epoch)
        writer.add_scalar("val_det/tp", detection_metrics["tp"], epoch)
        writer.add_scalar("val_det/fp", detection_metrics["fp"], epoch)

        if snn_enabled:
            writer.add_scalar("val_snn/avg_firing_rate", firing_metrics["avg_firing_rate"], epoch)
            writer.add_scalar("val_snn/total_spike_count", firing_metrics["total_spike_count"], epoch)
            writer.add_scalar("val_snn/sops_per_image", sops_metrics["sops_estimate_per_image"], epoch)
            writer.add_scalar("val_snn/sops_to_dense_ratio", sops_metrics["sops_to_dense_ratio"], epoch)

    return metrics


def parse_model_depths(value):
    """
    Accept:
        2
        "2"
        "2,2,2"
        [2, 2, 2]
        (2, 2, 2)
    """
    if isinstance(value, int):
        return value

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return int(value[0])
        if len(value) == 3:
            return tuple(int(x) for x in value)
        raise ValueError(f"model_depths must be int or 3 values, got {value}")

    if isinstance(value, str):
        value = value.strip()
        if "," not in value:
            return int(value)

        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"model_depths string must have 3 values, got {value}")
        return tuple(parts)

    raise TypeError(f"Unsupported model_depths type: {type(value)}")


def parse_v_reset(value):
    """
    Accept:
        0.0
        "0.0"
        None
        "none"
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"none", "null"}:
            return None
        return float(value)

    return float(value)


def build_neuron_cfg_from_args(args):
    return {
        "type": args.neuron_type,
        "tau": args.neuron_tau,
        "v_threshold": args.neuron_v_threshold,
        "attn_v_threshold": args.neuron_attn_v_threshold,
        "v_reset": parse_v_reset(args.neuron_v_reset),
        "detach_reset": args.neuron_detach_reset,
        "backend": args.neuron_backend,
    }


def main(args):
    device = torch.device(args.device)
    image_size = (args.input_height, args.input_width)

    depth_tag = args.model_depths
    if isinstance(depth_tag, (list, tuple)):
        depth_tag = "x".join(str(x) for x in depth_tag)

    output_dir = os.path.join(
        args.output_dir,
        (
            f"{args.dataset}_spikformer_{args.neuron_type}_yolox_"
            f"T{args.T}_win{args.window_ms}ms_"
            f"{args.input_height}x{args.input_width}_"
            f"ed{args.model_embed_dims}_d{depth_tag}_"
            f"lr{args.lr}"
        ),
    )
    utils.mkdir(output_dir)

    writer = SummaryWriter(os.path.join(output_dir, "tb"))

    print(args)
    print("Building datasets")

    if args.dataset == "evcivil":
        dataset_train = EVCivilDetectionDataset(
            root=args.data_path,
            split="train",
            T=args.T,
            input_size=image_size,
            window_ms=args.window_ms,
            max_samples=args.max_train_samples,
            max_sequences=args.max_train_sequences,
            verbose=True,
        )

        dataset_val = EVCivilDetectionDataset(
            root=args.data_path,
            split="val",
            T=args.T,
            input_size=image_size,
            window_ms=args.window_ms,
            max_samples=args.max_val_samples,
            max_sequences=args.max_val_sequences,
            verbose=True,
        )

    elif args.dataset == "detrac":
        dataset_train = DETRACDetectionDataset(
            root=args.data_path,
            split="train",
            T=args.T,
            input_size=image_size,
            frame_stride=args.frame_stride,
            representation=args.representation,
            one_class=(args.num_classes == 1),
            max_samples=args.max_train_samples,
            max_sequences=args.max_train_sequences,
            verbose=True,
        )

        dataset_val = DETRACDetectionDataset(
            root=args.data_path,
            split="val",
            T=args.T,
            input_size=image_size,
            frame_stride=args.frame_stride,
            representation=args.representation,
            one_class=(args.num_classes == 1),
            max_samples=args.max_val_samples,
            max_sequences=args.max_val_sequences,
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate_yolox_fn,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=detection_collate_yolox_fn,
    )

    print("Creating model")
    neuron_cfg = build_neuron_cfg_from_args(args)

    print("Neuron config:", neuron_cfg)

    model = SpikformerYOLOXDetector(
        num_classes=args.num_classes,
        in_channels=args.model_in_channels,
        embed_dims=args.model_embed_dims,
        num_heads=args.model_num_heads,
        depths=parse_model_depths(args.model_depths),
        mlp_ratio=args.model_mlp_ratio,
        drop_path_rate=args.model_drop_path_rate,
        neuron_cfg=neuron_cfg,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = amp.GradScaler() if args.amp else None

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_map50 = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        best_map50 = ckpt.get("best_map50", best_map50)

        # Override learning rate from command line after loading optimizer state.
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)

        if ckpt.get("finished_epoch", True):
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = ckpt["epoch"]

        print(
            f"Resumed from {args.resume}, "
            f"start_epoch={start_epoch}, global_step={global_step}, "
            f"best_val_loss={best_val_loss}, "
            f"new_lr={optimizer.param_groups[0]['lr']}"
        )

    print("Start training")
    start_time = time.time()

    # best_map50 = -1.0

    for epoch in range(start_epoch, args.epochs):
        train_loss, global_step = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            scaler=scaler,
            output_dir=output_dir,
            args=args,
            global_step=global_step,
            best_val_loss=best_val_loss,
            writer=writer,
        )

        val_loss = evaluate_loss(
            model=model,
            data_loader=data_loader_val,
            device=device,
            print_freq=args.print_freq,
        )

        eval_metrics = None

        should_eval_metrics = (
            not args.disable_eval_metrics
            and args.eval_metrics_every > 0
            and (
                epoch % args.eval_metrics_every == 0
                or epoch == args.epochs - 1
            )
        )

        if should_eval_metrics:
            eval_metrics = evaluate_detection_metrics(
                model=model,
                data_loader=data_loader_val,
                device=device,
                image_size=image_size,
                args=args,
                epoch=epoch,
                output_dir=output_dir,
                writer=writer,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best = True
        else:
            save_best = False

        writer.add_scalar("train_epoch/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/best_val_loss", best_val_loss, epoch)

        checkpoint = build_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            best_val_loss=best_val_loss,
            scaler=scaler,
            iter_in_epoch=None,
            finished_epoch=True,
        )
        checkpoint["best_map50"] = best_map50
        checkpoint["last_eval_metrics"] = eval_metrics

        atomic_save(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch}.pth"))
        atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pth"))

        if save_best:
            atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_best_val_loss.pth"))


        if eval_metrics is not None:
            current_map50 = eval_metrics["map50"]

            if current_map50 > best_map50:
                best_map50 = current_map50
                checkpoint["best_map50"] = best_map50
                atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_best_map50.pth"))
                print(f"[checkpoint] saved best mAP50 checkpoint: {best_map50:.6f}")

        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        if eval_metrics is None:
            det_msg = ""
        else:
            det_msg = (
                f", mAP50={eval_metrics['map50']:.4f}, "
                f"P={eval_metrics['precision_micro']:.4f}, "
                f"R={eval_metrics['recall_micro']:.4f}, "
                f"IoU={eval_metrics['mean_best_iou']:.4f}"
            )

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}, "
            f"best_map50={best_map50:.4f}, "
            f"global_step={global_step}{det_msg}, time={total_time}"
        )

    writer.close()
    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
