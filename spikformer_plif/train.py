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
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode

import utils
from spikformer.evcivil.evcivil_dataset_loader import EVCivilDetectionDataset, detection_collate_yolox_fn
from model_det_yolox_plif import SpikformerYOLOXDetectorPLIF
from val_det import decode_yolox_predictions, compute_ap_metrics


def parse_args():
    parser = argparse.ArgumentParser("ev-CIVIL Spikformer + YOLOX detection (PLIF)")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./logs_evcivil_det_yolox_plif")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--window-ms", type=float, default=30.0)

    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--class-names", type=str, default="crack,spalling")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True)

    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)

    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--log-every-iters", type=int, default=50)
    parser.add_argument("--save-every-iters", type=int, default=1000)
    parser.add_argument("--keep-last-k-iters", type=int, default=3)

    parser.add_argument("--resume", type=str, default="")

    return parser.parse_args()


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


def append_eval_log(output_dir, summary_lines):
    log_path = Path(output_dir) / "log.logs"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def register_spike_monitor(model):
    stats = {
        "spike_sum": 0.0,
        "spike_count": 0,
    }

    def hook(_module, _inp, out):
        if not torch.is_tensor(out):
            return
        stats["spike_sum"] += out.detach().float().sum().item()
        stats["spike_count"] += out.numel()

    handles = []
    for module in model.modules():
        if isinstance(module, MultiStepParametricLIFNode):
            handles.append(module.register_forward_hook(hook))

    return stats, handles


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
def evaluate_loss(model, data_loader, device, print_freq, args, output_dir, epoch):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    all_predictions = []
    all_targets = []
    image_size = (args.input_height, args.input_width)
    seen = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    spike_stats, spike_handles = register_spike_monitor(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_start = time.time()

    for images, targets, labels in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).float()

        outputs, loss_dict = model(images, labels=labels)
        if loss_dict is None:
            raise RuntimeError("YOLOXHead returned loss_dict=None in eval. Check yolox/head.py patch.")

        batch_preds = decode_yolox_predictions(
            outputs=outputs,
            image_size=image_size,
            score_thr=args.score_thr,
            nms_thr=args.nms_thr,
            max_det=args.max_det,
            num_classes=args.num_classes,
        )

        for i in range(len(batch_preds)):
            pred_i = batch_preds[i]
            tgt_i = {
                "boxes": targets[i]["boxes"].detach().cpu(),
                "labels": targets[i]["labels"].detach().cpu(),
            }
            all_predictions.append(pred_i)
            all_targets.append(tgt_i)
            seen += 1

        functional.reset_net(model)

        log_loss_dict(metric_logger, loss_dict)

    metric_logger.synchronize_between_processes()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    eval_time = time.time() - eval_start

    for handle in spike_handles:
        handle.remove()

    metrics = compute_ap_metrics(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=args.num_classes,
        iou_thr=args.iou_thr,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_mem_mb = None
    if device.type == "cuda":
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    firing_rate = 0.0
    if spike_stats["spike_count"] > 0:
        firing_rate = spike_stats["spike_sum"] / spike_stats["spike_count"]

    sops_est = firing_rate * float(n_params)
    latency_ms = 0.0
    if seen > 0:
        latency_ms = (eval_time / seen) * 1000.0

    print(
        f"* Val loss={metric_logger.loss.global_avg:.4f}, "
        f"giou={metric_logger.giou_loss.global_avg:.4f}, "
        f"conf={metric_logger.conf_loss.global_avg:.4f}, "
        f"cls={metric_logger.cls_loss.global_avg:.4f}, "
        f"num_fg={metric_logger.num_fg.global_avg:.4f}"
    )

    class_names = args.class_names.split(",") if args.class_names else []
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"Epoch {epoch} evaluation")
    summary_lines.append("=" * 80)
    summary_lines.append(f"val_loss: {metric_logger.loss.global_avg:.6f}")
    summary_lines.append(f"params: {n_params}")
    summary_lines.append(f"latency_ms: {latency_ms:.3f}")
    if max_mem_mb is None:
        summary_lines.append("max_mem_mb: N/A")
    else:
        summary_lines.append(f"max_mem_mb: {max_mem_mb:.2f}")
    summary_lines.append(f"firing_rate: {firing_rate:.6f}")
    summary_lines.append(f"sops_est: {sops_est:.3e}")
    summary_lines.append(f"mAP50: {metrics['map50']:.6f}")
    summary_lines.append(f"precision_micro: {metrics['precision_micro']:.6f}")
    summary_lines.append(f"recall_micro: {metrics['recall_micro']:.6f}")
    summary_lines.append(f"f1_micro: {metrics['f1_micro']:.6f}")
    summary_lines.append(f"num_gt: {metrics['num_gt']}")
    summary_lines.append(f"num_pred: {metrics['num_pred']}")
    summary_lines.append(f"tp: {metrics['tp']}")
    summary_lines.append(f"fp: {metrics['fp']}")
    summary_lines.append("")
    summary_lines.append("Per-class:")
    for cls_id, cls_m in metrics["classes"].items():
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else cls_id
        summary_lines.append(
            f"  class {cls_id} ({name}): "
            f"AP50={cls_m['ap50']:.6f}, "
            f"P={cls_m['precision']:.6f}, "
            f"R={cls_m['recall']:.6f}, "
            f"GT={cls_m['num_gt']}, "
            f"Pred={cls_m['num_pred']}, "
            f"TP={cls_m['tp']}, FP={cls_m['fp']}"
        )

    summary = "\n".join(summary_lines)
    print(summary)
    append_eval_log(output_dir, summary_lines)

    return metric_logger.loss.global_avg


def main(args):
    device = torch.device(args.device)
    image_size = (args.input_height, args.input_width)

    output_dir = os.path.join(
        args.output_dir,
        f"spikformer_yolox_plif_T{args.T}_{args.input_height}x{args.input_width}_lr{args.lr}",
    )
    utils.mkdir(output_dir)

    writer = SummaryWriter(os.path.join(output_dir, "tb"))

    print(args)
    print("Building datasets")

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
    model = SpikformerYOLOXDetectorPLIF(
        num_classes=args.num_classes,
        in_channels=2,
        embed_dims=256,
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

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

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
            args=args,
            output_dir=output_dir,
            epoch=epoch,
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

        atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pth"))
        atomic_save(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch}.pth"))

        if save_best:
            atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_best_val_loss.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
