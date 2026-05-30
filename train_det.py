import argparse
import datetime
import os
import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader
# from torchvision.ops import generalized_box_iou_loss

from spikingjelly.clock_driven import functional

import utils
from evcivil_dataset import EVCivilDetectionDataset, detection_collate_fn
from model_det import SpikformerDetectorOptionA

def generalized_box_iou_loss_local(boxes1, boxes2, reduction="mean", eps=1e-7):
    """
    boxes1, boxes2: [N, 4] in xyxy format, normalized or pixel.
    Returns GIoU loss = 1 - GIoU.
    """
    if boxes1.numel() == 0:
        loss = boxes1.sum() * 0.0
        return loss

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1 + area2 - inter + eps
    iou = inter / union

    cx1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    cy1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    cx2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    cy2 = torch.max(boxes1[:, 3], boxes2[:, 3])

    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps

    giou = iou - (c_area - union) / c_area
    loss = 1.0 - giou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unknown reduction: {reduction}")




def parse_args():
    parser = argparse.ArgumentParser("ev-CIVIL Spikformer detection Option A")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./logs_evcivil_det")

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

    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--max-val-sequences", type=int, default=None)

    parser.add_argument("--save-every-iters", type=int, default=2000)
    parser.add_argument("--keep-last-k-iters", type=int, default=3)
    parser.add_argument("--log-every-iters", type=int, default=50)

    return parser.parse_args()


def decode_boxes(raw_box, grid_h, grid_w):
    """
    raw_box: [B, 4, Gh, Gw]
    Returns normalized xyxy boxes: [B, 4, Gh, Gw], values in [0, 1].
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


class OptionADetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes=2,
        obj_weight=1.0,
        box_weight=5.0,
        cls_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight

        self.obj_loss_fn = nn.BCEWithLogitsLoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, pred, targets, image_size):
        """
        pred: [B, 5 + C, Gh, Gw]
        targets: list of dicts
        image_size: (H, W)

        Log key convention follows YOLOX version:
            conf_loss = objectness loss
            giou_loss = weighted box loss
            cls_loss  = classification loss
            l1_loss   = 0 for Option A
            num_fg    = number of positive cells per GT, roughly comparable to YOLOX num_fg
        """

        B, _, Gh, Gw = pred.shape
        img_h, img_w = image_size
        device = pred.device

        raw_box = pred[:, 0:4]
        obj_logit = pred[:, 4:5]
        cls_logit = pred[:, 5:]

        obj_target = torch.zeros((B, 1, Gh, Gw), device=device)
        box_target = torch.zeros((B, 4, Gh, Gw), device=device)
        cls_target = torch.zeros((B, Gh, Gw), dtype=torch.long, device=device)
        pos_mask = torch.zeros((B, Gh, Gw), dtype=torch.bool, device=device)

        total_gt = 0

        for b, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)

            if boxes.numel() == 0:
                continue

            total_gt += boxes.shape[0]

            # pixel xyxy -> normalized xyxy
            boxes_norm = boxes.clone()
            boxes_norm[:, [0, 2]] /= img_w
            boxes_norm[:, [1, 3]] /= img_h
            boxes_norm = boxes_norm.clamp(0, 1)

            cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
            cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2

            gx = torch.clamp((cx * Gw).long(), 0, Gw - 1)
            gy = torch.clamp((cy * Gh).long(), 0, Gh - 1)

            for i in range(len(boxes_norm)):
                y, x = gy[i], gx[i]

                # Naive one-cell assignment.
                # If multiple boxes fall into the same cell, this overwrites.
                obj_target[b, 0, y, x] = 1.0
                box_target[b, :, y, x] = boxes_norm[i]
                cls_target[b, y, x] = labels[i].long()
                pos_mask[b, y, x] = True

        obj_loss = self.obj_loss_fn(obj_logit, obj_target)

        if pos_mask.any():
            pred_xyxy = decode_boxes(raw_box, Gh, Gw)
            pred_pos = pred_xyxy.permute(0, 2, 3, 1)[pos_mask]
            tgt_pos = box_target.permute(0, 2, 3, 1)[pos_mask]

            raw_box_loss = generalized_box_iou_loss_local(
                pred_pos,
                tgt_pos,
                reduction="mean",
            )

            cls_pos = cls_logit.permute(0, 2, 3, 1)[pos_mask]
            cls_tgt = cls_target[pos_mask]
            cls_loss = self.cls_loss_fn(cls_pos, cls_tgt)
        else:
            raw_box_loss = raw_box.sum() * 0.0
            cls_loss = cls_logit.sum() * 0.0

        giou_loss = self.box_weight * raw_box_loss
        conf_loss = self.obj_weight * obj_loss
        cls_loss_weighted = self.cls_weight * cls_loss

        total = giou_loss + conf_loss + cls_loss_weighted

        num_pos = pos_mask.sum().float()
        denom = max(float(total_gt), 1.0)
        num_fg = num_pos / denom

        zero = total.detach() * 0.0

        loss_dict = {
            "loss": total.detach(),
            "giou_loss": giou_loss.detach(),
            "conf_loss": conf_loss.detach(),
            "cls_loss": cls_loss_weighted.detach(),
            "l1_loss": zero,
            "num_fg": num_fg.detach(),
        }

        return total, loss_dict


@torch.no_grad()
def evaluate_proxy(model, criterion, data_loader, device, print_freq=100, image_size=(256, 256)):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True).float()

        pred = model(images)
        loss, loss_dict = criterion(pred, targets, image_size)

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
    old_files = files[:-keep_last_k]

    for p in old_files:
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


def to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


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
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler,
    image_size,
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

    for iter_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        images = images.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with amp.autocast():
                pred = model(images)
                loss, loss_dict = criterion(pred, targets, image_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(images)
            loss, loss_dict = criterion(pred, targets, image_size)
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        global_step += 1

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

            print(f"[checkpoint] saved latest at epoch={epoch}, iter={iter_idx}, global_step={global_step}")

        batch_size = images.shape[0]

        log_loss_dict(
            metric_logger,
            loss_dict,
            lr=optimizer.param_groups[0]["lr"],
        )
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg, global_step


def main(args):
    device = torch.device(args.device)
    image_size = (args.input_height, args.input_width)

    output_dir = os.path.join(
        args.output_dir,
        f"optionA_T{args.T}_{args.input_height}x{args.input_width}_lr{args.lr}",
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
    collate_fn=detection_collate_fn,
    persistent_workers=args.workers > 0,
    prefetch_factor=2 if args.workers > 0 else None,
)

    data_loader_val = DataLoader(
    dataset_val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=detection_collate_fn,
    persistent_workers=args.workers > 0,
    prefetch_factor=2 if args.workers > 0 else None,
)

    print("Creating model")
    model = SpikformerDetectorOptionA(
        num_classes=args.num_classes,
        in_channels=2,
        embed_dims=256,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_params)

    criterion = OptionADetectionLoss(num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = amp.GradScaler() if args.amp else None

    start_epoch = 0
    best_val_loss = float("inf")

    global_step = 0

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
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            scaler=scaler,
            image_size=image_size,
            output_dir=output_dir,
            args=args,
            global_step=global_step,
            best_val_loss=best_val_loss,
            writer=writer,
        )

        val_loss = evaluate_proxy(
            model=model,
            criterion=criterion,
            data_loader=data_loader_val,
            device=device,
            print_freq=args.print_freq,
            image_size=image_size,
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

        atomic_save(checkpoint, os.path.join(output_dir, f"checkpoint_{epoch}.pth"))
        atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_latest.pth"))

        if save_best:
            atomic_save(checkpoint, os.path.join(output_dir, "checkpoint_best_val_loss.pth"))

        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}, "
            f"time={total_time}"
        )

    print("Done")

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)

#     python train_det.py \
#   --data-path /AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset \
#   --device cuda:2 \
#   --batch-size 4 \
#   --workers 2 \
#   --epochs 20 \
#   --T 16 \
#   --input-height 256 \
#   --input-width 256 \
#   --window-ms 30 \
#   --lr 1e-4 \
#   --max-train-sequences 80 \
#   --max-val-sequences 20  \
#   --output-dir ./logs_evcivil_det \