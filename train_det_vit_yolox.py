
import argparse
import datetime
import os
import time
from pathlib import Path

import torch
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.clock_driven import functional

import utils
from evcivil_dataset_yolox import EVCivilDetectionDataset, detection_collate_yolox_fn
from model_det_vit_yolox import ViTYOLOXDetector


def parse_args():
    parser = argparse.ArgumentParser("ev-CIVIL Spikformer + YOLOX detection")

    parser.add_argument("--data-path", type=str, required=True)
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


def main(args):
    device = torch.device(args.device)
    image_size = (args.input_height, args.input_width)

    output_dir = os.path.join(
        args.output_dir,
        f"spikformer_yolox_T{args.T}_{args.input_height}x{args.input_width}_lr{args.lr}",
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
    model = ViTYOLOXDetector(
    num_classes=args.num_classes,
    T=args.T,
    img_size=image_size,
    patch_size=16,
    embed_dim=256,
    depth=4,
    num_heads=8,
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
            f"global_step={global_step}, time={total_time}"
        )

    writer.close()
    print("Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
