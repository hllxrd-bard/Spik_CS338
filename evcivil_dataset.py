from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EVCivilDetectionDataset(Dataset):
    """
    ev-CIVIL detection dataset.

    Expected sequence files:
        *_events.h5:
            event_timestamp, x, y, polarity
        *_label.npy:
            [timestamp, class_id, bbox_x, bbox_y, bbox_w, bbox_h]

    Output:
        image: Tensor [T, 2, input_h, input_w]
        target:
            boxes:  Tensor [N, 4] in resized xyxy pixel coordinates
            labels: Tensor [N]
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_ratio: float = 0.8,
        T: int = 16,
        input_size: Tuple[int, int] = (256, 256),
        window_ms: float = 30.0,
        use_field: bool = True,
        use_lab: bool = True,
        seed: int = 2021,
        min_boxes: int = 1,
        verbose: bool = True,
        max_samples=None,
        max_sequences=None,
    ):
        self.root = Path(root)
        self.split = split
        self.train_ratio = train_ratio
        self.T = int(T)
        self.input_h = int(input_size[0])
        self.input_w = int(input_size[1])
        self.window_us = int(window_ms * 1000)
        self.use_field = use_field
        self.use_lab = use_lab
        self.seed = seed
        self.min_boxes = min_boxes

        if self.input_h % 16 != 0 or self.input_w % 16 != 0:
            raise ValueError(
                f"input_size must be divisible by 16, got {(self.input_h, self.input_w)}"
            )

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.seq_items = self._collect_sequences()
        self.seq_items = self._split_sequences(self.seq_items)

        if max_sequences is not None:
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(self.seq_items))
            rng.shuffle(indices)
            indices = indices[:max_sequences]
            indices = sorted(indices)
            self.seq_items = [self.seq_items[i] for i in indices]

        self.samples = self._build_samples()

        if max_samples is not None:
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(self.samples))
            rng.shuffle(indices)
            indices = indices[:max_samples]
            indices = sorted(indices)
            self.samples = [self.samples[i] for i in indices]

        if verbose:
            print(f"[EVCivilDetectionDataset] root={self.root}")
            print(f"[EVCivilDetectionDataset] split={self.split}")
            print(f"[EVCivilDetectionDataset] num_sequences={len(self.seq_items)}")
            print(f"[EVCivilDetectionDataset] num_samples={len(self.samples)}")
            print(f"[EVCivilDetectionDataset] T={self.T}, input={(self.input_h, self.input_w)}")
            print(f"[EVCivilDetectionDataset] window_us={self.window_us}")

        if len(self.samples) == 0:
            raise RuntimeError("No detection samples found. Check dataset root and label files.")

    def _collect_sequences(self) -> List[Dict]:
        label_files = sorted(self.root.rglob("*_label.npy"))

        items = []
        for label_path in label_files:
            # For event-based detection, skip frame-only labels.
            if label_path.name.endswith("_frame_label.npy"):
                continue

            parts = label_path.parts
            is_field = "Field_dataset" in parts
            is_lab = "Laboratory_dataset" in parts

            if is_field and not self.use_field:
                continue
            if is_lab and not self.use_lab:
                continue

            stem = label_path.name.replace("_label.npy", "")
            seq_dir = label_path.parent
            event_path = seq_dir / f"{stem}_events.h5"

            if not event_path.exists():
                # Some names can be unusual; fallback to glob.
                event_candidates = list(seq_dir.glob("*_events.h5"))
                if len(event_candidates) == 0:
                    continue
                event_path = event_candidates[0]

            items.append({
                "seq_dir": seq_dir,
                "event_path": event_path,
                "label_path": label_path,
                "seq_name": stem,
                "domain": "field" if is_field else "lab" if is_lab else "unknown",
            })

        return items

    def _split_sequences(self, items: List[Dict]) -> List[Dict]:
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(items))
        rng.shuffle(indices)

        cut = int(len(indices) * self.train_ratio)
        if self.split == "train":
            keep = set(indices[:cut].tolist())
        elif self.split in ["val", "test"]:
            keep = set(indices[cut:].tolist())
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return [item for i, item in enumerate(items) if i in keep]

    def _build_samples(self) -> List[Dict]:
        samples = []

        for item in self.seq_items:
            labels = np.load(item["label_path"], allow_pickle=True)

            if labels.ndim != 2 or labels.shape[1] < 6:
                continue

            # Unique annotation timestamps become sample centers.
            timestamps = np.unique(labels[:, 0].astype(np.int64))

            for ts in timestamps:
                lo = ts - self.window_us // 2
                hi = ts + self.window_us // 2

                mask = (labels[:, 0] >= lo) & (labels[:, 0] <= hi)
                active = labels[mask]

                if len(active) < self.min_boxes:
                    continue

                samples.append({
                    "event_path": str(item["event_path"]),
                    "label_path": str(item["label_path"]),
                    "timestamp": int(ts),
                    "t0": int(lo),
                    "t1": int(hi),
                    "seq_name": item["seq_name"],
                    "domain": item["domain"],
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        events = self._load_events_window(
            sample["event_path"],
            sample["t0"],
            sample["t1"],
        )

        image = self._events_to_tensor(events, sample["t0"], sample["t1"])

        labels_np = np.load(sample["label_path"], allow_pickle=True)
        mask = (labels_np[:, 0] >= sample["t0"]) & (labels_np[:, 0] <= sample["t1"])
        active = labels_np[mask]

        boxes, labels = self._labels_to_target(active)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.long),
            "timestamp": torch.tensor([sample["timestamp"]], dtype=torch.long),
        }

        return image, target

    def _load_events_window(self, event_path: str, t0: int, t1: int) -> Dict[str, np.ndarray]:
        with h5py.File(event_path, "r") as f:
            ts = f["event_timestamp"][:]

            left = np.searchsorted(ts, t0, side="left")
            right = np.searchsorted(ts, t1, side="right")

            return {
                "t": ts[left:right].astype(np.int64),
                "x": f["x"][left:right].astype(np.float32),
                "y": f["y"][left:right].astype(np.float32),
                "p": f["polarity"][left:right].astype(np.int64),
            }

    def _events_to_tensor(self, events: Dict[str, np.ndarray], t0: int, t1: int) -> torch.Tensor:
        frames = np.zeros((self.T, 2, self.input_h, self.input_w), dtype=np.float32)

        if len(events["t"]) == 0:
            return torch.from_numpy(frames)

        # Original DAVIS346 resolution from inspect/paper: W=346, H=260.
        orig_w, orig_h = 346.0, 260.0

        x = np.clip((events["x"] / orig_w) * self.input_w, 0, self.input_w - 1).astype(np.int64)
        y = np.clip((events["y"] / orig_h) * self.input_h, 0, self.input_h - 1).astype(np.int64)
        p = np.clip(events["p"], 0, 1).astype(np.int64)

        denom = max(t1 - t0, 1)
        tb = ((events["t"] - t0) / denom * self.T).astype(np.int64)
        tb = np.clip(tb, 0, self.T - 1)

        np.add.at(frames, (tb, p, y, x), 1.0)

        # Binary spikes are safer for a first SNN baseline.
        frames = (frames > 0).astype(np.float32)

        return torch.from_numpy(frames)

    def _labels_to_target(self, active: np.ndarray):
        boxes = []
        labels = []

        scale_x = self.input_w / 346.0
        scale_y = self.input_h / 260.0

        for row in active:
            cls = int(row[1])
            x, y, w, h = row[2], row[3], row[4], row[5]

            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            # Clip boxes. Some labels may exceed sensor boundary.
            x1 = float(np.clip(x1, 0, self.input_w - 1))
            y1 = float(np.clip(y1, 0, self.input_h - 1))
            x2 = float(np.clip(x2, 0, self.input_w - 1))
            y2 = float(np.clip(y2, 0, self.input_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
        )


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    targets = list(targets)
    return images, targets