from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DETRACDetectionDataset(Dataset):
    """
    UA-DETRAC frame-sequence detection dataset.

    This dataset mimics EVCivilDetectionDataset output:

        image: Tensor [T, 2, input_h, input_w]
        target:
            boxes:  Tensor [N, 4] in resized xyxy pixel coordinates
            labels: Tensor [N]

    For first sanity check, use representation="grayscale_dup":
        channel 0 = grayscale frame
        channel 1 = grayscale frame

    This keeps compatibility with current Spikformer/ViT YOLOX models:
        x: [B, T, 2, H, W]
    """

    VEHICLE_CLASS_MAP = {
    "car": 0,
    "van": 1,
    "bus": 2,
}

    def __init__(
        self,
        root: str,
        split: str = "train",
        T: int = 16,
        input_size: Tuple[int, int] = (256, 256),
        frame_stride: int = 1,
        representation: str = "grayscale_dup",
        one_class: bool = True,
        train_ratio: float = 0.8,
        seed: int = 2021,
        min_boxes: int = 1,
        max_samples: Optional[int] = None,
        max_sequences: Optional[int] = None,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.T = int(T)
        self.input_h = int(input_size[0])
        self.input_w = int(input_size[1])
        self.frame_stride = int(frame_stride)
        self.representation = representation
        self.one_class = bool(one_class)
        self.train_ratio = float(train_ratio)
        self.seed = int(seed)
        self.min_boxes = int(min_boxes)

        if self.input_h % 16 != 0 or self.input_w % 16 != 0:
            raise ValueError(
                f"input_size must be divisible by 16, got {(self.input_h, self.input_w)}"
            )

        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {self.frame_stride}")

        valid_reps = {"grayscale_dup", "grayscale_zero", "frame_diff"}
        if self.representation not in valid_reps:
            raise ValueError(f"Unknown representation={self.representation}, valid={valid_reps}")

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        self.images_dir = self._resolve_nested_dir("DETRAC-Images")
        self.train_ann_dir = self._resolve_nested_dir("DETRAC-Train-Annotations-XML")
        self.test_ann_dir = self._resolve_nested_dir("DETRAC-Test-Annotations-XML")

        self.seq_items = self._collect_sequences()
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
            print(f"[DETRACDetectionDataset] root={self.root}")
            print(f"[DETRACDetectionDataset] split={self.split}")
            print(f"[DETRACDetectionDataset] images_dir={self.images_dir}")
            print(f"[DETRACDetectionDataset] train_ann_dir={self.train_ann_dir}")
            print(f"[DETRACDetectionDataset] test_ann_dir={self.test_ann_dir}")
            print(f"[DETRACDetectionDataset] num_sequences={len(self.seq_items)}")
            print(f"[DETRACDetectionDataset] num_samples={len(self.samples)}")
            print(f"[DETRACDetectionDataset] T={self.T}, input={(self.input_h, self.input_w)}")
            print(f"[DETRACDetectionDataset] frame_stride={self.frame_stride}")
            print(f"[DETRACDetectionDataset] representation={self.representation}")
            print(f"[DETRACDetectionDataset] one_class={self.one_class}")

        if len(self.samples) == 0:
            raise RuntimeError("No DETRAC samples found. Check root, XML files, and image folders.")

    def _resolve_nested_dir(self, name: str) -> Path:
        """
        Handles both:
            root/DETRAC-Images/MVI_xxx
        and:
            root/DETRAC-Images/DETRAC-Images/MVI_xxx
        """
        p1 = self.root / name
        p2 = self.root / name / name

        if p2.exists():
            return p2
        if p1.exists():
            return p1

        candidates = list(self.root.rglob(name))
        candidates = [p for p in candidates if p.is_dir()]
        if candidates:
            candidates = sorted(candidates, key=lambda p: len(str(p)))
            return candidates[0]

        raise FileNotFoundError(f"Cannot find directory {name} under {self.root}")

    def _collect_sequences(self) -> List[Dict]:
        if self.split == "train":
            ann_dir = self.train_ann_dir
        elif self.split in {"val", "test"}:
            ann_dir = self.test_ann_dir
        else:
            raise ValueError(f"Unknown split: {self.split}")

        xml_files = sorted(ann_dir.glob("*.xml"))

        # Fallback: if test annotations are missing, split train XML by sequence.
        if len(xml_files) == 0 and self.split in {"val", "test"}:
            print("[DETRACDetectionDataset] No test XML found. Falling back to train-ratio split.")
            all_xml = sorted(self.train_ann_dir.glob("*.xml"))
            return self._split_xml_files(all_xml)

        items = []
        for xml_path in xml_files:
            seq_name = xml_path.stem
            seq_img_dir = self.images_dir / seq_name

            if not seq_img_dir.exists():
                # Some XML sequence names may differ from folder names.
                matches = list(self.images_dir.glob(seq_name))
                if len(matches) == 0:
                    continue
                seq_img_dir = matches[0]

            frame_paths = self._list_frame_paths(seq_img_dir)
            if len(frame_paths) == 0:
                continue

            items.append(
                {
                    "seq_name": seq_name,
                    "xml_path": xml_path,
                    "img_dir": seq_img_dir,
                    "frame_paths": frame_paths,
                }
            )

        return items

    def _split_xml_files(self, xml_files: List[Path]) -> List[Dict]:
        rng = np.random.RandomState(self.seed)
        indices = np.arange(len(xml_files))
        rng.shuffle(indices)

        cut = int(len(indices) * self.train_ratio)
        if self.split == "train":
            keep = set(indices[:cut].tolist())
        else:
            keep = set(indices[cut:].tolist())

        items = []
        for i, xml_path in enumerate(xml_files):
            if i not in keep:
                continue

            seq_name = xml_path.stem
            seq_img_dir = self.images_dir / seq_name
            if not seq_img_dir.exists():
                continue

            frame_paths = self._list_frame_paths(seq_img_dir)
            if len(frame_paths) == 0:
                continue

            items.append(
                {
                    "seq_name": seq_name,
                    "xml_path": xml_path,
                    "img_dir": seq_img_dir,
                    "frame_paths": frame_paths,
                }
            )

        return items

    @staticmethod
    def _list_frame_paths(seq_img_dir: Path) -> List[Path]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        paths = []
        for ext in exts:
            paths.extend(seq_img_dir.glob(ext))
        return sorted(paths)

    def _build_samples(self) -> List[Dict]:
        samples = []

        for item in self.seq_items:
            frame_to_boxes = self._parse_xml(item["xml_path"])

            for frame_num in sorted(frame_to_boxes.keys()):
                targets = frame_to_boxes[frame_num]

                if len(targets) < self.min_boxes:
                    continue

                frame_idx = frame_num - 1
                if frame_idx < 0 or frame_idx >= len(item["frame_paths"]):
                    continue

                samples.append(
                    {
                        "seq_name": item["seq_name"],
                        "frame_num": frame_num,
                        "frame_idx": frame_idx,
                        "frame_paths": item["frame_paths"],
                        "targets": targets,
                    }
                )

        return samples

    def _parse_xml(self, xml_path: Path) -> Dict[int, List[Dict]]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        frame_to_boxes: Dict[int, List[Dict]] = {}

        for frame_node in root.findall("frame"):
            frame_num = int(frame_node.attrib["num"])
            targets = []

            target_list = frame_node.find("target_list")
            if target_list is None:
                frame_to_boxes[frame_num] = targets
                continue

            for target_node in target_list.findall("target"):
                box_node = target_node.find("box")
                if box_node is None:
                    continue

                left = float(box_node.attrib["left"])
                top = float(box_node.attrib["top"])
                width = float(box_node.attrib["width"])
                height = float(box_node.attrib["height"])

                attr_node = target_node.find("attribute")
                vehicle_type = "unknown"

                if attr_node is not None:
                    vehicle_type = attr_node.attrib.get("vehicle_type", "unknown").lower()

                if self.one_class:
                    label = 0
                else:
                    # Multi-class mode: only keep car / van / bus.
                    # Skip unknown / others to avoid label id out of range.
                    if vehicle_type not in self.VEHICLE_CLASS_MAP:
                        continue

                    label = self.VEHICLE_CLASS_MAP[vehicle_type]

                if width <= 0 or height <= 0:
                    continue

                targets.append(
                    {
                        "bbox_xywh": [left, top, width, height],
                        "label": label,
                        "vehicle_type": vehicle_type,
                    }
                )

            frame_to_boxes[frame_num] = targets

        return frame_to_boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image = self._load_temporal_tensor(
            frame_paths=sample["frame_paths"],
            center_idx=sample["frame_idx"],
        )

        boxes, labels = self._targets_to_tensor(
            targets=sample["targets"],
            frame_path=sample["frame_paths"][sample["frame_idx"]],
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.long),
            "frame_num": torch.tensor([sample["frame_num"]], dtype=torch.long),
            "seq_name": sample["seq_name"],
        }

        return image, target

    def _temporal_indices(self, center_idx: int, n_frames: int) -> List[int]:
        # Use T frames ending at current frame:
        # center_idx - (T-1)*stride, ..., center_idx
        indices = []
        for k in range(self.T):
            offset = (self.T - 1 - k) * self.frame_stride
            idx = center_idx - offset
            idx = max(0, min(idx, n_frames - 1))
            indices.append(idx)
        return indices

    def _load_gray_resized(self, path: Path) -> np.ndarray:
        with Image.open(path) as img:
            img = img.convert("L")
            orig_w, orig_h = img.size
            img = img.resize((self.input_w, self.input_h), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _load_temporal_tensor(self, frame_paths: List[Path], center_idx: int) -> torch.Tensor:
        n_frames = len(frame_paths)
        indices = self._temporal_indices(center_idx, n_frames)

        gray_frames = [
            self._load_gray_resized(frame_paths[i])
            for i in indices
        ]

        frames = np.zeros((self.T, 2, self.input_h, self.input_w), dtype=np.float32)

        if self.representation == "grayscale_dup":
            for t, g in enumerate(gray_frames):
                frames[t, 0] = g
                frames[t, 1] = g

        elif self.representation == "grayscale_zero":
            for t, g in enumerate(gray_frames):
                frames[t, 0] = g
                frames[t, 1] = 0.0

        elif self.representation == "frame_diff":
            prev = gray_frames[0]
            for t, cur in enumerate(gray_frames):
                diff = cur - prev
                frames[t, 0] = np.clip(diff, 0.0, 1.0)
                frames[t, 1] = np.clip(-diff, 0.0, 1.0)
                prev = cur

        else:
            raise ValueError(f"Unknown representation={self.representation}")

        return torch.from_numpy(frames)

    def _targets_to_tensor(self, targets: List[Dict], frame_path: Path):
        with Image.open(frame_path) as img:
            orig_w, orig_h = img.size

        scale_x = self.input_w / float(orig_w)
        scale_y = self.input_h / float(orig_h)

        boxes = []
        labels = []

        for obj in targets:
            x, y, w, h = obj["bbox_xywh"]
            label = int(obj["label"])

            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            x1 = float(np.clip(x1, 0, self.input_w - 1))
            y1 = float(np.clip(y1, 0, self.input_h - 1))
            x2 = float(np.clip(x2, 0, self.input_w - 1))
            y2 = float(np.clip(y2, 0, self.input_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(label)

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