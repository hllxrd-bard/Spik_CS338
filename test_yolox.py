import torch
from torch.utils.data import DataLoader

from evcivil_dataset_yolox import EVCivilDetectionDataset, detection_collate_yolox_fn
from model_det_yolox import SpikformerYOLOXDetector
from spikingjelly.clock_driven import functional

root = "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset"

ds = EVCivilDetectionDataset(
    root=root,
    split="train",
    T=16,
    input_size=(256, 256),
    window_ms=30,
    max_samples=8,
    verbose=True,
)

loader = DataLoader(
    ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=detection_collate_yolox_fn,
)

device = "cuda:0"
model = SpikformerYOLOXDetector(num_classes=2).to(device)
model.train()

images, targets, labels = next(iter(loader))
images = images.to(device).float()
labels = labels.to(device).float()

outputs, losses = model(images, labels=labels)

print("images:", images.shape)
print("labels:", labels.shape)
print("outputs:", outputs.shape)
print("losses:", {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()})

functional.reset_net(model)