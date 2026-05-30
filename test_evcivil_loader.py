from evcivil_dataset import EVCivilDetectionDataset

root = "/AIClub_NAS/WorkingSpace/Personal/chinhnm/HLLXRD/hangdv_minion/dataset"

ds = EVCivilDetectionDataset(
    root=root,
    split="train",
    T=16,
    input_size=(256, 256),
    window_ms=30,
    verbose=True,
)

x, target = ds[0]

print("x:", x.shape, x.dtype, x.min().item(), x.max().item())
print("boxes:", target["boxes"].shape, target["boxes"][:5])
print("labels:", target["labels"].shape, target["labels"][:5])