_base_ = [
    "../../models/upernet_rn50_swin.py",
    "../../datasets/ems.py",
]
name = "upernet-swin"
trainer = dict(
    max_epochs=10,
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices="auto",
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=8,
    derivative_idx=True,
)
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices="auto",
)
reprojected = True
