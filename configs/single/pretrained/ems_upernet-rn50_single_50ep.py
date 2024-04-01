_base_ = [
    "../../models/upernet_rn50.py",
    "../../datasets/ems.py",
]
name = "upernet-rn50_single_ssl4eo_50ep"
trainer = dict(
    max_epochs=3,
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices="auto",
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=0,
)
evaluation = dict(
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices="auto",
)
