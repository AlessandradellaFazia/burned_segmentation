_base_ = [
    "../../models/upernet_rn50_copy.py",
    "../../datasets/ems.py",
]
name = "upernet-swin"
trainer = dict(
    max_epochs=3,
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices="auto",
)
data = dict(
    batch_size_train=2,
    batch_size_eval=2,
    num_workers=0,
)
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices="auto",
)
##
