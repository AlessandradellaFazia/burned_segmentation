_base_ = [
    "../../../models/upernet_rn50_6ch.py",
    "../../../datasets/ems.py",
]
name = "upernet-rn50_single_ssl4eo_50ep"
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
)
evaluation = dict(
    precision=16,
    accelerator="gpu",
    strategy=None,
    devices="auto",
)
reprojected = True
loss = "dice"
##configs\single\pretrained\ems_upernet-rn50_single_50ep.py
