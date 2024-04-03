_base_ = [
    "../../../models/upernet_swin.py",
    "../../../datasets/ems.py",
]
name = "upernet-swing2_10ep"
trainer = dict(
    max_epochs=2,
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices=1,
)
data = dict(
    batch_size_train=32,
    batch_size_eval=32,
    num_workers=8,
)
evaluation = dict(
    precision=16,
    accelerator="cpu",
    strategy=None,
    devices=1,
)
checkpoint_file = "pretrained\\mmseg_swin.pth"
model = dict(
    backbone=dict(
        pretrained=checkpoint_file,
        embed_dims=96,
        in_channels=12,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
