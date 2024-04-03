_base_ = [
    "../../../models/upernet_swin.py",
    "../../../datasets/ems.py",
]
name = "upernet-swing_10ep"
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
crop_size = (512, 512)
data_preprocessor = None
checkpoint_file = (
    "pretrained\swin_tiny_patch4_window7_224_20220317-1cdeb081.pth"  # noqa
)
reprojected = True
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        pretrained=checkpoint_file,
        # init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        in_channels=12,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
    ),
    decode_head=dict(in_channels=[96, 192, 384, 768], num_classes=512),
    auxiliary_head=dict(in_channels=384, num_classes=1),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]


##configs\single\pretrained\swin\swin.py
