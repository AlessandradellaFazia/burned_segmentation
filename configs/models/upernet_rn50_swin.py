# model settings
norm_cfg = dict(type="BN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
model = dict(
    type="CustomEncoderDecoder",
    data_preprocessor=None,
    backbone=dict(
        type="SwinTransformer",
        pretrained="pretrained/mmseg_swin.pth",
        pretrain_img_size=224,
        in_channels=12,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=backbone_norm_cfg,
    ),
    decode_head=dict(
        type="CustomUPerHead",
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
layer_to_reproject = "backbone.patch_embed.proj.weight"
