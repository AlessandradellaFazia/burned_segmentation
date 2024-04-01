_base_ = ["./segformer_mit-b0.py"]

model = dict(
    backbone=dict(embed_dims=64, num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]),
)
