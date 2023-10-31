# YOLOv6m model
model = dict(
    type='YOLOv6m',
    pretrained='./weights/yolov6l6.pt',
    depth_multiple=0.60,
    width_multiple=0.75,
    backbone=dict(
        type='CSPBepBackbone_P6',
        num_repeats=[1, 6, 12, 18, 6, 6],
        out_channels=[32, 64, 128, 256, 512, 768],
        csp_e=float(2)/3,
        fuse_P2=True,
        ),
    neck=dict(
        type='CSPRepBiFPANNeck_P6SIM',
        num_repeats=[12, 12, 12, 12, 12, 12, 12],
        out_channels=[256, 128, 64, 128, 258, 512, 1024],
        csp_e=float(2)/3,
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[64, 128, 256, 512, 1024],  # Add new in_channels for the fifth layer
        num_layers=4,
        begin_indices=24,
        anchors=4,
        anchors_init=[[11, 9, 17, 16, 29, 22],
                    [22, 35, 44, 42, 48, 69],
                    [80, 65, 70, 104, 187, 165],
                    [90, 74, 115, 130, 250, 213]],
        out_indices=[17, 20, 23],  # Add new out_index for the fifth layer
        strides=[8, 16, 32, 64],  # Add new stride for the fifth layer
        atss_warmup_epoch=0,
        iou_type='giou',
        use_dfl=True,
        reg_max=16, # if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 0.8,
            'dfl': 1.0,
        }
    )
)

solver=dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.9,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)