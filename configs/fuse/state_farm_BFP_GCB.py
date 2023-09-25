_base_ = '../cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco_n1.py'
# model settings
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(type='ContextBlock', ratio=0.25),
                stages=(False, True, True, True),
                position='after_conv3')
        ]),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')])