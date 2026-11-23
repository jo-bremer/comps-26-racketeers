# Training configuration for the original RTMDet-Tiny model.
# We fine-tuned the model to detect tennis balls using our COCO dataset.
# Images are resized and padded to 640x640 before training.
# This version uses 10 epochs, a batch size of 2, and a learning rate of 0.0001.
#
# This config uses the RTMDet-Tiny base configuration from MMDetection.

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = '/home/kaurh/RTMDet/'
classes = ('ball',)
metainfo = dict(classes=classes, palette=[(255, 0, 0)])
load_from = '/home/kaurh/RTMDet/checkpoints/rtmdet_tennis_init.pth'

# Detect only tennis balls
model = dict(
    bbox_head=dict(num_classes=1)
)

# Resize and pad the training images to 640x640
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640),
         pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# Training dataset
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations_full/ball_train_coco.json',
        data_prefix=dict(img='frames_full/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True, min_size=32)
    )
)

# Validation dataset
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations_full/ball_val_coco.json',
        data_prefix=dict(img='frames_full/'),
        metainfo=metainfo
    )
)

# Testing dataset
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations_full/ball_test_coco.json',
        data_prefix=dict(img='frames_full/'),
        metainfo=metainfo
    )
)

# Evaluation
val_evaluator = dict(
    ann_file=data_root + 'annotations_full/ball_val_coco.json'
)

test_evaluator = dict(
    ann_file=data_root + 'annotations_full/ball_test_coco.json'
)

# Train for 10 epochs and validate after each epoch
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1
)

# Start training from pretrained RTMDet weights
load_from = '/home/kaurh/RTMDet/checkpoints/rtmdet_tennis_init.pth'


# Save checkpoints and training results in this folder
work_dir = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis_full'

# Use a small learning rate for fine-tuning
optim_wrapper = dict(
    optimizer=dict(lr=0.0001)
)

# Use a fixed learning rate instead of the original schedule
param_scheduler = []

# Do not use the custom hooks from the base configuration
custom_hooks = []

# Save a checkpoint after every epoch
default_hooks = dict(
    checkpoint=dict(interval=1)
)
