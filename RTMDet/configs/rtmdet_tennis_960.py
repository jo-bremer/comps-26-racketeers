# Training configuration for the improved RTMDet-Tiny model.
# This version continues training from our original model.
# We increased the image resolution from 640x640 to 960x960
# to help the model detect smaller tennis balls.
#
# We trained for another 10 epochs using a batch size of 1
# and a learning rate of 0.0001.

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = '/home/kaurh/RTMDet/'
classes = ('ball',)
metainfo = dict(classes=classes, palette=[(255, 0, 0)])
load_from = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis_full/epoch_9.pth'

# Detect only tennis balls
model = dict(
    bbox_head=dict(num_classes=1)
)

# Resize and pad the training images to 960x960
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Pad', size=(960, 960),
         pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# Training dataset
train_dataloader = dict(
    batch_size=1,
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

# Continue training for 10 more epochs
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=10,
    val_interval=1
)

# Start from the original RTMDet model checkpoint
load_from = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis_full/epoch_9.pth'


# Save the improved model checkpoints separately
work_dir = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis_960'

# Use a small learning rate for fine-tuning
optim_wrapper = dict(
    optimizer=dict(lr=0.0001)
)

# Keep the learning rate fixed during fine-tuning
param_scheduler = []

# Disable the custom hooks from the base configuration
custom_hooks = []

# Save a checkpoint after every epoch
default_hooks = dict(
    checkpoint=dict(interval=1)
)


# Use 960x960 images for validation and testing
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 960), keep_ratio=True),
    dict(type='Pad', size=(960, 960),
         pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape',
                    'img_shape', 'scale_factor'))
]

val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)

test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)

# Use smaller batches to fit on our GPU
val_dataloader.update(batch_size=1)
test_dataloader.update(batch_size=1)

# Set the validation dataset paths
val_dataloader['dataset'].update(
    data_root='/home/kaurh/RTMDet/',
    ann_file='annotations_full/ball_val_coco.json',
    data_prefix=dict(img='frames_full/'),
    metainfo=metainfo
)

# Set the test dataset paths
test_dataloader['dataset'].update(
    data_root='/home/kaurh/RTMDet/',
    ann_file='annotations_full/ball_test_coco.json',
    data_prefix=dict(img='frames_full/'),
    metainfo=metainfo
)

# Reduce worker count
val_dataloader.update(num_workers=2)
test_dataloader.update(num_workers=2)
