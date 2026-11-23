
_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = '/home/kaurh/RTMDet/'
classes = ('ball',)
metainfo = dict(classes=classes, palette=[(255, 0, 0)])

# Detect only tennis balls
model = dict(
    bbox_head=dict(num_classes=1)
)

# Use a simpler pipeline for the initial training test
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
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='frames/'),
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
        ann_file='annotations/val_coco.json',
        data_prefix=dict(img='frames/'),
        metainfo=metainfo
    )
)

# Testing dataset
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test_coco.json',
        data_prefix=dict(img='frames/'),
        metainfo=metainfo
    )
)

# Evaluation
val_evaluator = dict(
    ann_file=data_root + 'annotations/val_coco.json'
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/test_coco.json'
)

# Short initial training run
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1
)

# Load pretrained RTMDet-Tiny weights
load_from = (
    '/home/kaurh/RTMDet/checkpoints/'
    'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
)

# Save results separately
work_dir = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis'

_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = '/home/kaurh/RTMDet/'
classes = ('ball',)
load_from = '/home/kaurh/RTMDet/checkpoints/rtmdet_tennis_init.pth'

# Detect only tennis balls
model = dict(
    bbox_head=dict(num_classes=1)
)

# Use a simpler pipeline for the initial training test
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
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='frames/'),
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
        ann_file='annotations/val_coco.json',
        data_prefix=dict(img='frames/'),
        metainfo=metainfo
    )
)

# Testing dataset
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test_coco.json',
        data_prefix=dict(img='frames/'),
        metainfo=metainfo
    )
)

# Evaluation
val_evaluator = dict(
    ann_file=data_root + 'annotations/val_coco.json'
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/test_coco.json'
)

# Short initial training run
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1
)

# Load compatible pretrained weights
load_from = '/home/kaurh/RTMDet/checkpoints/rtmdet_tennis_init.pth'


# Save results separately
work_dir = '/home/kaurh/RTMDet/work_dirs/rtmdet_tennis'

# Use a small learning rate for fine-tuning
optim_wrapper = dict(
    optimizer=dict(lr=0.0001)
)

# Disable the original 300-epoch learning rate schedule
param_scheduler = []

# Disable the original training hooks for this initial test
custom_hooks = []

# Save a checkpoint after every epoch
default_hooks = dict(
    checkpoint=dict(interval=1)
)
