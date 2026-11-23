"""
config for tracknetv5 with b2e30 and adamw lr1e-4
"""

from pathlib import Path

# ------------------- 1. Model definition (Model) -------------------
model = dict(
    type='TrackNetV5',
    backbone=dict(
        type='TrackNetV2Backbone', # OK
        in_channels=13
    ),
    neck=dict(
        type='TrackNetV2Neck'# OK
    ),
    head=dict(
        type='R_STRHead',
        in_channels=64,
        out_channels=3
    )
)

# ------------------- 2. Data definition (Data) -------------------
# --- 2.1 Common parameters ---
input_size = (1080, 1920)  # (height, width)
original_size = (1080, 1920) # Original image size (height, width)
# ‼️ Be sure to change this path to the correct path on your own machine
data_root = './../data/benchmark'

# --- 2.2 Data processing pipeline definition ---
pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    # dict(type='GenerateMotionAttention', threshold=40),
    dict(type='ConcatChannels',
        keys=['path_prev', 'path', 'path_next'],
        output_key='image'),
    dict(type='LoadAndFormatMultiTargets',  # Use the new multi-target loader
        keys=['gt_path_prev', 'gt_path', 'gt_path_next'],  # Specify the three GT paths
        output_key='target'),
    dict(type='Finalize',
        image_key='image',
        final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

# --- 2.3 Data loader configuration ---
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_train.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    ),
    val=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    )
)

# ------------------- 3. Loss function definition (Loss) -------------------

loss = dict(
    type='TrackNetV2Loss'
)

# ------------------- 4. Optimization strategy definition (Optimization) -------------------
optimizer = dict(type='AdamW', lr=1e-4)

# (2) Define the optimizer config: add gradient clipping (prevents exploding gradients)
optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
)

# (3) Define the learning rate config: Warmup + Step Decay
lr_config = dict(
    # Policy: use Step Decay
    policy='Step',
    # Linear warmup: ensures a stable start for the Transformer
    # warmup='linear',          # Use linear warmup
    # warmup_iters=50*200,          # Warmup length (the first 50 epochs)
    # warmup_ratio=1e-6,        # Initial learning rate (warm up starting from near 0)
    # Learning rate decay steps (epochs)
    step=[20, 25],          # Decay is triggered at the end of epoch 20 and epoch 25
    # Decay factor
    gamma=0.1                 # At each decay, the learning rate is multiplied by 0.1
)

# ------------------- 5. Evaluation strategy definition (Evaluation) -------------------
evaluation = dict(
    interval=1,
    metric=dict(
        type='TrackNetV2Metric',
        min_dist=4,
        original_size=original_size
    )
)

# ------------------- 6. Runtime definition (Runtime) -------------------
total_epochs = 30
work_dir = f'./workdirs/{Path(__file__).stem}' # Note: custom workdir

# ✨ Fix #3: per your request, add a maximum number of iterations per epoch
# steps_per_epoch = 200

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

custom_hooks = [
    dict(type='ValidationVisualizerV2Hook', num_samples_to_save=100, original_size=original_size)
]

seed = 42
resume_from = None