"""Debug config: single scene (Yuchen) for quick inference diagnosis."""
_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings - same as main config
num_channels = 32
num_instance_classes = 3
num_semantic_classes = 3
radius = 16
score_th = 0.4
chunk = 20_000
model = dict(
    type='ForAINetV2OneFormer3D_XAwarequery',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3,
    num_channels=num_channels,
    voxel_size=0.2,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[0],
    thing_cls=[1, 2],
    prepare_epoch=1000,
    query_point_num=300,
    radius=radius,
    score_th=score_th,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1,
        num_instance_queries=0,
        num_semantic_queries=num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion_XAwarequery',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=0.2),
        inst_criterion=dict(
            type='InstanceCriterionForAI_OneToManyMatch',
            matcher=dict(type='One2ManyMatcher'),
            loss_weight=[1.0, 1.0, 0.5],
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[0],
        thing_cls=[0]))

# dataset settings - single scene for debug
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = 'data/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
               'pts_instance_mask', 'instance_mask'])
]

test_dataloader = dict(
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_test_debug.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        box_type_3d='Depth',
        test_mode=True,
        backend_args=None))

# evaluator
class_names = ['ground', 'wood', 'leaf']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2')
sem_mapping = [0, 1, 2]
inst_mapping = sem_mapping[1:]
test_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0],
    thing_class_inds=list(range(1, num_semantic_classes)),
    min_num_points=1,
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)

# misc
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_optimizer=True),
    logger=dict(type='LoggerHook', interval=20),
    visualization=dict(type='Det3DVisualizationHook', draw=False))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
test_cfg = dict(type='TestLoop')
