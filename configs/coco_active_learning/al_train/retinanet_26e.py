_base_ = "../bases/al_retinanet_base.py"

labeled_data = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_labeled_1.json'
unlabeled_data = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_unlabeled_1.json'

model = dict(
    bbox_head=dict(
        type='RetinaQualityEMAHead',
        num_classes=1,
        
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file='/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json'
    ),
)

evaluation=dict(interval=1, metric='bbox') #changed from 99999
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005) #changed for puncta 0.001 0.00001
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2)) #changed for puncta

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 12]) #changed for puncta

runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=12, max_keep_ckpts=1, by_epoch=True)

