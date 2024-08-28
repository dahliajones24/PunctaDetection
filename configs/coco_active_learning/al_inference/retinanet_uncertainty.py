_base_ = "../bases/al_retinanet_inference_base.py"

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadUncertainty',
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=200)
)
data = dict(
    test=dict(ann_file='/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json')
)
unlabeled_data = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_unlabeled_1.json'

