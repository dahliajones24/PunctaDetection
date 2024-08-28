_base_ = "../bases/al_retinanet_inference_base.py"

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadFeat',
        total_images=1811,  
        max_det=100,
        feat_dim=256,
        output_path=''  
    ),
)
data = dict(
    test=dict(ann_file='/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json')
)
unlabeled_data = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_unlabeled_1.json'
