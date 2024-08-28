# Paths
config_dir  = '/home/djones/puncta_det/configs/coco_active_learning/'
work_dir    = '/home/djones/puncta_det/work_dirs/'

# Environment setting
python_path = 'python'
port        = 29500
gpus        = 1

# Data setting
oracle_path         = '/home/djones/puncta_det/data_puncta/puncta/annotations/instances_train.json'
init_label_json     = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_labeled_1.json'
init_unlabeled_json = '/home/djones/puncta_det/data_puncta/active_learning/coco_600_unlabeled_1.json'
init_model          = None

# Config setting
train_config             = config_dir + 'al_train/retinanet_26e.py'
uncertainty_infer_config = config_dir + 'al_inference/retinanet_uncertainty.py'
diversity_infer_config   = config_dir + 'al_inference/retinanet_diversity.py'

# Active learning setting
round_num             = 4
budget                = 50
budget_expand_ratio   = 4
uncertainty_pool_size = budget * budget_expand_ratio + gpus - (budget * budget_expand_ratio) % gpus

# Sampler setting
uncertainty_sampler_config = dict(
    type='DCUSSampler',
    n_sample_images=uncertainty_pool_size,
    oracle_annotation_path=oracle_path,
    score_thr=0.1,
    class_weight_ub=0.2,
    class_weight_alpha=0.3, #0.1 
    dataset_type='coco')
diversity_sampler_config = dict(
    type='DiversitySampler',
    n_sample_images=5,
    oracle_annotation_path=oracle_path,
    dataset_type='coco')

output_dir  = work_dir + 'active_learning_results_labelled1'