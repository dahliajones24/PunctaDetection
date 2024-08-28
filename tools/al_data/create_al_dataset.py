import os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='AL Dataset')
parser.add_argument('oracle_path', type=str, help='dataset root')
parser.add_argument('out_root', type=str, help='output json path')
parser.add_argument('n_diff', type=int, help='number of different initial sets')
parser.add_argument('n_labeled', type=int, help='number of labeled images')
parser.add_argument('dataset', choices=['coco', 'voc'], help='dataset type')
args = parser.parse_args()

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

voc_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def generate_active_learning_dataset(oracle_json, n_labeled_img, out_labeled_json, out_unlabeled_json, valid_classes):
    with open(oracle_json) as f:
        data = json.load(f)

    all_images = data['images']
    all_annotations = data['annotations']

    class_id2name = {c['id']: c['name'] for c in data['categories']}
    class_name2id = {c['name']: c['id'] for c in data['categories']}

    inds = np.random.permutation(len(all_images))
    labeled_inds = inds[:n_labeled_img].tolist()
    unlabeled_inds = inds[n_labeled_img:].tolist()

    labeled_images = [all_images[ind] for ind in labeled_inds]
    labeled_img_ids = [img['id'] for img in labeled_images]
    labeled_annotations = [ann for ann in all_annotations if (class_id2name[ann['category_id']] in valid_classes) and (ann['image_id'] in labeled_img_ids)]

    unlabeled_images = [all_images[ind] for ind in unlabeled_inds]

    out_labeled_data = dict(
        categories=data['categories'],
        images=labeled_images,
        annotations=labeled_annotations)

    out_unlabeled_data = dict(
        categories=data['categories'],
        images=unlabeled_images,
        annotations=[])

    with open(out_labeled_json, 'w') as fl:
        json.dump(out_labeled_data, fl)

    with open(out_unlabeled_json, 'w') as fu:
        json.dump(out_unlabeled_data, fu)

    print('------------------------------------------------------')
    print('Labeled data:')
    print('Output path: %s' % out_labeled_json)
    print('Number of images: %d' % len(labeled_images))
    print('Number of objects: %d' % len(labeled_annotations))

    print('Unlabeled data:')
    print('Output path: %s' % out_unlabeled_json)
    print('Number of images: %d' % len(unlabeled_images))
    print('------------------------------------------------------')

if __name__ == '__main__':
    valid_classes = CLASSES if args.dataset == 'coco' else voc_classes

    N = args.n_diff
    for i in range(N):
        data_prefix = f"{args.dataset}_{args.n_labeled}"
        generate_active_learning_dataset(
            oracle_json=args.oracle_path,
            n_labeled_img=args.n_labeled,
            out_labeled_json=os.path.join(args.out_root, f"{data_prefix}_labeled_{i + 1}.json"),
            out_unlabeled_json=os.path.join(args.out_root, f"{data_prefix}_unlabeled_{i + 1}.json"),
            valid_classes=valid_classes
        )
