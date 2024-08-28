VOCPATH=$1

mkdir -p data/VOC0712
mkdir -p data/VOC0712/annotations
mkdir -p data/VOC0712/images
cp /home/djones/puncta_det/data/VOCdevkit/VOC2007/JPEGImages/*.jpg data/VOC0712/images
cp /home/djones/puncta_det/data/VOCdevkit/VOC2012/JPEGImages/*.jpg data/VOC0712/images
cp $VOCPATH data/VOC0712/annotations

mkdir -p data/active_learning/coco/
mkdir -p data/active_learning/voc/

python tools/al_data/create_al_dataset.py \
       data/coco/annotations/instances_train2017.json \
       data/active_learning/coco \
       3 \
       2365 \
       coco

python tools/al_data/create_al_dataset.py \
       data/VOC0712/annotations/trainval_0712.json \
       data/active_learning/voc \
       3 \
       827 \
       voc