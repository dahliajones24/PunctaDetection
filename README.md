# Plug and Play Active Learning for Puncta Detection
<p align="center">
<img src="resources/intro_new.png" style="width:960px;"/>
</p>

The implementation of our paper can be found: [Active learning for puncta image detection in fluorescence microscopy (Machine Learning for Biomedical image analysis)](https://drive.google.com/file/d/1vUcAgI9RkuAwd6G2S9RmiFT8DdVqxAaz/view?usp=drive_link)

## Requirements

- Our codebase is built on top of [MMDetection](https://github.com/open-mmlab/mmdetection), which can be installed following the offcial instuctions.

## Usage

### Installation
```shell
python setup.py install
```

### Setup dataset
- Place your dataset as the following structure (Only vital files are shown). It should be easy because it's the default MMDetection data placement)
```
PPAL
|
`-- data_puncta
    |
    |--puncta
        |
        |--train
        |--val
        `--annotations
           |
           |--instances_train.json
            `--instances_val.json
``` 
- Install datasets into data_puncta/data_setup.
- Please download [groundtruth]([https://drive.google.com/file/d/1GIAmjGbg47dZFJjGYf2p-dU1z4V7pACQ/view?usp=sharing](https://drive.google.com/drive/folders/17i6LFeFjIkh8lkjx2L14xXGF_YzKojgh?usp=drive_link)).
- Please download [rgb_images](https://drive.google.com/drive/folders/1dj2ClENCNLw1tTh_XigfcgBpmJx0dQm5?usp=drive_link).

- - Set up active learning datasets
```shell
python data_processor.py
```

- - Set up active learning datasets
```shell
python instances_creation.py
```
  
- Set up active learning datasets
```shell
python al_setup.py 
```

- The above commands will set up the Puncta Datasets. The commands will also generate three different active learning initial annotations , where the COCO initial sets contain 2% of the original annotated images, and the Pascal VOC initial sets contains 5% of the original annotated images. 
- The resulted file structure is as following
```
PPAL
|
`-- data_puncta
    |
    |--puncta
    |   |
    |   |--train
    |   |--val
    |   `--annotations
    |      |
    |      |--instances_train.json
    |      `--instances_val.json
    |--data_setup
    |   |
    |   |--rgb_images
    |   |--groundtruth
    |   |--al_setup.py
    |   |--puncta_data_processor.py
    |   |--instances_creation.py
    |
    `--active_learning
        |--puncta_600_labeled_1.json
        |--puncta_600_unlabeled_1.json
        |--puncta_600_labeled_2.json
        |--puncta_600_unlabeled_2.json
        |--puncta_600_labeled_3.json
        |--puncta_600_unlabeled_3.json
      
```
- Please refer to [data_setup.sh](https://github.com/ChenhongyiYang/PPAL/blob/main/tools/al_data/data_setup.sh) and [create_al_dataset.py](https://github.com/ChenhongyiYang/PPAL/blob/main/tools/al_data/create_al_dataset.py) to generate you own active learning annotation.
### Run active learning
- You can run active learning using a single command with a config file. For example, you can run COCO and Pascal VOC RetinaNet experiments by
```shell
python tools/run_al_coco.py --config al_configs/puncta/ppal_retinanet_puncta.py --model retinanet
```
- Please check the config file to set up the data paths and environment settings before running the experiments.

```
