_base_ = "al_retinanet_base.py"
data_root = 'data_puncta/puncta/'
data = dict(
    test=dict(
        type='ALPunctaDataset',
        img_prefix=data_root + 'train/',
    )
)

