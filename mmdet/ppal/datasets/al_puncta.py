from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np

@DATASETS.register_module()
class ALPunctaDataset(CocoDataset):
    # Define the single class
    CLASSES = ('puncta',)

    def __init__(self, *args, **kwargs):
        super(ALPunctaDataset, self).__init__(*args, **kwargs)
        
        # Ensure img_infos are populated
        if not hasattr(self, 'img_infos') or not self.img_infos:
            self.img_infos = [
                {
                    'id': img_info['id'],
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height']
                }
                for img_info in self.coco.imgs.values()
            ]
            if not self.img_infos:
                raise AttributeError("The 'img_infos' attribute is still not initialized after attempting to load it. Check the dataset loading.")

        # Setup the flag attribute used for grouping
        self.flag = np.zeros(len(self.img_infos), dtype=np.uint8)
        for i, img_info in enumerate(self.img_infos):
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    if bboxes.shape[1] > 5:  # If there are additional fields like uncertainties
                        data['cls_uncertainty'] = float(bboxes[i][5])
                        data['box_uncertainty'] = float(bboxes[i][6])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results
