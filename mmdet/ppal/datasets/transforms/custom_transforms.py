from mmdet.datasets.builder import PIPELINES
import numpy as np
import cv2

@PIPELINES.register_module()
class RandomRotate:
    def __init__(self, level=10, prob=0.5):
        self.level = level
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() < self.prob:
            angle = np.random.uniform(-self.level, self.level)
            img = results['img']
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h))
            results['img'] = img

            # Rotate bounding boxes
            if 'gt_bboxes' in results:
                bboxes = results['gt_bboxes']
                new_bboxes = []
                for bbox in bboxes:
                    points = np.array([
                        [bbox[0], bbox[1]],  # Top-left
                        [bbox[2], bbox[1]],  # Top-right
                        [bbox[2], bbox[3]],  # Bottom-right
                        [bbox[0], bbox[3]],  # Bottom-left
                    ])
                    rotated_points = cv2.transform(np.array([points]), M)[0]
                    x_min = np.min(rotated_points[:, 0])
                    y_min = np.min(rotated_points[:, 1])
                    x_max = np.max(rotated_points[:, 0])
                    y_max = np.max(rotated_points[:, 1])
                    new_bboxes.append([x_min, y_min, x_max, y_max])
                results['gt_bboxes'] = np.array(new_bboxes, dtype=np.float32)

        return results

@PIPELINES.register_module()
class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, results):
        img = results['img']
        # Apply color jitter to the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)

        # Apply brightness adjustment
        v = cv2.add(v, int(self.brightness * 255))
        # Apply contrast adjustment
        v = cv2.multiply(v, 1 + self.contrast)
        # Apply saturation adjustment
        s = cv2.multiply(s, 1 + self.saturation)
        # Apply hue adjustment
        h = (h + int(self.hue * 180)) % 180

        img = cv2.merge([h, s, v])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        results['img'] = img

        return results
