import json
import numpy as np
import os
import torch
from collections import OrderedDict
from mmdet.ppal.builder import SAMPLER
from mmdet.ppal.sampler.al_sampler_base import BaseALSampler
from mmdet.ppal.utils.running_checks import sys_echo

eps = 1e-10

@SAMPLER.register_module()
class DCUSSampler(BaseALSampler):
    def __init__(self, n_sample_images, oracle_annotation_path, score_thr, class_weight_ub, class_weight_alpha, dataset_type):
        super(DCUSSampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.score_thr = score_thr
        self.log_init_info()

    def _get_classwise_weight(self, results_json):
        # Since this is a single-class dataset, we return a fixed weight of 1.0 for all calculations
        return {self.class_name2id[self.CLASSES[0]]: 1.0}

    def al_acquisition(self, result_json, last_label_path):
        # Adjust the method to work without class weights since we have only one class
        class_weights = self._get_classwise_weight(result_json)

        with open(result_json) as f:
            results = json.load(f)

        category_uncertainty = OrderedDict()
        category_count = OrderedDict()

        for res in results:
            img_id = res['image_id']
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']

            if label not in class_weights:
                print(f"Warning: Label {label} not found in class_weights. Skipping this entry.")
                continue

            if label not in category_uncertainty.keys():
                category_uncertainty[label] = 0.
                category_count[label] = 0.
            category_uncertainty[label] += uncertainty
            category_count[label] += 1

        category_avg_uncertainty = OrderedDict()
        for k in category_uncertainty.keys():
            category_avg_uncertainty[k] = category_uncertainty[k] / (category_count[k] + 1e-5)

        with open(last_label_path) as f:
            last_labeled_data = json.load(f)
            last_labeled_img_ids = [x['id'] for x in last_labeled_data['images']]

        image_hit = dict()
        for img_id in self.oracle_data.keys():
            image_hit[img_id] = 0
        for img_id in last_labeled_img_ids:
            image_hit[img_id] = 1

        image_uncertainties = OrderedDict()
        for img_id in self.oracle_data.keys():
            if image_hit[img_id] == 0:
                image_uncertainties[img_id] = [0.]

        for res in results:
            img_id = res['image_id']
            img_size = (self.oracle_data[img_id]['image']['width'], self.oracle_data[img_id]['image']['height'])
            if not self.is_box_valid(res['bbox'], img_size):
                continue
            if res['score'] < self.score_thr:
                continue
            uncertainty = float(res['cls_uncertainty'])
            label = res['category_id']

            if label not in class_weights:
                print(f"Warning: Label {label} not found in class_weights during image uncertainty calculation. Skipping this entry.")
                continue

            image_uncertainties[img_id].append(uncertainty * class_weights[label])

        for img_id in image_uncertainties.keys():
            _img_uncertainties = np.array(image_uncertainties[img_id])
            image_uncertainties[img_id] = _img_uncertainties.sum()

        img_ids = []
        merged_img_uncertainties = []
        for k, v in image_uncertainties.items():
            img_ids.append(k)
            merged_img_uncertainties.append(v)
        img_ids = np.array(img_ids)
        merged_img_uncertainties = np.array(merged_img_uncertainties)

        inds_sort = np.argsort(-1. * merged_img_uncertainties)
        sampled_inds = inds_sort[:self.n_images]
        unsampled_img_ids = inds_sort[self.n_images:]
        sampled_img_ids = img_ids[sampled_inds].tolist()
        unsampled_img_ids = img_ids[unsampled_img_ids].tolist()

        return sampled_img_ids, unsampled_img_ids

    def create_jsons(self, sampled_img_ids, unsampled_img_ids, last_labeled_json, out_label_path, out_unlabeled_path):
        with open(last_labeled_json) as f:
            last_labeled_data = json.load(f)

        last_labeled_img_ids = [x['id'] for x in last_labeled_data['images']]
        all_labeled_img_ids = last_labeled_img_ids + sampled_img_ids
        assert len(set(all_labeled_img_ids)) == len(last_labeled_img_ids) + len(sampled_img_ids)
        assert len(all_labeled_img_ids) + len(unsampled_img_ids) == self.image_pool_size

        sys_echo('---------------------------------------------')
        sys_echo('--->>> Creating new image sets:')
        sys_echo('--->>> Last round labeled set size: %d (%.2f%%)' % (len(last_labeled_img_ids),100.*float(len(last_labeled_img_ids))/self.image_pool_size))
        sys_echo('--->>> New uncertainty pool set size: %d (%.2f%%)'%(len(sampled_img_ids),100.*float(len(sampled_img_ids))/self.image_pool_size))
        sys_echo('---------------------------------------------')

        labeled_data = dict(images=[], annotations=[], categories=self.categories)
        unlabeled_data = dict(images=[], categories=self.categories)

        for img_id in sampled_img_ids:
            # no annotation here because the annotating happens in the diversity step
            labeled_data['images'].append(self.oracle_data[img_id]['image'])

        with open(out_label_path, 'w') as f:
            json.dump(labeled_data, f)

        self.latest_labeled = out_label_path

    def log_info(self, result_path, out_label_path, out_unlabeled_path):
        sys_echo('>>>> Round: %d' % self.round)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per Round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Unlabeled results path: %s' % result_path)
        sys_echo('>>>> Uncertainty pool image info file path: %s' % out_label_path)
        sys_echo('>>>> Score threshold: %s' % self.score_thr)

    def log_init_info(self):
        sys_echo('>> %s initialized:'%self.__class__.__name__)
        sys_echo('>>>> Dataset: %s' % self.dataset_type)
        sys_echo('>>>> Oracle annotation path: %s' % self.oracle_path)
        sys_echo('>>>> Image pool size: %d' % self.image_pool_size)
        sys_echo('>>>> Uncertainty pool size per round: %d (%.2f%%)'%(self.n_images, 100.*float(self.n_images)/self.image_pool_size))
        sys_echo('>>>> Score threshold: %s' % self.score_thr)
        sys_echo('\n')
