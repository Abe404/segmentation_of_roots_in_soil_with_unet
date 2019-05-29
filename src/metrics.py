# pylint: disable=C0111
"""
Computing metrics on output segmentations for root images

Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import math
import numpy as np
from skimage.io import imread

VALID_METRICS = ["accuracy", "TN", "FP", "FN", "TP",
                 "precision", "recall", "f1_score",
                 "iou", "true_mean", "pred_mean", "loss"]


def get_metrics_str(all_metrics, to_use=None):
    out_str = ""
    for name, val in all_metrics.items():
        if to_use is None or name in to_use:
            out_str += f" {name}: {val:.4g},"
    return out_str


def get_metrics(y_pred, y_true, loss=float('nan')):
    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    accuracy = (true_positives + true_negatives) / len(y_true)
    assert not np.isnan(true_negatives)
    assert not np.isnan(false_positives)
    assert not np.isnan(false_negatives)
    assert not np.isnan(true_positives)

    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = true_positives / (true_positives + false_positives + false_negatives)
    else:
        precision = recall = f1_score = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TN": true_negatives,
        "FP": false_positives,
        "FN": false_negatives,
        "TP": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
        "loss": loss
    }


def format_num(num):
    return f'{num:.4g}'


def print_metrics_from_dirs(annot_dir, seg_dir) -> None:
    print(f'Evaluating segmentations in {seg_dir} against annotations in {annot_dir}')
    metric_obj, f1s, recalls, precisions = evaluate_segmentations_from_dirs(annot_dir,
                                                                            seg_dir)
    print('All images combined:', get_metrics_str(metric_obj))

    f1s = [n for n in f1s if not math.isnan(n)]
    recalls = [n for n in recalls if not math.isnan(n)]
    precisions = [n for n in precisions if not math.isnan(n)]

    print('f1 mean', format_num(np.mean(f1s)))
    print('f1 std', format_num(np.std(f1s)))
    print('recall mean', format_num(np.mean(recalls)))
    print('recall std', format_num(np.std(recalls)))
    print('precision mean', format_num(np.mean(precisions)))
    print('precision std', format_num(np.std(precisions)))


def evaluate_segmentations_from_dirs(annot_dir, seg_dir):
    """ load all files from annot_dir
        load all files from the seg_dir (segmentations)

        Make sure they have the same names.
        Make sure they are images (PNG)
        Get the f1, recall and precision for all segmentations
    """
    annotation_paths = os.listdir(annot_dir)
    segmentation_paths = os.listdir(seg_dir)
    annotation_paths = sorted(annotation_paths)
    segmentation_paths = sorted(segmentation_paths)

    # names of files should be the same in both folders
    if len(annotation_paths) != len(segmentation_paths):
        msg = ("Should be same number of annotations as segmentations, "
               f"annotations: {len(annotation_paths)}, "
               f"segmentations: {len(segmentation_paths)}")
        raise Exception(msg)

    # now add the dirs to the path so they can actually be loaded from disk
    annotation_paths = [os.path.join(annot_dir, f) for f in annotation_paths]
    segmentation_paths = [os.path.join(seg_dir, f) for f in segmentation_paths]
    annots = [imread(f).astype(np.bool) for f in annotation_paths]
    segmentations = [imread(f).astype(np.bool) for f in segmentation_paths]

    # Metrics on all data combined (includes images with no roots)
    all_ground_truth = np.array(annots).reshape(-1)
    all_preds = np.array(segmentations).reshape(-1)
    metrics = get_metrics(all_preds, all_ground_truth)

    # Also compute metrics on a per-image basis
    f1s, recalls, precisions = evaluate_segmentations(annots, segmentations)
    return metrics, f1s, recalls, precisions


def evaluate_segmentations(annotations, segmentations):
    """ takes as input the annotations and segmentations
        outputs the f1 scores, recalls and precisions.
    """
    f1s = []
    recalls = []
    precisions = []
    for i, (annot, segmented) in enumerate(zip(annotations, segmentations)):
        print(f'Getting metrics for {i+1} of {len(annotations)}')
        metric = get_metrics(segmented.reshape(-1), annot.reshape(-1))
        f1s.append(metric['f1_score'])
        recalls.append(metric['recall'])
        precisions.append(metric['precision'])
    return f1s, recalls, precisions
