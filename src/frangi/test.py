# pylint: disable=W0612
"""
Take the best parameters from the frangi+cca segmentation system
and use them to segment the test set.
Then report the metrics.

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
from pathlib import Path
import numpy as np
from skimage.io import imread
import cma

import segment
import metrics
import cmaes_utils
DATA_DIR = Path('../data')
TEST_ANNOT_DIR = DATA_DIR / 'test' / 'annotations'
TEST_PHOTO_DIR = DATA_DIR / 'test' / 'photos'

TEST_CCA_SEG_DIR = Path('../output/frangi/test_segmentations')
RUNS_DIR = Path('../saved_output/frangi/cmaes_logs')


def load_best_params_from_log():
    log_path = RUNS_DIR / 'outcmaes'
    logger = cma.CMADataLogger(str(log_path)).load()
    best_function_val_at_each_iteration = logger.xrecent[:, 4]

    arg_min = np.argmin(best_function_val_at_each_iteration)
    min_val = best_function_val_at_each_iteration[arg_min]

    print('best_frangi_cca_obj_val', np.min(best_function_val_at_each_iteration))
    print('best_frangi_cca_f1', 1 - np.min(best_function_val_at_each_iteration))

    # param_names = ['Frangi sigma', 'beta1', 'beta2',
    #                'Frangi threshold', 'small object threshold']
    best_params, after_scaling = cmaes_utils.get_best_params(logger)

    print('best_params', best_params)
    print('after scaling', after_scaling)

    # after_scaling (best_params found)
    # [2.00000000e+00, 9.21819899e-02, 1.05175985e+00, 3.75864362e-05, 1.92000000e+02]
    return after_scaling


def print_frangi_metrics():
    annot_dir = TEST_ANNOT_DIR
    seg_dir = TEST_CCA_SEG_DIR
    metrics.print_metrics_from_dirs(annot_dir, seg_dir)


def seg_frangi_cca_test(best_params):
    image_names = [f for f in os.listdir(TEST_PHOTO_DIR) if 'png' in f]
    images = [imread(os.path.join(TEST_PHOTO_DIR, name)) for name in image_names]
    for i, im in enumerate(images):
        images[i] = np.array(im).astype(np.float32)
    segment.produce_segmentations(best_params, images,
                                  image_names, TEST_CCA_SEG_DIR)


if __name__ == '__main__':
    BEST_PARAMS = load_best_params_from_log()
    seg_frangi_cca_test(BEST_PARAMS)
    print_frangi_metrics()
