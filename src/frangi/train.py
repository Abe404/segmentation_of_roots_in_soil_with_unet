""""
Functionality to find the best parameters for the frangi baseline
method using CMA-ES

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

from typing import Tuple
from typing import Any
from functools import partial
from multiprocessing import Pool
import time
import sys

import numpy as np
import cma
from sklearn import metrics

# pylint: disable=C0413
sys.path.insert(0, '..')
from data_utils import load_train_data
from segment import segment_image
import cmaes_utils


def get_f1_fast(annot, pred) -> float:
    """ just does f1 instead of calculating all metrics
        converts to bools to speed it up """
    if np.sum(pred) == 0:
        return 0
    # round and convert to bool for speedup
    annot = np.round(annot).astype(bool)
    pred = np.round(pred).astype(bool)
    y_true = annot.reshape(-1)
    y_pred = pred.reshape(-1)
    f1 = metrics.f1_score(y_true, y_pred)
    return f1


def cmaes_metric_from_pool(params, photos, annots) -> float:
    """ segment all the images (mult processing) using the specified params
        get the mean_f1
        then return 1-mean_f1
    """
    params = np.array(params)
    inputs = []

    for i, (photo, annot) in enumerate(zip(photos, annots)):
        inputs.append([i, np.array(photo), np.array(annot), params])

    with Pool(len(photos)) as pool:
        f1_scores = pool.map(cmaes_image_f1_score, inputs)
        mean_f1 = np.mean(f1_scores)
        return 1 - mean_f1


def cmaes_image_f1_score(args):
    """
    Takes param list from CMA-ES with an image and annotation.
    Then returns the f1 score on that image.

    idx can be used for getting previously stored values from the cache
    """
    idx, photo, annot, params = args
    frangi_settings = params[:4]
    segmented = segment_image(photo,
                              frangi_settings,
                              small_object_threshold=params[4],
                              image_id=idx)

    f1_score = get_f1_fast(annot, segmented)
    return f1_score



def get_start_params() -> list:
    start_values = [4/10,  # x10 - Max frangi sigma
                    0.15, # x1 - frangi beta1
                    2.5/10, # x10 - frangi beta2
                    0.2, # / 1000 - Frangi threshold
                    40/100] # x100 - small_object_threshold

    return start_values


def find_cmaes_settings(start_sigma) -> Tuple[Any, Any]:
    """ Find best settings for baseline image segmentation
        using CMA-ES
    """
    photos, annots = load_train_data()
    photos = [p.astype(np.float32) for p in photos]
    objective = partial(cmaes_metric_from_pool, photos=photos, annots=annots)
    objective_wrapped = cmaes_utils.get_cmaes_params_warp(objective)
    start_values = get_start_params()
    return cma.fmin2(objective_wrapped, start_values, start_sigma)  # xopt, es


if __name__ == '__main__':
    START = time.time()
    SIGMA = 0.005
    print(f'started finding best (with SIGMA={SIGMA})')
    XOPT, _ = find_cmaes_settings(SIGMA)
    print(f'finished finding best (SIGMA={SIGMA})')
    print('total duration = ', time.time() - START)
    print('xopt = ', XOPT)
