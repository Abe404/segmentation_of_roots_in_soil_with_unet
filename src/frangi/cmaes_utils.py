"""
Utilities for working with co-variance matrix adaptation evolution strategy

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

import math
import random
from typing import Callable
import numpy as np
from cma.constraints_handler import BoundTransform


def get_best_params(logger):
    best_idx = np.argmin(logger.xrecent[:, 4])
    max_frangi_sigma = logger.xrecent[best_idx, 5]
    beta1 = logger.xrecent[best_idx, 6]
    beta2 = logger.xrecent[best_idx, 7]
    threshold = logger.xrecent[best_idx, 8]
    best = [max_frangi_sigma, beta1, beta2, threshold]
    best.append(logger.xrecent[best_idx, 9]) # small object threshold

    def identity(v):
        return v

    warper = get_cmaes_params_warp(identity, proba=False, repair=True)
    after_scaling = warper(best)
    return best, after_scaling


def assign_proba_round(params):
    """ map floating values to integers to avoid flat gradients when using CMAES """
    params[0] = get_proba_rounded(params[0]) # max_frangi_sigma
    params[4] = get_proba_rounded(params[4]) # small_object_threshold
    return params


def assign_round(params):
    """ map floating values to integers """
    params[0] = int(round(params[0])) # max_frangi_sigma
    params[4] = int(round(params[4])) # small_object_threshold
    return params


def rescale_cmaes_params(params):
    """ take from CMAES friendly scales to the scaled used by the
        actual image processing pipeline """
    params = np.array(params)
    params[0] *= 10 # frangi_max sigma
    params[2] *= 10 # frangi_beta2
    params[3] /= 10000 # frangi_threshold
    params[4] *= 100 # small_object_threshold
    return params


def get_cmaes_params_warp(objective, proba=True, repair=True) -> Callable:
    """ return a function which can be used to warp the parameters """

    # sigma should be between 1 and 6
    frangi_max_sigma_bound = BoundTransform([1, 6])

    def cmaes_params_warp_wrapper(params):
        """ rescale the parameters first and then
            map the floating poing values to integers

            return the objective function results when invoked on the
            parameters after.
        """
        params = rescale_cmaes_params(params)
        if repair:
            params[0] = frangi_max_sigma_bound.repair([params[0]])[0]
        if proba:
            params = assign_proba_round(params)
        else:
            params = assign_round(params)
        params = np.abs(params) # no negative params here.
        return objective(params)

    return cmaes_params_warp_wrapper


def get_proba_rounded(param: float) -> int:
    """ useful for CMAES.
        map floats to integers using uniform probabilty.

        if param is 3.7 then there will
        70% chance this function will return 4
        and 30% chance it will return 3.
    """
    if abs(param - round(param)) > 0:
        min_val = math.floor(param)
        max_val = math.ceil(param)
        chance_is_max = abs(param - min_val)
        if random.random() < chance_is_max:
            return int(max_val)
        return int(min_val)
    return int(param)
