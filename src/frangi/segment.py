"""
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

import time
import os
from typing import List
from multiprocessing import Pool

import numpy as np
from skimage import img_as_float
from skimage.morphology import remove_small_objects
from scipy.misc import imsave

from frangi_filter import get_frangi
from frangi_filter import get_thresholded
# pylint: disable=C0413
sys.path.append('.')
from data_utils import zero_one


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        print('Creating', dir_path)
        os.makedirs(dir_path)


def segment_image(image, frangi_settings: list, small_object_threshold: int,
                  image_id=None) -> np.ndarray:
    """ runs the whole segmentation pipeline
        image_id is used for caching, leave as None to disable cache
    """
    frangi_sigma_max, frangi_beta1, frangi_beta2, frangi_threshold = frangi_settings

    frangi_output = get_frangi(image_id, image, [frangi_sigma_max,
                                                 frangi_beta1, frangi_beta2])

    segmented = get_thresholded(frangi_output, frangi_threshold)

    segmented = remove_small_objects(np.round(segmented).astype(bool),
                                     small_object_threshold)

    segmented = zero_one(segmented)
    return segmented



def produce_segmentations(segment_params, images, image_names, output_dir) -> None:
    """
    Segment each image found in images with params and save
    them to the output_folder
    image_paths should be absolute paths
    """

    if not os.path.isdir(output_dir):
        print(f"Creating {output_dir}")
        os.makedirs(output_dir)

    params = np.array(segment_params)
    frangi_settings = params[:4]
    small_object_threshold = params[4]
    for photo, name in zip(images, image_names):
        seg_start = time.time()
        segmented = segment_image(photo,
                                  frangi_settings,
                                  small_object_threshold=small_object_threshold)
        print('Segmentation duration:', time.time() - seg_start)
        out_path = os.path.join(output_dir, name)
        print('Saving', out_path)
        imsave(out_path, img_as_float(segmented))


def segment_im_wrapper(args):
    """ convenient wrapper to convert the CMA-ES
        friendly param list into arguments that can be called
        by segment_image """
    idx, photo, params = args
    frangi_settings = params[:4]
    object_threshold = params[4]
    return segment_image(np.array(photo), frangi_settings,
                         small_object_threshold=object_threshold,
                         image_id=idx)


def produce_segmentations_pool(segment_params, images) -> List[np.ndarray]:
    """
    Similar to produce_segmentations but uses multiprocessing instead.
    Also this function returns the segmented images instead of saving them to disk
    """
    inputs = []
    for i, image in enumerate(images):
        inputs.append([i, np.array(image), np.array(segment_params)])

    with Pool(len(images)) as pool:
        segmented_images = pool.map(segment_im_wrapper, inputs)
    assert len(segmented_images) == len(images)
    return segmented_images


def save_segmentations_pool(segment_params, images, image_names, output_dir) -> None:
    """
    Similar to produce_segmentations_pool but saves them to disk
    """
    if not os.path.isdir(output_dir):
        raise Exception(f"{output_dir} needs to exist for segmentations to be output")

    inputs = []
    for i, image in enumerate(images):
        inputs.append([i, np.array(image), np.array(segment_params)])

    with Pool(len(images)) as pool:
        segmented_images = pool.map(segment_im_wrapper, inputs)

    for image, name in zip(segmented_images, image_names):
        out_path = os.path.join(output_dir, name)
        print('Saving', out_path)
        imsave(out_path, img_as_float(image))
