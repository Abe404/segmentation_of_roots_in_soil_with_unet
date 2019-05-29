# pylint: disable=C0111, R0914, R0915
"""
Utilities for loading and working with the image data.

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
from multiprocessing import Pool

import numpy as np
from skimage.io import imread


def load_annotations_pool(annotation_paths):
    print(f'Loading {len(annotation_paths)} annotations using Pool')
    with Pool(len(annotation_paths)) as pool:
        annot_list = pool.map(imread, annotation_paths)
    annot_list = [a.astype(bool).astype(np.float32) for a in annot_list]
    desired_shape = annot_list[0].shape + (1,) # add channel dimension
    annot_list = [a.reshape(desired_shape) for a in annot_list]
    return np.array(annot_list)


def load_images_pool(photo_paths):
    print(f'Loading {len(photo_paths)} photos using Pool')
    with Pool(len(photo_paths)) as pool:
        photo_list = pool.map(imread, photo_paths)
    return np.array(photo_list)


def get_paths(dir_name):
    return [os.path.join(dir_name, f) for f in os.listdir(dir_name)]


def zero_one(data):
    data = np.array(data)
    data[data > 0] = 1
    data[data < 1] = 0
    return data


def load_train_data():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_dir = os.path.join(cur_dir, '..', 'data', 'train')
    train_photo_dir = os.path.join(train_data_dir, 'photos')
    train_annot_dir = os.path.join(train_data_dir, 'annotations')
    # Sort so the annotations and photos correspond to each other.
    train_photo_paths = sorted(get_paths(train_photo_dir))
    train_annot_paths = sorted(get_paths(train_annot_dir))
    train_annotations = load_annotations_pool(train_annot_paths)
    train_photos = load_images_pool(train_photo_paths)
    return train_photos, train_annotations


def get_files_split():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    train_data_dir = os.path.join(cur_dir, '..', 'data', 'train')
    val_data_dir = os.path.join(cur_dir, '..', 'data', 'val')

    val_photo_dir = os.path.join(val_data_dir, 'photos')
    val_annot_dir = os.path.join(val_data_dir, 'annotations')

    train_photo_dir = os.path.join(train_data_dir, 'photos')
    train_annot_dir = os.path.join(train_data_dir, 'annotations')

    # Sort so the annotations and photos correspond to each other.
    val_photo_paths = sorted(get_paths(val_photo_dir))
    train_photo_paths = sorted(get_paths(train_photo_dir))
    val_annot_paths = sorted(get_paths(val_annot_dir))
    train_annot_paths = sorted(get_paths(train_annot_dir))

    return (val_photo_paths, train_photo_paths,
            val_annot_paths, train_annot_paths)
