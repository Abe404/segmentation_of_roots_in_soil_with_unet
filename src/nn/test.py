# pylint: disable=C0111, R0914, C0103
""" Get the metrics on the test data
    using the pre-trained model.

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
import argparse
import os
import sys
import warnings
import csv
from PIL import Image
import numpy as np


import torch

from torch.nn.functional import sigmoid
from skimage.io import (imread, imsave)
from skimage.exposure import rescale_intensity
from skimage.morphology import skeletonize
from skimage import img_as_float
import scipy
from unet import UNetGN
import im_utils
from sys_utils import multi_process
# pylint: disable=C0413
sys.path.append('.')
from metrics import print_metrics_from_dirs


def unet_segment(cnn, image, device):
    photo_shape = image.shape
    tiles, coords = im_utils.get_tiles(image,
                                       in_tile_shape=(572, 572, 3),
                                       out_tile_shape=(388, 388))
    for i, tile in enumerate(tiles):
        # Move channels from first to last
        tiles[i] = np.moveaxis(tile, -1, 0)

    cnn.eval()
    with torch.no_grad():
        output_tiles = []
        for i, tile in enumerate(tiles):
            tile = tile.astype(np.float32)
            tile = rescale_intensity(tile.astype(np.float32), out_range=(0, 1))
            tile -= 0.5
            tile = torch.from_numpy(np.array([tile]))
            if device.type != "cpu":
                tile = tile.to(device, non_blocking=True)

            outputs = cnn(tile)
            _, predicted = torch.max(outputs.data, 1)
            root_probs = sigmoid(outputs).squeeze(1)
            predicted = root_probs > 0.5
            predicted = predicted.view(-1).int()
            pred_np = predicted.data.cpu().numpy()
            out_tile = pred_np.reshape((388, 388))
            output_tiles.append(out_tile)

        reconstructed = im_utils.reconstruct_from_tiles(output_tiles, coords,
                                                        photo_shape[:-1])
    return reconstructed


def segment_dir_with_unet(checkpoint_path, in_dir, out_dir):
    """ segment data from in_dir using u-net """
    print('segment', in_dir, 'to', out_dir)
    assert os.path.isfile(checkpoint_path), \
            f'checkpoint {checkpoint_path}"  not found'
    assert os.path.isdir(in_dir), f'{in_dir} is not a dir'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    file_paths = os.listdir(in_dir)
    cnn = UNetGN()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    cnn.to(device)
    print('loading checkpoint', checkpoint_path)
    cnn.load_state_dict(torch.load(
        checkpoint_path, map_location=device, weights_only=True))
    for i, path in enumerate(file_paths):
        print('segmenting', i + 1, 'out of', len(file_paths))
        test_file = imread(os.path.join(in_dir, path))
        segmented = unet_segment(cnn, test_file, device)
        print('segmented sum', np.sum(segmented), 'unique', np.unique(segmented))
        out_path = os.path.join(out_dir, path)
        print('saving', out_path)
        # catch warnings as low contrast is ok here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = Image.fromarray(segmented * 255)
            if im.mode != '1':
                im = im.convert('1')
                im.save(out_path)


def skel_im(in_dir_path, out_dir_path, fname):
    """ skeletonize and save image """
    seg_im = img_as_float(imread(os.path.join(in_dir_path, fname)))
    skel = skeletonize(seg_im)
    skel = skel.astype(int)
    skel[skel > 0] = 255
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #imsave(os.path.join(out_dir_path, fname), skel)
        out_path = os.path.join(out_dir_path, fname)
        im = Image.fromarray(skel.astype(np.uint8))
        if im.mode != '1':
            im = im.convert('1')
            im.save(out_path)




def pixel_count(skel_dir, fname):
    image = imread(os.path.join(skel_dir, fname))
    return np.sum(image.astype(bool))


def save_skels_csv(skel_dir, output_csv_path):
    skel_fnames = sorted([f for f in os.listdir(skel_dir) if 'png' in f])
    assert skel_fnames, 'should be skel_fnames'
    root_intensities = multi_process(pixel_count, [skel_dir], skel_fnames)
    print('Saves skel len csv to ', output_csv_path)
    with open(output_csv_path, 'w')  as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for root_intensity, file_name in zip(root_intensities, skel_fnames):
            name = file_name.replace('.png', '')
            writer.writerow([name, root_intensity])
            print(name, root_intensity)


def csv_parts_list(fpath):
    lines = open(fpath).readlines()
    file_names = []
    root_intensities = []
    for line in lines:
        file_name, root_intensity = line.split(',')
        file_names.append(file_name.strip())
        root_intensities.append(root_intensity.strip())
    return file_names, root_intensities


def join_data(manual_ri_csv_path, cnn_skel_csv_path):
    man_fnames, man_ris = csv_parts_list(manual_ri_csv_path)
    cnn_fnames, cnn_skel_lens = csv_parts_list(cnn_skel_csv_path)
    man_fnames, man_ris = zip(*sorted(zip(man_fnames, man_ris)))
    cnn_fnames, cnn_skel_lens = zip(*sorted(zip(cnn_fnames, cnn_skel_lens)))
    assert man_fnames == cnn_fnames
    assert len(man_fnames) == len(cnn_skel_lens) == len(man_ris)
    cnn_skel_lens = [int(l) for l in cnn_skel_lens]
    man_ris = [float(i) for i in man_ris]
    return man_fnames, cnn_skel_lens, man_ris


def print_correlations(cnn_skel_lens, manual_ris):
    x = cnn_skel_lens
    y = manual_ris

    # slope, intercept, ...
    _, _, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    print('Spearman rank: ', scipy.stats.spearmanr(a=x, b=y, axis=0))
    print('Linear regression: r squared: ', r_value**2,
          'p_value: ', p_value, 'std_err: ', std_err)


def process_2016_grid_counted(checkpoint_path):

    input_dir = '../data/grid_counted'
    out_dir = '../output/unet/test_output'
    seg_dir = os.path.join(out_dir, 'grid_counted_seg')
    skel_dir = os.path.join(out_dir, 'grid_counted_skel')
    skel_csv_path = os.path.join(out_dir, 'grid_counted_skel_len.csv')

    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)

    if not os.path.isdir(skel_dir):
        os.makedirs(skel_dir)

    segment_dir_with_unet(
        checkpoint_path,
        input_dir,
        seg_dir
    )

    # Get skeletons
    multi_process(skel_im, [seg_dir, skel_dir], os.listdir(seg_dir))

    # Output skeleton length to csv
    save_skels_csv(skel_dir, skel_csv_path)

    manual_csv_path = os.path.join('../data/', 'grid_counted_manual_ri.csv')
    _, cnn_skel_lens, manual_ris = join_data(manual_csv_path, skel_csv_path)
    print_correlations(cnn_skel_lens, manual_ris)


def segment_and_eval(subset, checkpoint_path):
    segment_dir_with_unet(
        checkpoint_path,
        f'../data/{subset}/photos',
        f'../output/unet/{subset}_output/{subset}_segmentations'
    )
    print_metrics_from_dirs(
        f'../data/{subset}/annotations',
        f'../output/unet/{subset}_output/{subset}_segmentations'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict and Evaluate using a model checkpoint")
    parser.add_argument(
        "-c", "--checkpoint",
        default="../saved_output/unet/checkpoint_73.pkl"
    )
    parser.add_argument("-s", "--subsets", nargs="+",
                        help="One or more of train, val, test")
    args = parser.parse_args()
    for subset in args.subsets:
        segment_and_eval(subset, args.checkpoint)
