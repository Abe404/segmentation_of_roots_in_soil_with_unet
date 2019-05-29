"""
General functions for hanlding the frangi vesselness filter from skimage.

This code is a modified version of the scikit-image frangi filter.

Which can be viewed online at:
https://github.com/scikit-image/scikit-image/blob/v0.14.x/skimage/filters/_frangi.py

Original Work : Copyright (C) 2019, the scikit-image team
Modified work : Copyright (C) 2019 Abraham Smith
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
from skimage import exposure
from skimage.color import rgb2grey
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import numpy as np

# pylint: disable=C0413
sys.path.append('.')
from data_utils import zero_one


CACHE_DIR = '../output/frangi/frangi_lambda_cache'


def get_thresholded(image, thresh):
    """ sets any pixels below threshold to 0
        then sets any pixels above zero to 1
        Final image is just 0 and 1s
    """
    new_image = np.ones(image.shape)
    new_image[image <= thresh] = 0
    return zero_one(new_image)


def is_in_lambda_cache(key):
    """ checks lambda_cache folder for cached intermediary frangi calculations """
    return os.path.isfile(os.path.join(CACHE_DIR, key + '.npy'))


def save_to_cache(key, lambdas):
    """ saves lambdas to lambda_cache on disk """
    os.path.isfile(os.path.join(CACHE_DIR, key))
    np.save(os.path.join(CACHE_DIR, key), lambdas)


def ensure_cache_dir_exists():
    """ Create the cache directory if it doesn't exist already """
    if not os.path.exists(CACHE_DIR):
        print('Lambda cached dir did not exist, so creating it')
        os.makedirs(CACHE_DIR)


def load_from_cache(key):
    """ loads intermediary frangi calculations from lambda_cache folder """
    return np.load(os.path.join(CACHE_DIR, key + '.npy'))


def _frangi_hessian_common_filter(idx, image, sigmas, beta1, beta2):
    """
        See skimage implementation and documentation.

        Modified from skimage to facilitate intermediary caching
        and the exponential scale to improve performance of frangi
        when same images are being used multiple times.
        (As is the case when using CMA-ES optimisations).

    """
    # pylint: disable=too-many-locals
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    beta1 = 2 * beta1 ** 2
    beta2 = 2 * beta2 ** 2

    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros(sigmas.shape + image.shape)


    use_cache = idx is not None

    if use_cache:
        ensure_cache_dir_exists()
    # Disable naming warnings as this is skimage code that I don't have time to rewrite.
    # pylint: disable=invalid-name
    # Filtering for all sigmas
    for i, sigma in enumerate(sigmas):
        key = str(idx) + ':' + str(sigma)
        # idx of None implies do not use key
        if use_cache and is_in_lambda_cache(key):
            lambda1, rb, s2 = load_from_cache(key)
        else:
            # Make 2D hessian
            D = hessian_matrix(image, sigma, order='rc')
            # Correct for scale
            D = np.array(D) * (sigma ** 2)
            # Calculate (abs sorted) eigenvalues and vectors
            lambda1, lambda2 = hessian_matrix_eigvals(D)

            # Compute some similarity measures
            lambda1[lambda1 == 0] = 1e-10
            rb = (lambda2 / lambda1) ** 2
            s2 = lambda1 ** 2 + lambda2 ** 2
            if use_cache:
                save_to_cache(key, (lambda1, rb, s2))

        # Compute the output image
        filtered = np.exp(-rb / beta1) * (np.ones(np.shape(image)) - np.exp(-s2 / beta2))

        # Store the results in 3D matrices
        filtered_array[i] = filtered
        lambdas_array[i] = lambda1
    # pylint: enable=invalid-name
    # pylint: enable=too-many-locals
    return filtered_array, lambdas_array


def frangi(key, image, sigmas, beta1=0.5, beta2=15):
    """ Filter an image with the Frangi filter.
        This filter can be used to detect continuous edges, e.g. vessels,
        wrinkles, rivers. It can be used to calculate the fraction of the
        whole image containing such objects.
    """
    sigmas = np.array(sigmas)
    filtered, lambdas = _frangi_hessian_common_filter(key, image, sigmas, beta1, beta2)
    filtered[lambdas < 0] = 0 # when black ridges
    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    return np.max(filtered, axis=0)


def get_sigmas(sigma_max: int):
    """ designed to get the exponential scale instead of linear scale for the frangis """
    assert sigma_max > 0
    return [1] + list(np.logspace(start=1, stop=sigma_max-1, num=sigma_max-1, base=2))


def get_frangi(image_id, photo, params):
    """ returns the frangi intensities for the photo with the specified params
        params are a list to facilitate easy optimisation with CMA-ES.
    """
    assert len(params) == 3
    sigma_max, beta1, beta2 = params

    # frangi needs greyscale and rgb2grey needs image in 0-1 range
    photo = exposure.rescale_intensity(photo, out_range=(0, 1))
    grey_photo = rgb2grey(photo)
    inverted_photo = 1 - grey_photo
    sigmas = get_sigmas(sigma_max)
    frangi_output = frangi(image_id,
                           inverted_photo,
                           sigmas,
                           beta1=abs(beta1),
                           beta2=abs(beta2))
    return frangi_output
