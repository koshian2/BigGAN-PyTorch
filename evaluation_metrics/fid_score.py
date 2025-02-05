#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# https://github.com/mseitzer/pytorch-fid

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d
import torchvision
import pickle

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from evaluation_metrics.inception import InceptionV3
from evaluation_metrics.eval_preprocess import IgnoreLabelDataset, tensor_to_dataset

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def get_activations(imgs, model, batch_size=256, dims=2048,
                    cuda=True):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- imgs        : Dataloader of images
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    assert type(imgs) is IgnoreLabelDataset
    model.eval()

    N = len(imgs)
    pred_arr = np.empty((N, dims))
    up = torch.nn.UpsamplingBilinear2d(size = (299, 299))
    if cuda:
        up = up.cuda()

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            if cuda:
                batch = batch.cuda()
            batch_size_i = batch.size()[0]

            pred = model(up(batch))[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            start = i * batch_size
            end = i * batch_size + batch_size_i
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size_i, -1)

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(imgs, batch_size=256, dims=2048, cuda=True):
    """Calculation of the statistics used by the FID.
    Params:
    -- imgs        : Data loader of images (IgnoreLabelDataset)
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model = torch.nn.DataParallel(model.cuda())

    act = get_activations(imgs, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


## main function ##
def calculate_fid_given_tensor(generated_tensor, original_statistics_filename,
                               batch_size=256, cuda=True, dims=2048):
    """Calculates the FID of tensor"""
    pkl_path = f"./evaluation_metrics/{original_statistics_filename}"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError("Invalid path: %s" % pkl_path)

    # generated image statstics
    dataset = tensor_to_dataset(generated_tensor)
    m1, s1 = calculate_activation_statistics(dataset,
                        batch_size=batch_size, dims=dims, cuda=cuda)

    # original image statictics
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
    m2, s2 = data["mu"], data["sigma"]

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

## cache stastics (original images)

def cifar10_fid_statistics(train=True, overwrite=False):
    dataset = torchvision.datasets.CIFAR10(root="./data", download=True, train=train)
    X = np.asarray(dataset.data, dtype=np.float32).transpose([0, 3, 1, 2])  # (50000, 3, 32, 32), [0-255]    
    X = X / 127.5 - 1.0
    # dataset
    dataset = tensor_to_dataset(torch.as_tensor(X))
    # stastics
    mu, sigma = calculate_activation_statistics(dataset)
    file_suffix = "train" if train else "test"

    if overwrite:
        with open(f"evaluation_metrics/cifar10_{file_suffix}.pkl", "wb") as fp:
            pickle.dump({"mu": mu, "sigma": sigma}, fp)
