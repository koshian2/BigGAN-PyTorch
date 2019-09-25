import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

import torchvision
from torchvision.models.inception import inception_v3
from evaluation_metrics.eval_preprocess import tensor_to_dataset

import numpy as np
import os
from scipy.stats import entropy
from tqdm import tqdm

# https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

def inception_score(imgs, cuda=True, batch_size=256, resize=True, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    if cuda:
        inception_model = torch.nn.DataParallel(inception_model)
    inception_model.eval()

    up = nn.UpsamplingBilinear2d(size=(299, 299)).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# 10.254
def calculate_inception_score_given_tensor(image_tensor):
    assert image_tensor.size(1) == 3
    dataset = tensor_to_dataset(image_tensor)
    result = inception_score(dataset)

    return result


# checking for cifar
def cifar_test():
    cifar = torchvision.datasets.CIFAR10(root="./data", download=True, train=True)
    X = np.asarray(cifar.data, dtype=np.float32).transpose([0, 3, 1, 2])  # (50000, 3, 32, 32), [0-255]    
    X = X / 127.5 - 1.0
    is_score = calculate_inception_score_given_tensor(torch.as_tensor(X))
    print("Inception score of CIFAR-10 train set")
    print(is_score)  # should be around 10.25

