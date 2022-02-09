# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import glob
import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from tqdm import tqdm
from .im2ndarray import npy2ndarray
from .inceptionv3 import construct_inceptionv3
from .common import get_data_iterator
from nnabla.ext_utils import get_extension_context


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='paths to the directory containing fake images, \
                              or the text file listing fake images.')
    parser.add_argument('--params-path',
                        default=os.path.dirname(os.path.abspath(__file__)) +
                        '/original_inception_v3.h5', type=str,
                        help='path to the weight file (.h5).')
    parser.add_argument('--batch-size', '-b', default=16, type=int,
                        help='batch-size. automatically adjusted. see the code.')
    parser.add_argument('--splits', '-s', default=1, type=int,
                        help='number of image sets to compute scores.')
    parser.add_argument('--epsilon', '-e', default=0.0, type=float,
                        help='epsilon value used to avoid zero-division error.')
    parser.add_argument('--context', '-c', default="cudnn", type=str,
                        help='Backend name.')
    parser.add_argument('--device-id', '-d', default=0, type=int,
                        help='Device ID.')
    parser.add_argument('--faster-resize', action="store_true",
                        help='If specified, nnabla.interpolate is used to perform resize on GPU.')

    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Compute KL-divergence between 2 distributions.
        Args:
            p (np.ndarray): p(y|x). (B, 1008)
            q (np.ndarray): p(y).   (1, 1008)
            eps (float): small value used to avoid zero division error.
                         increase its value if error raises.
        Returns:
            kld (np.ndarray): KL-divergence. (B, 1, 1, 1)
    """
    kld = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)
    return kld


def marginal_dist(py_given_x: np.ndarray) -> np.ndarray:
    """Compute a marginal distribution of p(y|x) over x.
        Args:
            py_given_x (np.ndarray): p(y|x). (B, 1008)

        Returns:
            _ (np.ndarray): p(y). a marginal distribution of y. (1, 1008)
    """
    return np.mean(py_given_x, axis=0, keepdims=True)


def get_conditional_dist(fake_images):
    """Get the prediction score using Inception v3.
        Args:
            fake_images (nn.NdArray):  NdArrays representing images.
                                       Shape must be (B, 3, 299, 299).
                                       Must be pre-normalized, i.e. its values must lie in [-1., +1.]

        Returns:
            py_given_x (nn.NdArray): Class probabilities of given images. (B, 1008)
    """
    py_given_x = construct_inceptionv3(fake_images)
    py_given_x = PF.affine(py_given_x, 1008,
                           name="Affine", with_bias=False)  # strangely, 1008 is correct, and no bias.
    py_given_x = F.softmax(py_given_x)
    return py_given_x


def get_all_features_on_imagepaths(image_paths, batch_size, faster_resize=False):
    """Extract all the given images' feature.
        Args:
            image_paths (list): list of image file's paths.
            batch_size (int): batch size.
            faster_resize (bool): If true, resize is performed on GPU by nnabla.interpolate.
                                  Otherwise, slower but the same resize implementation with tensorflow by numpy will be used.

        Returns:
            all_py_given_x (np.ndarray): (N) images' class probabilities. shape: (N, 1008)
    """
    print("loading images...")
    num_images = len(image_paths)

    pbar = tqdm(total=num_images)
    data_iter = get_data_iterator(image_paths, batch_size,
                                  shuffle=False, stop_exhausted=False, channel_first=True, num_channels=3)
    samples = 0
    all_py_given_x = []
    while samples < num_images:
        d_npy = data_iter.next()[0]
        if samples + len(d_npy) > num_images:
            d_npy = d_npy[:num_images]

        images = npy2ndarray(d_npy, imsize=(299, 299),
                             normalize=True, use_tf_resize=not faster_resize)
        py_given_x = get_conditional_dist(images)

        all_py_given_x.append(py_given_x.data)

        pbar.update(len(d_npy))
        samples += len(d_npy)

    return np.concatenate(all_py_given_x, axis=0)


def load_parameters(params_path):
    if not os.path.isfile(params_path):
        from nnabla.utils.download import download
        url = "https://nnabla.org/pretrained-models/nnabla-examples/eval_metrics/inceptions/original_inception_v3.h5"
        download(url, params_path, False)
    nn.load_parameters(params_path)


def main(args):
    # Context
    ctx = get_extension_context(args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    batch_size = args.batch_size
    eps = args.epsilon

    load_parameters(args.params_path)  # overwrite

    path = args.path
    ext = os.path.splitext(path)[-1]
    if ext == "":
        # path points to a directory
        assert os.path.isdir(path), "specified directory is not found."
        fake_image_paths = glob.glob(
            f"{path}/*.png") + glob.glob(f"{path}/*.jpg")
        # or simply glob.glob(f"{path}/*"). but may retrieve some other files.
    elif ext == ".txt":
        # path points to a text file
        assert os.path.isfile(path), "specified file is not found."
        with open(path, "r") as f:
            fake_image_paths = [_.rstrip("\n") for _ in f.readlines()]
    else:
        raise RuntimeError(f"Invalid path: {path}")

    print("calculating all features of fake data...")
    faster_resize = args.faster_resize
    all_py_given_x = get_all_features_on_imagepaths(
        fake_image_paths, batch_size, faster_resize)

    print("Finished extracting features. Calculating inception Score...")

    splits = args.splits
    interval = all_py_given_x.shape[0] // splits
    scores = []
    for i in range(splits):
        part = all_py_given_x[(i * interval):((i + 1) * interval), :]
        kl = kl_divergence(part, marginal_dist(part), eps)
        scores.append(np.exp(np.mean(kl, axis=0)))

    print(
        f"Image Sets: {path}\nbatch size: {batch_size}\nsplit size: {splits}")
    print(f"Inception Score: {np.mean(scores):.3f}\nstd: {np.std(scores):.3f}")


if __name__ == '__main__':
    args = get_args()
    main(args)
