# Copyright 2022 Sony Group Corporation.
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

import pickle
from os import path, mkdir
import sys
import os
import numpy as np
import nnabla as nn
from nnabla.utils.image_utils import imsave
from nnabla.ext_utils import get_extension_context
import args
from utils import utils
sys.path.append(os.path.abspath('../../image-generation/pggan/'))
from functions import pixel_wise_feature_vector_normalization
from helpers import load_gen


def convert_images_to_uint8(images, drange=[-1, 1]):
    """
        convert float32 -> uint8
    """
    if isinstance(images, nn.Variable):
        images = images.d
    if isinstance(images, nn.NdArray):
        images = images.data

    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return np.uint8(np.clip(images, 0, 255))


def generate_images(gen, num_images, n_latent=512,
                    hyper_sphere=True, save_dir=None, latent_vector=None):
    """
    generate the images
    Args:
        gen : load generator
        num_images (int) : number of images to generate
        n_latent (int) : 512-D latent space trained on the CelebA
        hyper_sphere (bool) : default True
        save_dir (str) : directory to save the images
        latent_vector (str) : path to save the latent vectors(.pkl file)
    """

    if not path.isdir(save_dir):
        mkdir(save_dir)
    z_data = np.random.randn(num_images, n_latent, 1, 1)
    # Saving latent vectors
    with open(latent_vector, 'wb+') as f:
        pickle.dump(z_data.reshape((num_images, n_latent)), f)
    z = nn.Variable.from_numpy_array(z_data)
    z = pixel_wise_feature_vector_normalization(z) if hyper_sphere else z
    batch_size = 64
    iterations = int(num_images/batch_size)
    if num_images % batch_size != 0:
        iterations += 1
    count = 0
    for ell in range(iterations):
        y = gen(z[ell * batch_size:(ell + 1) * batch_size], test=True)
        images = convert_images_to_uint8(y, drange=[-1, 1])
        for i in range(images.shape[0]):
            imsave(save_dir+'/gen_'+str(count)+'.jpg',
                   images[i], channel_first=True)
            count += 1
    print("images are generated")


def generate_flipped_images(gen, latent_vector, hyper_sphere=True, save_dir=None):
    """
    generate flipped images
    Args:
        gen : generator
        latent_vector(numpy.ndarray) : latent_vector
        hyper_sphere (bool) : default True
        save_dir (str) : directory to save the images
    """
    if not path.isdir(save_dir):
        mkdir(save_dir)
    z_data = np.reshape(
        latent_vector, (latent_vector.shape[0], latent_vector.shape[1], 1, 1))
    z = nn.Variable.from_numpy_array(z_data)
    z = pixel_wise_feature_vector_normalization(z) if hyper_sphere else z
    batch_size = 64  # we have taken batch size of 64
    num_images = latent_vector.shape[0]
    iterations = int(num_images / batch_size)
    if num_images % batch_size != 0:
        iterations += 1
    count = 0
    for ell in range(iterations):
        y = gen(z[ell * batch_size:(ell + 1) * batch_size], test=True)
        images = convert_images_to_uint8(y, drange=[-1, 1])
        for i in range(images.shape[0]):
            imsave(save_dir + '/gen_' + str(count) +
                   '.jpg', images[i], channel_first=True)
            count += 1

    print("all paired images generated")


if __name__ == "__main__":
    # args
    opt = args.get_args()
    experiment = opt['generate']
    num_images = opt['num_images']
    attr_list = utils.get_all_attr()
    # Context
    ctx = get_extension_context(
        opt['context'], device_id=opt['device_id'], type_config=opt['type_config'])
    nn.set_default_context(ctx)
    nn.set_auto_forward(True)

    # Generate config
    model_load_path = opt['generator_model']
    use_bn = False
    last_act = 'tanh'
    use_wscale = True
    use_he_backward = False
    # Load generator
    gen = load_gen(model_load_path, use_bn=use_bn, last_act=last_act,
                   use_wscale=use_wscale, use_he_backward=use_he_backward)
    if experiment == 'orig':
        save_dir = "{}/AllGenImages".format(opt["fake_data_dir"])
        latent_vector = r"{}/latent_vectors.pkl".format(
            opt['record_latent_vector'])
        generate_images(gen, num_images,
                        save_dir=save_dir, latent_vector=latent_vector)

    if experiment == 'flip':
        save_dir = "{}/{}/".format(
            opt["fake_data_dir"], attr_list[opt['attribute']])
        latent = pickle.load(open(r"{}/latent_vectors_{}.pkl".format(
            opt['record_latent_vector'], attr_list[opt['attribute']]), 'rb'))
        generate_flipped_images(gen, latent, save_dir=save_dir)
