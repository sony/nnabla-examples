# Copyright 2021 Sony Group Corporation
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
import argparse
import numpy as np
from tqdm import trange

import nnabla as nn
import nnabla.functions as F
from nnabla.logger import logger
from nnabla.utils.image_utils import imsave

from models import TCVCGenerator, TCVCGenerator_feat
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", "-L", required=True, type=str)
    parser.add_argument("--no_prev", action="store_true")
    parser.add_argument("--gen_feat", action="store_true")
    parser.add_argument("--dataset_mode", default="test", type=str)
    args, subargs = parser.parse_known_args()

    conf_path = os.path.join(os.path.dirname(args.load_path), "config.yaml")
    conf = read_yaml(conf_path)

    conf.load_path = args.load_path
    conf.no_prev = args.no_prev 
    conf.gen_feat = args.gen_feat
    conf.dataset_mode=args.dataset_mode
    return conf

class Generator(object):
    def __init__(self, image_shape, mconf, no_prev=False, gen_feat=False):
        self.image_shape = image_shape
        self.model_conf = mconf

        self.real_prev = nn.Variable(shape=(1, 3) + self.image_shape)
        self.sketch = nn.Variable(shape=(1, 1) + self.image_shape)

        self.no_prev = no_prev
        self.gen_feat = gen_feat

        self.fake = self.define_network()

    def define_network(self):

        if self.no_prev or self.gen_feat:
            x = self.sketch
        else:
            x = F.concatenate(self.sketch, self.real_prev, axis=1)


        if self.gen_feat:
            print("using TCVC_feat generator")
            from nnabla.models.imagenet import VGG16
            self.vgg = VGG16()
            generator = TCVCGenerator_feat()
            prev_feat= vgg16_get_feat(self.real_prev, self.vgg)
            fake, _ = generator(x, prev_feat,
                                channels=self.model_conf.gg_channels,
                                downsample_input=False, 
                                n_residual_layers=self.model_conf.gg_num_residual_loop)
        else:
            generator = TCVCGenerator()
            fake, _ = generator(x,
                                channels=self.model_conf.gg_channels,
                                downsample_input=False, 
                             n_residual_layers=self.model_conf.gg_num_residual_loop)

        return fake

    @staticmethod
    def _check_ndarray(x):
        if not isinstance(x, np.ndarray):
            raise ValueError("image must be np.ndarray.")

    def __call__(self, sketch_np, real_prev_np):
        self._check_ndarray(sketch_np)
        self._check_ndarray(real_prev_np)

        self.sketch.d = sketch_np
        self.real_prev.d = real_prev_np
        self.fake.forward(clear_buffer=True)

        return self.fake.d


def generate():
    conf = get_config()

    # batch_size is forced to be 1
    conf.train.batch_size = 1

    image_shape = tuple(x * conf.model.g_n_scales for x in [256, 256])
    # set context
    comm = init_nnabla(conf.nnabla_context)

    dataset_path = conf.tcvc_dataset.data_dir
    data_iter = create_tcvc_iterator(conf.train.batch_size, dataset_path,
                                            image_shape=image_shape, shuffle=False,
                                            rng=None, flip=False, dataset_mode=conf.dataset_mode, is_train=False)
    if comm.n_procs > 1:
        data_iter = data_iter.slice(
            rng=None, num_of_slices=comm.n_procs, slice_pos=comm.rank)

    # define generator
    generator = Generator(image_shape=image_shape, mconf=conf.model, no_prev=conf.no_prev, gen_feat=conf.gen_feat)

    # load parameters
    if not os.path.exists(conf.load_path):
        logger.warn("Path to load params is not found."
                    " Loading params is skipped and generated result will be unreasonable. ({})".format(conf.load_path))

    nn.load_parameters(conf.load_path)

    progress_iterator = trange(data_iter._size // conf.train.batch_size,
                               desc="[Generating Images]", disable=comm.rank > 0)


    save_path = os.path.join(conf.train.save_path, "gen_results", conf.dataset_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for i in progress_iterator:
        image, image_prev, sketch_id = data_iter.next()
        
        # for reference-based generator
        if conf.gen_feat:
            if i % 12 == 0:
                ref = image
            gen = generator(sketch_id, ref)
        else: #original tcvc generating method (better give a image_prev every 12 frames as well)
            if i!=0:
                image_prev=temp
            gen = generator(sketch_id, image_prev)
            temp = gen
        '''
        else: #give a ground truth every 12 frames
            if i==1 or i % 12 == 0:
                gen = generator(sketch_id, image_prev)
            else:
                image_prev=temp
                gen = generator(sketch_id, image_prev)
            temp = gen

        '''

        gen = (gen - gen.min()) / (gen.max() - gen.min())

        gen_image_path = os.path.join(
            save_path, "res_{}.png".format(str(i).zfill(5)))

        imsave(gen_image_path, gen[0], channel_first=True)

if __name__ == '__main__':
    generate()
