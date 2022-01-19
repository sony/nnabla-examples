# Copyright 2021 Sony Corporation.
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

import nnabla as nn

import os
import warnings

from models import Generator


class BaseExecution(object):

    def __init__(self, monitor, config, args, comm, few_shot_config):

        self.config = config['train']
        self.comm = comm

        self.few_shot_config = few_shot_config

        self.img_size = args.img_size
        self.batch_size = args.batch_size

        self.auto_forward = args.auto_forward
        self.device_id = args.device_id

        os.makedirs(args.results_dir, exist_ok=True)
        self.results_dir = args.results_dir

        global_scope = 'Generator' if args.train else 'GeneratorEMA'
        # global_scope = 'Generator'

        self.generator = Generator(
            config['generator'], self.img_size, config['train']['mix_after'], global_scope=global_scope)

        self.load_checkpoint(args)

    def load_checkpoint(self, args):
        """Load pretrained parameters and solver states

        Args:
                args (ArgumentParser): To check if tensorflow trained weights are to be used for testing and to get the path of the folder 
                                                                from where the parameter and solver states are to be loaded
        """

        if args.use_tf_weights:
            if not os.path.isfile(os.path.join(args.weights_path, 'gen_params.h5')):
                os.makedirs(args.weights_path, exist_ok=True)
                print("Downloading the pretrained tf-converted weights. Please wait...")
                url = "https://nnabla.org/pretrained-models/nnabla-examples/GANs/stylegan2/styleGAN2_G_params.h5"
                from nnabla.utils.data_source_loader import download
                download(url, os.path.join(
                    args.weights_path, 'gen_params.h5'), False)
            nn.load_parameters(os.path.join(
                args.weights_path, 'gen_params.h5'))

            print('Loaded pretrained weights from tensorflow!')

        else:
            try:
                if args.pre_trained_model is not None:
                    if os.path.isfile(args.pre_trained_model):
                        nn.load_parameters(args.pre_trained_model)
                    elif os.path.isfile(os.path.join(args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5')):
                        if args.train:
                            with nn.parameter_scope('Generator'):
                                nn.load_parameters(os.path.join(
                                    args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5'))
                            with nn.parameter_scope('GeneratorEMA'):
                                nn.load_parameters(os.path.join(
                                    args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5'))
                            if self.few_shot_config['common']['type'] == 'cdc':
                                with nn.parameter_scope('Source'):
                                    nn.load_parameters(os.path.join(
                                        args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5'))
                        else:
                            nn.load_parameters(os.path.join(
                                args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5'))
                            nn.load_parameters(os.path.join(
                                args.pre_trained_model, 'ffhq-slim-gen-256-config-e.h5'))
                        if os.path.isfile(os.path.join(args.pre_trained_model, 'ffhq-slim-disc-256-config-e-corrected.h5')):
                            nn.load_parameters(os.path.join(
                                args.pre_trained_model, 'ffhq-slim-disc-256-config-e-corrected.h5'))
                if os.path.isdir(args.weights_path):
                    with nn.parameter_scope('Discriminator'):
                        nn.load_parameters(os.path.join(
                            args.weights_path, 'disc_params.h5'))
                    with nn.parameter_scope('Generator'):
                        nn.load_parameters(os.path.join(
                            args.weights_path, 'gen_params.h5'))
                    with nn.parameter_scope('GeneratorEMA'):
                        nn.load_parameters(os.path.join(
                            args.weights_path, 'gen_ema_params.h5'))
            except:
                if args.test:
                    warnings.warn(
                        "Testing Model without pretrained weights!!!")
                else:
                    print('No Pretrained weights loaded.')

    def save_weights(self, path, epoch):
        """Save parameters and solver states

        Args:
                path (str): Location of the storage path
                epoch (int): Training epochs
        """

        file_name = os.path.join(path, 'epoch_'+str(epoch))
        os.makedirs(file_name, exist_ok=True)

        with nn.parameter_scope('Discriminator'):
            nn.save_parameters(os.path.join(file_name, 'disc_params.h5'))
        with nn.parameter_scope('Generator'):
            nn.save_parameters(os.path.join(file_name, 'gen_params.h5'))
        with nn.parameter_scope('GeneratorEMA'):
            nn.save_parameters(os.path.join(file_name, 'gen_ema_params.h5'))

        self.gen_solver.save_states(os.path.join(file_name, 'gen_solver.h5'))
        self.disc_solver.save_states(os.path.join(file_name, 'disc_solver.h5'))

        print(f'Model weights and Solver states saved at {file_name}!')
