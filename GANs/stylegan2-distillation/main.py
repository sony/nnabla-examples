import os
import sys

import nnabla as nn
from nnabla.ext_utils import get_extension_context
from nnabla.monitor import Monitor

common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.comm import CommunicatorWrapper
from neu.yaml_wrapper import read_yaml, write_yaml

from argparse import ArgumentParser
import time

from train import Trainer
from data_iterators.attribute_faces import get_data_iterator


def make_parser():
    parser = ArgumentParser(description='StyleGAN2: Nnabla implementation')
    
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to the folder containing the images')
    parser.add_argument("--device-id", default=1, type=int, 
                        help='Device ID of the GPU for training')
    
    return parser

if __name__ == '__main__':

    parser = make_parser()
    config = read_yaml(os.path.join('configs', 'gender.yaml'))
    args = parser.parse_args()
    config.nnabla_context.device_id = args.device_id
    config.gender_faces.data_dir = args.data_root
    
    ctx = get_extension_context(config.nnabla_context.ext_name, device_id=args.device_id)
    nn.set_auto_forward(True)
    
    comm = CommunicatorWrapper(ctx)
    nn.set_default_context(ctx)
          
    di = get_data_iterator(args.data_root, comm, config.train.batch_size, config.model.base_image_shape, 
                            img_exts=['_o.png', '_y.png'])
    
    trainer = Trainer(config.train, config.model, comm, di)
    
    trainer.train()

    if comm.rank == 0:
        print('Completed!')
          