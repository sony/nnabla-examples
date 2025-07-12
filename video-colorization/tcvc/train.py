import argparse
import os

from trainer import Trainer
from utils import *


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config.yaml")
    args, subargs = parser.parse_known_args()

    conf = read_yaml(args.cfg)

    # nnabla execution context args
    parser.add_argument(
        "--device-id", default=conf.nnabla_context.device_id, type=int)
    parser.add_argument("--ext_name", default=conf.nnabla_context.ext_name)
    parser.add_argument(
        "--type-config", default=conf.nnabla_context.type_config)


    # training args
    parser.add_argument("--fix-global-epoch", default=conf.train.fix_global_epoch, type=int,
                        help="Number of epochs where the global generator's parameters are fixed.")
    parser.add_argument("--save-path", default=conf.train.save_path)
    parser.add_argument("--load-path", default=conf.train.load_path)

    parser.add_argument("--no_prev", default=conf.train.no_prev, action="store_true", help='if specified, do *not* use previous frame during training')
    parser.add_argument("--gen_feat", default=conf.train.gen_geat, action="store_true", 
                        help='if specified, instead of using prev as input, use VGG-16 to extract features from prev for mid-level res-blocks')
    parser.add_argument("--full_prev", default=conf.train.full_prev, action="store_true", 
                        help='always give ground truth previous frame, can only be True when gen_feat=True')

    # model args
    parser.add_argument("--d-n-scales", default=conf.model.d_n_scales, type=int,
                        help="Number of layers of discriminator pyramids")
    parser.add_argument("--g-n-scales", default=conf.model.g_n_scales, type=int,
                        help="A number of generator resolution stacks. If 1, only global generator is used.")

    args = parser.parse_args()

    # refine config
    conf.nnabla_context.update(
        {"device_id": args.device_id, "ext_name": args.ext_name, "type_config": args.type_config})
    conf.train.fix_global_epoch = args.fix_global_epoch
    conf.model.d_n_scales = args.d_n_scales
    conf.model.g_n_scales = args.g_n_scales

    conf.train.save_path = args.save_path
    conf.train.no_prev = args.no_prev
    conf.train.gen_feat = args.gen_feat
    return conf


if __name__ == '__main__':
    conf = get_config()

    comm = init_nnabla(conf.nnabla_context)

    dataset_path = conf.tcvc_dataset.data_dir

    # dump conf to file
    if comm.rank == 0:
        write_yaml(os.path.join(conf.train.save_path, "config.yaml"), conf)

    # define trainer
    trainer = Trainer(conf.train, conf.model, comm, dataset_path)

    trainer.train()
