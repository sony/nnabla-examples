from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datasets.dataset.pascal_config import PascalVOCDefaultParams
from datasets.dataset.coco_config import COCODefaultParams

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # basic experiment setting
        self.parser.add_argument('task', default='ctdet',
                                 help='ctdet')
        self.parser.add_argument('--dataset', default='coco',
                                 help='coco | pascal')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', type=int, default=0,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: save all visualizations to disk')
        self.parser.add_argument('--demo', default=None,
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')
        self.parser.add_argument('--extension_module', default='cudnn',
                                 help='NNabla extension module. '
                                      'cpu | cuda | cudnn')
        self.parser.add_argument('--data_dir', type=str, default='/home/ubuntu/data/',
                                 help='Path to root directory of dataset.')
        self.parser.add_argument('--root_output_dir', type=str,
                                 default=os.path.join(os.path.dirname(__file__), '..', '..'),
                                 help='Path to root directory of output data.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # Used for debug/result confirmation

        # log
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--save_intervals', type=int, default=5,
                                 help='number of epochs to save weights.')
        self.parser.add_argument('--val_calc_map', action='store_true',
                                 help='Disable AP calculation during validation.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.2,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'dlav0'],
                                 help='model architecture.')
        self.parser.add_argument('--num_layers', type=int, choices=[18, 34], default=18,
                                 help='number of layers of feature extractor architecture.')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=5e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                 help='drop learning rate by at given epochs.')
        self.parser.add_argument('--lr_decay', type=float, default=0.1,
                                 help='Multiply LR by given value every drop.')
        self.parser.add_argument('--weight_decay', type=float, default=0.0,
                                 help='weight decay parameter.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='batch size')
        self.parser.add_argument('--warmup', type=int, default=0,
                                 help='number of warmup epochs')
        self.parser.add_argument('--checkpoint', type=str, default='',
                                 help='checkpoint file to resume training.')
        self.parser.add_argument('--mixed_precision', action='store_true',
                                 help='Mixed Precision training using NVIDIA tensor cores.')
        self.parser.add_argument('--channel_last', action='store_true',
                                 help='Channel last models. Currently only DLAv0 is supported')
        self.parser.add_argument('--checkpoint_dir', type=str, default='',
                                 help='Root folder that includes checkpoint(s) for test.')
        # test
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')

        self.parser.add_argument('--K', type=int, default=100,
                             help='max number of output objects.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')

        # dataset
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation'
                                      'from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.1,
                                 help='when not using random crop'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='when not using random crop'
                                      'apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='not use the color augmenation '
                                      'from CornerNet')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        # task
        # ctdet
        self.parser.add_argument('--reg_offset', action='store_false',
                                 help='Regress local offset.')

        # monitor setting
        self.parser.add_argument('--train_monitor_interval', type=int, default=1)
        self.parser.add_argument('--eval_monitor_interval', type=int, default=1)


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(
            len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dlav0' in opt.arch else 64
        opt.down_ratio = 4
        opt.pad = 31
        opt.num_stacks = 1

        opt.exp_dir = os.path.join(opt.root_output_dir, "exp", opt.task)
        opt.save_dir = os.path.join(opt.exp_dir,opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)
        os.makedirs(opt.save_dir, exist_ok=True)

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        opt.train_size = dataset.train_size
        opt.eval_size = dataset.eval_size
        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)
        if opt.task == 'ctdet':
            # assert opt.dataset in ['pascal', 'coco']
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2}  # if not opt.cat_spec_wh else 2 * opt.num_classes}
            opt.heads.update({'reg': 2})
        else:
            assert 0, 'task {} not defined!'.format(opt.task)
        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        default_dataset_info = {
                'coco': COCODefaultParams,
                'pascal': PascalVOCDefaultParams,
                }
        opt = self.parse(args)
        dataset = default_dataset_info[opt.dataset]
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
