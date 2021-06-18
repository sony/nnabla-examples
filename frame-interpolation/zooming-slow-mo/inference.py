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

import os.path as osp
import argparse
import glob
import cv2
import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from models import zooming_slo_mo_network
import utils.utils as util


parser = argparse.ArgumentParser(
    description='Zooming-SloMo or only Slo-Mo Inference')
parser.add_argument('--input-dir', type=str, default='test_example/',
                    help='input data directory, expected to have input frames')
parser.add_argument('--model', type=str, default='ZoomingSloMo_NNabla.h5',
                    help='model path')
parser.add_argument('--context', type=str, default='cudnn',
                    help="Extension modules. ex) 'cpu', 'cudnn'.")
parser.add_argument('--metrics', action='store_true', default=False,
                    help='calculate metrics i.e. SSIM and PSNR')
parser.add_argument('--only-slomo', action='store_true', default=False,
                    help='If True, Slo-Mo only Inference (No Zooming)')

args = parser.parse_args()


def test():
    """
    Test(Zooming SloMo) - inference on set of input data or Vid4 data
    """
    # set context and load the model
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    nn.load_parameters(args.model)
    input_dir = args.input_dir
    n_ot = 7

    # list all input sequence folders containing input frames
    inp_dir_list = sorted(glob.glob(input_dir + '/*'))
    inp_dir_name_list = []
    avg_psnr_l = []
    avg_psnr_y_l = []
    avg_ssim_y_l = []
    sub_folder_name_l = []
    save_folder = 'results'
    # for each sub-folder
    for inp_dir in inp_dir_list:
        gt_tested_list = []
        inp_dir_name = inp_dir.split('/')[-1]
        sub_folder_name_l.append(inp_dir_name)

        inp_dir_name_list.append(inp_dir_name)
        save_inp_folder = osp.join(save_folder, inp_dir_name)
        img_low_res_list = sorted(glob.glob(inp_dir + '/*'))

        util.mkdirs(save_inp_folder)
        imgs = util.read_seq_imgs_(inp_dir)

        img_gt_l = []
        if args.metrics:
            replace_str = 'LR'
            for img_gt_path in sorted(glob.glob(osp.join(inp_dir.replace(replace_str, 'HR'), '*'))):
                img_gt_l.append(util.read_image(img_gt_path))

        avg_psnr, avg_psnr_sum, cal_n = 0, 0, 0
        avg_psnr_y, avg_psnr_sum_y = 0, 0
        avg_ssim_y, avg_ssim_sum_y = 0, 0

        skip = args.metrics

        select_idx_list = util.test_index_generation(
            skip, n_ot, len(img_low_res_list))

        # process each image
        for select_idxs in select_idx_list:
            # get input images
            select_idx = [select_idxs[0]]
            gt_idx = select_idxs[1]
            imgs_in = F.gather_nd(
                imgs, indices=nn.Variable.from_numpy_array(select_idx))
            imgs_in = F.reshape(x=imgs_in, shape=(1,) + imgs_in.shape)
            output = zooming_slo_mo_network(imgs_in, args.only_slomo)
            outputs = output[0]
            outputs.forward(clear_buffer=True)

            for idx, name_idx in enumerate(gt_idx):
                if name_idx in gt_tested_list:
                    continue
                gt_tested_list.append(name_idx)
                output_f = outputs.d[idx, :, :, :]
                output = util.tensor2img(output_f)
                cv2.imwrite(osp.join(save_inp_folder,
                                     '{:08d}.png'.format(name_idx + 1)), output)
                print("Saving :", osp.join(save_inp_folder,
                                           '{:08d}.png'.format(name_idx + 1)))

                if args.metrics:
                    # calculate PSNR
                    output = output / 255.
                    ground_truth = np.copy(img_gt_l[name_idx])
                    cropped_output = output
                    cropped_gt = ground_truth

                    crt_psnr = util.calculate_psnr(
                        cropped_output * 255, cropped_gt * 255)
                    cropped_gt_y = util.bgr2ycbcr(cropped_gt, only_y=True)
                    cropped_output_y = util.bgr2ycbcr(
                        cropped_output, only_y=True)
                    crt_psnr_y = util.calculate_psnr(
                        cropped_output_y * 255, cropped_gt_y * 255)
                    crt_ssim_y = util.calculate_ssim(
                        cropped_output_y * 255, cropped_gt_y * 255)

                    avg_psnr_sum += crt_psnr
                    avg_psnr_sum_y += crt_psnr_y
                    avg_ssim_sum_y += crt_ssim_y
                    cal_n += 1

        if args.metrics:
            avg_psnr = avg_psnr_sum / cal_n
            avg_psnr_y = avg_psnr_sum_y / cal_n
            avg_ssim_y = avg_ssim_sum_y / cal_n

            avg_psnr_l.append(avg_psnr)
            avg_psnr_y_l.append(avg_psnr_y)
            avg_ssim_y_l.append(avg_ssim_y)

    if args.metrics:
        print('################ Tidy Outputs ################')
        for name, ssim, psnr_y in zip(sub_folder_name_l, avg_ssim_y_l, avg_psnr_y_l):
            print(
                'Folder {} - Average SSIM: {:.6f}  PSNR-Y: {:.6f} dB. '.format(name, ssim, psnr_y))
        print('################ Final Results ################')
        print('Total Average SSIM: {:.6f}  PSNR-Y: {:.6f} dB for {} clips. '.format(
            sum(avg_ssim_y_l) / len(avg_ssim_y_l), sum(avg_psnr_y_l) /
            len(avg_psnr_y_l),
            len(inp_dir_list)))


if __name__ == '__main__':
    test()
