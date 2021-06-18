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

import time
import os
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
import numpy as np
from utils import get_hw_boundary, trim_patch_boundary, compute_psnr, save_results_yuv
from args import get_config
from ops import model
from data_loader import read_mat_file


def inference():
    """
    Inference function to generate high resolution hdr images
    """
    conf = get_config()
    ctx = get_extension_context(
        conf.nnabla_context.context, device_id=conf.nnabla_context.device_id)
    nn.set_default_context(ctx)

    data, target = read_mat_file(conf.data.lr_sdr_test, conf.data.hr_hdr_test, conf.data.d_name_test,
                                 conf.data.l_name_test, train=False)

    if not os.path.exists(conf.test_img_dir):
        os.makedirs(conf.test_img_dir)

    data_sz = data.shape
    target_sz = target.shape
    PATCH_BOUNDARY = 10  # set patch boundary to reduce edge effect around patch edges
    test_loss_PSNR_list_for_epoch = []
    inf_time = []
    start_time = time.time()

    test_pred_full = np.zeros((target_sz[1], target_sz[2], target_sz[3]))

    print("Loading pre trained model.........", conf.pre_trained_model)
    nn.load_parameters(conf.pre_trained_model)

    for index in range(data_sz[0]):
        ###======== Divide Into Patches ========###
        for p in range(conf.test_patch ** 2):
            pH = p // conf.test_patch
            pW = p % conf.test_patch
            sH = data_sz[1] // conf.test_patch
            sW = data_sz[2] // conf.test_patch
            H_low_ind, H_high_ind, W_low_ind, W_high_ind = \
                get_hw_boundary(
                    PATCH_BOUNDARY, data_sz[1], data_sz[2], pH, sH, pW, sW)
            data_test_p = nn.Variable.from_numpy_array(
                data.d[index, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :])
            data_test_sz = data_test_p.shape
            data_test_p = F.reshape(
                data_test_p, (1, data_test_sz[0], data_test_sz[1], data_test_sz[2]))
            st = time.time()
            net = model(data_test_p, conf.scaling_factor)
            net.pred.forward()
            test_pred_temp = net.pred.d
            inf_time.append(time.time() - st)
            test_pred_t = trim_patch_boundary(test_pred_temp, PATCH_BOUNDARY, data_sz[1], data_sz[2],
                                              pH, sH, pW, sW, conf.scaling_factor)
            #pred_sz = test_pred_t.shape
            test_pred_t = np.squeeze(test_pred_t)
            test_pred_full[pH * sH * conf.scaling_factor: (pH + 1) * sH * conf.scaling_factor,
                           pW * sW * conf.scaling_factor: (pW + 1) * sW * conf.scaling_factor, :] = test_pred_t

        ###======== Compute PSNR & Print Results========###
        test_GT = np.squeeze(target.d[index, :, :, :])
        test_PSNR = compute_psnr(test_pred_full, test_GT, 1.)
        test_loss_PSNR_list_for_epoch.append(test_PSNR)
        print(" <Test> [%4d/%4d]-th images, time: %4.4f(minutes), test_PSNR: %.8f[dB]  "
              % (int(index), int(data_sz[0]), (time.time() - start_time) / 60, test_PSNR))
        if conf.save_images:
            # comment for faster testing
            save_results_yuv(test_pred_full, index, conf.test_img_dir)
    test_PSNR_per_epoch = np.mean(test_loss_PSNR_list_for_epoch)

    print("######### Average Test PSNR: %.8f[dB]  #########" % (
        test_PSNR_per_epoch))
    print("######### Estimated Inference Time (per 4K frame): %.8f[s]  #########" %
          (np.mean(inf_time) * conf.test_patch * conf.test_patch))


if __name__ == '__main__':
    inference()
