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
import time
import cv2

import numpy as np
import nnabla as nn
import nnabla.functions as F
from nnabla.ext_utils import get_extension_context
from PIL import Image

from args import get_config
import preprocess
from vggnet import vgg_net
from nonlocal_net import frame_colorization


def transform(image, image_size):
    '''
    Preprocess the given image
    Args:
        image(cv2/numpy) : image to be preprocessed
        image_size: Output image size
    Returns:
        numpy/cv2 image of shape image_size

    '''
    image = preprocess.centerpad(image, image_size)
    image = preprocess.center_crop(image, image_size)
    image = preprocess.rgb2lab(image)
    image = np.transpose(image, [2, 0, 1])
    image = preprocess.normalize(image)
    return np.expand_dims(image, 0)


def interpolate(image, scale):
    '''
    Resize the given input with linear interpolation
    Args:
        image: numpy image of ndim 4 (B,C,H,W)
        scale: scale factor
    Returns:
        batched resized image(numpy)
    '''
    # Change order  [B, C, H, W] -> [H, W, B*C]
    image = np.transpose(image.squeeze(), (1, 2, 0))
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR)
    # Change order  [H, W, C] -> [B, C, H, W]
    image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
    return image

def interpolate_nn(image, frame, scale):
    '''
    Linear Interpolation on Variable image and frame
    Args:
        image  : image (Variable) 
        frame  : frame (Variable) 
    Returns
        linear interpolated image and frame 
    '''
    image = F.interpolate(image, scale)
    frame  = F.interpolate(frame, scale)
    return image, frame

def get_rgb_frame(ia_lab_large,i_current_ab_predict_nn, conf):

    curr_bs_l = ia_lab_large[:, 0:1, :, :]
    curr_predict = interpolate(
        i_current_ab_predict_nn.data.data,
        scale=2) * 1.25
    if conf.wls_filter_on:
        guide_image = preprocess.uncenter_l(curr_bs_l, conf) * 255 / 100
        wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
            guide_image[0, 0, :, :].astype(np.uint8), conf.lambda_value, conf.sigma_color)
        curr_predict_a = wls_filter.filter(
            curr_predict[0, 0, :, :])
        curr_predict_b = wls_filter.filter(
            curr_predict[0, 1, :, :])
        curr_predict_a = np.reshape(
            curr_predict_a, (1, 1, curr_predict_a.shape[0], curr_predict_a.shape[1]))
        curr_predict_b = np.reshape(
            curr_predict_b, (1, 1, curr_predict_b.shape[0], curr_predict_b.shape[1]))
        curr_predict_filter = np.concatenate(
            (curr_predict_a, curr_predict_b), axis=1)
        ia_predict_rgb = preprocess.batch_lab2rgb_transpose(
            conf, curr_bs_l[:32], curr_predict_filter[:32, ...])
    else:
        ia_predict_rgb = preprocess.batch_lab2rgb_transpose(
            conf, curr_bs_l[:32], curr_predict[:32, ...])
    return ia_predict_rgb

def colorize_video(conf, ref):
    '''
    Colorize the input frames and save the output as colorized frames and video
    Args:
        conf: conf object
        ref: refrence image
    '''
    def load_weights():
        nn.load_parameters('../../devc_vgg19_conv.h5')
        nn.load_parameters('../../devc_nonlocal.h5')
        nn.load_parameters('../../devc_colornet.h5')
    
    reference_file = os.path.join(conf.data.ref_path, ref)
    output_path = os.path.join(conf.data.output_path, 'out_' + ref.split(".")[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filenames = [f for f in os.listdir(conf.data.input_path)
                 if os.path.isfile(os.path.join(conf.data.input_path, f))]
    print(f"processing the folder: {conf.data.input_path}")
    # sort the frames in order as in video
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))
    # read reference name from reference input else first frame assuming it's
    # colorized
    ref_name = conf.data.input_path + \
        filenames[0] if conf.data.frame_propagation else reference_file
    i_last_lab_predict = None
    # Load the Weights
    nn.clear_parameters()
    load_weights()
    print(f"reference = {ref_name}")
    # Preprocess reference image
    frame_ref = np.array(Image.open(ref_name))
    ib_lab_large = nn.Variable.from_numpy_array(transform(frame_ref, conf.data.image_size))
    for iter_num, frame_name in enumerate(filenames):
        print("input =", frame_name)
        frame = Image.open(os.path.join(conf.data.input_path, frame_name))
        ia_lab_large = nn.Variable.from_numpy_array(transform(np.array(frame), conf.data.image_size))
        ia_lab, ib_lab = interpolate_nn(ia_lab_large, ib_lab_large, scale = (0.5,0.5))
        ia_l = ia_lab[:, 0:1, :, :]
        if i_last_lab_predict is None:
            if conf.data.frame_propagation:
                i_last_lab_predict = ib_lab
            else:
                i_last_lab_predict = nn.Variable(ia_lab.shape)
        i_reference_l = ib_lab[:, 0:1, :, :]
        i_reference_ab = ib_lab[:, 1:3, :, :]
        i_reference_rgb = preprocess.lab2rgb(
            F.concatenate(
                preprocess.uncenter_l(i_reference_l, conf),
                i_reference_ab,
                axis=1))
        if type(i_last_lab_predict).__module__ == "numpy":
            i_last_lab_predict_nn = nn.Variable.from_numpy_array(
                i_last_lab_predict)
        else:
            i_last_lab_predict_nn = i_last_lab_predict
        t_start = time.time()
        features_b_nn = vgg_net(i_reference_rgb, pre_process=True, fix=True)
        i_current_ab_predict, _i_current_nonlocal_lab, _features_gray = frame_colorization(
            ia_lab, ib_lab, i_last_lab_predict_nn,
            features_b_nn, feature_noise=0, temperature=1e-10)
        # forward the network
        nn.forward_all([i_current_ab_predict])
        i_last_lab_predict = np.concatenate(
            (ia_l.data.data, i_current_ab_predict.data.data), axis=1)
        print(f"Runtime: {time.time() - t_start:.2g} second")
        rgb_frame = get_rgb_frame(ia_lab_large.d, i_current_ab_predict, conf)
        preprocess.save_frames(rgb_frame, output_path, iter_num)
        iter_num = iter_num + 1
    # save the video
    preprocess.frames2vid(
        frame_folder=output_path,
        frame_shape=conf.data.image_size,
        output_dir=output_path,
        filename=conf.data.output_video)

def main():
    conf = get_config()
    ctx = get_extension_context(
        conf.nnabla_context.context, device_id=conf.nnabla_context.device_id)
    nn.set_default_context(ctx)
    refs = sorted(os.listdir(conf.data.ref_path))
    # sort the reference images in order
    # Inference the input frames taking each reference image
    for ref in refs:
        colorize_video(conf, ref)


if __name__ == "__main__":
    main()
