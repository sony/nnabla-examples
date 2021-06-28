# Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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

import sys
import os
import time
import cv2
import nnabla as nn
import numpy as np
from nnabla.ext_utils import get_extension_context, import_extension_module
from PIL import Image
from args import get_args
from preprocess import centerpad, lab2rgb, uncenter_l, center_crop, rgb2lab, normalize,batch_lab2rgb_transpose, save_frames,frames2vid

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
    image = centerpad(image,image_size)
    image = center_crop(image,image_size)
    image = rgb2lab(image)
    image = np.transpose(image,[2,0,1])
    image = normalize(image)
    image = np.expand_dims(image,0)
    return image.astype(np.float32)

def interpolate(image, scale):
    '''
    Resize the given input using linear interpolation
    Args:
        image: batch numpy image of ndim 4
        scale: scale factor
    Returns:
        batched resized image(numpy)
    '''
    image = np.transpose(image.squeeze(), (1,2,0))
    image = cv2.resize(image, None, fx = scale, fy = scale, interpolation=cv2.INTER_LINEAR)
    image = np.expand_dims(np.transpose(image, (2,0,1)),0)
    return image
def colorize_video(args, ref):
    '''
    Colorize the input frames and save the output as colorized frames and video
    Args:
        args: argparser object
        ref: refrence image
    '''
    ext = import_extension_module(args.context)
    reference_file = os.path.join(args.ref_path, ref)
    output_path = os.path.join(args.output_path, 'out_' + ref.split(".")[0])
    if not os.path.exists(output_path):
        try:
            os.makedirs(output_path)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit()
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4
    IA_predict_rgb = None
    filenames = [f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path,f))]
    print(f"processing the folder: {args.input_path}")
    #sort the frames in order as in video
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))
    # read reference name from reference input else first frame assuming it's colorized
    ref_name = args.input_path + filenames[0] if args.frame_propagate else reference_file
    frame_ref = np.array(Image.open(ref_name))
    total_time = 0
    I_last_lab_predict = None
    #Load the Weights
    nn.clear_parameters()
    nn.load_parameters('../devc_vgg19_conv.h5')
    nn.load_parameters('../devc_nonlocal.h5')
    nn.load_parameters('../devc_colornet.h5')
    print(f"reference = {ref_name}")
    for iter_num, frame_name in enumerate(filenames):
        print("input =", frame_name)
        frame = Image.open(os.path.join(args.input_path, frame_name))
        IA_lab_large = transform(np.array(frame),args.image_size) ##
        IB_lab_large = transform(np.array(frame_ref), args.image_size)
        IA_lab = interpolate(
            IA_lab_large, scale=0.5)
        IB_lab = interpolate(
            IB_lab_large, scale=0.5)
        IA_l = IA_lab[:, 0:1, :, :]
        if I_last_lab_predict is None:
            if args.frame_propagate:
                I_last_lab_predict = IB_lab
            else:
                I_last_lab_predict = np.zeros_like(IA_lab)
        I_current_lab = IA_lab
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = lab2rgb(np.concatenate((uncenter_l(I_reference_l), I_reference_ab), axis=1))
        # Copy to nn Variabes
        I_reference_rgb_nn = nn.Variable.from_numpy_array(I_reference_rgb)
        I_current_lab_nn = nn.Variable.from_numpy_array(I_current_lab)
        I_reference_lab_nn = nn.Variable.from_numpy_array(I_reference_lab)
        I_last_lab_predict_nn = nn.Variable.from_numpy_array(I_last_lab_predict)
        t_start = time.clock()
        features_B_nn = vgg_net(I_reference_rgb_nn, pre_process = True, fix = True)
        I_current_ab_predict_nn, I_current_nonlocal_lab_predict_nn, features_current_gray_nn = frame_colorization(
                I_current_lab_nn, I_reference_lab_nn, I_last_lab_predict_nn, features_B_nn, feature_noise=0, temperature=1e-10)
        # forward the network
        I_current_ab_predict_nn.forward()
        I_last_lab_predict =np.concatenate((IA_l, I_current_ab_predict_nn.data.data), axis=1)
        delta_t = time.clock() - t_start
        print(f"runtime: {delta_t}")
        if iter_num > 0:
            total_time += delta_t
            print(f"{(total_time/iter_num):.2g} second")
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = interpolate(I_current_ab_predict_nn.data.data, scale=2) * 1.25
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].astype(np.uint8), lambda_value, sigma_color)
            curr_predict_a = wls_filter.filter(
                curr_predict[0, 0, :, :])
            curr_predict_b = wls_filter.filter(
                curr_predict[0, 1, :, :])
            curr_predict_a = np.reshape(curr_predict_a, (1,1,curr_predict_a.shape[0],curr_predict_a.shape[1]))
            curr_predict_b = np.reshape(curr_predict_b,(1,1, curr_predict_b.shape[0], curr_predict_b.shape[1]))
            curr_predict_filter = np.concatenate(
                    (curr_predict_a, curr_predict_b), axis=1)
            IA_predict_rgb = batch_lab2rgb_transpose(
            curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose(
                curr_bs_l[:32], curr_predict[:32, ...])

        # save the frames
        save_frames(IA_predict_rgb, output_path, iter_num)
        iter_num = iter_num + 1
        #clear memory cache
        ext.clear_memory_cache()
    #save the video
    if IA_predict_rgb is not None:
        output_shape = IA_predict_rgb.shape[:-1]
        frames2vid(frame_folder=output_path, frame_shape = output_shape,  output_dir=output_path, filename=args.output_video)

def main():
    args = get_args()
    ctx = get_extension_context(
        args.context, device_id=args.device_id)
    nn.set_default_context(ctx)
    refs = os.listdir(args.ref_path)
    # sort the reference images in order
    refs.sort()
    #Inference the input frames taking each reference image
    for ref in refs:
        colorize_video(args,ref)
if __name__ == "__main__":
    main()
