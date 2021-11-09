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

import os
import logging
import argparse
import numpy as np

import nnabla as nn
import nnabla.monitor as nm
import nnabla.functions as F
import nnabla.logger as logger

from nnabla.utils.image_utils import imread
from nnabla.ext_utils import get_extension_context
from nnabla.utils.data_source_loader import download

from utils import read_yaml
from frames_dataset import read_video
from model import unlink_all, persistent_all
from keypoint_detector import detect_keypoint
from generator import occlusion_aware_generator

from tqdm import tqdm
from imageio import mimsave
from scipy.spatial import ConvexHull

from external_utils import Visualizer
# this class is provided under CC BY-NC 4.0.
# for more details, see external_utils.py.


def download_provided_file(url, filepath=None, verbose=True):
    if not filepath:
        filepath = os.path.basename(url)
    if not os.path.exists(filepath):
        if verbose:
            logger.info(f"{filepath} not found. Downloading...")
        download(url, filepath, False)
        if verbose:
            logger.info(f"Downloaded {filepath}.")
    return


def adjust_kp(kp_source, kp_driving, kp_driving_initial,
              adapt_movement_scale=1,
              use_relative_movement=False,
              use_relative_jacobian=False):

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = F.batch_matmul(kp_driving['jacobian'],
                                           F.reshape(
                                                F.batch_inv(
                                                    F.reshape(kp_driving_initial['jacobian'],
                                                              (-1,) +
                                                              kp_driving_initial['jacobian'].shape[-2:],
                                                              inplace=False)),
                                                kp_driving_initial['jacobian'].shape))
            kp_new['jacobian'] = F.batch_matmul(
                jacobian_diff, kp_source['jacobian'])

    return kp_new


def reshape_result(visualization):
    h, w = 2 * [visualization.shape[0]]
    if visualization.shape != (h, w*29, 3):
        assert visualization.shape == (h, w*28, 3)
        # in case that generated images with keypoints is missing
        visualization = np.concatenate([visualization[:, :3*w],
                                        np.zeros((h, w, 3)).astype(np.uint8),
                                        visualization[:, 3*w:]])
    vis_r1 = visualization[:, :6*w]
    vis_r2 = visualization[:, 6*w:12*w]
    vis_r3 = visualization[:, 12*w:18*w]
    vis_r4 = visualization[:, 18*w:24*w]
    vis_r5 = np.concatenate(
        [visualization[:, 24*w:], np.zeros((h, w, 3)).astype(np.uint8)], axis=1)
    vis_reshaped = np.concatenate(
        [vis_r1, vis_r2, vis_r3, vis_r4, vis_r5], axis=0)
    return vis_reshaped


def animate(args):

    # get context
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    logger.setLevel(logging.ERROR)  # to supress minor messages

    if not args.config:
        assert not args.params, "pretrained weights file is given, but corresponding config file is not. Please give both."
        download_provided_file(
            "https://nnabla.org/pretrained-models/nnabla-examples/GANs/first-order-model/voxceleb_trained_info.yaml")
        args.config = 'voxceleb_trained_info.yaml'

        download_provided_file(
            "https://nnabla.org/pretrained-models/nnabla-examples/GANs/first-order-model/pretrained_fomm_params.h5")

    config = read_yaml(args.config)

    dataset_params = config.dataset_params
    model_params = config.model_params

    if args.detailed:
        vis_params = config.visualizer_params
        visualizer = Visualizer(**vis_params)

    if not args.params:
        assert "log_dir" in config, "no log_dir found in config. therefore failed to locate pretrained parameters."
        param_file = os.path.join(
            config.log_dir, config.saved_parameters)
    else:
        param_file = args.params
    print(f"Loading {param_file} for image animation...")
    nn.load_parameters(param_file)

    bs, h, w, c = [1] + dataset_params.frame_shape
    source = nn.Variable((bs, c, h, w))
    driving_initial = nn.Variable((bs, c, h, w))
    driving = nn.Variable((bs, c, h, w))

    filename = args.driving

    # process repeated until all the test data is used
    driving_video = read_video(
        filename, dataset_params.frame_shape)  # (#frames, h, w, 3)
    driving_video = np.transpose(
        driving_video, (0, 3, 1, 2))  # (#frames, 3, h, w)

    source_img = imread(args.source, channel_first=True,
                        size=(256, 256)) / 255.
    source_img = source_img[:3]

    source.d = np.expand_dims(source_img, 0)
    driving_initial.d = driving_video[0][:3, ]

    with nn.parameter_scope("kp_detector"):
        kp_source = detect_keypoint(source,
                                    **model_params.kp_detector_params,
                                    **model_params.common_params,
                                    test=True, comm=False)
        persistent_all(kp_source)

    with nn.parameter_scope("kp_detector"):
        kp_driving_initial = detect_keypoint(driving_initial,
                                             **model_params.kp_detector_params,
                                             **model_params.common_params,
                                             test=True, comm=False)
        persistent_all(kp_driving_initial)

    with nn.parameter_scope("kp_detector"):
        kp_driving = detect_keypoint(driving,
                                     **model_params.kp_detector_params,
                                     **model_params.common_params,
                                     test=True, comm=False)
        persistent_all(kp_driving)

    if args.adapt_movement_scale:
        nn.forward_all([kp_source["value"],
                        kp_source["jacobian"],
                        kp_driving_initial["value"],
                        kp_driving_initial["jacobian"]])
        source_area = ConvexHull(kp_source['value'].d[0]).volume
        driving_area = ConvexHull(kp_driving_initial['value'].d[0]).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_norm = adjust_kp(kp_source=unlink_all(kp_source), kp_driving=kp_driving,
                        kp_driving_initial=unlink_all(kp_driving_initial),
                        adapt_movement_scale=adapt_movement_scale,
                        use_relative_movement=args.unuse_relative_movement,
                        use_relative_jacobian=args.unuse_relative_jacobian)
    persistent_all(kp_norm)

    with nn.parameter_scope("generator"):
        generated = occlusion_aware_generator(source,
                                              kp_source=unlink_all(kp_source),
                                              kp_driving=kp_norm,
                                              **model_params.generator_params,
                                              **model_params.common_params,
                                              test=True, comm=False)

    if not args.full and 'sparse_deformed' in generated:
        del generated['sparse_deformed']  # remove needless info

    persistent_all(generated)

    generated['kp_driving'] = kp_driving
    generated['kp_source'] = kp_source
    generated['kp_norm'] = kp_norm

    # generated contains these values;
    # 'mask': <Variable((bs, num_kp+1, h/4, w/4)) when scale_factor=0.25
    # 'sparse_deformed': <Variable((bs, num_kp+1, num_channel, h/4, w/4))  # (bs, num_kp + 1, c, h, w)
    # 'occlusion_map': <Variable((bs, 1, h/4, w/4))
    # 'deformed': <Variable((bs, c, h, w))
    # 'prediction': <Variable((bs, c, h, w))

    mode = "arbitrary"
    if "log_dir" in config:
        result_dir = os.path.join(
            args.out_dir, os.path.basename(config.log_dir), f"{mode}")
    else:
        result_dir = os.path.join(args.out_dir, "test_result", f"{mode}")

    # create an empty directory to save generated results
    _ = nm.Monitor(result_dir)

    # load the header images.
    header = imread("imgs/header_combined.png", channel_first=True)
    generated_images = list()

    # compute these in advance and reuse
    nn.forward_all([kp_source["value"],
                    kp_source["jacobian"]],
                   clear_buffer=True)
    nn.forward_all([kp_driving_initial["value"],
                    kp_driving_initial["jacobian"]],
                   clear_buffer=True)

    num_of_driving_frames = driving_video.shape[0]

    for frame_idx in tqdm(range(num_of_driving_frames)):
        driving.d = driving_video[frame_idx][:3, ]
        nn.forward_all([generated["prediction"],
                        generated["deformed"]], clear_buffer=True)

        if args.detailed:
            # visualize source w/kp, driving w/kp, deformed source, generated w/kp, generated image, occlusion map
            visualization = visualizer.visualize(
                source=source.d, driving=driving.d, out=generated)
            if args.full:
                visualization = reshape_result(visualization)  # (H, W, C)
            combined_image = visualization.transpose(2, 0, 1)  # (C, H, W)

        elif args.only_generated:
            combined_image = np.clip(generated["prediction"].d[0], 0.0, 1.0)
            combined_image = (255*combined_image).astype(np.uint8)  # (C, H, W)

        else:
            # visualize source, driving, and generated image
            driving_fake = np.concatenate([np.clip(driving.d[0], 0.0, 1.0),
                                           np.clip(generated["prediction"].d[0], 0.0, 1.0)], axis=2)
            header_source = np.concatenate([np.clip(header / 255., 0.0, 1.0),
                                            np.clip(source.d[0], 0.0, 1.0)], axis=2)
            combined_image = np.concatenate(
                [header_source, driving_fake], axis=1)
            combined_image = (255*combined_image).astype(np.uint8)

        generated_images.append(combined_image)

    # once each video is generated, save it.
    output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.mp4"
    output_filename = f"{os.path.basename(args.source)}_by_{output_filename}"
    output_filename = output_filename.replace("#", "_")
    if args.output_png:
        monitor_vis = nm.MonitorImage(output_filename, nm.Monitor(result_dir),
                                      interval=1, num_images=1,
                                      normalize_method=lambda x: x)
        for frame_idx, img in enumerate(generated_images):
            monitor_vis.add(frame_idx, img)
    else:
        generated_images = [_.transpose(1, 2, 0) for _ in generated_images]
        # you might need to change ffmpeg_params according to your environment.
        mimsave(f'{os.path.join(result_dir, output_filename)}', generated_images,
                fps=args.fps,
                ffmpeg_params=["-pix_fmt", "yuv420p",
                               "-vcodec", "libx264",
                               "-f", "mp4",
                               "-q", "0"])

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--params', default=None, type=str)
    parser.add_argument('--source', default="", type=str)
    parser.add_argument('--driving', default="", type=str)
    parser.add_argument('--out-dir', '-o', default="result", type=str)
    parser.add_argument('--context', '-c', default='cudnn',
                        type=str, choices=['cudnn', 'cpu'])
    parser.add_argument('--output-png', action='store_true',
                        help="if chosen, outputs .png image file.")
    parser.add_argument('--fps', default=10, type=int,
                        help="framerate for making video. In effect when --output-video is specified.")
    parser.add_argument('--only-generated', action='store_true',
                        help="if chosen, outputs generated images only.")
    parser.add_argument('--detailed', action='store_true',
                        help="if chosen, visualizes keypoints and occlusion map as well.")
    parser.add_argument('--full', action='store_true',
                        help="if chosen, visualizes all the generated elements.")
    # animation params
    parser.add_argument('--adapt-movement-scale', action='store_true',
                        help="Adapt movement scale between source and driving image.")
    parser.add_argument('--unuse-relative-movement', action='store_false',
                        help="DO NOT consider relative movement between source and driving image.")
    parser.add_argument('--unuse-relative-jacobian', action='store_false',
                        help="DO NOT consider relative jacobian between source and driving image.")

    args = parser.parse_args()

    assert args.source and args.driving, "you need to have source and driving images for animation."

    if args.only_generated:
        assert not args.detailed, "--only-generated flag is used, but --detailed is also used, disable the latter option."

    if args.full:
        assert args.detailed, "specify --detailed to enable --full option."

    animate(args)


if __name__ == '__main__':
    main()
