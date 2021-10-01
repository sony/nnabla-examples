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
import glob
import logging
import argparse
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.monitor as nm
import nnabla.logger as logger

from nnabla.utils.image_utils import imread
from nnabla.ext_utils import get_extension_context

from frames_dataset import read_video
from generator import occlusion_aware_generator
from keypoint_detector import detect_keypoint
from model import unlink_all, persistent_all
from utils import read_yaml
from imageio import mimsave, imsave
from tqdm import tqdm

from external_utils import Visualizer
# this class is provided under CC BY-NC 4.0.
# for more details, see external_utils.py.


def reshape_result(visualization):
    h, w = 2 * [visualization.shape[0]]
    if visualization.shape != (h, w*29, 3):
        assert visualization.shape == (h, w*28, 3)
        # in case that generated images with keypoints is missing
        visualization = np.concatenate([visualization[:, :3*w],
                                        np.zeros((h, w, 3)).astype(np.uint8),
                                        visualization[:, 3*w:]], axis=1)
    vis_r1 = visualization[:, :6*w]
    vis_r2 = visualization[:, 6*w:12*w]
    vis_r3 = visualization[:, 12*w:18*w]
    vis_r4 = visualization[:, 18*w:24*w]
    vis_r5 = np.concatenate(
        [visualization[:, 24*w:], np.zeros((h, w, 3)).astype(np.uint8)], axis=1)
    vis_reshaped = np.concatenate(
        [vis_r1, vis_r2, vis_r3, vis_r4, vis_r5], axis=0)
    return vis_reshaped


def reconstruct(args):

    # get context
    ctx = get_extension_context(args.context)
    nn.set_default_context(ctx)
    logger.setLevel(logging.ERROR)  # to supress minor messages

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
    nn.load_parameters(param_file)

    bs, h, w, c = [1] + dataset_params.frame_shape
    source = nn.Variable((bs, c, h, w))
    driving_initial = nn.Variable((bs, c, h, w))
    driving = nn.Variable((bs, c, h, w))

    with nn.parameter_scope("kp_detector"):
        kp_source = detect_keypoint(source,
                                    **model_params.kp_detector_params,
                                    **model_params.common_params,
                                    test=True, comm=False)
        persistent_all(kp_source)

    with nn.parameter_scope("kp_detector"):
        kp_driving = detect_keypoint(driving,
                                     **model_params.kp_detector_params,
                                     **model_params.common_params,
                                     test=True, comm=False)
        persistent_all(kp_driving)

    with nn.parameter_scope("generator"):
        generated = occlusion_aware_generator(source,
                                              kp_source=unlink_all(kp_source),
                                              kp_driving=kp_driving,
                                              **model_params.generator_params,
                                              **model_params.common_params,
                                              test=True, comm=False)

    if not args.full and 'sparse_deformed' in generated:
        del generated['sparse_deformed']  # remove needless info

    persistent_all(generated)

    generated['kp_driving'] = kp_driving
    generated['kp_source'] = kp_source

    # generated contains these values;
    # 'mask': <Variable((bs, num_kp+1, h/4, w/4)) when scale_factor=0.25
    # 'sparse_deformed': <Variable((bs, num_kp+1, num_channel, h/4, w/4))  # (bs, num_kp + 1, c, h, w)
    # 'occlusion_map': <Variable((bs, 1, h/4, w/4))
    # 'deformed': <Variable((bs, c, h, w))
    # 'prediction': <Variable((bs, c, h, w))

    mode = "reconstruction"
    if "log_dir" in config:
        result_dir = os.path.join(
            args.out_dir, os.path.basename(config.log_dir), f"{mode}")
    else:
        result_dir = os.path.join(args.out_dir, "test_result", f"{mode}")

    # create an empty directory to save generated results
    _ = nm.Monitor(result_dir)
    if args.eval:
        os.makedirs(os.path.join(result_dir, "png"), exist_ok=True)

    # load the header images.
    header = imread("imgs/header_combined.png", channel_first=True)

    filenames = sorted(glob.glob(os.path.join(
        dataset_params.root_dir, "test", "*")))
    recon_loss_list = list()

    for filename in tqdm(filenames):
        # process repeated until all the test data is used
        driving_video = read_video(
            filename, dataset_params.frame_shape)  # (#frames, h, w, 3)
        driving_video = np.transpose(
            driving_video, (0, 3, 1, 2))  # (#frames, 3, h, w)

        generated_images = list()
        source_img = driving_video[0]

        source.d = np.expand_dims(source_img, 0)
        driving_initial.d = driving_video[0]

        # compute these in advance and reuse
        nn.forward_all(
            [kp_source["value"], kp_source["jacobian"]], clear_buffer=True)

        num_of_driving_frames = driving_video.shape[0]

        for frame_idx in tqdm(range(num_of_driving_frames)):
            driving.d = driving_video[frame_idx]
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
                combined_image = np.clip(
                    generated["prediction"].d[0], 0.0, 1.0)
                combined_image = (
                    255*combined_image).astype(np.uint8)  # (C, H, W)

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
            # compute L1 distance per frame.
            recon_loss_list.append(
                np.mean(np.abs(generated["prediction"].d[0] - driving.d[0])))

        # post process only for reconstruction evaluation.
        if args.eval:
            # crop generated images region only.
            if args.only_generated:
                eval_images = generated_images
            elif args.full:
                eval_images = [_[:, :h, 4*w:5*w] for _ in generated_images]
            elif args.detailed:
                assert generated_images[0].shape == (c, h, 5*w)
                eval_images = [_[:, :, 3*w:4*w] for _ in generated_images]
            else:
                eval_images = [_[:, h:, w:] for _ in generated_images]
            # place them horizontally and save for evaluation.
            image_for_eval = np.concatenate(
                eval_images, axis=2).transpose(1, 2, 0)
            imsave(os.path.join(result_dir, "png", f"{os.path.basename(filename)}.png"),
                   image_for_eval)

        # once each video is generated, save it.
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.mp4"
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
    print(f"Reconstruction loss: {np.mean(recon_loss_list)}")

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--params', default=None, type=str)
    parser.add_argument('--eval', action='store_true',
                        help="outputs files in a format suited for evaluation.")
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

    args = parser.parse_args()

    if args.only_generated:
        assert not args.detailed, "--only-generated flag is used, but --detailed is also used, disable the latter option."

    if args.full:
        assert args.detailed, "specify --detailed to enable --full option."

    reconstruct(args)


if __name__ == '__main__':
    main()
