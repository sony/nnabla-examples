# Copyright 2022 Sony Group Corporation.
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
import nnabla.ext_utils
import argparse

import pathlib

import numpy as np

import json

import cv2

import cliport
from cliport import tasks
from cliport.tasks import cameras
from cliport.environments.environment import Environment
from cliport.utils import utils

import nnabla as nn
import nnabla.functions as F
import nnabla_cliport.clip.clip as nnabla_clip
from nnabla_cliport.models.clip_text_encoder import CLIPTextEncoder
from nnabla_cliport.models.clip_image_encoder import CLIPImageEncoder
from nnabla_cliport.models.cliport_attention import CLIPortAttention
from nnabla_cliport.models.cliport_transport import CLIPortTransport


def build_cliport_environment(args):
    scriptdir = pathlib.Path(__file__).parent
    cliport_root = pathlib.Path(cliport.__file__).resolve().parent
    cliport_env = Environment(
        assets_root=str(cliport_root / 'environments/assets'),
        disp=args.render,
        shared_memory=False,
        hz=480,
        record_cfg={
            'save_video_path': f'{scriptdir}/video',
            'video_height': 640,
            'video_width': 720,
            'fps': 20,
            'add_text': True
        }
    )

    return cliport_env


def cliport_with_nnabla(image, text, attention, transport, args):
    # HWC -> CHW
    rgbd = np.transpose(image, axes=(2, 0, 1))
    rgbd = nn.Variable.from_numpy_array(rgbd)
    rgbd = F.reshape(rgbd, shape=(1, *rgbd.shape))
    token = nn.Variable.from_numpy_array(nnabla_clip.tokenize([text]))

    print(f'token: {token.d}')
    attention_pixels = attention.compute_features(
        rgbd, token)[0]  # take the first batch
    print(f'attention_pixels shape: {attention_pixels.shape}')
    attention_pixels.forward(clear_no_need_grad=True)
    pick_argmax_q = np.argmax(attention_pixels.d)
    pick_argmax_q = np.unravel_index(
        pick_argmax_q, shape=attention_pixels.shape)
    pick_pixel = pick_argmax_q[:2]
    pick_theta = pick_argmax_q[2] * (2 * np.pi / attention_pixels.shape[2])
    print(f'pick pixel: {pick_pixel}')

    transport_pixels = transport.compute_features(
        rgbd, token)[0]  # take the first batch
    pivot = np.asarray([pick_pixel])
    transport.set_pivot(pivot=pivot)
    # this transpose is redundant. just to follow the original code
    transport_pixels = F.transpose(transport_pixels, axes=(1, 2, 0))
    transport_pixels.forward(clear_buffer=True)
    place_argmax_q = np.argmax(transport_pixels.d)
    place_argmax_q = np.unravel_index(
        place_argmax_q, shape=transport_pixels.shape)
    place_pixel = place_argmax_q[:2]
    place_theta = place_argmax_q[2] * (2 * np.pi / transport_pixels.shape[2])
    print(f'place pixel: {place_pixel}')

    pixel_size = 0.003125
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    height_map = image[:, :, 3]
    pick_xyz = utils.pix_to_xyz(pick_pixel, height_map, bounds, pixel_size)
    place_xyz = utils.pix_to_xyz(place_pixel, height_map, bounds, pixel_size)
    pick_quaternion = utils.eulerXYZ_to_quatXYZW((0, 0, -pick_theta))
    place_quaternion = utils.eulerXYZ_to_quatXYZW((0, 0, -place_theta))

    if args.show_attention:
        cv2.imshow('attention', np.squeeze(attention_pixels.d * 255))
        cv2.imshow('transport', np.squeeze(
            transport_pixels.d[:, :, place_argmax_q[2]] * 255))
        cv2.waitKey(100)
    return {
        'pose0': (np.asarray(pick_xyz), np.asarray(pick_quaternion)),
        'pose1': (np.asarray(place_xyz), np.asarray(place_quaternion)),
        'pick': (pick_pixel[0], pick_pixel[1], pick_theta),
        'place': (place_pixel[0], place_pixel[1], place_theta)
    }


def reconstruct_rgbd(state):
    pixel_size = 0.003125
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    camera_config = cameras.RealSenseD415.CONFIG
    cmap, hmap = utils.get_fused_heightmap(
        state, camera_config, bounds, pixel_size)
    rgbd = np.concatenate(
        (cmap, hmap[Ellipsis, None], hmap[Ellipsis, None], hmap[Ellipsis, None]), axis=2)
    return rgbd


def run_cliport_task(env, seed, task_name, demo_number, args):
    scriptdir = pathlib.Path(__file__).parent
    snapshotdir = pathlib.Path(args.snapshot_dir)
    text_encoder = CLIPTextEncoder('text_encoder')
    text_encoder.load_parameters(f'{scriptdir}/data/text_encoder.h5')
    attention_image_encoder = CLIPImageEncoder('attention_image_encoder')
    if args.use_original_attention_params:
        attention_image_encoder.load_parameters(
            f'{scriptdir}/data/attention_image_encoder.h5')
    else:
        attention_image_encoder.load_parameters(
            f'{snapshotdir}/attention_image_encoder.h5')

    transport_key_image_encoder = CLIPImageEncoder(
        'transport_key_image_encoder')
    transport_query_image_encoder = CLIPImageEncoder(
        'transport_query_image_encoder')
    if args.use_original_transport_params:
        transport_key_image_encoder.load_parameters(
            f'{scriptdir}/data/transport_key_image_encoder.h5')
        transport_query_image_encoder.load_parameters(
            f'{scriptdir}/data/transport_query_image_encoder.h5')
    else:
        transport_key_image_encoder.load_parameters(
            f'{snapshotdir}/transport_key_image_encoder.h5')
        transport_query_image_encoder.load_parameters(
            f'{snapshotdir}/transport_query_image_encoder.h5')

    attention = CLIPortAttention(
        'attention', image_encoder=attention_image_encoder, text_encoder=text_encoder)
    if args.use_original_attention_params:
        attention.load_parameters(f'{scriptdir}/data/attention.h5')
    else:
        attention.load_parameters(f'{snapshotdir}/attention.h5')

    transport = CLIPortTransport('transport',
                                 key_image_encoder=transport_key_image_encoder,
                                 query_image_encoder=transport_query_image_encoder,
                                 text_encoder=text_encoder)
    if args.use_original_transport_params:
        transport.load_parameters(f'{scriptdir}/data/transport.h5')
    else:
        transport.load_parameters(f'{snapshotdir}/transport.h5')

    np.random.seed(seed)
    task = tasks.names[task_name]()
    task.mode = 'val'
    env.seed(seed)
    env.set_task(task)
    state = env.reset()
    env_info = env.info

    if args.record:
        video_name = f'{task_name}-{demo_number+1:06d}'
        env.start_rec(video_name)

    total_reward = 0
    for _ in range(task.max_steps):
        rgbd = reconstruct_rgbd(state)
        text = env_info["lang_goal"]
        print(f'text: {text}')

        action = cliport_with_nnabla(rgbd, text, attention, transport, args)
        next_state, reward, done, env_info = env.step(action)
        state = next_state

        total_reward += reward
        if done:
            break

    if args.record:
        env.end_rec()

    return total_reward


def run_cliport(args):
    gpu_context = nnabla.ext_utils.get_extension_context(
        'cudnn', device_id=args.gpu)
    print(f'evaluating snapshot: {args.snapshot_dir}')

    with nn.context_scope(gpu_context):
        task = args.task
        env = build_cliport_environment(args)
        n_episodes = args.num_eval_episodes

        total_rewards = []
        total_success = []
        seed = 5000
        for i in range(0, n_episodes):
            nn.seed(seed)
            task_reward = run_cliport_task(env, seed, task, i, args)
            task_success = task_reward > 0.99
            total_rewards.append(task_reward)
            total_success.append(task_success)
            seed += 1

        mean_reward = str(np.mean(total_rewards))
        successes = str(np.sum(total_success))
        print(f'mean_reward: {mean_reward}')
        print(f'successes: {successes}')

        # save eval result
        save_dir = pathlib.Path(args.snapshot_dir)
        save_file = save_dir / 'result.json'
        results = {
            'mean_reward': mean_reward,
            'successes': successes
        }
        with open(str(save_file), 'w') as f:
            json.dump(results, f, indent=4)


def run_evaluation(args):
    run_cliport(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-dir', type=str, required=True)
    parser.add_argument('--use-original-attention-params', action='store_true')
    parser.add_argument('--use-original-transport-params', action='store_true')
    parser.add_argument('--num-eval-episodes', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--show-attention', action='store_true')
    parser.add_argument('--task', type=str, default='stack-block-pyramid-seq-seen-colors',
                        choices=['align-rope',
                                 'assembling-kits-seq-seen-colors',
                                 'packing-boxes-pairs-seen-colors',
                                 'packing-shapes',
                                 'packing-seen-google-objects-seq',
                                 'packing-seen-google-objects-group',
                                 'put-block-in-bowl-seen-colors',
                                 'stack-block-pyramid-seq-seen-colors',
                                 'separating-piles-seen-colors',
                                 'towers-of-hanoi-seq-seen-colors'])

    args = parser.parse_args()

    run_evaluation(args)


if __name__ == '__main__':
    main()
