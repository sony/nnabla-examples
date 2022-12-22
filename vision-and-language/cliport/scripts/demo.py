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

from PIL import Image

import numpy as np

import cliport
from cliport import dataset
from cliport import tasks
from cliport.environments.environment import Environment
from cliport.utils import utils

import nnabla as nn
import nnabla.functions as F
import nnabla_cliport.clip.clip as nnabla_clip
from nnabla_cliport.models.clip_text_encoder import CLIPTextEncoder
from nnabla_cliport.models.clip_image_encoder import CLIPImageEncoder
from nnabla_cliport.models.cliport_attention import CLIPortAttention
from nnabla_cliport.models.cliport_transport import CLIPortTransport


def encode_with_nnabla(image, text, args):
    import nnabla.ext_utils
    gpu_context = nnabla.ext_utils.get_extension_context('cudnn', device_id=0)
    with nn.context_scope(gpu_context):
        # load params
        scriptdir = pathlib.Path(__file__).parent
        image = nnabla_clip.preprocess(image)
        image = nn.Variable.from_numpy_array(image)

        image_encoder = CLIPImageEncoder('image_encoder')
        image_encoder.load_parameters(
            f'{scriptdir}/{args.param_dir}/image_encoder.h5')
        encoded_image, mid_features = image_encoder.encode_image(image)

        for i, mid_feature in enumerate(mid_features):
            print(f'mid_feature[{i}] shape: {mid_feature.shape}')

        text_encoder = CLIPTextEncoder('text_encoder')
        text_encoder.load_parameters(
            f'{scriptdir}/{args.param_dir}/text_encoder.h5')
        token = nn.Variable.from_numpy_array(nnabla_clip.tokenize(text))
        print(f'nnabla token: {token.shape}')
        encoded_text = text_encoder.encode_text(token)

        nn.forward_all([encoded_text, encoded_image])

        print(f'nnabla encoded text shape: {encoded_text.shape}')
        print(f'nnabla encoded image shape: {encoded_image.shape}')

        return encoded_image, encoded_text


def encode_with_pytorch(image, text):
    import torch
    import clip as pytorch_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = pytorch_clip.load("ViT-B/32", device=device)
    model, preprocess = pytorch_clip.load("RN50", device=device)

    image = preprocess(image).unsqueeze(0).to(device)
    text = pytorch_clip.tokenize(text).to(device)
    print(f'pytorch token: {text.shape}')

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    print(f'pytorch encoded text shape: {text_features.shape}')
    print(f'pytorch encoded image shape: {image_features.shape}')

    return image_features, text_features


def run_clip(args):
    scriptdir = pathlib.Path(__file__).parent
    image = Image.open(f'{scriptdir}/CLIP.png')
    text = ['hello japan', 'hello world', 'hello sony']

    nnabla_im_feature, nnabla_text_feature = encode_with_nnabla(
        image, text, args)
    pytorch_im_feature, pytorch_text_feature = encode_with_pytorch(image, text)

    print(f'nnabla im shape: {nnabla_im_feature.shape}')
    print(f'pytorch im shape: {pytorch_im_feature.shape}')
    print(f'nnabla im feature: {nnabla_im_feature.d[0]}')
    print(f'pytorch im feature: {pytorch_im_feature[0]}')
    # np.testing.assert_allclose(nnabla_im_feature.d[0], pytorch_im_feature.cpu().numpy(), atol=1e-1)

    print(f'nnabla text shape: {nnabla_text_feature.shape}')
    print(f'pytorch text shape: {pytorch_text_feature.shape}')
    print(f'nnabla text feature: {nnabla_text_feature.d[1]}')
    print(f'pytorch text feature: {pytorch_text_feature[1]}')
    np.testing.assert_allclose(
        nnabla_text_feature.d, pytorch_text_feature.cpu().numpy(), atol=1e-1)
    assert nnabla_im_feature.shape == tuple(pytorch_im_feature.shape)
    assert nnabla_text_feature.shape == tuple(pytorch_text_feature.shape)


def build_cliport_environment(task):
    import hydra  # noqa
    scriptdir = pathlib.Path(__file__).parent
    cliport_root = pathlib.Path(cliport.__file__).resolve().parent
    cliport_env = Environment(
        assets_root=str(cliport_root / 'environments/assets'),
        disp=True,
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

    train_config = utils.load_hydra_config(
        str(scriptdir / 'cliport_quickstart' / 'multi-language-conditioned-cliport-n1000-train' / '.hydra' / 'config.yaml'))
    datadir = str(pathlib.Path(__file__).parent / f'data/{task}-test')
    env_data = dataset.RavensDataset(
        datadir, train_config, n_demos=10, augment=False)

    return cliport_env, env_data


def cliport_with_nnabla(image, text, args):
    scriptdir = pathlib.Path(__file__).parent

    text_encoder = CLIPTextEncoder('text_encoder')
    text_encoder.load_parameters(f'{scriptdir}/data/text_encoder.h5')
    # NOTE: cliport's underlying clip batch norm parameters are different from original clip model
    attention_image_encoder = CLIPImageEncoder('attention_image_encoder')
    attention_image_encoder.load_parameters(
        f'{scriptdir}/{args.param_dir}/attention_image_encoder.h5')
    transport_key_image_encoder = CLIPImageEncoder(
        'transport_key_image_encoder')
    transport_key_image_encoder.load_parameters(
        f'{scriptdir}/{args.param_dir}/transport_key_image_encoder.h5')
    transport_query_image_encoder = CLIPImageEncoder(
        'transport_query_image_encoder')
    transport_query_image_encoder.load_parameters(
        f'{scriptdir}/{args.param_dir}/transport_query_image_encoder.h5')

    attention = CLIPortAttention(
        'attention', image_encoder=attention_image_encoder, text_encoder=text_encoder)
    attention.load_parameters(f'{scriptdir}/{args.param_dir}/attention.h5')

    transport = CLIPortTransport('transport',
                                 key_image_encoder=transport_key_image_encoder,
                                 query_image_encoder=transport_query_image_encoder,
                                 text_encoder=text_encoder)
    transport.load_parameters(f'{scriptdir}/{args.param_dir}/transport.h5')

    # HWC -> CHW
    rgbd = np.transpose(image, axes=(2, 0, 1))
    rgbd = nn.Variable.from_numpy_array(rgbd)
    rgbd = F.reshape(rgbd, shape=(1, *rgbd.shape))
    token = nn.Variable.from_numpy_array(nnabla_clip.tokenize([text]))
    print(f'token: {token.d}')
    attention_pixels = attention.compute_features(
        rgbd, token)[0]  # take the first batch
    # this transpose is redundant. just to follow the original code
    attention_pixels.forward(clear_no_need_grad=True)
    pick_argmax_q = np.argmax(attention_pixels.d)
    pick_argmax_q = np.unravel_index(
        pick_argmax_q, shape=attention_pixels.shape)
    pick_pixel = pick_argmax_q[:2]
    pick_theta = pick_argmax_q[2] * (2 * np.pi / attention_pixels.shape[2])
    print(f'pick pixel: {pick_pixel}')
    print(f'pick theta: {pick_theta}')

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
    print(f'place theta: {place_theta}')

    pixel_size = 0.003125
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
    height_map = image[:, :, 3]
    pick_xyz = utils.pix_to_xyz(pick_pixel, height_map, bounds, pixel_size)
    place_xyz = utils.pix_to_xyz(place_pixel, height_map, bounds, pixel_size)
    pick_quaternion = utils.eulerXYZ_to_quatXYZW((0, 0, -pick_theta))
    place_quaternion = utils.eulerXYZ_to_quatXYZW((0, 0, -place_theta))

    return {
        'pose0': (np.asarray(pick_xyz), np.asarray(pick_quaternion)),
        'pose1': (np.asarray(place_xyz), np.asarray(place_quaternion)),
        'pick': (pick_pixel[0], pick_pixel[1], pick_theta),
        'place': (place_pixel[0], place_pixel[1], place_theta)
    }


def run_cliport_task(env, env_data, task_name, demo_number, args):
    import cv2
    _, seed = env_data.load(demo_number)
    np.random.seed(seed)
    task = tasks.names[task_name]()
    task.mode = 'test'
    env.seed(seed)
    env.set_task(task)
    state = env.reset()
    env_info = env.info
    rgbd = env_data.get_image(state)

    if args.record:
        video_name = f'{task_name}-{demo_number+1:06d}'
        env.start_rec(video_name)

    total_reward = 0
    print(f'max_steps: {task.max_steps}')
    for _ in range(task.max_steps):
        rgbd = env_data.get_image(state)
        text = env_info["lang_goal"]
        print(f'lang goal: {text}')
        bgrd = rgbd[..., [2, 0, 1, 3, 4, 5]]
        cv2.imshow('observation', bgrd[:, :, :3] / 255.0)
        cv2.waitKey(1)

        action = cliport_with_nnabla(rgbd, text, args)
        next_state, reward, done, env_info = env.step(action)
        state = next_state

        total_reward += reward
        print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
        if done:
            break

    if args.record:
        env.end_rec()


def run_cliport(args):
    gpu_context = nnabla.ext_utils.get_extension_context('cudnn', device_id=0)

    with nn.context_scope(gpu_context):
        task = args.task
        env, env_data = build_cliport_environment(task)
        n_demos = 10
        utils.set_seed(0, torch=True)
        for i in range(0, n_demos):
            nn.seed(0)
            run_cliport_task(env, env_data, task, i, args)


def run_demo(args):
    if args.clip:
        run_clip()
    else:
        run_cliport(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--param-dir', type=str, default='data')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--task', type=str, default='stack-block-pyramid-seq-seen-colors',
                        choices=['align-rope',
                                 'assembling-kits-seq-seen-colors',
                                 'assembling-kits-seq-unseen-colors',
                                 'packing-boxes-pairs-seen-colors',
                                 'packing-boxes-pairs-unseen-colors',
                                 'packing-shapes',
                                 'packing-seen-google-objects-seq',
                                 'packing-unseen-google-objects-seq',
                                 'packing-seen-google-objects-group',
                                 'packing-unseen-google-objects-group',
                                 'put-block-in-bowl-seen-colors',
                                 'put-block-in-bowl-unseen-colors',
                                 'stack-block-pyramid-seq-seen-colors',
                                 'stack-block-pyramid-seq-unseen-colors',
                                 'separating-piles-seen-colors',
                                 'separating-piles-unseen-colors',
                                 'towers-of-hanoi-seq-seen-colors',
                                 'towers-of-hanoi-seq-unseen-colors'])

    args = parser.parse_args()

    run_demo(args)


if __name__ == '__main__':
    main()
