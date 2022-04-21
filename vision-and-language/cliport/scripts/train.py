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

import argparse

import numpy as np

import re

import datetime

import json

import sys

import pathlib

import pickle

from cliport.utils import utils
from cliport.tasks import cameras

from tensorboardX import SummaryWriter

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context

import nnabla_cliport.clip.clip as nnabla_clip
from nnabla_cliport.models.clip_text_encoder import CLIPTextEncoder
from nnabla_cliport.models.clip_image_encoder import CLIPImageEncoder
from nnabla_cliport.models.cliport_attention import CLIPortAttention
from nnabla_cliport.models.cliport_transport import CLIPortTransport


def reconstruct_rgbd(state, bounds, pixel_size):
    camera_config = cameras.RealSenseD415.CONFIG
    cmap, hmap = utils.get_fused_heightmap(
        state, camera_config, bounds, pixel_size)
    rgbd = np.concatenate(
        (cmap, hmap[Ellipsis, None], hmap[Ellipsis, None], hmap[Ellipsis, None]), axis=2)
    return rgbd


def process_sampled_data(data, augment):
    pixel_size = 0.003125
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

    (state, action, reward, info) = data
    rgbd = reconstruct_rgbd(state, bounds, pixel_size)

    pick_xyz, pick_xyzw = action['pose0']
    place_xyz, place_xyzw = action['pose1']
    pick_pixel = utils.xyz_to_pix(pick_xyz, bounds, pixel_size)
    pick_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(pick_xyzw)[2])
    place_pixel = utils.xyz_to_pix(place_xyz, bounds, pixel_size)
    place_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(place_xyzw)[2])
    place_theta = place_theta - pick_theta
    pick_theta = 0  # Fix to 0 rad

    # augment data
    if augment:
        rgbd, _, (pick_pixel, place_pixel), _ = utils.perturb(
            rgbd, [pick_pixel, place_pixel], theta_sigma=60)

    processed_data = {
        'rgbd': rgbd,
        'pick_pixel': pick_pixel,
        'pick_theta': pick_theta,
        'place_pixel': place_pixel,
        'place_theta': place_theta,
        'goal_text': info['lang_goal']
    }
    return processed_data


def data_to_attention_label(data):
    theta = data['pick_theta'] / (2.0 * np.pi)
    theta = np.int32(np.round(theta))
    rgbd = data['rgbd']
    label_shape = rgbd.shape[:2] + (1, )
    label = np.zeros(label_shape)
    pick_pixel = data['pick_pixel']
    label[pick_pixel[0], pick_pixel[1], theta] = 1.0
    label = label.transpose((2, 0, 1))
    label = label.reshape(np.prod(label.shape))

    return label


def data_to_transport_label(data):
    rotations = 36
    theta = data['place_theta'] / (2.0 * np.pi / rotations)
    theta = np.int32(np.round(theta)) % rotations
    rgbd = data['rgbd']
    label_shape = rgbd.shape[:2] + (rotations, )
    label = np.zeros(label_shape)
    place_pixel = data['place_pixel']
    label[place_pixel[0], place_pixel[1], theta] = 1.0
    label = label.transpose((2, 0, 1))
    label = label.reshape(np.prod(label.shape))

    return label


def unzip(zipped):
    return list(zip(*zipped))


def save_snapshot(outdir, models):
    outdir.mkdir(parents=True, exist_ok=True)
    for model in models:
        filename = model.scope_name + '.h5'
        model.save_parameters(outdir / filename)


def sample_to_training_data(sample, augment):
    sample = process_sampled_data(sample, augment)
    rgbd = sample['rgbd']
    goal_text = sample['goal_text']
    goal_text = nnabla_clip.tokenize([goal_text])
    pivot = sample['pick_pixel']
    attention_label = data_to_attention_label(sample)
    transport_label = data_to_transport_label(sample)
    return (rgbd, goal_text, pivot, attention_label, transport_label)


def sample_batch(batch_size, task_episodes, augment=True):
    batch = []
    for _ in range(batch_size):
        task_number = np.random.choice(len(task_episodes))
        episodes = task_episodes[task_number]
        episode_number = np.random.choice(len(episodes))
        episode = episodes[episode_number]
        sample_index = np.random.choice(range(len(episode) - 1))
        sample = episode[sample_index]
        data = sample_to_training_data(sample, augment)
        batch.append(data)
    return batch


def build_training_graph(rgbd,
                         text,
                         attention_label,
                         transport_label,
                         attention: CLIPortAttention,
                         transport: CLIPortTransport,
                         args):
    attention_logits = attention.compute_features(
        rgbd, text, apply_softmax=False)
    attention_loss = -attention_label * F.log_softmax(attention_logits)
    attention_loss = F.sum(
        attention_loss) if args.sum else F.mean(attention_loss)

    transport_logits = transport.compute_features(
        rgbd, text, apply_softmax=False)
    transport_loss = -transport_label * F.log_softmax(transport_logits)
    transport_loss = F.sum(
        transport_loss) if args.sum else F.mean(transport_loss)

    return attention_loss, transport_loss


def build_cliport_model(args):
    scriptdir = pathlib.Path(__file__).parent

    image_encoder = CLIPImageEncoder('image_encoder', training=True)
    image_encoder.load_parameters(f'{scriptdir}/data/image_encoder.h5')
    attention_image_encoder = image_encoder.deepcopy('attention_image_encoder')
    transport_key_image_encoder = image_encoder.deepcopy(
        'transport_key_image_encoder')
    transport_query_image_encoder = image_encoder.deepcopy(
        'transport_query_image_encoder')
    text_encoder = CLIPTextEncoder('text_encoder')
    text_encoder.load_parameters(f'{scriptdir}/data/text_encoder.h5')

    attention = CLIPortAttention('attention',
                                 image_encoder=attention_image_encoder,
                                 text_encoder=text_encoder,
                                 training=True,
                                 half_precision=args.gpu if args.half else None)
    transport = CLIPortTransport('transport',
                                 key_image_encoder=transport_key_image_encoder,
                                 query_image_encoder=transport_query_image_encoder,
                                 text_encoder=text_encoder,
                                 training=True,
                                 half_precision=args.gpu if args.half else None)
    return attention, transport


def write_json(outpath, dictionary):
    info_json = json.dumps(dictionary, indent=4)
    with open(str(outpath), 'w') as outfile:
        outfile.write(info_json)


class Trainer():
    def __init__(self, attention, transport, args):
        rotations = 36
        self._rgbd_var = nn.Variable(shape=(args.batch_size, 6, 320, 160))
        self._text_var = nn.Variable(shape=(args.batch_size, 77))  # text token
        attention_shape = (1, *self._rgbd_var.shape[-2:])
        self._attention_label_var = nn.Variable(
            shape=(args.batch_size, np.prod(attention_shape)))
        transport_shape = (rotations, *self._rgbd_var.shape[-2:])
        self._transport_label_var = nn.Variable(
            shape=(args.batch_size, np.prod(transport_shape)))

        self._attention = attention
        self._transport = transport
        self._attention_loss, self._transport_loss = build_training_graph(self._rgbd_var,
                                                                          self._text_var,
                                                                          self._attention_label_var,
                                                                          self._transport_label_var,
                                                                          attention,
                                                                          transport,
                                                                          args)
        self._total_loss = self._attention_loss + self._transport_loss
        self._attention_loss.persistent = True
        self._transport_loss.persistent = True
        self._total_loss.persistent = True

        self._attention_solver = S.Adam(alpha=1.0e-4)
        self._attention_solver.set_parameters(attention.get_parameters())
        self._transport_solver = S.Adam(alpha=1.0e-4)
        self._transport_solver.set_parameters(transport.get_parameters())
        self._clear_buffer = not args.no_clear_buffer

    def train(self, batch_size, task_episodes):
        batch = sample_batch(batch_size=batch_size,
                             task_episodes=task_episodes)
        (rgbd, text, pivot, attention_label, transport_label) = unzip(batch)

        self._rgbd_var.d = np.asarray(rgbd).transpose((0, 3, 1, 2))
        self._text_var.d = np.asarray(text)
        self._attention_label_var.d = np.asarray(attention_label)
        self._transport_label_var.d = np.asarray(transport_label)
        self._transport.set_pivot(np.asarray(pivot))

        self._attention_solver.zero_grad()
        self._transport_solver.zero_grad()
        self._total_loss.forward(clear_no_need_grad=True)
        self._total_loss.backward(clear_buffer=self._clear_buffer)
        self._attention_solver.update()
        self._transport_solver.update()

        return self._attention_loss.d, self._transport_loss.d, self._total_loss.d

    def evaluate(self, batch_size, eval_episodes):
        num_samples = 100
        total_attention_loss = 0
        total_transport_loss = 0
        for _ in range(num_samples):
            batch = sample_batch(batch_size, eval_episodes, augment=False)

            (rgbd, text, pivot, attention_label, transport_label) = unzip(batch)

            self._rgbd_var.d = np.asarray(rgbd).transpose((0, 3, 1, 2))
            self._text_var.d = np.asarray(text)
            self._attention_label_var.d = np.asarray(attention_label)
            self._transport_label_var.d = np.asarray(transport_label)
            self._transport.set_pivot(np.asarray(pivot))

            self._total_loss.forward(clear_buffer=True)
            total_attention_loss += self._attention_loss.d
            total_transport_loss += self._transport_loss.d

        return total_attention_loss, total_transport_loss


def prepare_output_dir(base_dir, args, time_format='%Y-%m-%d-%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    outdir = base_dir / time_str
    outdir.mkdir(parents=True, exist_ok=False)

    # Save all the arguments
    args_file_path = outdir / 'args.json'
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    write_json(args_file_path, json.dumps(args))

    # Save the command
    argv_file_path = outdir / 'command.json'
    argv = ' '.join(sys.argv)
    write_json(argv_file_path, json.dumps(argv))

    return outdir


def run_training(task_episodes, eval_episodes, args):
    attention, transport = build_cliport_model(args)
    trainer = Trainer(attention, transport, args)

    prev_attention_eval_result = None
    prev_transport_eval_result = None

    iterations = args.iterations
    batch_size = args.batch_size
    scriptdir = pathlib.Path(__file__).parent
    outdir = prepare_output_dir(scriptdir / args.outdir, args=args)
    writer = SummaryWriter()
    for iteration in range(1, iterations+1):
        # fill training data
        attention_loss, transport_loss, total_loss = trainer.train(
            batch_size, task_episodes)

        print(f'iteration #{iteration}')
        writer.add_scalar('attention_loss', np.squeeze(
            attention_loss), iteration)
        writer.add_scalar('transport_loss', np.squeeze(
            transport_loss), iteration)
        writer.add_scalar('total_loss', np.squeeze(total_loss), iteration)
        if iteration % args.save_interval == 0:
            if args.save_best:
                attention_eval_result, transport_eval_result = trainer.evaluate(
                    batch_size, eval_episodes)
                savedir = outdir / 'best-iteration'
                if prev_attention_eval_result is None or attention_eval_result < prev_attention_eval_result:
                    info = {'iteration': f'{iteration}',
                            'loss': f'{attention_eval_result}'}
                    save_snapshot(savedir, [attention,
                                            attention._image_encoder])
                    write_json(savedir / 'best-attention.json', info)
                    prev_attention_eval_result = attention_eval_result
                if prev_transport_eval_result is None or transport_eval_result < prev_transport_eval_result:
                    info = {'iteration': f'{iteration}',
                            'loss': f'{transport_eval_result}'}
                    save_snapshot(savedir, [transport,
                                            transport._key_image_encoder,
                                            transport._query_image_encoder])
                    write_json(savedir / 'best-transport.json', info)
                    prev_transport_eval_result = transport_eval_result
            else:
                savedir = outdir / f'iteration-{iteration}'
                save_snapshot(savedir, [attention,
                                        transport,
                                        attention._image_encoder,
                                        transport._key_image_encoder,
                                        transport._query_image_encoder])


def load_data(load_file):
    with open(str(load_file), 'rb') as f:
        return pickle.load(f)


def load_episodes(dataset_root, task_name, mode):
    scriptdir = pathlib.Path(__file__).parent
    dataset_root = scriptdir / pathlib.Path(dataset_root)
    rgb_dir = dataset_root / f'{task_name}-{mode}' / 'color'

    filename_pattern = r'([0-9]+)-([0-9]+).pkl'
    pattern_regex = re.compile(filename_pattern)

    episodes = []
    for episode_file in sorted(rgb_dir.iterdir()):
        print(f'episode_file: {episode_file.name}')
        result = pattern_regex.match(str(episode_file.name))
        groups = result.groups()
        episode_number, seed = int(groups[0]), int(groups[1])
        print(f'episode_number: {episode_number}, seed: {seed}')
        episode = load_episode(dataset_root, task_name,
                               mode, episode_number, seed)
        episodes.append(episode)
    return episodes


def load_episode(load_path, task_name, mode, episode_number, seed):
    load_path = pathlib.Path(load_path)
    assert load_path.exists()

    file_name = f'{episode_number:06d}-{seed}.pkl'
    episode_data = {}
    file_types = ['color', 'depth', 'action', 'reward', 'info']
    for file_type in file_types:
        load_file = load_path / f'{task_name}-{mode}' / file_type / file_name
        data = load_data(load_file)
        episode_data[file_type] = data
    # reconstruct episode
    episode = []
    for (rgb, depth, action, reward, info) in zip(*[episode_data[type] for type in file_types]):
        state = {'color': rgb, 'depth': depth}
        episode.append((state, action, reward, info))
    return episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--task', type=str, nargs='+',
                        default=['stack-block-pyramid-seq-seen-colors'],
                        choices=['align-rope',
                                 'assembling-kits-seq-seen-colors'
                                 'packing-boxes-pairs-seen-colors'
                                 'packing-shapes',
                                 'packing-seen-google-objects-seq'
                                 'packing-seen-google-objects-group',
                                 'put-block-in-bowl-seen-colors',
                                 'stack-block-pyramid-seq-seen-colors',
                                 'separating-piles-seen-colors',
                                 'towers-of-hanoi-seq-seen-colors'])
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'test'])
    parser.add_argument('--iterations', type=int, default=200 * 1000)
    parser.add_argument('--outdir', type=str, default='training_results')
    parser.add_argument('--save-interval', type=int, default=2500)
    parser.add_argument('--sum', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--save-best', action='store_true')
    parser.add_argument('--no-clear-buffer', action='store_true')

    args = parser.parse_args()

    task_episodes = []
    eval_episodes = []
    for task in args.task:
        episodes = load_episodes(args.dataset_path, task, args.mode)
        task_episodes.append(episodes)
        if args.save_best:
            episodes = load_episodes(args.dataset_path, task, 'val')
            eval_episodes.append(episodes)
    context = get_extension_context('cudnn', device_id=args.gpu)
    with nn.context_scope(context):
        run_training(task_episodes, eval_episodes, args)


if __name__ == '__main__':
    main()
