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
import random

import pathlib

import pickle

import cliport
from cliport import tasks
from cliport.environments.environment import Environment


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


def pickle_data(save_file, data):
    with open(str(save_file), 'wb') as f:
        pickle.dump(data, f)


def save_episode(save_path, task_name, mode, seed, episode, episode_number):
    rgb_images = []
    depth_images = []
    actions = []
    rewards = []
    info_list = []
    for state, action, reward, info in episode:
        rgb_images.append(np.uint8(state['color']))
        depth_images.append(np.float32(state['depth']))
        actions.append(action)
        rewards.append(reward)
        info_list.append(info)
    save_path = pathlib.Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    episode_data = {
        'color': rgb_images,
        'depth': depth_images,
        'action': actions,
        'reward': rewards,
        'info': info_list
    }
    file_name = f'{episode_number:06d}-{seed}.pkl'
    for (file_type, data) in episode_data.items():
        save_file = save_path / f'{task_name}-{mode}' / file_type / file_name
        save_file.parent.mkdir(parents=True, exist_ok=True)
        pickle_data(save_file, data)


def collect_dataset(args):
    env = build_cliport_environment(args)

    task = tasks.names[args.task]()
    task.mode = args.mode

    oracle = task.oracle(env)

    episodes = []
    if args.mode == 'train':
        seed = 0
    elif args.mode == 'val':
        seed = 5000
    elif args.mode == 'test':
        seed = 10000
    else:
        raise NotImplementedError
    while len(episodes) < args.num_episodes:
        np.random.seed(seed)
        random.seed(seed)
        env.set_task(task)
        state = env.reset()
        info = env.info

        reward = 0
        total_reward = 0
        episode = []
        for _ in range(task.max_steps):
            action = oracle.act(state, info)
            episode.append((state, action, reward, info))
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        episode.append((state, None, reward, info))
        print(
            f'episode No.{len(episodes)} finished. total reward: {total_reward}')

        if total_reward > 0.99:
            episodes.append((episode, seed))
        seed += 1

    for episode_number, (episode, seed) in enumerate(episodes):
        save_episode(args.save_path, args.task, args.mode,
                     seed, episode, episode_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, required=True)
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
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    collect_dataset(args)


if __name__ == '__main__':
    main()
