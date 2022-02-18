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

import gym

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla_rl.hooks as H
from nnabla_rl.algorithms import CategoricalDQN, CategoricalDQNConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder, ReplayBufferBuilder
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.models import DiscreteValueDistributionFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.replay_buffers import MemoryEfficientAtariBuffer
from nnabla_rl.utils.reproductions import build_atari_env  # noqa
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.writers import FileWriter


def build_classic_control_env(env_name, render=False):
    env = gym.make(env_name)
    env = NumpyFloat32Env(env)
    if render:
        # render environment if render is True
        env = ScreenRenderEnv(env)
    return env


class ExampleClassicControlValueDistributionFunction(DiscreteValueDistributionFunction):
    def __init__(self, scope_name: str, n_action: int, n_atom: int, v_min: float, v_max: float):
        super(ExampleClassicControlValueDistributionFunction, self).__init__(
            scope_name, n_action, n_atom, v_min, v_max)

    def all_probs(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine4"):
                h = PF.affine(h, n_outmaps=self._n_action * self._n_atom)
            h = F.reshape(h, (-1, self._n_action, self._n_atom))
        # Output must sum up to 1
        return F.softmax(h, axis=2)


class ExampleAtariValueDistributionFunction(DiscreteValueDistributionFunction):
    def __init__(self, scope_name: str, n_action: int, n_atom: int, v_min: float, v_max: float):
        super(ExampleAtariValueDistributionFunction, self).__init__(
            scope_name, n_action, n_atom, v_min, v_max)

    def all_probs(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = PF.convolution(s, 32, (8, 8), stride=(4, 4))
                h = F.relu(h)
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, 64, (4, 4), stride=(2, 2))
                h = F.relu(h)
            with nn.parameter_scope("conv3"):
                h = PF.convolution(h, 64, (3, 3), stride=(1, 1))
                h = F.relu(h)
            h = F.reshape(h, (-1, 3136))
            with nn.parameter_scope("affine1"):
                h = PF.affine(h, 512)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=self._n_action * self._n_atom)
            h = F.reshape(h, (-1, self._n_action, self._n_atom))
        # Output must sum up to 1
        return F.softmax(h, axis=2)


class ExampleValueDistributionFunctionBuilder(ModelBuilder):
    def __init__(self, is_atari=False):
        self._is_atari = is_atari

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_atari:
            return ExampleAtariValueDistributionFunction(scope_name,
                                                         env_info.action_dim,
                                                         algorithm_config.num_atoms,
                                                         algorithm_config.v_min,
                                                         algorithm_config.v_max)
        else:
            return ExampleClassicControlValueDistributionFunction(scope_name,
                                                                  env_info.action_dim,
                                                                  algorithm_config.num_atoms,
                                                                  algorithm_config.v_min,
                                                                  algorithm_config.v_max)


class ExampleValueDistributionSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: CategoricalDQNConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate,
                        eps=1e-2 / algorithm_config.batch_size)
        return solver


class ExampleReplayBufferBuilder(ReplayBufferBuilder):
    def __init__(self, is_atari=False):
        self._is_atari = is_atari

    def build_replay_buffer(self, env_info, algorithm_config, **kwargs):
        config: CategoricalDQNConfig = algorithm_config
        if self._is_atari:
            return MemoryEfficientAtariBuffer(capacity=config.replay_buffer_size)
        else:
            return ReplayBuffer(capacity=config.replay_buffer_size)


def train():
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = 'CartPole-v1'
    train_env = build_classic_control_env(env_name)
    # evaluation env is used only for running the evaluation of models during the training.
    # if you do not evaluate the model during the training, this environment is not necessary.
    eval_env = build_classic_control_env(env_name, render=True)
    is_atari = False
    start_timesteps = 5000
    max_explore_steps = 10000
    evaluation_timing = 10000
    total_iterations = 100000
    num_atoms = 20

    # If you want to train on atari games, uncomment below
    # You can change the name of environment to change the game to train.
    # For the list of available games see: https://gym.openai.com/envs/#atari
    # Your machine must at least have more than 20GB of memory to run the training.
    # Adjust the replay_buffer_size through CategoricalDQNConfig if you do not have enough memory on your machine.
    # env_name = 'BreakoutNoFrameskip-v4'
    # train_env = build_atari_env(env_name)
    # eval_env = build_atari_env(env_name, test=True, render=True)
    # is_atari = True
    # start_timesteps = 50000
    # max_explore_steps = 1000000
    # evaluation_timing = 250000
    # total_iterations = 50000000
    # num_atoms = 51

    # Will output evaluation results and model snapshots to the outdir
    outdir = f'{env_name}_results'

    # Writer will save the evaluation results to file.
    # If you set writer=None, evaluator will only print the evaluation results on terminal.
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicEvaluator(run_per_evaluation=5)
    # evaluate the trained model with eval_env every 5000 iterations
    # change the timing to 250000 on atari games.
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=evaluation_timing, writer=writer)

    # This will print the iteration number every 100 iteration.
    # Printing iteration number is convenient for checking the training progress.
    # You can change this number to any number of your choice.
    iteration_num_hook = H.IterationNumHook(timing=100)

    # save the trained model every 5000 iterations
    # change the timing to 250000 on atari games.
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=evaluation_timing)

    # Set gpu_id to -1 to train on cpu.
    gpu_id = 0
    config = CategoricalDQNConfig(gpu_id=gpu_id,
                                  learning_rate=2.5e-4,
                                  num_atoms=num_atoms,
                                  start_timesteps=start_timesteps,
                                  max_explore_steps=max_explore_steps)
    c51 = CategoricalDQN(train_env,
                         config=config,
                         value_distribution_builder=ExampleValueDistributionFunctionBuilder(
                             is_atari=is_atari),
                         value_distribution_solver_builder=ExampleValueDistributionSolverBuilder(),
                         replay_buffer_builder=ExampleReplayBufferBuilder(is_atari=is_atari))
    # Set instanciated hooks to periodically run additional jobs
    c51.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    c51.train(train_env, total_iterations=total_iterations)


if __name__ == '__main__':
    train()
