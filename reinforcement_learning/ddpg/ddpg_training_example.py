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
from nnabla_rl.algorithms import DDPG, DDPGConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder, ReplayBufferBuilder
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.models import DeterministicPolicy, ContinuousQFunction
from nnabla_rl.replay_buffer import ReplayBuffer
from nnabla_rl.utils.reproductions import build_mujoco_env  # noqa
from nnabla_rl.utils.evaluator import EpisodicEvaluator
from nnabla_rl.writers import FileWriter


def build_classic_control_env(env_name, render=False):
    env = gym.make(env_name)
    env = NumpyFloat32Env(env)
    if render:
        # render environment if render is True
        env = ScreenRenderEnv(env)
    return env


class ExampleClassicControlQFunction(ContinuousQFunction):
    def __init__(self, scope_name: str):
        super(ExampleClassicControlQFunction, self).__init__(scope_name)

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        h = F.concatenate(s, a)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(h, n_outmaps=200)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=1)
        return h


class ExampleClassicControlPolicy(DeterministicPolicy):
    def __init__(self, scope_name: str, action_dim: int):
        super(ExampleClassicControlPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=200)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=self._action_dim)
        return h


class ExampleMujocoQFunction(ContinuousQFunction):
    def __init__(self, scope_name: str):
        super(ExampleMujocoQFunction, self).__init__(scope_name)

    def q(self, s: nn.Variable, a: nn.Variable) -> nn.Variable:
        h = F.concatenate(s, a)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(h, n_outmaps=400)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=300)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=1)
        return h


class ExampleMujocoPolicy(DeterministicPolicy):
    def __init__(self, scope_name: str, action_dim: int, max_action_value: float):
        super(ExampleMujocoPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def pi(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=400)
                h = F.relu(x=h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=300)
                h = F.relu(x=h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=self._action_dim)
        return F.tanh(h) * self._max_action_value


class ExamplePolicyBuilder(ModelBuilder):
    def __init__(self, is_mujoco=False):
        self._is_mujoco = is_mujoco

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_mujoco:
            max_action_value = float(env_info.action_space.high[0])
            return ExampleMujocoPolicy(scope_name, env_info.action_dim, max_action_value)
        else:
            return ExampleClassicControlPolicy(scope_name, env_info.action_dim)


class ExampleQFunctionBuilder(ModelBuilder):
    def __init__(self, is_mujoco=False):
        self._is_mujoco = is_mujoco

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_mujoco:
            return ExampleMujocoQFunction(scope_name)
        else:
            return ExampleClassicControlQFunction(scope_name)


class ExamplePolicySolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: DDPGConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleQSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: DDPGConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self, env_info, algorithm_config, **kwargs):
        config: DDPGConfig = algorithm_config
        return ReplayBuffer(capacity=config.replay_buffer_size)


def train():
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = 'Pendulum-v1'
    train_env = build_classic_control_env(env_name)
    # evaluation env is used only for running the evaluation of models during the training.
    # if you do not evaluate the model during the training, this environment is not necessary.
    eval_env = build_classic_control_env(env_name, render=True)
    evaluation_timing = 1000
    start_timesteps = 200
    total_iterations = 10000
    is_mujoco = False

    # If you want to train on mujoco, uncomment below
    # You can change the name of environment to change the environment to train.
    # env_name = 'HalfCheetah-v2'
    # train_env = build_mujoco_env(env_name)
    # eval_env = build_mujoco_env(env_name, test=True, render=True)
    # evaluation_timing = 5000
    # start_timesteps: int = 10000
    # total_iterations = 1000000
    # is_mujoco = True

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
    config = DDPGConfig(gpu_id=gpu_id, learning_rate=1e-3,
                        start_timesteps=start_timesteps)
    ddpg = DDPG(train_env,
                config=config,
                actor_builder=ExamplePolicyBuilder(is_mujoco=is_mujoco),
                actor_solver_builder=ExamplePolicySolverBuilder(),
                critic_builder=ExampleQFunctionBuilder(is_mujoco=is_mujoco),
                critic_solver_builder=ExampleQSolverBuilder(),
                replay_buffer_builder=ExampleReplayBufferBuilder())
    # Set instanciated hooks to periodically run additional jobs
    ddpg.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    ddpg.train(train_env, total_iterations=total_iterations)


if __name__ == '__main__':
    train()
