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
import nnabla_rl.distributions as D
import nnabla_rl.hooks as H
from nnabla_rl.algorithms import A2C, A2CConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.distributions import Distribution
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.models import Model, StochasticPolicy, VFunction
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


class ExampleClassicControlPolicy(StochasticPolicy):
    def __init__(self, scope_name: str, action_dim: int):
        super(ExampleClassicControlPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                z = PF.affine(h, n_outmaps=self._action_dim)
        return D.Softmax(z)


class ExampleClassicControlVFunction(VFunction):
    def __init__(self, scope_name: str):
        super(ExampleClassicControlVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=100)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                v = PF.affine(h, n_outmaps=1)
        return v


class ExampleAtariSharedFunctionHead(Model):
    def __init__(self, scope_name):
        super(ExampleAtariSharedFunctionHead,
              self).__init__(scope_name=scope_name)

    def __call__(self, s):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("conv1"):
                h = PF.convolution(s, outmaps=16, kernel=(8, 8), stride=(4, 4))
            h = F.relu(x=h)
            with nn.parameter_scope("conv2"):
                h = PF.convolution(h, outmaps=32, kernel=(4, 4), stride=(2, 2))
            h = F.relu(x=h)
            h = F.reshape(h, shape=(h.shape[0], -1))
            with nn.parameter_scope("linear1"):
                h = PF.affine(h, n_outmaps=256)
            h = F.relu(x=h)
        return h


class EXampleAtariPolicy(StochasticPolicy):
    def __init__(self, head, scope_name, action_dim):
        super(EXampleAtariPolicy, self).__init__(scope_name=scope_name)
        self._action_dim = action_dim
        self._head = head

    def pi(self, s: nn.Variable) -> Distribution:
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_pi"):
                z = PF.affine(h, n_outmaps=self._action_dim)
        return D.Softmax(z=z)

    def _hidden(self, s):
        return self._head(s)


class EXampleAtariVFunction(VFunction):
    def __init__(self, head, scope_name):
        super(EXampleAtariVFunction, self).__init__(scope_name=scope_name)
        self._head = head

    def v(self, s):
        h = self._hidden(s)
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("linear_v"):
                v = PF.affine(h, n_outmaps=1)
        return v

    def _hidden(self, s):
        return self._head(s)


class ExamplePolicyBuilder(ModelBuilder):
    def __init__(self, is_atari=False):
        self._is_atari = is_atari

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_atari:
            scope_name = "shared"
            head = ExampleAtariSharedFunctionHead(scope_name)
            return EXampleAtariPolicy(head, scope_name, env_info.action_dim)
        else:
            return ExampleClassicControlPolicy(scope_name, env_info.action_dim)


class ExampleVFunctionBuilder(ModelBuilder):
    def __init__(self, is_atari=False):
        self._is_atari = is_atari

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_atari:
            scope_name = "shared"
            head = ExampleAtariSharedFunctionHead(scope_name)
            return EXampleAtariVFunction(head, scope_name)
        else:
            return ExampleClassicControlVFunction(scope_name)


class ExamplePolicySolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: A2CConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleVFunctionSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: A2CConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


def train():
    # 1. prepare environment
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = 'CartPole-v1'
    train_env = build_classic_control_env(env_name)
    # Evaluation env is used only for running the evaluation of models during the training.
    # If you do not evaluate the model during the training, this environment is not necessary.
    eval_env = build_classic_control_env(env_name, render=True)
    # Set some parameters used in training model
    is_atari = False
    start_timesteps = 1
    evaluation_timing = 10000
    total_iterations = 100000

    # If you want to train on atari games, uncomment below
    # You can change the name of environment to change the game to train.
    # For the list of available games see: https://gym.openai.com/envs/#atari
    # Your machine must at least have more than 20GB of memory to run the training.
    # Adjust the actor_num through A2CConfig if you do not have enough memory on your machine.
    # env_name = 'BreakoutNoFrameskip-v4'
    # train_env = build_atari_env(env_name)
    # eval_env = build_atari_env(env_name, test=True, render=True)
    # is_atari = True
    # start_timesteps = 1
    # evaluation_timing = 250000
    # total_iterations = 50000000

    # Will output evaluation results and model snapshots to the outdir
    outdir = f'{env_name}_results'

    # Writer will save the evaluation results to file.
    # If you set writer=None, evaluator will only print the evaluation results on terminal.
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicEvaluator(run_per_evaluation=5)
    # Evaluate the trained model with eval_env every 5000 iterations
    # Change the timing to 250000 on atari games.
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
    config = A2CConfig(gpu_id=gpu_id, start_timesteps=start_timesteps)
    a2c = A2C(train_env,
              config=config,
              v_function_builder=ExampleVFunctionBuilder(is_atari=is_atari),
              v_solver_builder=ExampleVFunctionSolverBuilder(),
              policy_builder=ExamplePolicyBuilder(is_atari=is_atari),
              policy_solver_builder=ExamplePolicySolverBuilder())
    # Set instanciated hooks to periodically run additional jobs
    a2c.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    a2c.train(train_env, total_iterations=total_iterations)


if __name__ == '__main__':
    train()
