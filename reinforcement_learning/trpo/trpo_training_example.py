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
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import nnabla.solvers as S
import nnabla_rl.hooks as H
from nnabla_rl.distributions import Gaussian, Distribution
from nnabla_rl.algorithms import TRPO, TRPOConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.models import StochasticPolicy, VFunction
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


class ExampleClassicControlVFunction(VFunction):
    def __init__(self, scope_name: str):
        super(ExampleClassicControlVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=64)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=64)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=1)
        return h


class ExampleMujocoVFunction(VFunction):
    def __init__(self, scope_name: str):
        super(ExampleMujocoVFunction, self).__init__(scope_name)

    def v(self, s: nn.Variable) -> nn.Variable:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=64)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=64)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=1)
        return h


class ExampleClassicControlPolicy(StochasticPolicy):
    def __init__(self, scope_name: str, action_dim: int):
        super(ExampleClassicControlPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=64,
                              w_init=I.OrthogonalInitializer(np.sqrt(2.0)))
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=64,
                              w_init=I.OrthogonalInitializer(np.sqrt(2.0)))
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=self._action_dim * 2,
                              w_init=I.OrthogonalInitializer(np.sqrt(0.01)))
            reshaped = F.reshape(h, shape=(-1, 2, self._action_dim))
            mean, ln_sigma = F.split(reshaped, axis=1)
            ln_var = ln_sigma * 2.0
        return Gaussian(mean=mean, ln_var=ln_var)


class ExampleMujocoPolicy(StochasticPolicy):
    def __init__(self, scope_name: str, action_dim: int):
        super(ExampleMujocoPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable) -> Distribution:
        with nn.parameter_scope(self.scope_name):
            h = PF.affine(s, n_outmaps=64, name="linear1",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.0)))
            h = F.tanh(x=h)
            h = PF.affine(h, n_outmaps=64, name="linear2",
                          w_init=I.OrthogonalInitializer(np.sqrt(2.0)))
            h = F.tanh(x=h)
            mean = PF.affine(
                h, n_outmaps=self._action_dim, name="linear3", w_init=I.OrthogonalInitializer(np.sqrt(2.0))
            )
            assert mean.shape == (s.shape[0], self._action_dim)

            ln_sigma = nn.parameter.get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=I.ConstantInitializer(0.0)
            )
            ln_var = F.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
        return Gaussian(mean, ln_var)


class ExamplePolicyBuilder(ModelBuilder):
    def __init__(self, is_mujoco=False):
        self._is_mujoco = is_mujoco

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_mujoco:
            return ExampleMujocoPolicy(scope_name, env_info.action_dim)
        else:
            return ExampleClassicControlPolicy(scope_name, env_info.action_dim)


class ExampleVFunctionBuilder(ModelBuilder):
    def __init__(self, is_mujoco=False):
        self._is_mujoco = is_mujoco

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_mujoco:
            return ExampleMujocoVFunction(scope_name)
        else:
            return ExampleClassicControlVFunction(scope_name)


class ExampleVSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: TRPOConfig = algorithm_config
        solver = S.Adam(alpha=config.vf_learning_rate)
        return solver


def train():
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = "Pendulum-v1"
    train_env = build_classic_control_env(env_name)
    # evaluation env is used only for running the evaluation of models during the training.
    # if you do not evaluate the model during the training, this environment is not necessary.
    eval_env = build_classic_control_env(env_name, render=True)
    evaluation_timing = 5000
    total_iterations = 250000
    pi_batch_size = 5000
    num_steps_per_iteration = 5000
    is_mujoco = False

    # If you want to train on mujoco, uncomment below
    # You can change the name of environment to change the environment to train.
    # env_name = "HalfCheetah-v2"
    # train_env = build_mujoco_env(env_name)
    # eval_env = build_mujoco_env(env_name, test=True, render=True)
    # evaluation_timing = 5000
    # total_iterations = 1000000
    # pi_batch_size = 5000
    # num_steps_per_iteration = 5000
    # is_mujoco = True

    # Will output evaluation results and model snapshots to the outdir
    outdir = f"{env_name}_results"

    # Writer will save the evaluation results to file.
    # If you set writer=None, evaluator will only print the evaluation results on terminal.
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicEvaluator(run_per_evaluation=5)
    # evaluate the trained model with eval_env every 5000 iterations
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=evaluation_timing, writer=writer)

    # This will print the iteration number every 100 iteration.
    # Printing iteration number is convenient for checking the training progress.
    # You can change this number to any number of your choice.
    iteration_num_hook = H.IterationNumHook(timing=100)

    # save the trained model every 5000 iterations
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=evaluation_timing)

    # Set gpu_id to -1 to train on cpu.
    gpu_id = 0
    config = TRPOConfig(gpu_id=gpu_id, pi_batch_size=pi_batch_size,
                        num_steps_per_iteration=num_steps_per_iteration)
    trpo = TRPO(
        train_env,
        config=config,
        policy_builder=ExamplePolicyBuilder(is_mujoco=is_mujoco),
        v_function_builder=ExampleVFunctionBuilder(is_mujoco=is_mujoco),
        v_solver_builder=ExampleVSolverBuilder(),
    )
    # Set instanciated hooks to periodically run additional jobs
    trpo.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    trpo.train(train_env, total_iterations=total_iterations)


if __name__ == "__main__":
    train()
