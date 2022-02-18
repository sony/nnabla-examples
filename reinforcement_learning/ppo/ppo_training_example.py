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
import nnabla.initializer as I
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla_rl.hooks as H
from nnabla_rl.algorithms import PPO, PPOConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder
from nnabla_rl.distributions import Gaussian, Softmax
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.models import StochasticPolicy, VFunction
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
                h = PF.affine(s, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine4"):
                h = PF.affine(h, n_outmaps=1)
        return h


class ExamplePendulumPolicy(StochasticPolicy):
    def __init__(self, scope_name: str, action_dim: int):
        super(ExamplePendulumPolicy, self).__init__(scope_name)
        self._action_dim = action_dim

    def pi(self, s: nn.Variable):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(h)
            with nn.parameter_scope("affine4"):
                mean = PF.affine(h, n_outmaps=self._action_dim)
            ln_sigma = nn.parameter.get_parameter_or_create(
                "ln_sigma", shape=(1, self._action_dim), initializer=I.ConstantInitializer(0.))
            ln_var = F.broadcast(
                ln_sigma, (s.shape[0], self._action_dim)) * 2.0
        return Gaussian(mean=mean, ln_var=ln_var)


class ExampleCartPolePolicy(StochasticPolicy):
    def __init__(self, scope_name: str, n_action: int):
        super(ExampleCartPolePolicy, self).__init__(scope_name)
        self._n_action = n_action

    def pi(self, s: nn.Variable):
        with nn.parameter_scope(self.scope_name):
            with nn.parameter_scope("affine1"):
                h = PF.affine(s, n_outmaps=256)
                h = F.relu(x=h)
            with nn.parameter_scope("affine2"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(x=h)
            with nn.parameter_scope("affine3"):
                h = PF.affine(h, n_outmaps=256)
                h = F.relu(x=h)
            with nn.parameter_scope("affine4"):
                z = PF.affine(h, n_outmaps=self._n_action)
        return Softmax(z=z)


class ExamplePolicyBuilder(ModelBuilder):
    def __init__(self, is_discrete=False):
        self._is_discrete = is_discrete

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        if self._is_discrete:
            return ExampleCartPolePolicy(scope_name, env_info.action_dim)
        else:
            return ExamplePendulumPolicy(scope_name, env_info.action_dim)


class ExampleVFunctionBuilder(ModelBuilder):
    def __init__(self):
        pass

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        return ExampleClassicControlVFunction(scope_name)


class ExamplePolicySolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: PPOConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleVSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: PPOConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


def train():
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = 'Pendulum-v1'
    train_env = build_classic_control_env(env_name)
    # evaluation env is used only for running the evaluation of models during the training.
    # if you do not evaluate the model during the training, this environment is not necessary.
    eval_env = build_classic_control_env(env_name, render=True)
    actor_timesteps = 2048
    actor_num = 8
    batch_size = 32
    lmb = 0.95
    gamma = 0.99
    evaluation_timing = 1000
    entropy_coefficient = 0.0
    learning_rate = 3e-4
    epochs = 10
    total_iterations = 200000
    is_discrete = False

    # If you want to train on discrete action environment, uncomment below
    # You can change the name of environment to change the environment to train.
    # env_name = 'CartPole-v1'
    # train_env = build_classic_control_env(env_name)
    # eval_env = build_classic_control_env(env_name, render=True)
    # actor_timesteps = 32
    # actor_num = 8
    # batch_size = 8
    # lmb = 0.8
    # gamma = 0.98
    # evaluation_timing = 1000
    # entropy_coefficient = 0.0
    # learning_rate = 1e-3
    # epochs = 20
    # total_iterations = 10000
    # is_discrete = True

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
    config = PPOConfig(gpu_id=gpu_id,
                       actor_timesteps=actor_timesteps,
                       learning_rate=learning_rate,
                       batch_size=batch_size,
                       gamma=gamma,
                       lmb=lmb,
                       epsilon=0.2,
                       entropy_coefficient=entropy_coefficient,
                       epochs=epochs,
                       decrease_alpha=False,
                       actor_num=actor_num,
                       preprocess_state=False)
    ppo = PPO(train_env,
              config=config,
              policy_builder=ExamplePolicyBuilder(is_discrete=is_discrete),
              policy_solver_builder=ExamplePolicySolverBuilder(),
              v_function_builder=ExampleVFunctionBuilder(),
              v_solver_builder=ExampleVSolverBuilder())
    # Set instanciated hooks to periodically run additional jobs
    ppo.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    ppo.train(train_env, total_iterations=total_iterations)


if __name__ == '__main__':
    train()
