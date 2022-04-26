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
import nnabla_rl.initializers as RI
from nnabla_rl.algorithms import HER, HERConfig
from nnabla_rl.builders import ModelBuilder, SolverBuilder, ReplayBufferBuilder
from nnabla_rl.environments.wrappers import ScreenRenderEnv, NumpyFloat32Env
from nnabla_rl.environments.wrappers.goal_conditioned import GoalConditionedTupleObservationEnv
from nnabla_rl.models import DeterministicPolicy, ContinuousQFunction
from nnabla_rl.replay_buffers.hindsight_replay_buffer import HindsightReplayBuffer

from nnabla_rl.utils.evaluator import EpisodicSuccessEvaluator
from nnabla_rl.writers import FileWriter


def build_mujoco_goal_conditioned_env(id_or_env, test=False, seed=None, render=False):
    if isinstance(id_or_env, gym.Env):
        env = id_or_env
    else:
        env = gym.make(id_or_env)
    env = GoalConditionedTupleObservationEnv(env)
    env = NumpyFloat32Env(env)
    if render:
        env = ScreenRenderEnv(env)

    env.seed(seed)
    return env


class ExampleMujocoQFunction(ContinuousQFunction):
    def __init__(self, scope_name, optimal_policy=None):
        super(ExampleMujocoQFunction, self).__init__(scope_name)
        self._optimal_policy = optimal_policy

    def q(self, s, a):
        obs, goal, _ = s
        with nn.parameter_scope(self.scope_name):
            h = F.concatenate(obs, goal, a, axis=1)
            linear1_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear1', w_init=linear1_init)
            h = F.relu(h)
            linear2_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear2', w_init=linear2_init)
            h = F.relu(h)
            linear3_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear3', w_init=linear3_init)
            h = F.relu(h)
            pred_q_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=1)
            q = PF.affine(h, n_outmaps=1, name='pred_q', w_init=pred_q_init)
        return q

    def max_q(self, s: nn.Variable) -> nn.Variable:
        assert self._optimal_policy, 'Optimal policy is not set!'
        optimal_action = self._optimal_policy.pi(s)
        return self.q(s, optimal_action)


class ExampleMujocoPolicy(DeterministicPolicy):
    def __init__(self, scope_name: str, action_dim: int, max_action_value: float):
        super(ExampleMujocoPolicy, self).__init__(scope_name)
        self._action_dim = action_dim
        self._max_action_value = max_action_value

    def pi(self, s) -> nn.Variable:
        # s = (observation, goal, achieved_goal)
        obs, goal, _ = s
        with nn.parameter_scope(self.scope_name):
            h = F.concatenate(obs, goal, axis=1)
            linear1_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear1', w_init=linear1_init)
            h = F.relu(h)
            linear2_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear2', w_init=linear2_init)
            h = F.relu(h)
            linear3_init = RI.GlorotUniform(inmaps=h.shape[1], outmaps=64)
            h = PF.affine(h, n_outmaps=64, name='linear3', w_init=linear3_init)
            h = F.relu(h)
            action_init = RI.GlorotUniform(
                inmaps=h.shape[1], outmaps=self._action_dim)
            h = PF.affine(h, n_outmaps=self._action_dim,
                          name='action', w_init=action_init)
        return F.tanh(h) * self._max_action_value


class ExamplePolicyBuilder(ModelBuilder):
    def __init__(self):
        super(ExamplePolicyBuilder, self).__init__()

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        max_action_value = float(env_info.action_space.high[0])
        return ExampleMujocoPolicy(scope_name, env_info.action_dim, max_action_value)


class ExampleQFunctionBuilder(ModelBuilder):
    def __init__(self, is_mujoco=False):
        super(ExampleQFunctionBuilder, self).__init__()

    def build_model(self, scope_name, env_info, algorithm_config, **kwargs):
        return ExampleMujocoQFunction(scope_name)


class ExamplePolicySolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: HERConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleQSolverBuilder(SolverBuilder):
    def build_solver(self, env_info, algorithm_config, **kwargs):
        config: HERConfig = algorithm_config
        solver = S.Adam(alpha=config.learning_rate)
        return solver


class ExampleReplayBufferBuilder(ReplayBufferBuilder):
    def build_replay_buffer(self, env_info, algorithm_config, **kwargs):
        config: HERConfig = algorithm_config
        return HindsightReplayBuffer(reward_function=env_info.reward_function,
                                     hindsight_prob=config.hindsight_prob,
                                     capacity=config.replay_buffer_size)


def check_success(experiences) -> bool:
    last_info = experiences[-1][-1]
    if last_info['is_success'] == 1.0:
        return True
    else:
        return False


def train():
    # nnabla-rl's Reinforcement learning algorithm requires environment that implements gym.Env interface
    # for the details of gym.Env see: https://github.com/openai/gym
    env_name = 'FetchReach-v1'
    train_env = build_mujoco_goal_conditioned_env(env_name)
    eval_env = build_mujoco_goal_conditioned_env(
        env_name, test=True, render=True)
    start_timesteps: int = 1
    evaluation_timing = 20000
    total_iterations = 200000

    # Will output evaluation results and model snapshots to the outdir
    outdir = f'{env_name}_results'

    # Writer will save the evaluation results to file.
    # If you set writer=None, evaluator will only print the evaluation results on terminal.
    writer = FileWriter(outdir, "evaluation_result")
    evaluator = EpisodicSuccessEvaluator(
        check_success=check_success, run_per_evaluation=5)
    # evaluate the trained model with eval_env every 20000 iterations
    evaluation_hook = H.EvaluationHook(
        eval_env, evaluator, timing=evaluation_timing, writer=writer)

    # This will print the iteration number every 20000 iteration.
    # Printing iteration number is convenient for checking the training progress.
    # You can change this number to any number of your choice.
    iteration_num_hook = H.IterationNumHook(timing=20000)

    # Save the trained model every 20000 iterations
    save_snapshot_hook = H.SaveSnapshotHook(outdir, timing=evaluation_timing)

    # TODO: Now, it does not work to train HER on GPU.
    # So, please train HER on CPU.
    gpu_id = -1
    # Set some parameters for training HER
    config = HERConfig(gpu_id=gpu_id,
                       batch_size=256,
                       gamma=0.98,
                       tau=0.05,
                       exploration_noise_sigma=0.2,
                       n_cycles=10,
                       start_timesteps=start_timesteps)
    her = HER(train_env,
              config=config,
              actor_builder=ExamplePolicyBuilder(),
              actor_solver_builder=ExamplePolicySolverBuilder(),
              critic_builder=ExampleQFunctionBuilder(),
              critic_solver_builder=ExampleQSolverBuilder(),
              replay_buffer_builder=ExampleReplayBufferBuilder())
    # Set instanciated hooks to periodically run additional jobs
    her.set_hooks(
        hooks=[evaluation_hook, iteration_num_hook, save_snapshot_hook])
    her.train(train_env, total_iterations=total_iterations)


if __name__ == '__main__':
    train()
