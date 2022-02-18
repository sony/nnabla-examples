# PPO training example

This is an example code for training models with [PPO algorithm](https://arxiv.org/abs/1812.05905) on nnabla.

## Prerequisites

This example uses [nnabla-rl](https://github.com/sony/nnabla-rl).
Install nnabla-rl with:

```sh
$ pip install nnabla-rl
```

For the details of nnabla-rl, visit [nnabla-rl's project page](https://github.com/sony/nnabla-rl).

## How to start the training

Just run the below command.
Wait for a while (1 hour approx. It depends on your machine's spec)
You will see the [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) controlled by the reinforcement learning agent!

```
$ python ppo_training_example.py
```

You can also train the models on mujoco. Check the comments in the training script for details.
