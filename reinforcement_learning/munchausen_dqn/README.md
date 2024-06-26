# MunchausenDQN training example

This is an example code for training models with [MunchausenDQN algorithm](https://proceedings.neurips.cc/paper/2020/file/2c6a0bae0f071cbbf0bb3d5b11d90a82-Paper.pdf) on nnabla.

## Prerequisites

This example uses [nnabla-rl](https://github.com/sony/nnabla-rl).
Install nnabla-rl with:

```sh
$ pip install nnabla-rl
```

For the details of nnabla-rl, visit [nnabla-rl's project page](https://github.com/sony/nnabla-rl).

This example render's the training environment. Also install pyglet.

```sh
$ pip install pyglet
```

### Optional

If you want to train the models on Atari games, also run:

```sh
$ pip install gym[atari]
```

and also accept the atari rom license and install the roms with:

```sh
$ pip install gym[accept-rom-license]
```

## How to start the training

Just run the below command. You will see the [cartpole](https://gym.openai.com/envs/CartPole-v1/) controlled by the reinforcement learning agent in a couple of minutes!

```
$ python m_dqn_training_example.py
```

You can also train the models on atari games. Check the comments in the training script for details.