# A2C training example

This is an example code for training models with A2C algorithm on nnabla.
A2C is a synchronous version of A3C (Asynchronous Advantage Actor-Critic) proposed by V. Mnih, et al. in the paper: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).

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
$ python a2c_training_example.py
```

You can also train the models on atari games. Check the comments in the training script for details.