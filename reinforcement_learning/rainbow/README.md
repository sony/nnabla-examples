# Rainbow training example

This is an example code for training models with [Rainbow algorithm](https://arxiv.org/abs/1710.02298) on nnabla.

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
$ python rainbow_training_example.py
```

You can also train the models on atari games. Check the comments in the training script for details.
