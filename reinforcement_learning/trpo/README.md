# TRPO training example

This is an example code for training models with [TRPO algorithm](https://arxiv.org/abs/1502.05477) on nnabla.

## Prerequisites

This example uses [nnabla-rl](https://github.com/sony/nnabla-rl).
Install nnabla-rl with:

```sh
$ pip install nnabla-rl
```

For the details of nnabla-rl, visit [nnabla-rl's project page](https://github.com/sony/nnabla-rl).

### Optional

If you want to train the models on MuJoCo, install [MuJoCo](https://mujoco.org).

and also run:

```sh
$ pip install mujoco-py
```

If you face any rendering issue related to opengl try below before starting the script

```sh
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## How to start the training

Just run the below command. You will see the [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) controlled by the reinforcement learning agent in a couple of minutes!

```
$ python trpo_training_example.py
```

You can also train the models on mujoco. Check the comments in the training script for details.
