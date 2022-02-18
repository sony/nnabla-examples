# HER training example

This is an example code for training models with [HER algorithm](https://arxiv.org/abs/1707.01495) on nnabla.

## Prerequisites

This example uses [nnabla-rl](https://github.com/sony/nnabla-rl).
Install nnabla-rl with:

```sh
$ pip install nnabla-rl
```

For the details of nnabla-rl, visit [nnabla-rl's project page](https://github.com/sony/nnabla-rl).

HER algorithm is used for Goal-conditioned Environment.
Now, this example supports only Robotics environment in MuJoco, so please install [MuJoCo](https://mujoco.org).

and also run:

```sh
$ pip install mujoco-py
```

If you face any rendering issue related to opengl try below before starting the script

```sh
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## How to start the training

Just run the below command. You will see the [FetchReach](https://gym.openai.com/envs/FetchReach-v1/) controlled by the reinforcement learning agent in a couple of minutes!

```
$ python her_training_example.py
```
