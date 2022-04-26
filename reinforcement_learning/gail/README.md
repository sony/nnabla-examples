# GAIL training example

This is an example code for training models with [GAIL algorithm](https://arxiv.org/abs/1606.03476) on nnabla.

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

For training GAIL, an expert dataset is required, so install [d4rl](https://github.com/rail-berkeley/d4rl).

```sh
$ pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

If you face install errors, try to clone [d4rl repositry](https://github.com/rail-berkeley/d4rl) and modify `dm_control @ git+git://github.com/deepmind/dm_control@master#egg=dm_control` to `dm_control @ git+https://github.com/deepmind/dm_control@4f1a9944bf74066b1ffe982632f20e6c687d45f1` in their setup.py. (See the d4rl [issue](https://github.com/rail-berkeley/d4rl/issues/141))
After modifying the setup.py, you can install d4rl.

```sh
# cd to <d4rl root dir>
$ pip install .
```

If you face any rendering issue related to opengl try below before starting the script

```sh
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## How to start the training

Just run the below command. You will see the [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) controlled by the reinforcement learning agent in a couple of minutes!

```
$ python gail_training_example.py
```

You can also train the models on mujoco. Check the comments in the training script for details.
