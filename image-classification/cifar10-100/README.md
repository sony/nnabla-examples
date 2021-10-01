# Training Example using CIFAR-10 and CIFAR-100

---

## Overview

The examples listed below demonstrate several deep learning algorithms on
CIFAR-10 dataset and CIFAR-100 dataset, which are one of the most popular image
classification datasets in the machine learning community.
The datasets will be automatically downloaded when running any of the examples.

---


## Classification task (`classification.py`)

This example demonstrates the training of image classification on
CIFAR-10 dataset(CIFAR-100 dataset). The convolutional neural network takes
32x32 pixel color images as input and outputs the predictions of 10-way
classification(100-way classification).  By default, CIFAR-10 dataset is selected.

When you run the example by

```
python classification.py

```

you will see the training progress in the console (decreasing training and
validation error).

By default, the script will be executed with GPU.
If you prefer to run with CPU, try

```
python classification.py -c cpu
```

After the learning completes successfully, the results will be saved in
"tmp.monitor/". In this folder you will find model files "\*.h5" and result
files "\*.txt".

The classification example provides two choices of neural network architectures
to train, CIFAR10 dataset with 23-layers ResNet and CIFAR100 with 23-layers ResNet.
You can select it with the `-n` option. For more details see the source code and
the help produced by running with the `-h` option.

## Multi-Device Multi-Process Training

This example shows the naive `Data Parallel Distributed Training` for
the object recognition task using CIFAR-10 dataset and 23-layers ResNet with
[NCCL](https://github.com/NVIDIA/nccl) using `multi-process` in a single node.

NOTE that if you would like to run this example, please follow the build
instruction to enable the multi-device training and make sure to prepare
environment where you can use multiple GPUs.

When you run the script like the following,

```
mpirun -n 4 python classification.py --context "cudnn" -b 64

```

you can execute the training of 23-layers ResNet in the
`Data Parallel Distributed Training` manner with the batch size being 64
and 4 GPUs.

## Multi-Node Training

This example shows the naive `Data Parallel Distributed Training` for
the object recognition task using CIFAR-10 dataset and 23-layers ResNet with
[NCCL](https://github.com/NVIDIA/nccl) using `multi-process` over multiple nodes.

NOTE that if you would like to run this example, please follow the build
instruction to enable the multi-device training and make sure to prepare
environment where you can use multiple GPUs over multiple nodes.

When you run the script like the following,

```
mpirun --hostfile hostfile python classification.py --context "cudnn" -b 64

```

you can execute the training of 23-layers ResNet in the
`Data Parallel Distributed Training` manner with the batch size being 64
and N-GPUs and M-Nodes specified by the hostfile.

## Overlapping All-Reduce with Backward

This example shows the naive `Data Parallel Distributed Training` for
the object recognition task using CIFAR-10 dataset and 23-layers ResNet with
[NCCL](https://github.com/NVIDIA/nccl) using `all_reduce_callback` API.
All-reduce introduces communication overhead
because it requires inter-process and inter-node communications.
The `all_reduce_callback` API overlaps these all-reduce communications
with backward computation in order to decrease the execution time.

Warning: This API does not support shared parameters currently.
Thus, you cannot use this API when you train RNN.

NOTE that if you would like to run this example,
please follow the build instruction to enable the multi-device training
and make sure to prepare environment where you can use multiple GPUs
over multiple nodes.

When you run the script like the following,

```
mpirun --hostfile hostfile python classification.py --context "cudnn" -b 64 --with-all-reduce-callback

```

you can execute the training of 23-layers ResNet in the
`Data Parallel Distributed Training` manner with the batch size being 64.
And the all-reduce is pipelined with backward computation.
