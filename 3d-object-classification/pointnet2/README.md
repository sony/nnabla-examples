# Pointnet for 3d object classification

This is a nnabla implementation of the [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413).

## Requirements

### Python environment

Install `python >= 3.7`, then set up python dependencies from [requirements.txt](./requirements.txt):

```bash
pip install -r ./requirements.txt
```

Note that this requirements.txt dose not contain `nnabla-ext-cuda`.
If you have CUDA environment, we highly recommend to install `nnabla-ext-cuda` and use GPU devices.
See [NNabla CUDA extension package installation guide](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

After installing python dependencies, install neu (Nnable Examples Utils).

```bash
# move to this directory.
$ cd /path/to/nnabla-examples

$ pip install -e .
```

See [NEU](https://github.com/nnabla/nnabla-examples/tree/master/utils) to get more details.

## Dataset

Follow the data download instruction in [author's github](https://github.com/charlesq34/pointnet2#shape-classification).
Create an empty `. /data` directory and move the uncompressed data folder to `./data/modelnet40_normal_resampled`.

## Train

### Training ssg model

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "ssg"
```

If you want to train with normal vector, 

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "ssg" --with_normal
```

This will take about 10 hours using one Tesla V100, and you can find the result in pointnet2_classification_result/seed_<your seed>/monitors

### Training msg model

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "msg"
```

If you want to train with normal vector, 

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "msg" --with_normal
```

This will take about 18 hours using one Tesla V100, and you can find the result in pointnet2_classification_result/seed_<your seed>/monitors


## Evaluation

```bash
python evaluate.py --device_id <device id> --context "cudnn" --checkpoint_json_path <your checkpoint json path>
```

The evaluation results will be shown in your console.

## Result

| Model | Accuracy |
|:---:|:---:|
| Ours (Pointnet2 msg with normal) | 92.4 % |
| Ours (Pointnet2 msg without normal) | 91.8% |
| Ours (Pointenet2 ssg with normal) | 92.2% |
| Ours (Pointenet2 ssg without normal) | 91.2% |
| Ours (Pointenet1) | 89.6% |
| Reported Score (with normal) | 91.7% |
| Reported Score | 90.7% |

## PreTrained Weight
You can download the h5 file of each pre-trained model from the following links.

When you use the weights that we provide, please refer to and follow the license agreement or the terms of use for [ModelNet40 dataset]((https://github.com/charlesq34/pointnet2#shape-classification)). 

| Model | Weight | Accuracy |
|:---:|:---:|:---:|
| Pointnet2 msg with normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-classification/pointnet2/params_msg_normal.h5) | 92.4 % |
| Pointnet2 msg without normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-classification/pointnet2/params_msg.h5) | 91.8% |
| Pointnet2 ssg with normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-classification/pointnet2/params_ssg_normal.h5) | 92.2% |
| Pointnet2 ssg without normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-classification/pointnet2/params_ssg.h5) | 91.2% |

# References

1. https://github.com/charlesq34/pointnet2
2. https://github.com/yanx27/Pointnet_Pointnet2_pytorch
3. Qi, Charles R., et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." Proceedings of the Advances in Neural Information Processing Systems (NeurIPS). 2017.
