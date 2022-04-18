# Pointnet for 3d object part segmentation

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

Follow the data download instruction in [author's github](https://github.com/charlesq34/pointnet2#object-part-segmentation).
Create an empty `. /data` directory and move the uncompressed data folder to `./data/shapenetcore_partanno_segmentation_benchmark_v0_normal`.

## Train

### Training ssg model

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "ssg"
```

This will take about 24 hours using one Tesla V100, and you can find the result in pointnet2_part-segmentation_result/seed_<your seed>

### Training msg model

```bash
python train.py --device_id <device id> --context "cudnn" --model_type "msg"
```

This will take about 30 hours using one Tesla V100, and you can find the result in pointnet2_part-segmentation_result/seed_<your seed>


## Evaluation

```bash
python evaluate.py --device_id <device id> --context "cudnn" --checkpoint_json_path <your checkpoint json path>
```

The evaluation results will be shown in your console.

## Result

| Model | Average IoU (Shape) | Average IoU (Class) |
|:---:|:---:|:---:|
| Ours (Pointnet2 msg with normal) | 85.1 | 82.1 |
| Ours (Pointenet2 ssg with normal) | 85.0 | 81.6 | 
| Reported Score (with normal) | 85.1 | 81.9 |

## PreTrained Weight
You can download the h5 file of each pre-trained model from the following links.

When you use the weights that we provide, please refer to and follow the license agreement or the terms of use for [ShapeNet part dataset](https://github.com/charlesq34/pointnet2#object-part-segmentation). 

| Model | Weight | Average IoU (Shape) | Average IoU (Class) |
|:---:|:---:|:---:|:---:|
| Pointnet2 msg with normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-part-segmentaion/pointnet2/params_msg_normal.h5) | 85.1 | 82.1 |
| Pointnet2 ssg with normal | [link](https://nnabla.org/pretrained-models/nnabla-examples/3d-object-part-segmentaion/pointnet2/params_ssg_normal.h5) | 85.0 | 81.6 | 

# References

1. https://github.com/charlesq34/pointnet2
2. https://github.com/yanx27/Pointnet_Pointnet2_pytorch
3. Qi, Charles R., et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." Proceedings of the Advances in Neural Information Processing Systems (NeurIPS). 2017.
