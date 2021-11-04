# Pointnet for 3d object classification

This is a nnabla implementation of the [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593).

## Requirements

### Python environment

Install `python >= 3.6`, then set up python dependencies from [requirements.txt](./requirements.txt):

```bash
pip install -r ./requirements.txt
```

Note that this requirements.txt dose not contain `nnabla-ext-cuda`.
If you have CUDA environment, we highly recommend to install `nnabla-ext-cuda` and use GPU devices.
See [NNabla CUDA extension package installation guide](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html).

## Dataset

To download the [ModelNet40 dataset](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), run the following script.

```bash
$ cd ./data
$ ./download_data.sh
```

This script should download and unpack the data automatically into the data/ folder.

## Train

```bash
python train.py --device_id <device id> --context "cudnn"
```

This will take about 2.5 hours using one GeForce RTX 3060, and you can find the result in pointnet_classification_result/seed_<your seed>/monitors

## Evaluation

```bash
python evaluate.py --device_id <device id> --context "cudnn" --snapshot_dir <your snapshot dir>
```

The evaluation results will be shown in your console.

## Result

| Ours | Reported Score |
|:---|:---|
| 89.6% | 89.2% |

# References

1. https://github.com/charlesq34/pointnet
2. https://github.com/charlesq34/pointnet2
3. https://github.com/yanx27/Pointnet_Pointnet2_pytorch
4. Qi, Charles R., et al. "Pointnet: Deep learning on point sets for 3d classification and segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
