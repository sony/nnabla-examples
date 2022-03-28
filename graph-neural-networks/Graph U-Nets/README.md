# Graph U-Nets

This is nnabla implementation of Graph U-Nets

Reference
- Gao, Hongyang, and Shuiwang Ji. "Graph U-Nets." ICML. 2019.
- [PyTorch Implementation of Graph U-Nets](https://github.com/HongyangGao/Graph-U-Nets)

## Dataset and Task

We use [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA) for semi-supervised node classification.

## Usage

The dataset will be downloaded automatically.

```sh
$python train.py --epoch 50 --seed 915
```

## Requirements

- nnabla >= 1.21
- scikit-learn
- networkx
- pandas

## Notes

- This implementaion does not use public split but random split.
