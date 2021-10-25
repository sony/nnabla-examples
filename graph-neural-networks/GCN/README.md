# A nnabla implementaion of Graph Convolutional Network

This is a re-implementation of the Graph Convolutional Network(GCN) with nnabla.

Reference
> Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl) (ICLR 2017)

## Dataset and Task

Semi-supervised node classification with [Cora dataset](https://relational.fit.cvut.cz/dataset/CORA).
The Cora dataset is a citation network of sceientific papers.  It contains 2708 papers labeled with 7 subjects as nodes. There are 5429 edges, which represent the citation relationship of the paper. Each paper has a feature vector of 1433 dimensions based on word occurrences.

We use GCN to semi-supervised node classification of papers. That is, we assume that only some of the papers are labeled, and predict the subjects of the remaining papers.

## How to use

NetworkX, Pandas and scikit-learn are required for loading and preprocessing data. You will also need nnabla>=1.21.0 to use the F.dot function.
Run `train.py` to training. The dataset will be downloaded automatically.

```sh
$python train.py
```

## Notes

- This implementaion does not use public split but random split.
- Currently, sparse matrix operations are not supported. If nnabla supports sparse matrix computations in the future, it might be better to use them in GCN layers for more efficient calculation.
- In the paper, weight decay is only applied to the first layer, but in this implementation, it is applied to all layers.
