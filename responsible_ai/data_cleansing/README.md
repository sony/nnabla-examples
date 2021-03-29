# Data Cleansing with Storage-efficient Approximation of Influence Functions

This is the code for data cleansing : Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions.

> [**Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions**]((https://arxiv.org/abs/2103.11807).),
> Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira
> *arXiv technical report ([arXiv 2103.11807]( https://arxiv.org/abs/2103.11807))*            

![](./imgs/datacleansing.png)

## Abstract 
Identifying the influence of training data for data cleansing can improve the accuracy of deep learning. An approach with stochastic gradient descent (SGD) called SGD-influence to calculate the influence scores was proposed, but, the calculation costs are expensive. It is necessary to temporally store the parameters of the model during training phase for inference phase to calculate influence sores. In close connection with the previous method, we propose a method to reduce cache files to store the parameters in training phase for calculating inference score. We only adopt the final parameters in last epoch for influence functions calculation. In our experiments on classification, the cache size of training using MNIST dataset with our approach is 1.236 MB. On the other hand, the previous method used cache size of 1.932 GB in last epoch. It means that cache size has been reduced to 1/1,563. We also observed the accuracy improvement by data cleansing with removal of negatively influential data using our approach as well as the previous method. Moreover, our *simple* and *general* proposed method to calculate influence scores is available on our auto ML tool without programing, Neural Network Console. The source code is also available.

## Citation
@misc{suzuki2021data,
      title={Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions}, 
      author={Kenji Suzuki and Yoshiyuki Kobayashi and Takuya Narihira},
      year={2021},
      eprint={2103.11807},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

## Installation
* Neural Network Libraries
Please follow the installation guide of the Neural Network Libraries on https://nnabla.org/download/

* In order to install related package, please use the requirements.rxt

```
pip install requirements.txt
```

## How to Run
All processes are executed with the following example command.

```
python train_infl_eval.py -o output -ds mnist -nt 1
```
 | option | input parameters | items |
 |--------|------------------|-------|
 | -o  | {user setting}   | output folder |
 | -ds |  mnist, cifar10, stl10 | datasets|
 | -nt  |  {number(int)}  | number of trials   |

The detailed infomation for each option is described in train_infl_eval.py
This consists of the processes below. 

### Training
In training phase (sgd_influence/train.py), the order of the instances used in each epoch and batch is recorded for influence calculation while saving parameters.
A conventional CNN model with six convolution layers is used in this example.

### SGD-Influence Calculation
In influence calculation phase (sgd_influence/infl.py), the effect of each instance in training is calculated with the saved information in training.
The parameter of every loop in each epoch is used for the original algorithm ([1] Hara *et al.*), while only the last weight is used to calculate approximate influence in modified method (Ours).

### Evaluation
Retraining is executed in almost the same way as in training phase to evaluate the result.
In evaluation phase (sgd_influence/eval.py), the worst n instances of influence are removed while training the model.
This is repeated assigning different values for n, which enables to evaluate the effect of particular instances on model accuracy.
Random skip is also evaluated as control.

### Visualize the Results
In visualization phase (sgd_influence/analyze.py), results of the retraining are summarized and visualized.
This helps you to understand the effect of influence in data cleansing.

## Reference
[1] Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data cleansing for models trained with SGD. In [*Advances in  Neural Information Processing Systems*, pages 4215-4224, 2019.](https://proceedings.neurips.cc/paper/2019/hash/5f14615696649541a025d3d0f8e0447f-Abstract.html)
