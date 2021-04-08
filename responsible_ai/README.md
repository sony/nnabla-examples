# Responsible AI

![](./imgs/responsible_ai.png)

## Overview

These are examples of responsible AI that consists of eXplainable AI and fairness of machine learning. 


In order to utilize AI technology to enrich human's life style and contribute to the development of society, we will pursue fairness,  transparency, and accountability while actively engaging in dialogue with stakeholders. We will continue to contribute responsible AI in order to maintain the trustworthy of our products and services.

---

## Data cleansing with with Storage-efficient Approximation of Influence Functions [code](./data_cleansing/)

> [**Data Cleansing for Deep Neural Networks with Storage-efficient Approximation of Influence Functions**]((https://arxiv.org/abs/2103.11807).),
> Kenji Suzuki, Yoshiyuki Kobayashi, Takuya Narihira
> *arXiv technical report ([arXiv 2103.11807]( https://arxiv.org/abs/2103.11807))*            

![](./data_cleansing/imgs/datacleansing.png)

## TracIn [code](./tracin/)
> [**Estimating Training Data Influence by Tracing Gradient Descent**]((https://arxiv.org/abs/2002.08484).),
> Garima Pruthi, Frederick Liu, Mukund Sundararajan, Satyen Kale
> *arXiv technical report ([arXiv 2002.08484](https://arxiv.org/abs/2002.08484))* 

This is an implementation of Neural Network Libraries for estimating training data influence by tracing gradient descent called TracIn. The goal is to identify the influence of training data points. TracIn computes the influence of a training sample by tracing the training loss changes. This code is an example of mislabeled data detection with TracIn. 

<img src="./tracin/figure/self_influence_distribution.png" width="440px"><img src="./tracin/figure/score_curve.png" width="440px">
