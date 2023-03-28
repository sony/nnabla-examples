# Fairness metrics
The following is an interactive demo in Colab that showcases various fairness metrics:

* Demographic parity
: Prediction of a machine learning model must be independent of protected class A. Demographic parity is defined by the following equation:

$$ P(\hat{Y} = 1 | A=0) = P(\hat{Y} = 1 | A=1) $$

* Disparate Impact
: This metric is calculated by dividing the propotion of positive outcomes for the unprivileged group by the propotion of positive outcomes for the privileged group. Disparate Impact is defined by the following equation:
$$\frac{P(\hat{Y}=1 \mid A=0)}{P(\hat{Y}=1 \mid A=1)}$$

* Equal opportunity
: True positive rate outcome must be independent of the protected class A . Equal opportunity  is defined by the following equation:

$$P(\hat{Y}=1 \mid A=0, Y=1) = P(\hat{Y}=1 \mid A=1,Y=1)$$

* Equalised odds
: Positive outcome of the model must be independent of the protected class  A. Equalised odds is defined by the following equation:

$$P(\hat{Y}=1 \mid A=0, Y=y) = P(\hat{Y}=1 \mid A=1,Y=y),\ \text{where}\ y \in {0,1}$$


<p align="center">
<img src='images/fairness_meric_workflow_daigram.png'>
</p>
<p align="center">
Figure: Overview of Fairness metrics.
</p>


# Interactive demo

|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 Metrics tutorial | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/fairness_metrics.ipynb) | Fairness definition|<a href="url"><img src="images/fairness_meric_workflow_daigram.png" align="center" height="90" ></a>|
 
# References
1. Hardt, M., Price, E., and Srebro, N. (2016). “[Equality of opportunity in supervised learning](https://arxiv.org/pdf/1906.09688.pdf)” in Advances in Neural Information Processing Systems, eds D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (Barcelona: Curran Associates, Inc.), 3315–3323.
2. Chouldechova, A. (2017). [Fair prediction with disparate impact: a study of bias in recidivism prediction instruments.](https://arxiv.org/abs/1610.07524)
3. Zafar, M. B., Valera, I., Gomez Rodriguez, M., and Gummadi, K. P. (2017a). [Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment.](https://arxiv.org/abs/1610.08452)
4. [The Impossibility Theorem of Machine Fairness -- A Causal Perspective.](https://arxiv.org/abs/2007.06024)
