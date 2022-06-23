# PRR: Prejudice Remover Regularizer

This is a colab interactive demo of Prejudice Remover Regularizer technique.

<p align="center">
<img src='images/Prejudice_Remover_Regularizer_workflow_diagram.png'>
</p>
<p align="center">
Figure: Overview of Prejudice Remover Regularizer
</p>

In this demo, we show how to reduce Indirect Prejudice, using the Prejudice Removal Technique.
In this PRR technique, two types of regularizers are used to enforce fair classification. 
First regularizer is a standard technique to avoid over-fitting. We use an L2 regularizer $||θ||^2$,
Second regularizer,$R(D,θ)$, is introduced to enforce fair classification. 
These two regularizers are added to the objective function for minimization.
<p align="center">
<img src='images/Prejudice_Remover_Regularizer_Equation.png'>
</p>
<p align="center">
Figure: Objective function of Prejudice Remover Regularizer
</p>


# Interactive demo

|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 [Prejudice Remover Regularizer](https://doi.org/10.1007/978-3-642-33486-3_3) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/prejudice_remover_regularizer.ipynb) | Mitigate the model bias with Prejudice Removal Technique |<a href="url"><img src="images/Prejudice_Remover_Regularizer_workflow_diagram.png" align="center" height="90" ></a>|
 
 

# Citation
This is based on Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2012.

# References
1. Toshihiro Kamishima, Shotaro Akaho, Hideki Asoh & Jun Sakuma. "Fairness-aware classifier with prejudice remover regularizer. " Joint European Conference on Machine Learning and Knowledge Discovery in Databases ECML PKDD 2012: Machine Learning and Knowledge Discovery in Databases pp 35–50
2. Celis, L. Elisa, et al. "Classification with fairness constraints: A meta-algorithm with provable guarantees." Proceedings of the conference on fairness, accountability, and transparency. 2019.
3. Zhang, Brian Hu, Blake Lemoine, and Margaret Mitchell. "Mitigating unwanted biases with adversarial learning." Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society. 2018.
4. https://archive.ics.uci.edu/ml/datasets/Adult
5. Zafar, Muhammad Bilal, et al. "Fairness constraints: Mechanisms for fair classification." Artificial Intelligence and Statistics. PMLR, 2017.
