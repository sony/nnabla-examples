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
 
 
# Citation
This is based on Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2012.

# References
1. Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2012.
2. Celis, L. Elisa, et al. "Classification with fairness constraints: A meta-algorithm with provable guarantees." Proceedings of the conference on fairness, accountability, and transparency. 2019.
3. Zhang, Brian Hu, Blake Lemoine, and Margaret Mitchell. "Mitigating unwanted biases with adversarial learning." Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society. 2018.
4. https://archive.ics.uci.edu/ml/datasets/Adult
5. Zafar, Muhammad Bilal, et al. "Fairness constraints: Mechanisms for fair classification." Artificial Intelligence and Statistics. PMLR, 2017.
