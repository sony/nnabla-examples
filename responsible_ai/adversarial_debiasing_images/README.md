## Adversarial debiasing for Images
This is a colab interactive demo of application of adversarial debiasing in classification task on real-world CelebA dataset. GANs generally involve orchestration of two neural networks in which the two neural networks compete with each other to become more accurate  in their predictions. This fundamental idea is used in adversarial debiasing also:
 
 1. First model is a classifier which predicts target variable based on input features (training data).
 2. Second model is an adversary and it tries to predict sensitive attribute based on the predictions of the classifier model.
![adversarial_debiasing_workflow daigram](images/adversarial_debiasing_workflow_diagram.png)

### Interactive demo

|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 [Adversarial Debiasingn for images](https://arxiv.org/abs/1801.07593) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/adversarial_debiasing_images.ipynb) | Mitigate the model bias with Adversarial networks |<a href="url"><img src="images/adversarial_debiasing_workflow_diagram.png" align="center" height="90" ></a>|
 


### Dataset
* This example uses CelebA dataset for training & validation. 
Please download the dataset and store it in a directory named 'data'.
Authors kindly offer an easy way to prepare the dataset. Follow these instructions:
 ```sh
URL=https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=0
ZIP_FILE=./data/celeba.zip
mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE
```
* Once dataset is downloaded, you will find a text file named `list_attr_celeba.txt`. 
It must be like this: in the first line, attribute names are listed followed by image name and its attribute table (1 for true, -1 for false)

```
5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose ...
000001.jpg -1  1  1 -1 -1 -1 -1 -1 ...
000002.jpg -1 -1 -1  1 -1 -1 -1  1 ...
000003.jpg -1 -1 -1 -1 -1 -1  1 -1 ...
...
```
* Once the dataset ready, split the dataset into train , valid and test set with the following command:
`python split_dataset.py --out_dir ./data/train --split train`

### Configuration
* In `args.py`, you can find many configuration. You can choose attributes to use for training the Baseline model and 
Adversarial model. `--attribute` & `--protected_attribute` options to select the target & protected attributes of the model.
(by default, "Attractive" as target attribute & "Male" as a protected attribute are used). Available attributes are listed in `celeba_attributes_list.txt`
Note that you need to specify the dataset of training directory (with `--celeba_image_train_dir` option) and validation directory (with `--celeba_image_valid_dir` option).

### Baseline model Training
* Train the baseline model with the following command: 
`python main.py --model_train baseline --celeba_image_train_dir [PATH to train dataset] 
--celeba_image_valid_dir [PATH to valid dataset] --attr_path [Path to list_attr_celeba.txt] 
--attribute [target attribute] --protected_attribute [protected attribute] --model-save-path [PATH to save the model parmeters]`

### Adversarial model Training
* Train the Adversarial model with the following command: 
`python main.py --model_train adversarial --celeba_image_train_dir [PATH to train dataset] 
--celeba_image_valid_dir [PATH to valid dataset] --attr_path [Path to list_attr_celeba.txt] 
--attribute [target attribute] --protected_attribute [protected attribute] 
--training_ratio [training ratio b/w classifier & adversary model] --lamda [adversarial loss of each class] 
--model-save-path [PATH to save the model parmeters]`

After running, the experiment results and fairness metrics will be saved under `--model-save-path` directory
### References
1. Zhang, Brian Hu, Blake Lemoine, and Margaret Mitchell. "Mitigating unwanted biases with adversarial learning." Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society. 2018.
2. Beutel, Alex, et al. "Data decisions and theoretical implications when adversarially learning fair representations." arXiv preprint arXiv:1707.00075 (2017).
3. Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016).
4. Wang, Zeyu, et al. "Towards fairness in visual recognition: Effective strategies for bias mitigation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
5. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).
6. Liu, Ziwei, et al. "Large-scale celebfaces attributes (celeba) dataset." Retrieved August 15.2018 (2018): 11.
7. Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
8. Lokhande, Vishnu Suresh, et al. "Fairalm: Augmented lagrangian method for training fair models with little regret." European Conference on Computer Vision. Springer, Cham, 2020.
9. Wang, Angelina, and Olga Russakovsky. "Directional bias amplification." International Conference on Machine Learning. PMLR, 2021.
10. Chen, Mingliang, and Min Wu. "Towards threshold invariant fair classification." Conference on Uncertainty in Artificial Intelligence. PMLR, 2020.


 
 



 
