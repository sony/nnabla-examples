# GAN Data Debiasing
This is a colab interactive demo of GAN Data Debiasing.
In this tutorial, we will build a fair dataset by using GAN to generate images that remove inappropriate correlations.

<p align="center">
<img src='images/gan_data_debiasing_workflow.png'>
</p>
<p align="center">
Figure: Overview of GAN Data Debiasing.
</p>

# Interactive demo

|Name| Notebook           | Task  | Example                       |
|:---------------------------------:|:-------------:|:-----:|:------------:|
 [GAN Data Debiasing](https://arxiv.org/abs/2012.01469) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/nnabla-examples/blob/master/interactive-demos/gan_data_debiasing.ipynb) | Mitigate the dataset bias with GAN|<a href="url"><img src="images/gan_data_debiasing_workflow.png" align="center" height="90" ></a>|



#### If you want to train the GAN-debiased model from the scratch please follow the below steps.
***
### Dataset
* Please download celebA dataset and decompress for training
* Run `python crop_images.py` file to crop the aligned & cropped 178×218 images to 128x128.

### Configuration
* In `args.py`, there are configurations for training the baseline & gan-debiased,
generating images, and validating trained models using a certain metric.

### Training
* Train the baseline model with the following command 
`python main.py --model_train baseline
--base-img-path [PATH to dataset] --attribute [target attribute] 
--protected_attribute [protected attribute]
--model-save-path [PATH to save the model parmeters]`

### Check fairness
* Check the Fairness of the model
`python compute_fairness.py --base-img-path [PATH to dataset] --attribute [target attribute] 
--protected_attribute [protected attribute]
--model-save-path [PATH to save the model parmeters]`

if the bias occurs(check the fairness metric) in the baseline attribute classifier model try to use the GAN-based data augmentation method.  

### Generation
* Train the progressive growing of GANs on the celebA dataset, please refer to our GitHub page.
* Run `python generate_images.py --generate orig --attribute [target attribute]
--record_latent_vector [PATH save latent vector] --fake_data_dir [PATH save generated image]
--generator_model [PATH Generator model]`

Train the protected attribute classifier (generate the labels for generated images)

### Generate hallucinate labels
* Generate hallucinate labels for the generated images with the trained baseline models for computing the complementary latent vectors, using the below command
`python hallucinate_labels.py --model-save-path [PATH saved model] --attribute [target attribute] --fake_data_dir [PATH save generated image]`

### Compute complementary latent vectors
* Estimate hyperplanes and compute complementary latent vectors `python eastimate_hyperplane.py --record_latent_vector [PATH save latent vector] --attribute [target attribute]
 --protected_attribute [protected attribute] --fake_data_dir [PATH generated hallcucinate labels]`
* Generate flipped image `python generate_images.py --generate flip --attribute [target attribute]
--record_latent_vector [PATH save latent vector] --fake_data_dir [PATH save generated image]
--generator_model [PATH Generator model]`

### Train
* Tain the attribute classifier with augmented data
`python main.py --model_train gan_debiased
--base-img-path [PATH to dataset] --fake_data_dir [PATH to generated images] --attribute [target attribute] 
--protected_attribute [protected attribute]
--model-save-path [PATH to save the model parmeters]`

Once the training got completed , check the model fairness.
***
# References
1. [Fair Attribute Classification through Latent Space De-biasing ](https://arxiv.org/abs/2012.01469)
2. [FairALM: Augmented Lagrangian Method for Training Fair Models with Little Regret](https://arxiv.org/abs/2004.01355)
3. [Directional Bias Amplification](https://arxiv.org/abs/2102.12594)
4. [Towards Threshold Invariant Fair Classification](https://arxiv.org/abs/2006.10667)


 
 



 
