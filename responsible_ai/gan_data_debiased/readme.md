## NNabla-GAN-Debiasing

### Dataset
* Pleas download celebA dataset and decompress for training
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

### References
1. [Fair Attribute Classification through Latent Space De-biasing ](https://arxiv.org/abs/2012.01469)
2. [FairALM: Augmented Lagrangian Method for Training Fair Models with Little Regret](https://arxiv.org/abs/2004.01355)
3. [Directional Bias Amplification](https://arxiv.org/abs/2102.12594)
4. [Towards Threshold Invariant Fair Classification](https://arxiv.org/abs/2006.10667)


 
 



 
