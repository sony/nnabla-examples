# Utilities for NNabla examples

NEU, nnabla examples utils, provides a bunch of reusable components for writing training and inference scripts in nnabla-examples. Please note that this package is not organized very well and not stable so far, the API might change without any notice.

## How to use

This package is not provided for package managers such as pip and conda so far. You have to set a python path to this folder to use `neu` package.

We usually set a path to this folder at a utils package under each training example folder when you import it. See [Pix2PixHD/utils](../GANs/pix2pixHD/utils/__init__.py) for an example.

Alternatively, you can install neu through pip:
```
# move to this directory.
cd /path/to/nnabla-examples

# Recommend to install with editable option so that you can easily manipulate this package for your project.
pip install -e .
```

After installing neu through pip, you can easily import neu like below:
```python
from neu.misc import init_nnabla
comm = init_nnabla(ext_name="cudnn", device_id="0", type_config="float", random_pseed=True)
```

# Documentation 

NEU (nnabla-example utils) provides the following set of functionalities:

## callbacks 

These are the set of of callback helper functions that can be used in conjunction with standard neural network layers. At present, two function callbacks are supported.

Functions:
- [`neu.callbacks.spectral_norm_callback(dim)`](neu/callbacks.py?plain=1#L25)   
    Args:
    - `dim`: Dimension along which spectral normalization needs to be applied. For more details on spectral normalization, check [nnabla docs](https://nnabla.readthedocs.io/en/latest/python/api/parametric_function.html#nnabla.parametric_functions.spectral_norm)
- [`neu.callbacks.weight_standardization_callback(dim)`](neu/callbacks.py?plain=1#L39)   
Args:
    - `dim`: Dimension along which weight standardization needs to be applied. For more details on weight standardization, check [nnabla docs](https://nnabla.readthedocs.io/en/latest/python/api/function.html#nnabla.functions.weight_standardization)

Example: 

```python
from neu.callbacks import spectral_norm_callback
y = PF.convolution(x, 64, (3,3), apply_w=spectral_norm_callback(dim=0))
# This callback applies spectral normalization to the convolution weight parameter before the convolution operation
``` 

## checkpoint_util 

Functions to load and save network checkpoints - weights and solver state. 

Functions:
- [`neu.checkpoint_util.save_checkpoint(path, current_iter, solvers, n_keeps=-1)`](neu/checkpoint_util.py?plain=1#L27)   
    Args:
    - `path`: Path to the directory the checkpoint file is stored in.
    - `current_iter`: Current iteretion of the training loop.
    - `solvers`: A dictionary about solver's info, which is like;   
                     `solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}`   
                     The keys are used just for state's filenames, so can be anything.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.
    - `n_keeps`: Number of latest checkpoints to keep. If -1, all checkpoints are kept. Note that we assume save_checkpoint is called from a single line in your script. When you have to call this from multiple lines, n_keeps must be -1 (you have to disable n_keeps).
- [`neu.checkpoint_util.load_checkpoint(path, solvers)`](neu/checkpoint_util.py?plain=1#L152)   
Args:
    - `path`: Path to the checkpoint file.
    - `solvers`: A dictionary about solver's info, which is like;    
                     `solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}`    
                     The keys are used for retrieving proper info from the checkpoint.
                     so must be the same as the one used when saved.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

Example:

- Saving a checkpoint
```python
from neu.checkpoint_util import save_checkpoint

# Create computation graph with parameters.
pred = construct_pred_net(input_Variable, ...)
# Create solver and set parameters.
solver = S.Adam(learning_rate)
solver.set_parameters(nn.get_parameters())
# If you have another_solver like,
# another_solver = S.Sgd(learning_rate)
# another_solver.set_parameters(nn.get_parameters())
# Training loop.
for i in range(start_point, max_iter):
    pred.forward()
    pred.backward()
    solver.zero_grad()
    solver.update()
    save_checkpoint(path, i, solver)
    # If you have another_solver,
    # save_checkpoint(path, i,
    {"solver": solver, "another_solver": another})
```

Notes:
    It generates the checkpoint file (.json) which is like;
```python
checkpoint_1000 = {
    "":{
        "states_path": <path to the states file>
        "params_names":["conv1/conv/W", ...],
        "num_update":1000
        },
    "current_iter": 1000
    }
    If you have multiple solvers.
    checkpoint_1000 = {
            "generator":{
                "states_path": <path to the states file>,
                "params_names":["deconv1/conv/W", ...],
                "num_update":1000
                },
            "discriminator":{
                "states_path": <path to the states file>,
                "params_names":["conv1/conv/W", ...],
                "num_update":1000
                },
            "current_iter": 1000
            }
```

- Loading a checkpoint 
```python
# Create computation graph with parameters.
pred = construct_pred_net(input_Variable, ...)
# Create solver and set parameters.
solver = S.Adam(learning_rate)
solver.set_parameters(nn.get_parameters())
# AFTER setting parameters.
start_point = load_checkpoint(path, solver)
# Training loop.
```

Notes: This function requires the checkpoint file generated by `neu.checkpoint_util.save_checkpoint()`

## CLI

Command Line Interface to calculate Inception Score and FID (Fréchet Inception Distance).

Usage:
- For Inception score: 
```
python -m neu.metrics.gan_eval.inception_score <path to the directory or text file> 
```
Then you get the results like;
```
2020-04-15 04:46:13,219 [nnabla][INFO]: Initializing CPU extension...
2020-04-15 04:46:16,921 [nnabla][INFO]: Initializing CUDA extension...
2020-04-15 04:46:16,923 [nnabla][INFO]: Initializing cuDNN extension...
calculating all features of fake data...
loading images...
Finished extracting features. Calculating inception Score...
Image Sets: <image set used for calculation>
batch size: 16
split size: 1
Inception Score: 38.896
std: 0.000
```
- For FID: 

```
python -m neu.metrics.gan_eval.fid <path to the directory, text file, or .npz file of REAL data> \
                               <path to the directory, text file, or .npz file of FAKE data> \
                               --params-path <path to the pretrained weights, can be omitted>
```

Then you get the results like;
```
2020-04-10 10:08:47,272 [nnabla][INFO]: Initializing CPU extension...
2020-04-10 10:08:47,518 [nnabla][INFO]: Initializing CUDA extension...
2020-04-10 10:08:47,519 [nnabla][INFO]: Initializing cuDNN extension...
Computing statistics...
loading images...
100%|##################################################################################################################| 10000/10000 [07:36<00:00, 22.07it/s]
Image Set 1: <image set 1 used for calculation>
Image Set 2: <image set 2 used for calculation>
batch size: 16
Frechet Inception Distance: 55.671
```

For more details on this CLI usage, inception score and FID, check [here](https://github.com/nnabla/nnabla-examples/tree/master/utils/neu/metrics/gan_eval). 

## comm

[`CommunicatorWrapper`](neu/comm.py?plain=1#L28) for all multi-gpu execution. 

Usage: 

```python
from nnabla.ext_utils import get_extension_context
from neu.comm import CommunicatorWrapper
ctx = get_extension_context('cudnn')
comm = CommunicatorWrapper(ctx)

# Train loop for multi-gpu training
for i in range(n_iter):
    x, y = get_data()
    y_pred = network(x)
    loss = loss_fn(y, y_pred)

    loss.backward(clear_buffer=True)

    params = [p.grad for p in nn.get_parameters().values()]
    comm.all_reduce(params, division=False, inplace=True)

    solver.weight_decay(self.weight_decay)
    solver.update()
```

# gan_losses

Classes: 
- [`neu.gan_losses.BaseGanLoss(object)`](neu/gan_losses.py?plain=1#L111)       
    A base class of GAN loss functions.
    This class object offers a callable method which takes discriminator
    output variables from both real and fake, and returns a `GanLossContainer` which
    holds discreminator and generator loss values as computation graph
    variables.
    GAN loss functions for discriminator $L_D$ and generator $L_G$ can be written in the following generalized form
    $L_D = \mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[L^r_D \left(D(x_r)\right)\right] + \mathop{\mathbb{E}}_{x_f \sim G}\left[L^f_D \left( D(x_f) \right)\right]\\
    L_G = \mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[L^r_G \left( D(x_r) \right)\right] + \mathop{\mathbb{E}}_{x_f \sim G} \left[L^f_G \left( D(x_f) \right)\right]$
    where :math:$L^r_D$ and :math: $L^r_G$ are loss functions of real data :math:`x_r` sampled from dataset :math:$P_{\rm data}$ for discriminator and generator respectively, and $L^f_D$ and :math:$L^f_G$ are for fake data :math:$x_f$ generated from the current generator $G$. Those functions take discriminator outputs $D(\cdot)$ as inputs.
    In most of GAN variants (with some exceptions), those loss functions can be defined as the following symmetric form
    $
        L^r_D(d) = l^+(d)\\
        L^f_D(d) = l^-(d)\\
        L^r_G(d) = l^-(d)\\
        L^f_G(d) = l^+(d)\\
    $
    where :math:$l^+$ is a loss function which encourages the discriminator
    output to be high while $`l^-$ to be low.
    Different $l^+$ and $l^-$ give different types of GAN losses.
    For example, the Least Square GAN (LSGAN) is derived from
    $
        l^+(d) = (d - 1)^2 \\
        l^-(d) = (d - 0)^2.
    $
    Any derived class must implement both $l^+(d)$ $l^-(d)$ as `def _loss_plus(self, d)` and `def _loss_minus(self, d)` for $l^+(d)$ and $l^-(d)$ respectively, then the overall loss function is defined by the symmetric form explained above.
    Note:
        The loss term for real data of generator loss $\mathop{\mathbb{E}}_{x_r \sim P_{\rm data}} \left[l^- \left( D(x_r) \right)\right]$ is usually omitted at computation graph because generator model is not dependent of that term. If you want to obtain it as outputs, call a method `use_generator_loss_real(True)` to enable it.

    `__call__(self, d_r, d_f)`            
    Get GAN losses given disriminator outputs of both real and fake.      
    Args:            
        `d_r (~nnabla.Variable)`: Discriminator output of real data, `D(x_real)`          
        `d_f (~nnabla.Variable)`: Discriminator output of fake data, `D(x_fake)`      
    Note:           
        The discriminator scores which are fed into this must be
        pre-activation values, that is `[-inf, inf]`.        
    Returns:              
    GanLossContainer

- [`neu.gan_losses.GanLossContainer(object)`](neu/gan_losses.py?plain=1#L29)

    A container class of GAN outputs from `GanLoss` classes.    
    Attributes:       
    - `loss_dr (nnabla.Variable)`: Discriminator loss for real data.
    - `loss_df (nnabla.Variable)`: Discriminator loss for fake data.
    - `loss_gr (nnabla.Variable)`: Generator loss for real data. This is usually set as `None`
            because generator parameters are independent of real data in standard GAN losses. Exceptions for examples are relativistic GAN losses.
    - `loss_gf (nnabla.Variable)`: Generator loss for fake data.          
    Property:
    - `discriminator_loss (nnabla.Variable)`: Returns `loss_dr + loss_df`. It is devided by 2 if loss_gr is
            not `None`.
        generator_loss (~nnabla.Variable): Returns `loss_gr + loss_gf`
    This class implements `+` (add) operators (radd as well) for the
    following operands.          
    * `GanLossContainer + GanLossContainer`: Losses from containers are
      added up for each of loss_dr, loss_df, loss_gr, and loss_gf
      and a new GanLossContainer object is returned.
    * `GanLossContainer + None`: `None` is ignored and the given
      `GanLossContainer` is returned.

- [`neu.gan_losses.GanLoss(BaseGanLoss)`](neu/gan_losses.py?plain=1#L205)        
Standard GAN loss defined as
    $
        l^+(d) = \ln \sigma (d) \\
        l^-(d) = \ln \left(1 - \sigma (d )\right)
    $
    in a generalized form described in `BaseGanLoss` documentation. Here, $\sigma$ is Sigmoid function $\sigma(d) = \frac{1}{1 + e^{-d}}$ to interpret input as probability.       

    References:       
    Ian J. Goodfellow et. al. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [`neu.gan_losses.LsGanLoss(BaseGanLoss)`](neu/gan_losses.py?plain=1#L230)
    Least Square GAN loss defined as
    $
        l^+(d) = (d - 1)^2 \\
        l^-(d) = (d - 0)^2
    $
    in a generalized form described in `BaseGanLoss` documentation.      
    References:             
        Xudong Mao, et al. [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
- [`neu.gan_losses.WassersteinGanLoss(BaseGanLoss)`](neu/gan_losses.py?plain=1#L257)            
    Wasserstein GAN loss defined as
    $
        l^+(d) = d \\
        l^-(d) = -d
    $
    in a generalized form described in `BaseGanLoss` documentation.       
    References:
        Martin Arjovsky, et al. [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

- [`neu.gan_losses.GeometricGanLoss(object)`](neu/gan_losses.py?plain=1#L282)            
    Geometric GAN loss with SVM hyperplane defined as
    $
        L^r_D = \max(0, 1 - d) \\
        L^f_D = \max(0, 1 + d) \\
        L^r_G = d \\
        L^f_G = -d
    $
    in a generalized form described in `BaseGanLoss` documentation, but note that it's not symmetric.             
    References:
        Jae Hyun Lim, et al.
        [Geometric GAN](https://arxiv.org/abs/1705.02894)    

    Note:
        This is somestimes called as Hinge GAN.

- [`neu.gan_losses.HingeGanLoss(object)`](neu/gan_losses.py?plain=1#L318)        
 An alias of `GeometricGanLoss`
- [`neu.gan_losses.SymmetricHingeGanLoss(object)`](neu/gan_losses.py?plain=1#L326)      
    Symmetric hinge GAN loss defined as
    $
        l^+ = \max(0, 1 - d) \\
        l^- = \max(0, 1 + d)
    $
    in a generalized form described in `BaseGanLoss` documentation.
    The loss function of RaHingeGAN can be created by passing this to `RelativisticAverageGanLoss`.            
    References:           
        `Alexia Jolicoeur-Martineau [Relativistic Average GAN](https://arxiv.org/abs/1807.00734)
    
- [`neu.gan_losses.RelativisticAverageGanLoss(object)`](neu/gan_losses.py?plain=1#L352)      
    Relativistic Average GAN (RaGAN) Loss        
    Args:        
        gan_loss (BaseGanLoss): A GAN loss.           
        average (bool): If False, averaging is omitted. Hence, it becomes Relativistic GAN.        
    References:
        Alexia Jolicoeur-Martineau. [Relativistic Average GAN](https://arxiv.org/pdf/1807.00734.pdf)

## html_creator

Classes:       
- [`neu.html_creator.HtmlCreator(object)`](neu/html_creator.py?plain=1#L32)      
Utilizes [dominate](https://github.com/Knio/dominate) to create an html page to display all intermediate image generation for an image generation model.
Args:     
- `root (str)`: Path containing the folder `imgs` which contains intermediate image generations.
- `page_title (str)`: `default="no title"`
- `redirect_interval (int)`: `default=0`


## initializer

Weight Initialization functions. At present, two function initialization are supported.

Functions:
- [`neu.initializer.w_init(x, out_dims, gain=0.02, type="xavier")`](neu/initializer.py?plain=1#L21)  
Xavier initilization in nnabla.  
- [`neu.initializer.pytorch_conv_init(inmaps, kernel)`](neu/initializer.py?plain=1#L28)      
For convolution weight initialization in the exact same way as done in pytorch   

Example: 
```python
import nnabla as nn, nnabla.parametric_functions as PF
from neu.initializer import pytorch_conv_init

x = nn.Variable((32,3,28,28))
kernel = (4,4)
outmaps = 64 
y = PF.convolution(x, outmaps=outmaps, kernel=kernel, w_init=pytorch_conv_init(x.shape[1], (4,4))
```

## layers

Functions:

- [`neu.layers._normalize(x, norm_type, channel_axis=1)`](neu/layers.py?plain=1#L34)   
Args: 
    - `x (nn.Variable)`: Input variable for normlization
    - `norm_type (str)` : A type of normalization. `["in", "bn"]` are supported now, which correspond to Instance and Batch normalization respectively.
    - `channel_axis (int)` : Channel axis for normalization

- [`neu.layers.spade(x, m, hidden_dim=128, kernel=(3, 3), norm_type="in")`](neu/layers.py?plain=1#L43)       
Spatially-Adaptive Normalization proposed in Semantic Image Synthesis with Spatially-Adaptive Normalization (https://arxiv.org/pdf/1903.07291.pdf)     
Args:     
    - `x (nn.Variable)`: Input variable for spade layer.
    - `m (nn.Variable)`:
            Spatial condition variable like object_id mask segmentation.
            This is for generating adaptive scale(gamma) and adaptive bias(beta) applied after normalization.
    - `hidden_dim (int)`: Hidden dims for first convolution applied to m.
    - `kernel (list of int)`: Kernel shapes for convolutions.
    - `norm_type (str)` : A type of normalization. `["in", "bn"]` are supported now.

- [`neu.layers.rescale_values(x, input_min=-1, input_max=1, output_min=0, output_max=255)`](neu/layers.py?plain=1#L83)  
Rescale the range of values of `x` from [input_min, input_max] -> [output_min, output_max].

## learning_rate_scheduler

Classes: 
- [`neu.learning_rate_scheduler.BaseLearningRateSchedule(object)`](neu/learning_rate_scheduler.py?plain=1#L61)    
    Base class of Learning rate scheduler.    
    This gives a current learning rate according to a scheduling logic
    implemented as a method `_get_lr` in a derived class. It internally
    holds the current epoch and the current iteration to calculate a
    scheduled learning rate. You can get the current learning rate by
    calling `get_lr`. You have to set the current epoch which will be
    used in `_get_lr` by manually calling  `set_epoch(self, epoch)`
    while it updates the current iteration when you call
    `get_lr_and_update`.

    - `set_epoch(self, epoch)`    
    Set current epoch number.
    - `get_lr_and_update(self)`    
        Get current learning rate and update itereation count. The iteration count is calculated by how many times this method is called.

    - `get_lr(self)`      
    Get current learning rate according to the schedule.
    - `_get_lr(self, current_epoch, current_iter)`     
        Get learning rate by current iteration. A derived class must override this method.
        Args:
        - current_epoch(int): Epoch count.
        - current_iter(int):
                Current iteration count from the beginning of training.

- [`neu.learning_rate_scheduler.EpochStepLearningRateScheduler(BaseLearningRateSchedule)`](neu/learning_rate_scheduler.py?plain=1#L159)    
    Learning rate scheduler with step decay.    
    Args:    
    - `base_lr (float)`: Base learning rate
    - `decay_at (list of ints)`: It decays the lr by a factor of `decay_rate` at given iteration/epochs specified via `decay_at` (`default=[30,60,80]`).
    - `decay_rate (float)`: See above (`default=0.1`).
    - `warmup_epochs (int)`: It performs warmup during this period (`default=5`).
    - `legacy_warmup (bool)`: 
        We add 1 in the denominator to be consistent with previous
            implementation (`default=False`).

- [`neu.learning_rate_scheduler.EpochCosineLearningRateScheduler(BaseLearningRateSchedule)`](neu/learning_rate_scheduler.py?plain=1#L195)    
    Cosine Annealing Decay with warmup.
    The learning rate gradually increases linearly towards `base_lr` during
    `warmup_epochs`, then gradually decreases with cosine decay towards 0 for
    `epochs - warmup_epochs`.

    
    Args:        
    - `base_lr (float)`: Base learning rate
    - `epochs (int)`: See description above.
    - `warmup_epochs (int)`: It performs warmup during this period (`default=5`).

- [`neu.learning_rate_scheduler.PolynomialLearningRateScheduler(BaseLearningRateSchedule)`](neu/learning_rate_scheduler.py?plain=1#L233)      
Polynomial decay schedule. The learning rate gradually increases linearly towards `base_lr` during `warmup_epochs`, then gradually decreases with polynomial (degree specified by `power`) decay towards 0 for
    `epochs - warmup_epochs`.
    - `base_lr (float)`: Base learning rate
    - `epochs (int)`: See description above.
    - `warmup_epochs (int)`: See description above (`default=5`).
    - `power (float)`: : See description above (`default=0.1`).

Example: 

```python
from neu.learning_rate_scheduler import EpochStepLearningRateScheduler

def train(...):
    ...
    solver = Momentum()
    lr_sched = EpochStepLearningRateScheduler(1e-1)
    for epoch in range(max_epoch):
        lr_sched.set_epoch(epoch)
        for it in range(max_iter_in_epoch):
            lr = lr_sched.get_lr_and_update()
            solver.set_learning_rate(lr)
            ...
```

## lms

Functions:
- [`neu.lms.lms_scheduler(ctx, use_lms, gpu_memory_size=8 << 30, window_length=12 << 30)`](neu/lms.py?plain=1#L29)  
returns [nnabla.lms.SwapInOutScheduler](https://nnabla.readthedocs.io/en/latest/python/api/lms.html?highlight=lms#nnabla.lms.SwapInOutScheduler) for large model training (whose size is larger than alloted GPU memory). If `use_lms==False` or cuda/cudnn backend is not used, then a dummy scheduler (which does not do anything) is returned. 

Args:   
- `ctx`: nnabla extension context. Returned by `nnabla.ext_utils.get_extension_context()`.
- `use_lms (bool)`
- `gpu_memory_size (float)`
- `window_length (float)`

## losses:

Functions:    

Classification/Regression losses
- [`neu.losses.sigmoid_ce(logits, value, mask=None, eps=1e-5)`](neu/losses.py?plain=1#L25)       
sigmoid cross entropy and reduce_mean 
- [`neu.losses.softmax_ce(logits, targets, mask=None, eps=1e-5)`](neu/losses.py?plain=1#L40)     
softmax cross entropy and reduce_mean
- [`neu.losses.mae(x, y, mask=None, eps=1e-5)`](neu/losses.py?plain=1#L57)                 
l1 distance and reduce mean
- [`neu.losses.mse(x, y, mask=None, eps=1e-5)`](neu/losses.py?plain=1#L71)   
l2 distance and reduce mean                     

Likelyhood based losses
- [`neu.losses.kl_snd(mu, logvar)`](neu/losses.py?plain=1#L89)    
kl divergence with standard normal distribution
- [`neu.losses.kl_normal(mean1, logvar1, mean2, logvar2)`](neu/losses.py?plain=1#L99)       
kl divergence between two gaussians
- [`neu.losses.approx_standard_normal_cdf(x)`](neu/losses.py?plain=1#L105)           
A fast approximation of the cumulative distibution function of the standard normal
- [`neu.losses.gaussian_log_likelihood(x, mean, logstd, orig_max_val=255)`](neu/losses.py?plain=1#L113)           
    Compute the log-likelihood of a Gaussian distribution for given data `x`.     
    Args:        
    - `x (nn.Variable)`: Target data. It is assumed that the values are ranged [-1, 1],
                         which are originally [0, orig_max_val].         
    - `means (nn.Variable)`: Gaussian mean. Must be the same shape as x.            
    - `logstd (nn.Variable)`: Gaussian log standard deviation. Must be the same shape as x.       
    - `orig_max_val (int)`: The maximum value that x originally has before being rescaled.       
    
    Return:       
        A log probabilies of `x` in nats.

GAN based losses
- [`neu.losses.ls_gan_loss(r_out, f_out)`](neu/losses.py?plain=1#L165)     
Least Squares loss for GAN. Using common notations: `r_out` refers to $D\left(x, y\right)$ and `f_out` refers to $D\left(G\left(z\right), y\right)$. For details, check [here](https://paperswithcode.com/method/gan-least-squares-loss). 
- [`neu.losses.hinge_gan_loss(r_out, f_out)`](neu/losses.py?plain=1#L180)   
    Hinge loss for GAN. Using common notations: `r_out` refers to $D\left(x, y\right)$ and `f_out` refers to $D\left(G\left(z\right), y\right)$. For details, check [here](https://paperswithcode.com/method/gan-hinge-loss). 
- [`neu.losses.get_gan_loss(type)`](neu/losses.py?plain=1#L191)  
At present, supported types are `"ls"` and `"hinge"`
- [`neu.losses.vgg16_perceptual_loss(fake, real)`](neu/losses.py?plain=1#L205) 
VGG perceptual loss based on VGG-16 network. Assuming the values in fake and real are in [0, 255]. Features are obtained from all ReLU activations of the first convolution after each downsampling (maxpooling) layer (including the first convolution applied to an image).

## misc
Functions:   
- [`neu.misc.set_random_pseed(comm)`](neu/misc.py?plain=1#L24)    
Sets parameter seed (this does not set the seed for nnabla functions). 
- [`neu.misc.init_nnabla(conf=None, ext_name=None, device_id=None, type_config=None, random_pseed=True)`](neu/misc.py?plain=1#L35)        
Essentially, this function is a wrapper to set up context (using `ext_name`, `device_id` and `type_config`), parameter seed and in case of multi-gpu execution, disable outputs from loggers except `rank==0`.
- [`neu.misc.makedirs(dirpath)`](neu/misc.py?plain=1#L176)     
Creates a directory specified by `dirpath`. It does not do anything if `dirpath` already exists as a directory and throws an error if `dirpath` exists as a file. 
- [`neu.misc.get_current_time()`](neu/misc.py?plain=1#L187)      
returns `datetime.datetime.now().strftime('%m%d_%H%M%S')`
- [`neu.misc.get_iterations_per_epoch(dataset_size, batch_size, round="ceil")`](neu/misc.py?plain=1#L193)      
    Returns number of iterations to see whole images in dataset (= 1 epoch).     
    Args:     
    - dataset_size (int): A number of images in dataset
    - batch_size (int): A number of batch_size.
    - round (str): Round method. One of ["ceil", "floor"].

Classes:
- [`neu.misc.DictInterfaceFactory(object)`](neu/misc.py?plain=1#L110)      
Creating a single dict interface of any function or class. See example for usage.
- [`neu.misc.AttrDict(dict)`](neu/misc.py?plain=1#L70)               
Very convenient wrapper for a dictionary that sets a key of a dictionary as an attribute of that dictionary. See example for usage.

Example:
```python
# Example of neu.misc.DictInterfaceFactory
from neu.misc import DictInterfaceFactory

# Define a function.
def foo(a, b=1, c=None):
    for k, v in locals():
        print(k, v)

# Register the function to the factory.
dictif = DictInterfaceFactory()
dictif.register(foo)

# You can call the registered function by name and a dict representing the arguments.
cfg = dict(a=1, c='hello')
dictif.call('foo', cfg)

# The following will fail because the `foo` function requires `a`.
#     cfg = dict(c='hello')
#     dictif.call('foo', cfg)

# Any argument not required will be just ignored.
cfg = dict(a=1, aaa=0)
dictif.call('foo', cfg)

# You can also use it for class initializer (we use it as a class decorator).
@dictif.register
class Bar:
    def __init__(self, a, b, c=None):
        for k, v in locals():
            print(k, v)
bar = dictif.call('Bar', dict(a=0, b=0))
```

```python
# Example of neu.misc.AttrDict
from neu.misc import AttrDict
dict_ = {'a': 10, 'b': (5,6), 'c': [2,3,5]}
attrdict_ = AttrDict(dict_)

print(attrdict_.a) # sample as dict_['a'] and attrdict_['a']
print(attrdict_.b) # sample as dict_['b'] and attrdict_['b']
print(attrdict_.c) # sample as dict_['c'] and attrdict_['c']

```

## post_processing

Functions:    
- [`neu.post_processing.uint82bin(n, count=8)`](neu/post_processing.py?plain=1#L27)    
returns the binary of integer `n`, count refers to amount of bits
- [`neu.post_processing.labelcolormap(N)`](neu/post_processing.py?plain=1#L32)
Generates a colormap for generating image from a labelmap (such as segmentation output). `N` refers to number of labels in the labelmap. This function is used internally by `neu.post_processing.Colorize`.   
Classes:
- [`neu.post_processing.Colorize`](neu/post_processing.py?plain=1#L63)         
Parameters:               
    - `n`: Number of labels in a labelmap (`default=35`)            

    `__call__(self, label_image, channel_first=False)`:        
Generates image using label_image where each image is mapped to a color by using a `colormap` generated by `neu.post_processing.labelcolormap()`.


## reporter

Functions:
- [`neu.reporter.get_value(val, dtype=float, reduction=True)`](neu/reporter.py?plain=1#L32)    
get float value from `nn.NdArray / nn.Variable / np.ndarray / float`
- [`neu.reporter.get_tiled_image(img, channel_last=False)`](neu/reporter.py?plain=1#L63)     
Generates a tiled image given batched images    
Args:    
    -  `img (np.ndarray)`: Input batched images. The shape should be (B, C, H, W) or (B, H, W, C) depending on `channel_last`. dtype must be np.uint8.        
    - `channel_last (bool)`: If True, the last axis (=3) will be handled as channel.
- [`neu.reporter.save_tiled_image(img, path, channel_last=False)`](neu/reporter.py?plain=1#L82)     
Save given batched images as tiled image. The first axis will be handled as batch. This function internally uses `neu.reporter.get_tiled_image()` to generate tiled image.   
Args:    
    - `img (np.ndarray)`: Images to save. The shape should be (B, C, H, W) or (B, H, W, C) depending on `channel_last`. dtype must be np.uint8.        
    - `path (str)`:  Path to save.     
    - `channel_last (bool)`: If True, the last axis (=3) will be handled as channel.

Classes: 
- [`neu.reporter.MonitorWrapper`](neu/reporter.py?plain=1#L106)     
Wrapper class that combines `nnabla.monitor.MonitorSeries` and `nnabla.monitor.MonitorTimeElapsed`. This class is internally used by other classes in `neu.reporter`. It can be used independently as well. For more details on `nnabla.monitor`, check [docs](https://nnabla.readthedocs.io/en/latest/python/api/monitor.html). Check example for usage.   
Parameters:     
    - `save_path`: Path to save the logged values
    - `interval`: Interval of flush the outputs  
    - `save_time`: Set `True` to log the elapsed time (`default=True`)    
    - `silent`: Set `False` to print the logged values to output screen (`default=False`) 
       
   `set_series(self, name)`      
Set a new series name. `name` is the name of the monitor which is used in the log. This function is internally called by `__call__` function.  
`__call__(self, name, series_val, epoch)`
    - `name (str)`: Name with which to track a particular value
    - `series_val (nn.NdArray/nn.Variable/np.ndarray/float)`: Value of the variable/parameter
    - `epoch (int)`: Current epoch at which the `series_val` has been calculated.

- [`neu.reporter.Reporter`](neu/reporter.py?plain=1#L152)    
For simultaneous tracking of multiple loss values. This can be used to log some value in multi-gpu execution as well. In multi-gpu mode, the logged value is the average of the individual values in different GPUs.

This class can also be used to save images. The images are stored individually and are also used to create a html image using the library [dominate](https://github.com/Knio/dominate). Saving images is particularly useful in case of generative models.

Parameters:     
    - `comm`: `nnabla.communicators.Communicator` object. For details, check [docs](https://nnabla.readthedocs.io/en/latest/python/api/communicator.html).
    - `losses`: Losses dictionary/list in the form of `{"loss_name": loss_Variable, ...} or list of tuple(key, value)`
    - `save_path`: Path to save the logged values
    - `interval`: Interval of flush the outputs  
    - `save_time`: Set `True` to log the elapsed time (`default=True`)    
    - `silent`: Set `False` to print the logged values to output screen (`default=False`)     
`__call__(self)`:
To log loss values
`step(self, iteration, images=None)`:   
To save images.

Check example for further details.

- [`neu.reporter.AverageLogger`](neu/reporter.py?plain=1#L305)
To track average value of a variable. This class is used internally by `neu.reporter.KVReporter`.

- [`neu.reporter.KVReporter`](neu/reporter.py?plain=1#L330)
Extends `neu.reporter.MonitorWrapper` by allowing for simultaneous tracking of multiple values, value tracking in multi-gpu execution and tracking running average of values. 
    - `comm`: `nnabla.communicators.Communicator` object. For details, check [docs](https://nnabla.readthedocs.io/en/latest/python/api/communicator.html) (`default=None`).
    - `losses`: Losses dictionary/list in the form of `{"loss_name": loss_Variable, ...} or list of tuple(key, value)`
    - `save_path`: Path to save the logged values (`default=None`)
    - `interval`: Interval of flush the outputs  
    - `skip_kv_to_monitor`: Set `True` to log the elapsed time (`default=True`)    
    - `monitor_silent`: Set `False` to print the logged values to output screen (`default=True`)  

Check example for other functions and usage.

Example:

```python
from neu.reporter import Reporter, KVReporter
from neu.misc import init_nnabla
from tqdm import trange

# MonitorWrapper Usage

monitor = MonitorWrapper("path/to/save", interval=10, save_time=True)
# Currently supports three types below
v1: nn.Variable()
v2: nn.NdArray()
v3: np.ndarray() or float
vars = {"name1": v1, "name2": v2, "name3", v3}
for epoch in range(max_epoch):
    for k, v in vars.items():
        monitor(k, v, epoch)

# Reporter usage
comm = init_nnabla(ext_name="cudnn", device_id=device_id, type_config="float")
losses = {"gen_loss": gen_loss, "disc_loss": disc_loss}
reporter = Reporter(comm, losses, "path/to/save")
for epoch in range(num_epochs):
    progress_iterator = trange(self.data_iter._size // self.batch_size, 
                    desc=f"[epoch {epoch}]", disable=self.comm.rank > 0) 
    reporter.start(progress_iterator)

    for i in progress_iterator:
        // Do forward backward pass

        reporter() # Log generator and discriminator loss
        tracked_images = {"real_image": x, "fake_image": gen_output}
        reporter.step(epoch, tracked_images)

# KVReporter Usage

comm = init_nnabla(ext_name="cudnn", device_id=device_id, type_config="float")
reporter = KVReporter(comm, save_path="path/to/logdir", show_interval=20, force_persistent=True)
x = nn.Variable(...)
h1 = F.affine(x, c1, name="aff1")
h2 = F.affine(h1, c2, name="aff2")
loss = F.mean(F.squared_error(h2, t))
solver = S.Adam()
solver.set_parameters(nn.get_parameters())
# show loss
for i in range(max_iter):
    loss.forward()
    loss.backward()
    solver.zero_grad()
    solver.update()
    
    # KVReporter can handle nn.Variable and nn.NdArray as well as np.ndarry.
    # Using kv_mean(name, val), you can calculate moving average.
    reporter.kv_mean("loss", loss)
    # If you don't need to take moving average for the value, use kv(name, val) instead.
    reporter.kv("iterations", i)
    # get all traced values.
    # If sync=True, all values will be synced across devices via `comm`.
    # If reset=True, KVReporter resets the current average for all values as zero.
    reporter.dump(file=sys.stdout, sync=True, reset=True)
    # save values through nnabla.Monitor
    reporter.flush_monitor(i)

```

## save_args

Function:
- [`neu.save_args.save_args(args, config=None)`](neu/save_args.py?plain=1#L20)  
Saves `ArgumentParser.parse_args()` in `ArgumentParser.parse_args().monitor_path/Arguments.txt` and config file (if provided) in `ArgumentParser.parse_args().<>.yaml`. For this function to work, `ArgumentParser.parse_args()` must have `monitor_path` as one of the arguments and `<>.yaml` argument to store the config file in `.yaml` format.

## Save nnp 

Function:
- [`neu.save_nnp.save_nnp(input, output, batchsize)`](neu/save_nnp.py?plain=1#L17)    
returns `contents` dictionary that can be saved in nnp format. For details, check [nnabla documentation](https://nnabla.readthedocs.io/en/latest/python/api/utils/save_load.html#nnabla.utils.save.save).    

## variable_utils

Function:
- [`neu.variable_utils.get_params_startswith(str)`](neu/variable_utils.py?plain=1#L19)   
returns a subset of `nn.get_parameters()` such that all the variable scope name within that subset starts with `str`
- [`neu.variable_utils.set_persistent_all(*variables)`](neu/variable_utils.py?plain=1#L23)
sets the `persistent` flag of all the passed `variables` as `True`
- [`neu.variable_utils.set_need_grad_all(*variables, need_grad)`](neu/variable_utils.py?plain=1#L34)  
sets `need_grad` of all the passed `variables` to a specified boolean `need_grad` value
- [`neu.variable_utils.get_unlinked_all(*variables)`](neu/variable_utils.py?plain=1#L47)  
returns a list of all the passed `variables` after unlinking them from the computation graph
- [`neu.variable_utils.zero_grads_all(*variables)`](neu/variable_utils.py?plain=1#L62)           
Sets the gradient of all the variables as 0.     
- [`neu.variable_utils.fill_all(*variables, value=0)`](neu/variable_utils.py?plain=1#L73)  
sets the value of all variable elements to a specified `value`

Example:
```python
from neu.variable_utils import *

x = nn.Variable(shape=...)
y = nn.Variable(shape=...)
z = nn.Variable(shape=...)

with nn.parameter_scope('scope_1'): # registers 'scope_1/conv/W' and 'scope_1/conv/bias'
    fx = PF.convolution(x, ...)
with nn.parameter_scope('scope_1_1'): # registers 'scope_1_1/conv/W' and 'scope_1_1/conv/bias'
    fy = PF.convolution(y, ...)
with nn.parameter_scope('scope_2'): # registers 'scope_2/conv/W' and 'scope_2/conv/bias'
    fz = PF.convolution(z, ...)

fill_all(x,y,z, value=10) # same as for v in [x,y,z]: v.data.fill(10)
set_need_grad_all(x,y,z,True) # same as for v in [x,y,z]: v.need_grad=True

xy_params = get_params_startswith('scope_1')
# contains 4 parameters corresponding to scope names:
# 'scope_1/conv/W', 'scope_1/conv/bias', 'scope_1_1/conv/W' and 'scope_1_1/conv/bias'

```

## yaml_wrapper

Utility functions to read and write yaml file. 

Functions:
- [`neu.yaml_wrapper.read_yaml(filepath)`](neu/yaml_wrapper.py?plain=1#L42)  
Read `yaml` from `filepath`
- [`neu.yaml_wrapper.write_yaml(filepath, obj)`](neu/yaml_wrapper.py?plain=1#L49)    
Write `obj` (in `yaml` format) to `filepath`

Example:
```python
from neu.yaml_wrapper import read_yaml, write_yaml
config_path = ...
config = read_yaml(config_path)

# update config


write_yaml(config_path, config)

```

## datasets         

`neu` also provided data iterator for [ade20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and [cityscapes](https://www.cityscapes-dataset.com/) datasets         

- [`neu.datasets.ade20k.create_data_iterator(batch_size, data_list, image_shape, comm=None, shuffle=True, rng=None, with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True)`](neu/datasets/ade20k.py?plain=1#L207)
- [`neu.datasets.city_scapes.create_data_iterator(batch_size, data_list, image_shape, comm=None, shuffle=True, rng=None, with_memory_cache=False, with_parallel=False, with_file_cache=False, flip=True)`](neu/datasets/city_scapes.py?plain=1#L153)

For details on usage, check [Pix2PixHD](../GANs/pix2pixHD)

## metrics

Check [metrics README](neu/metrics).

## tts

Based on the [nnabla implementation of NVC-Net](https://github.com/sony/ai-research-code/tree/master/nvcnet), neu also provides some utility functions related to Text to Speech (tts).    

### audio

Functions: 
- [`neu.tts.audio.amp_2_db(x)`](neu/tts/audio.py?plain=1#L23)  
Converts a signal from amplitude to decibel.
- [`neu.tts.audio.db_2_amp(x)`](neu/tts/audio.py?plain=1#L28)                 
Converts a signal from decibel to amplitude.
- [`neu.tts.preemphasis(x, factor=0.97)`](neu/tts/audio.py?plain=1#L33)    
Applies preemphasis to a signal.
- [`neu.tts.rev_preemphasis(x, factor=0.97)`](neu/tts/audio.py?plain=1#L38)          
Applies inverse preemphasis to a signal
- [`neu.tts.normalize(x, hp)`](neu/tts/audio.py?plain=1#L43)                          
Normalize spectrogram into a range of [0, 1].
Args:            
    - `x (numpy.ndarray)`: The input spectrogram of shape (freq x time).
    - `hp (HParams)`: A container for hyperparameters.
        - `max_db (float)`: Maximum intensity in decibel.
        - `ref_db (float)`: Reference intensity in decibel.
Returns:             
    `numpy.ndarray`: An (freq x time) array of values in [0, 1].
- [`neu.tts.denormalize(x, hp)`](neu/tts/audio.py?plain=1#L59)       
Denormalize spectrogram.             
Args:          
    - `x (numpy.ndarray)`: The input spectrogram of shape (freq x time).             
    - `hp (HParams)`: A container for hyperparameters.          
        - `max_db (float)`: Maximum intensity in decibel.        
        - `ref_db (float)`: Reference intensity in decibel.       
Returns:       
    - `numpy.ndarray`: Spectrogram of shape (freq x time).
- [`neu.tts.spec2mel(spectrogram, sr, n_fft, n_mels)`](neu/tts/audio.py?plain=1#L76)       
Convert a spectrogram to a mel-spectrogram.         
Args:         
    - `spectrogram (numpy.ndarray)`: A spectrogram of shape (freq x time).
    - `sr (int)`: Sampling rate of the incoming signal.
    - `n_fft (int)`: Number of FFT components.
    - `n_mels (int)`: number of Mel bands to generate.       
Returns:        
    - `numpy.ndarray`: Mel spectrogram of shape (n_mels x time).
- [`neu.tts.wave2spec(wave, hp)`](neu/tts/audio.py?plain=1#L93)          
Convert a waveform to spectrogram.
Args:              
    - `wave (np.ndarray)`: An input waveform in 1D array.
    - `hp (HParams)`: A container for hyperparameters.
        - `n_fft (int)`: Length of the windowed signal after padding with zeros.
        - `hop_length (int)`: Number of audio samples between adjacent STFT columns.
        - `win_length (int)`: Each frame of audio is windowed by `window()` of length `win_length` and then padded with zeros to match `n_fft`.
Returns:         
    - `numpy.ndarray`: Spectrogram of shape (1+n_fft//2, time)
- [`neu.tts.spec2wave(spectrogram, hp)`](neu/tts/audio.py?plain=1#L114)        
Griffin-Lim's algorithm.     
Args:     
    - `spectrogram (numpy.ndarray)`: A spectrogram input of shape
        `(1+n_fft//2, time)`.
    - `hp (HParams)`: A container for hyperparameters.            
        - `n_fft (int)`: Number of FFT components.
        - `hop_length (int)`: Number of audio samples between adjacent STFT columns.
        - `win_length (int)`: Each frame of audio is windowed by `window()` of length `win_length` and then padded with zeros to match `n_fft`.
        - `n_iter (int, optional)`: [description]. Defaults to 50.
Returns:        
    `np.ndarray`: An 1D array representing the waveform.
- [`neu.tts.synthesize_from_spec(spec, hp)`](neu/tts/audio.py?plain=1#L145)      
Convert a waveform from its spectrogram.           
Args:        
    - `spec (numpy.ndarray)`: A spectrogram of shape (time, (n_fft//2+1)).
    - `hp (HParams)`: A container for hyperparameters.
        - `max_db (float)`: Maximum intensity in decibel.
        - `ref_db (float)`: Reference intensity in decibels.
        - `preemphasis (float)`: A pre-emphasis factor.
        - `n_fft (int)`: Length of the windowed signal after padding with zeros.
        - `hop_length (int)`: Number of audio samples between adjacent STFT columns.
        - `win_length (int)`: Each frame of audio is windowed by `window()` of
            length `win_length` and then padded with zeros to match `n_fft`.
        - `n_iter (int, optional)`: The number of iterations used in the Griffin-Lim
            algorithm. Defaults to 50.
Returns:       
    - `numpy.ndarray`: An 1D array of float values.

### hparams

- [`class neu.tts.hparams.HParams(object)`](neu/tts/hparams.py?plain=1#L19)     
A `HParams` object holds hyperparameters used to build and train a model, such as the learning rate, batch size, etc. The hyperparameter are available as attributes of the HParams object as follow: 
```python
from neu.tts.hparams import HParams
hp = HParams(learning_rate=0.1, num_hidden_units=100) 
# any number of hyperparameters can be passed for initialization
hp.learning_rate    # ==> 0.1
hp.num_hidden_units # ==> 100
```
The class also provides `save(self, filename)` function  to save the hyperparameter dictionary. 

### logger

- [`class neu.tts.logger.ProgressMeter(object)`](neu/tts/logger.py?plain=1#L21)         
A Progress Meter.                
Args:                   
    - `num_batches(int)`: The number of batches per epoch.                   
    - `path(str, optional)`: Path to save tensorboard and log file. Defaults to None.                 
    - `quiet(bool, optional)`: If quite == True, no message will be shown.    

    Functions:                                
    - `info(self, message, view=True)`            
        Shows a message.         
        Args:                                
        - `message(str)`: The message. 
        - `view(bool, optional)`: If shows to terminal. Defaults to True. 

    - `display(self, batch, key=None)`              
        Displays current values for meters.               
        Args:                
        - `batch(int)`: The number of batch.                                   
        - `key([type], optional)`: [description]. Defaults to None.  
    - `update(self, tag, value, n=1)`          
        Updates the meter.         
        Args:           
        - `tag(str)`: The tag name.
        - `value(number)`: The value to update.
        - `n(int, optional)`: The len of minibatch. Defaults to 1.
    - `close(self)`         
    Closes all the file descriptors       
    - `reset(self)`           
    Resets the ProgressMeter        

- [`class neu.tts.logger.AverageMeter(object)`](neu/tts/logger.py?plain=1#L98)
Computes and stores the average and current value.
Args:                   
    - `name(str)`: The number of batches per epoch.                   
    - `fmt(str, optional)`: Formatting option for printing the value. Defaults to ':f'.                 

    Functions:                                
    - `update(self, value, n=1)`          
        Updates the meter.                   
        Args:                       
        - `value(number)`: The value to update (whose current value and running average needs to be tracked).               
        - `n(int, optional)`: The len of minibatch. Defaults to 1.      
    - `reset(self)`                  
    Resets the AverageMeter           

### module

Functions:    
- [`neu.tts.module.insert_parent_name(name, params)`](neu/tts/module.py?plain=1#L21)    
Inserts `f@{name}` in the bedninning of all the keys of `params`   

Classes 
- [`neu.tts.module.ParamMemo(object)`](neu/tts/module.py?plain=1#L28)                       
To get newly added parameters from a set of existing parameters. Example:                 
```python
from neu.tts.module import ParamMemo
import nnabla as nn

p = ParamMemo()

params = {'a': nn.Variable((2,3)), 'b': nn.Variable((4,5))}
p.filter_and_update(params)
# OrderedDict([('a', <Variable((2, 3), need_grad=False) at 0x7fcfd5fb4b30>),
#             ('b', <Variable((4, 5), need_grad=False) at 0x7fcfd5fbbf50>)])

params['c'] = nn.Variable((2,3))
p.filter_and_update(params)
# OrderedDict([('c', <Variable((2, 3), need_grad=False) at 0x7fcfd6043dd0>)])

params['d'] = nn.Variable((2,3))
p.filter_and_update(params)
# OrderedDict([('d', <Variable((2, 3), need_grad=False) at 0x7fcfd6043dd0>)])
```

- [`neu.tts.module.Module(object)`](neu/tts/module.py?plain=1#43) 
Pytorch like class for defining neural network architecture. Architecture classes inheriting from this class need to define `call()` function for the network forward pass.   
    Functions:                                
    - `get_parameters(self, recursive=True, grad_only=False, memo=None)`              
    returns parameters associated with the object of the network class         
    - `set_parameter(self, key, param, raise_if_missing=False)`                  
    Recursively sets parameter. This function is called internally by `set_parameters`                    
    - `set_parameters(self, params, raise_if_missing=False)`           
    Registers weights of a network in a hierarchical manner in a similar way as done in a pytorch class that has been dereived from `torch.nn.Module`. 
    
    - `load_parameters(self, path, raise_if_missing=False)`                
    Loads parameters from a file with the specified format.    
    Args:     
        - `path (str)`: Path to file                   
        - `raise_if_missing (bool, optional)`: Raise exception if some parameters are missing. Defaults to `False`.
    - `save_parameters(self, path, grad_only=False)`    
    Saves the parameters to a file.     
    Args:     
        - `path (str)`: Path to file.
        - `grad_only (bool, optional)`: If `need_grad=True` is required for parameters which will be saved. Defaults to False.
    - `__setattr__(self, name, value)`    
    Sets a particular attribute of the object which can be accessed via `name` and stores `value`   
    - `__getattr__(self, name)` 
    To access a particular attribute (`name`) of the object 
    - `__call__(self, *args, **kwargs)`        
    calls `call()` function
    - `call(self, *args, **kwargs)`    
         Not defined here. Needs to be defined in the child class.

### optimizer

- [`class neu.tts.optimizer.Optimizer(object)`](neu/tts/optimizer.py?plain=1#L20)       
An Optimizer class.            
Args:
    - `weight_decay (float, optional)`: Weight decay (L2 penalty). Should be a positive value. Defaults to 0.              
    - `max_norm (float, optional)`: An input scalar of float value. Should be a positive value. Defaults to 0.             
    - `lr_scheduler (BaseLearningRateScheduler, optional)`: Learning rate scheduler. Defaults to None (no learning rate scheduler is applied).            
    - `name (str, optional)`: Name of the solver. Defaults to 'Sgd'.                 
Raises:         
    `NotImplementedError`: If the solver is not supported in nnabla.

### text
- [`neu.tts.text.text_normalize(text, vocab)`](neu/tts/text.py?plain=1#L24)             
Normalize an input text.             
Args:                   
    - `text (str)`: An input text.
    - `vocab (str)`: A string containing alphabets.    

    Returns:                
    - `str`: A text containing only given alphabets.            

### trainer
- [`class neu.tts.trainer.Trainer(ABC)`](neu/tts/trainer.py?plain=1#L28)        
Implementation of Trainer.                        
Args:                 
    - `model (model.module.Module)`: WaveGlow model.              
    - `dataloader (dict)`: A dataloader.                                 
    - `optimizer (Optimizer)`: An optimizer used to update the parameters.              
    - `hparams (HParams)`: Hyper-parameters.           

## Additional Information
## Loading Intermediate h5 to nnp 
The below written code demonstrate how the intermediate h5 file can be used for all the examples present in the NNabla-examples repository. When you run the training script, the network file(.nnp) will be saved before the training begins to obatin the architecture of the network even though weights and biases are randomly initialized. The network parameters(.h5) will also be saved. Using these two files you can get the network along with its parameters to do inference.

## Steps to load the h5 file to nnp
* Load the nnp file at epoch 0.
* Load the desired .h5 file.
* Use nnp.get_network_names() to fetch the network name. 
* Create a variable graph of the network by name.
* Load the inputs to the input variable.
* Execute the network.


### Demonstration of the code 
```python
import nnabla as nn
from nnabla.utils.image_utils import imread
from nnabla.utils.nnp_graph import NnpLoader
nn.clear_parameters()
nn.load_parameters('Path to saved h5 file')
nnp = NnpLoader('Path to saved nnp file')
img = imread('Path to image file')
net = nnp.get_network('name of the network', batch_size=1)
y = net.outputs['y']
x = net.inputs['x']
x.d = img
y.forward(clear_buffer=True)
```
