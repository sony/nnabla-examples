# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
import numpy as np
import nnabla as nn
import nnabla.functions as F

from layers import chunk
from neu.losses import kl_normal, gaussian_log_likelihood
from neu.misc import AttrDict

# Better to implement ModelVarType as a module?


class ModelVarType(enum.Enum):
    """
    Enum class for the type of model variance.

    Each entry represents
    - FIXED_SMALL: variance of posterior q(x_{t-1}|x_t, x_0) corresponding to beta_t in the paper.
    - FIXED_LARGE: variance of forward noising process q(x_t | x_{t-1}) corresponding to beta_tilde_t in the paper
    - LEARNED_RANGE: interpolation between FIXED_SMALL and FIXED_LARGE proposed by "Improved Denoising Diffusion Probabilistic Models".
    """
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

    @staticmethod
    def get_supported_keys():
        return [x.name for x in ModelVarType]

    @staticmethod
    def get_vartype_from_key(key):
        for elem in ModelVarType:
            if elem.name.lower() == key.lower():
                return elem

        raise ValueError(
            f"key '{key}' is not supported. Key must be one of {ModelVarType.get_supported_keys()}.")


def is_learn_sigma(model_var_type: ModelVarType):
    # If we add model variance type, modify this also.
    return model_var_type == ModelVarType.LEARNED_RANGE


def _get_alpha_bar_from_time(t):
    """
    Noise scheduling method proposed by Nichol et. al to avoid too noisy image especially for smaller resolution.

    This strategy creates beta as follows:
        alpha_bar(t) = f(t) / f(0)
        f(t) = cos((t / T + s) / (1 + s) * PI / 2) ** 2
        beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1)

    where s = 0.008 (~= 1 / 127.5), which ensures sqrt(beta_0) is slightly smaller than the pixel bin size.
    """

    assert 0 <= t <= 1, "t must be normalized by max diffusion timestep (T)."

    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


def get_beta_schedule(strategy, num_timesteps):
    """
    Get a pre-defined beta schedule for the given strategy.
    Based on a scheduling proposed by Ho et. al, scale it to any timesteps.

    Args:
        strategy (string): Strategy to create noise schedule. Should be one of {"linear", "cosine"}.
        num_timesteps (int): Max timestep for the diffusion process.
    """
    if strategy == "linear":
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end,
                            num_timesteps, dtype=np.float64)
    elif strategy == "cosine":
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            max_beta = 0.999  # make beta lower than 1 to prevent singularities
            alpha_bar = 1 - \
                _get_alpha_bar_from_time(t2) / _get_alpha_bar_from_time(t1)
            betas.append(min(alpha_bar, max_beta))
            assert betas[-1] <= max_beta

        betas = np.array(betas)
    else:
        raise NotImplementedError(strategy)

    assert betas.shape == (num_timesteps,)
    return betas


def const_var(np_array):
    """
    Create constant nn.Variable from numpy array.
    """
    ret = nn.Variable.from_numpy_array(np_array, need_grad=False)
    ret.persistent = True
    return ret


def noise_like(shape, noise_function=F.randn, repeat=False):
    """
    Create noise with given shape.
    """
    if repeat:
        noise = noise_function(shape=(1, ) + shape[1:])
        return F.tile(noise, (shape[0], ) + (1, ) * (len(shape) - 1))

    return noise_function(shape=shape)


def mean_along_except_batch(x, batch_axis=0):
    return F.mean(x, axis=list(range(1, len(x.shape))))


class GaussianDiffusion(object):
    """
    An interface for gaussian diffusion process.

    Args:
        betas (numpy.ndarray): 
            A 1-D tensor of noise scales at each timestep. 
            Assume this is created by get_beta_schedule() defined above.
    """

    def __init__(self,
                 betas,
                 model_var_type=ModelVarType.FIXED_SMALL):
        self.model_var_type = model_var_type

        # check input betas
        assert (isinstance(betas, np.ndarray))
        assert (betas > 0).all() and (betas <= 1).all()

        betas = betas.astype(np.float64)
        self.np_betas = betas
        T, = betas.shape
        self.num_timesteps = int(T)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)
        assert alphas_cumprod_prev.shape == (T, )

        self.betas = const_var(betas)
        self.alphas_cumprod = const_var(alphas_cumprod)
        self.alphas_cumprod_prev = const_var(alphas_cumprod_prev)
        self.alphas_cumprod_next = const_var(alphas_cumprod_next)

        # for q(x_t | x{t-1})
        self.sqrt_alphas_cumprod = const_var(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = const_var(
            np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = const_var(
            np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = const_var(
            np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = const_var(
            np.sqrt(1. / alphas_cumprod - 1.))

        # for q(x_{t-1} | x_t, x_0)
        posterior_var = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_var = const_var(posterior_var)
        self.posterior_log_var_clipped = const_var(
            np.log(np.append(posterior_var[1], posterior_var[1:])))
        self.posterior_mean_coef1 = const_var(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = const_var(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # for FIXED_LARGE
        # According to the origianl author's implementation,
        # initial (log-)variance should be the posterior's one to get a better decoder log likelihood.
        np_betas_clipped = np.append(posterior_var[1], betas[1:])
        self.betas_clipped = const_var(np_betas_clipped)
        self.log_betas_clipped = const_var(np.log(np_betas_clipped))

    @staticmethod
    def _extract(a, t, x_shape=None):
        """
        Extract some coefficients at specified timesteps.
        If x_shape is not None, 
        reshape the extracted one to [batch_size, 1, 1, 1, 1, ...] corrensponding to x_shape for broadcasting purposes.
        """

        B, = t.shape
        out = F.gather(a, t, axis=0)

        if x_shape is None:
            return out

        assert x_shape[0] == B
        assert out.shape == (B, )
        out_shape = np.array(list(x_shape))
        out_shape[1:] = 1
        return F.reshape(out, out_shape)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step), which samples from q(x_t | x_0).
        xt = sqrt(cumprod(alpha_0, ..., alpha_t)) * x_0 + sqrt(1 - cumprod(alpha_0, ..., alpha_t)) * epsilon

        Args:
            x_start (nn.Variable): The (B, C, ...) tensor of x_0.
            t (nn.Variable): A 1-D tensor of timesteps.

        Return:
            x_t (nn.Variable): 
                The (B, C, ...) tensor of x_t.
                Each sample x_t[i] corresponds to the noisy image at timestep t[i] constructed from x_start[i].
        """
        if noise is None:
            noise = F.randn(shape=x_start.shape)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod,
                          t, x_start.shape) * noise
        )

    def predict_xstart_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t.
        x0 = x_t / sqrt(cumprod(alpha_0, ..., aplha_t)) - sqrt(1 / cumprod(alpha_0, ..., alpha_t) - 1) * epsilon

        Args:
            x_t (nn.Variable): The (B, C, ...) tensor whose i-th element is an noisy data at timestep t[i].
            t (nn.Variable): A 1-D tensor of timesteps.
            noise (nn.Variable): The (B, C, ...) tensor of predicted noise in x_t.

        Return:
            x_0 (nn.Variable): The (B, C, ...) tensor of predicted x_0.
        """
        assert x_t.shape == noise.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod,
                          t, x_t.shape) * noise
        )

    def predict_noise_from_xstart(self, x_t, t, pred_xstart):
        """
        Predict epsilon from x_t and predicted x_0.
        epsilon = (x_t / sqrt(cumprod(alpha_0, ..., alpha_t)) - x_0) / (sqrt(1 / cumprod(alpha_0, ..., alpha_t) - 1))

        Args:
            x_t (nn.Variable): The (B, C, ...) tensor whose i-th element is an noisy data at timestep t[i].
            t (nn.Variable): A 1-D tensor of timesteps.
            noise (nn.Variable): The (B, C, ...) tensor of predicted noise in x_t.

        Return:
            x_0 (nn.Variable): The (B, C, ...) tensor of predicted x_0.
        """
        assert x_t.shape == pred_xstart.shape

        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = self._extract(self.posterior_var, t, x_t.shape)
        posterior_log_var_clipped = self._extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_var.shape[0] == posterior_log_var_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def _vlb_in_bits_per_dims(self, model, x_start, x_t, t,
                              clip_denoised=True):
        """
        Calculate variational lower bound in bits/dims.
        """
        B, C, H, W = x_start.shape
        assert x_start.shape == x_t.shape
        assert t.shape == (B, )

        # true parameters
        mean, _, log_var_clipped = self.q_posterior(x_start, x_t, t)

        # pred parameters
        preds = self.p_mean_var(model, x_t, t, clip_denoised=clip_denoised)

        # Negative log-likelihood
        nll = -gaussian_log_likelihood(x_start,
                                       mean=preds.mean, logstd=0.5 * preds.log_var)
        nll_bits = mean_along_except_batch(nll) / np.log(2.0)
        assert nll.shape == x_start.shape
        assert nll_bits.shape == (B, )

        # kl between true and pred in bits
        kl = kl_normal(mean, log_var_clipped, preds.mean, preds.log_var)
        kl_bits = mean_along_except_batch(kl) / np.log(2.0)
        assert kl.shape == x_start.shape
        assert kl_bits.shape == (B, )

        # Return nll at t = 0, otherwise KL(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t))
        return F.where(F.equal_scalar(t, 0), nll_bits, kl_bits)

    def train_loss(self, model, x_start, t, noise=None):
        """
        Calculate training loss for given data and model.

        Args:
            model (callable): 
                A trainable model to predict noise in data conditioned by timestep.
                This function should perform like pred_noise = model(x_noisy, t).
                If self.model_var_type is the one that requires prediction for sigma, model has to output them as well.
            x_start (nn.Variable): The (B, C, ...) tensor of x_0.
            t (nn.Variable): A 1-D tensor of timesteps.
            noise (callable or None): A noise generator. If None, F.randn(shape=x_start.shape) will be used.

        Return:
            loss (dict of {string: nn.Variable}): 
                Return dict that has losses to train the `model`.
                You can access each loss by a name that will be:
                    - `vlb`: Variational Lower Bound for learning sigma. 
                             This will be included only if self.model_var_type requires to learn sigma.
                    - `mse`: MSE between actual and predicted noise.
                Each entry is the (B, ) tensor of batched loss computed from given inputs.
                Note that this function doesn't reduce batch dim
                in order to make it easy to trace the loss value at each timestep.
                Therefore, you should take average for returned Variable over batch dim to train the model.
        """
        B, C, H, W = x_start.shape
        assert t.shape == (B, )

        if noise is None:
            noise = F.randn(shape=x_start.shape)
        assert noise.shape == x_start.shape

        # Calculate x_t from x_start, t, and noise.
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        assert x_noisy.shape == x_start.shape

        # Predict noise.
        # According to the original DDPM, this is superior than reconstructing x_0.
        # If model_var_type requires to learn sigma, model must output pred_sigma as well.
        pred = model(x_noisy, t)

        # Calculate losses
        ret = AttrDict()

        if is_learn_sigma(self.model_var_type):
            # split pred into 2 variables along channel axis.
            pred_noise, pred_sigma = chunk(pred, num_chunk=2, axis=1)
            assert pred_sigma.shape == x_start.shape, \
                f"Shape mismutch between pred_sigma {pred_sigma.shape} and x_start {x_start.shape}"

            # Variational lower bound for sigma
            # Use dummy function as model, since we already got prediction from model.
            var = F.concatenate(pred_noise.get_unlinked_variable(
                need_grad=False), pred_sigma, axis=1)
            ret.vlb = self._vlb_in_bits_per_dims(model=lambda x_t, t: var,
                                                 x_start=x_start,
                                                 x_t=x_noisy,
                                                 t=t)
        else:
            pred_noise = pred

        assert pred_noise.shape == x_start.shape, \
            f"Shape mismutch between pred_noise {pred_sigma.shape} and x_start {x_start.shape}"

        ret.mse = mean_along_except_batch(F.squared_error(noise, pred_noise))

        # shape check for all losses
        for name, loss in ret.items():
            assert loss.shape == (B, ), \
                 f"A Variabla for loss `{name}` has a wrong shape ({loss.shape} != {(B, )})"

        return ret

    def p_mean_var(self, model, x_t, t, clip_denoised=True):
        """
        Compute mean and var of p(x_{t-1}|x_t) from model.

        Args:
            model (Callable): A callbale that takes x_t and t and predict noise (and more).
            x_t (nn.Variable): The (B, C, ...) tensor at timestep t (x_t).
            t (nn.Variable): A 1-D tensor of timesteps. The first axis represents batchsize.
            clip_denoised (bool): If True, clip the denoised signal into [-1, 1].

        Returns:
            An AttrDict containing the following items:
                "mean": the mean predicted by model.
                "var": the variance predicted by model (or pre-defined variance).
                "log_var": the log of "var".
                "xstart": the x_0 predicted from x_t and t by model.
        """
        B, C, H, W = x_t.shape
        assert t.shape == (B, )
        pred = model(x_t, t)

        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            assert pred.shape == (B, 2 * C, H, W)
            pred_noise, pred_var_coeff = chunk(pred, num_chunk=2, axis=1)

            min_log = self._extract(
                self.posterior_log_var_clipped, t, x_t.shape)
            max_log = F.log(self._extract(self.betas, t, x_t.shape))

            # pred_var_coeff should be [0, 1]
            v = F.sigmoid(pred_var_coeff)
            model_log_var = v * max_log + (1 - v) * min_log
            model_var = F.exp(model_log_var)
        else:
            # Model only predicts noise
            pred_noise = pred

            model_log_var, model_var = {
                ModelVarType.FIXED_LARGE: lambda: (
                    self._extract(self.log_betas_clipped, t, x_t.shape),
                    self._extract(self.betas_clipped, t, x_t.shape)
                ),
                ModelVarType.FIXED_SMALL: lambda: (
                    self._extract(
                        self.posterior_log_var_clipped, t, x_t.shape),
                    self._extract(self.posterior_var, t, x_t.shape)
                )
            }[self.model_var_type]()

        x_recon = self.predict_xstart_from_noise(
            x_t=x_t, t=t, noise=pred_noise)

        if clip_denoised:
            x_recon = F.clip_by_value(x_recon, -1, 1)

        model_mean, _, _ = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)

        assert model_mean.shape == x_recon.shape == x_t.shape

        assert model_mean.shape == model_var.shape == model_log_var.shape or \
            (model_mean.shape[0] == model_var.shape[0] == model_log_var.shape[0] and model_var.shape[1:] == (
                1, 1, 1) and model_log_var.shape[1:] == (1, 1, 1))

        # returns
        ret = AttrDict()
        ret.mean = model_mean
        ret.var = model_var
        ret.log_var = model_log_var
        ret.xstart = x_recon

        return ret

    def p_sample(self,
                 model,
                 x_t,
                 t,
                 clip_denoised=True,
                 noise_function=F.randn,
                 repeat_noise=False,
                 no_noise=False):
        """
        Sample from the model for one step.
        Also return predicted x_start.
        """
        preds = self.p_mean_var(model=model, x_t=x_t,
                                t=t, clip_denoised=clip_denoised)

        # no noise when t == 0
        if no_noise:
            return preds.mean, preds.xstart

        noise = noise_like(x_t.shape, noise_function, repeat_noise)
        assert noise.shape == x_t.shape

        # sample from gaussian N(model_mean, )
        return preds.mean + F.exp(0.5 * preds.log_var) * noise, preds.xstart

    # DDIM sampler
    def ddim_sample(self,
                    model,
                    x_t,
                    t,
                    clip_denoised=True,
                    noise_function=F.randn,
                    repeat_noise=False,
                    no_noise=False,
                    eta=0.):
        """
        sample x_{t-1} from x_{t} by the model using DDIM sampler.
        Also return predicted x_start.
        """
        preds = self.p_mean_var(model, x_t, t, clip_denoised=clip_denoised)

        pred_noise = self.predict_noise_from_xstart(x_t, t, preds.xstart)

        from layers import sqrt
        alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)
        sigma = (
            eta
            * sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        mean_pred = (
            preds.xstart * sqrt(alpha_bar_prev)
            + sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_noise
        )

        if no_noise:
            return mean_pred, preds.xstart

        noise = noise_like(x_t.shape, noise_function, repeat_noise)
        return mean_pred + sigma * noise, preds.xstart

    def ddim_rev_sample(self, model, x_t, t, clip_denoised=True, eta=0.0):
        """
        sample x_{t+1} from x_{t} by the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "ReverseODE only for deterministic path"
        preds = self.p_mean_var(model, x_t, t, clip_denoised=clip_denoised)

        pred_noise = self.predict_noise_from_xstart(x_t, t, preds.xstart)
        alpha_bar_next = self._extract(self.alphas_cumprod_next, t, x_t.shape)

        from layers import sqrt
        return (
            preds.xstart * sqrt(alpha_bar_next)
            + sqrt(1 - alpha_bar_next) * pred_noise
        )

    def sample_loop(self, model, shape, sampler,
                    noise=None,
                    dump_interval=-1,
                    progress=False,
                    without_auto_forward=False):
        """
        Iteratively Sample data from model from t=T to t=0.
        T is specified as the length of betas given to __init__().

        Args:
            model (collable): 
                A callable that takes x_t and t and predict noise (and sigma related parameters).
            shape (list like object): A data shape.
            sampler (callable): A function to sample x_{t-1} given x_{t} and t. Typically, self.p_sample or self.ddim_sample.
            noise (collable): A noise generator. If None, F.randn(shape) will be used.
            interval (int): 
                If > 0, all intermediate results at every `interval` step will be returned as a list.
                e.g. if interval = 10, the predicted results at {10, 20, 30, ...} will be returned.
            progress (bool): If True, tqdm will be used to show the sampling progress.

        Returns:
            - x_0 (nn.Variable): the final sampled result of x_0
            - samples (a list of nn.Variable): the sampled results at every `interval`
            - pred_x_starts (a list of nn.Variable): the predicted x_0 from each x_t at every `interval`: 
        """
        T = self.num_timesteps
        indices = list(range(T))[::-1]

        samples = []
        pred_x_starts = []

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        if without_auto_forward:
            if noise is None:
                noise = np.random.randn(*shape)
            else:
                assert isinstance(noise, np.ndarray)
                assert noise.shape == shape

            x_t = nn.Variable.from_numpy_array(noise)
            t = nn.Variable.from_numpy_array([T - 1 for _ in range(shape[0])])

            # build graph
            y, pred_x_start = sampler(model, x_t, t)
            up_x_t = F.assign(x_t, y)
            up_t = F.assign(t, t - 1)
            update = F.sink(up_x_t, up_t)

            cnt = 0
            for step in indices:
                y.forward(clear_buffer=True)
                update.forward(clear_buffer=True)

                cnt += 1
                if dump_interval > 0 and cnt % dump_interval == 0:
                    samples.append((step, y.d.copy()))
                    pred_x_starts.append((step, pred_x_start.d.copy()))
        else:
            with nn.auto_forward():
                if noise is None:
                    x_t = F.randn(shape=shape)
                else:
                    assert isinstance(noise, np.ndarray)
                    assert noise.shape == shape
                    x_t = nn.Variable.from_numpy_array(noise)
                cnt = 0
                for step in indices:
                    t = F.constant(step, shape=(shape[0], ))
                    x_t, pred_x_start = sampler(
                        model, x_t, t, no_noise=step == 0)
                    cnt += 1
                    if dump_interval > 0 and cnt % dump_interval == 0:
                        samples.append((step, x_t.d.copy()))
                        pred_x_starts.append((step, pred_x_start.d.copy()))

        assert x_t.shape == shape
        return x_t.d.copy(), samples, pred_x_starts

    def p_sample_loop(self, *args, **kwargs):
        """
        Sample data from x_T ~ N(0, I) with p(x_{t-1}|x_{t}) proposed by "Denoising Diffusion Probabilistic Models".
        See self.sample_loop for more details about sampling process.

        """

        return self.sample_loop(*args, sampler=self.p_sample, **kwargs)

    def ddim_sample_loop(self, *args, **kwargs):
        """
        Sample data from x_T ~ N(0, I) with p(x_{t-1}|x_{t}, x_{0}) proposed by "Denoising Diffusion Implicit Models".
        See self.sample_loop for more details about sampling process.

        """
        from functools import partial

        return self.sample_loop(*args,
                                sampler=partial(self.ddim_sample, eta=0.),
                                **kwargs)
