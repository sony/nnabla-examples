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
from functools import partial
from typing import Union

import nnabla as nn
import nnabla.functions as F
import numpy as np
from neu.losses import gaussian_log_likelihood, kl_normal
from neu.misc import AttrDict

from config import DiffusionConfig

from .layers import chunk, sqrt
from .utils import Shape4D, context_scope, force_float

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
    def get_vartype_from_key(key: str):
        for elem in ModelVarType:
            if elem.name.lower() == key.lower():
                return elem

        raise ValueError(
            f"key '{key}' is not supported. Key must be one of {ModelVarType.get_supported_keys()}.")


def is_learn_sigma(model_var_type: Union[ModelVarType, str]):
    # convert string to ModelVarType
    if isinstance(model_var_type, str):
        model_var_type = ModelVarType.get_vartype_from_key(model_var_type)

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


def respace_betas(betas, use_timesteps):
    """
    Given betas and use_timestpes, returns respaced betas.

    Args:
        betas (np.array): A T elements of array containing beta_t at index t.
        use_timesteps (tuple):
            A tuple indicates which timesteps are used for generation process.
            For example, if use_timestpes = (0, 100, 300, 500, 700, 900, 1000), only ts included in use_timestpes will be used.

    Returns:
        new_betas (np.array): A respaced betas.
        timestep_map (np.array): 
            A list indicating how to map a new index to original index.
            It will be the same as use_timesteps sorted in increasing order.
    """

    assert hasattr(
        use_timesteps, "__iter__"), "use_timesteps must be iterable."
    use_timesteps = tuple(use_timesteps)

    T = len(betas)

    new_betas = []
    prev_alphas_cumprod = 1.
    alphas_cumprod = 1.
    timestep_map = []
    for t in range(T):
        alphas_cumprod *= 1. - betas[t]
        if t in use_timesteps:
            new_beta = 1. - alphas_cumprod / prev_alphas_cumprod
            new_betas.append(new_beta)
            timestep_map.append(t)
            prev_alphas_cumprod = alphas_cumprod

    return np.asarray(new_betas, dtype=np.float64), np.asarray(timestep_map, dtype=int)


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
    """

    def __init__(self, conf: DiffusionConfig):
        self.model_var_type = ModelVarType.get_vartype_from_key(
            conf.model_var_type)

        # generate betas from strategy and max timesteps
        betas = get_beta_schedule(conf.beta_strategy, conf.max_timesteps)
        self.beta_strategy = conf.beta_strategy

        # setup timestep to start sampling
        assert 0 < conf.t_start <= conf.max_timesteps, \
            f"Invalid t_start. t_start (= {conf.t_start}) must be an integer between [1, {conf.max_timesteps}]."
        self.max_timesteps = conf.max_timesteps
        self.t_start = conf.t_start

        # setup respacing
        self.timestep_map = None

        if conf.respacing_step > 1 or conf.t_start < conf.max_timesteps - 1:
            # Note: timestep is shifted one step ahead because of 0-indexing (e.g q_sample(x_start, 0) samples x_1 insted of x_0.)
            # According to guided-diffusion, we should always use t = 0 and T - 1.
            # In addition to them, add (T / respacing_step - 2) timesteps.
            max_timesteps = min(conf.t_start, conf.max_timesteps)
            num_use_timesteps = max_timesteps // conf.respacing_step

            frac_steps = float(max_timesteps - 1) / \
                (num_use_timesteps - 1)

            start = 0
            cur_idx = 0.
            use_timesteps = []
            for _ in range(num_use_timesteps):
                use_timesteps.append(start + round(cur_idx))
                cur_idx += frac_steps

            assert use_timesteps[0] == 0
            assert use_timesteps[-1] == max_timesteps - 1

            betas, timestep_map = respace_betas(betas, use_timesteps)
            self.timestep_map = const_var(timestep_map)

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
        assert alphas_cumprod_prev.shape == (T, )

        self.betas = const_var(betas)
        self.alphas_cumprod = const_var(alphas_cumprod)
        self.alphas_cumprod_prev = const_var(alphas_cumprod_prev)

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
    @force_float
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

    def _rescale_timestep(self, t):
        if self.timestep_map is None:
            return t

        assert isinstance(self.timestep_map, nn.Variable)

        # revert t to the original timestep
        return self._extract(self.timestep_map, t)

    @staticmethod
    def _pre_compute_model_kwargs(model_kwargs):
        if model_kwargs is None:
            return

        assert isinstance(
            model_kwargs, dict), f"model_kwargs must be dict but `{type(model_kwargs)}` is given"

        var_list = []
        for x in model_kwargs.values():
            if not isinstance(x, nn.Variable):
                continue

            if x.parent is None:
                continue

            x.apply(persistent=True)
            var_list.append(x)

        if len(var_list) == 0:
            return

        # clear_buffer=True is maybe unsafe.
        if len(var_list) > 1:
            vars = F.sink(*var_list)
        else:
            vars = var_list[0]

        vars.forward(clear_buffer=True)

    @force_float
    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data, which samples from q(x_t | x_0).
        xt = sqrt(cumprod(alpha_0, ..., alpha_t)) * x_0 + sqrt(1 - cumprod(alpha_0, ..., alpha_t)) * epsilon
        Note that t is 0-indexed and shifted one step, which is t == 0 means sampling x_1 and t == T-1 means sampling x_T.

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

    @force_float
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

    @force_float
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

    @force_float
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

    def _vlb_in_bits_per_dims(self, model, x_start, x_t, t, *,
                              model_kwargs=None,
                              clip_denoised=True, channel_last=False):
        """
        Calculate variational lower bound in bits/dims.
        """
        B = x_start.shape[0]
        assert x_start.shape == x_t.shape
        assert t.shape == (B, )

        # true parameters
        mean, _, log_var_clipped = self.q_posterior(x_start, x_t, t)

        # pred parameters
        preds = self.p_mean_var(model, x_t, t,
                                model_kwargs=model_kwargs,
                                clip_denoised=clip_denoised,
                                channel_last=channel_last)

        with context_scope("float"):
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

    def train_loss(self, *, model, x_start, t, model_kwargs=None, noise=None, channel_last=False):
        """
        Calculate training loss for given data and model.

        Args:
            model (callable): 
                A trainable model to predict noise in data conditioned by timestep.
                This function should perform like pred_noise = model(x_noisy, t).
                If self.model_var_type is the one that requires prediction for sigma, model has to output them as well.
            x_start (nn.Variable): A (B, C, ...) tensor of x_0.
            t (nn.Variable): A 1-D tensor of timesteps.
            model_kwargs (Dict): Additional model arguments. 
            noise (callable or None): A noise generator. If None, F.randn(shape=x_start.shape) will be used.
            channel_last (boolean): Whether channel axis is the last axis of an array or not.

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
        B = x_start.shape[0]
        c_axis = 3 if channel_last else 1
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
        if model_kwargs is None:
            model_kwargs = {}
        pred = model(x_noisy, t, **model_kwargs)

        # Calculate losses
        ret = AttrDict()

        if is_learn_sigma(self.model_var_type):
            assert pred.shape[c_axis] == x_start.shape[c_axis] * 2

            # split pred into 2 variables along channel axis.
            pred_noise, pred_sigma = chunk(pred, num_chunk=2, axis=c_axis)
            assert pred_sigma.shape == x_start.shape, \
                f"Shape mismutch between pred_sigma {pred_sigma.shape} and x_start {x_start.shape}"

            # Variational lower bound for sigma
            # Use dummy function as model, since we already got prediction from model.
            var = F.concatenate(pred_noise.get_unlinked_variable(
                need_grad=False), pred_sigma, axis=3 if channel_last else 1)
            ret.vlb = self._vlb_in_bits_per_dims(model=lambda x_t, t, **kwargs: var,
                                                 x_start=x_start,
                                                 x_t=x_noisy,
                                                 model_kwargs=model_kwargs,
                                                 t=t,
                                                 channel_last=channel_last)
        else:
            assert pred.shape[c_axis] == x_start.shape[c_axis]
            pred_noise = pred

        assert pred_noise.shape == x_start.shape, \
            f"Shape mismutch between pred_noise {pred_sigma.shape} and x_start {x_start.shape}"

        ret.mse = mean_along_except_batch(F.squared_error(noise, pred_noise))

        # shape check for all losses
        for name, loss in ret.items():
            assert loss.shape == (B, ), \
                 f"A Variabla for loss `{name}` has a wrong shape ({loss.shape} != {(B, )})"

        return ret

    def p_mean_var(self, model, x_t, t, *, model_kwargs=None, clip_denoised=True, channel_last=False, classifier_free_guidance_weight=None):
        """
        Compute mean and var of p(x_{t-1}|x_t) from model.

        Args:
            model (Callable): A callbale that takes x_t and t and predict noise (and more).
            x_t (nn.Variable): The (B, C, ...) tensor at timestep t (x_t).
            t (nn.Variable): A 1-D tensor of timesteps. The first axis represents batchsize.
            clip_denoised (boolean): If True, clip the denoised signal into [-1, 1].
            channel_last (boolean): Whether the channel axis is the last axis of an Array or not.
            classifier_free_guidance_weight (float): A weight for classifier-free guidance.

        Returns:
            An AttrDict containing the following items:
                "mean": the mean predicted by model.
                "var": the variance predicted by model (or pre-defined variance).
                "log_var": the log of "var".
                "xstart": the x_0 predicted from x_t and t by model.
        """
        B, C, H, W = Shape4D(
            x_t.shape, channel_last=channel_last).get_as_tuple("bchw")
        assert t.shape == (B, )

        if model_kwargs is None:
            model_kwargs = {}

        pred = model(x_t, t, **model_kwargs)

        if self.model_var_type == ModelVarType.LEARNED_RANGE:
            pred_noise, pred_var_coeff = chunk(
                pred, num_chunk=2, axis=3 if channel_last else 1)

            min_log = self._extract(
                self.posterior_log_var_clipped, t, x_t.shape)
            max_log = F.log(self._extract(self.betas, t, x_t.shape))

            # No need to constrain v, according to the "improved DDPM" paper.
            v = pred_var_coeff
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

        # classifier-free guidance
        if classifier_free_guidance_weight is not None and classifier_free_guidance_weight > 0:
            model_kwargs_uncond = model_kwargs.copy()
            model_kwargs_uncond["cond_drop_rate"] = 1
            pred_uncond = model(x_t, t, **model_kwargs_uncond)

            if self.model_var_type == ModelVarType.LEARNED_RANGE:
                pred_noise_uncond = pred_uncond[...,
                                                :3] if channel_last else pred_uncond[:, :3]
            else:
                pred_noise_uncond = pred_uncond

            # (1 + w) * eps(t, c) - w * eps(t)
            w = classifier_free_guidance_weight
            pred_noise = (1 + w) * pred_noise - w * pred_noise_uncond

        x_recon = self.predict_xstart_from_noise(
            x_t=x_t, t=t, noise=pred_noise)

        if clip_denoised:
            x_recon = F.clip_by_value(x_recon, -1, 1)

        model_mean, _, _ = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)

        assert model_mean.shape == x_recon.shape

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
                 *,
                 model_kwargs=None,
                 clip_denoised=True,
                 noise_function=F.randn,
                 repeat_noise=False,
                 no_noise=False,
                 channel_last=False,
                 classifier_free_guidance_weight=None):
        """
        Sample from the model for one step.
        Also return predicted x_start.
        """
        preds = self.p_mean_var(model=model,
                                x_t=x_t,
                                t=t,
                                model_kwargs=model_kwargs,
                                clip_denoised=clip_denoised,
                                channel_last=channel_last,
                                classifier_free_guidance_weight=classifier_free_guidance_weight)

        # no noise when t == 0
        if no_noise:
            return preds.mean, preds.xstart

        noise = noise_like(x_t.shape, noise_function, repeat_noise)
        assert noise.shape == x_t.shape

        # sample from gaussian N(model_mean, model_var)
        return preds.mean + F.exp(0.5 * preds.log_var) * noise, preds.xstart

    # DDIM sampler
    def ddim_sample(self,
                    model,
                    x_t,
                    t,
                    *,
                    model_kwargs=None,
                    clip_denoised=True,
                    noise_function=F.randn,
                    repeat_noise=False,
                    no_noise=False,
                    eta=0.,
                    channel_last=False,
                    classifier_free_guidance_weight=None):
        """
        sample x_{t-1} from x_{t} by the model using DDIM sampler.
        Also return predicted x_start.
        """

        preds = self.p_mean_var(model, x_t, t,
                                model_kwargs=model_kwargs,
                                clip_denoised=clip_denoised,
                                channel_last=channel_last,
                                classifier_free_guidance_weight=classifier_free_guidance_weight)

        pred_noise = self.predict_noise_from_xstart(x_t, t, preds.xstart)

        alpha_bar = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev, t, x_t.shape)

        with context_scope("float"):
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

    def ddim_rev_sample(self,
                        model,
                        x_tm1,
                        t,
                        *,
                        model_kwargs=None,
                        clip_denoised=True,
                        eta=0.0,
                        channel_last=False,
                        classifier_free_guidance_weight=None,
                        no_noise=False):
        """
        sample x_{t} from x_{t-1} by the model using DDIM reverse ODE.
        Because we design self.betas[t] is a noise scale for x_{t+1} (timestep is shifted 1 step),
        ddim rev
        """
        assert eta == 0.0, "ReverseODE only for deterministic path"

        alpha_bar = self._extract(self.alphas_cumprod, t, x_tm1.shape)

        if no_noise:
            # Special case for x_0
            return x_tm1 * sqrt(alpha_bar), None

        # Note: if t == 0, no_noise must be True so that t - 1 >= 0 is always satisfied.
        # Todo: replace no_noise with a multipliation by zero.
        preds = self.p_mean_var(model, x_tm1, t - 1,
                                model_kwargs=model_kwargs,
                                clip_denoised=clip_denoised,
                                channel_last=channel_last,
                                classifier_free_guidance_weight=classifier_free_guidance_weight)

        pred_noise = self.predict_noise_from_xstart(x_tm1, t - 1, preds.xstart)

        with context_scope("float"):
            return (
                preds.xstart * sqrt(alpha_bar)
                + sqrt(1 - alpha_bar) * pred_noise
            ), None

    def sample_loop(self, model, shape, sampler, *,
                    x_init=None,
                    model_kwargs=None,
                    dump_interval=-1,
                    progress=False,
                    reverse=True):
        """
        Iteratively sample data from model from t=T to t=0.
        T is specified as the length of betas given to __init__().

        Args:
            model (collable): 
                A callable that takes x_t and t and predict noise (and sigma related parameters).
            shape (list like object): A data shape.
            sampler (callable): A function to sample x_{t-1} given x_{t} and t. Typically, self.p_sample or self.ddim_sample.
            x_init (numpy.ndarray or nn.variable): 
                An initial (noisy) image.
                If reverse is True, x_init should be x_T of a noisy image.
                If reverse is False, x_init should be x_0 of a clean image.
            interval (int): 
                If > 0, all intermediate results at every `interval` step will be returned as a list.
                e.g. if interval = 10, the predicted results at {10, 20, 30, ...} will be returned.
            progress (boolean): If True, tqdm will be used to show the sampling progress.
            reverse (boolean): 
                If True, reverse sampling process (= generation process) will be performed.
                If False, forward sampling process (= diffusion process) will be performed.

        Returns:
            - x_0 (nn.Variable): the final sampled result of x_0
            - samples (a list of nn.Variable): the sampled results at every `interval`
            - pred_x_starts (a list of nn.Variable): the predicted x_0 from each x_t at every `interval`: 
        """
        # setup timesteps
        T = self.num_timesteps
        indices = list(range(T))
        if reverse:
            indices = indices[::-1]

        samples = []
        pred_x_starts = []

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        # update model to enable respacing if needed
        def timestep_rescaled_model(x, t, **kwargs):
            return model(x, self._rescale_timestep(t), **kwargs)

        # make sure input_cond is already computed
        self._pre_compute_model_kwargs(model_kwargs)

        with nn.auto_forward():
            # setup initial image
            if x_init is None:
                if not reverse:
                    raise ValueError(
                        "x_init must be given for the forward process (reverse=False).")

                # case reverse=True
                if self.max_timesteps != self.t_start:
                    raise ValueError(
                        f"x_init must be given when the case with max_timesteps ({self.max_timesteps}) != t_start ({self.t_start})")

                # if T == t_start for the reverse process, we can rondamly sample an initial noise.
                x_init = F.randn(shape=shape)
            else:
                if isinstance(x_init, np.ndarray):
                    x_init = nn.Variable.from_numpy_array(x_init)
                else:
                    assert isinstance(x_init, nn.Variable), \
                        "noise must be an instance of np.ndarray or nn.Variable, or None."
                    x_init.persistent = True
                assert x_init.shape == shape
            x_t = x_init

            cnt = 0
            for step in indices:
                t = F.constant(step, shape=(shape[0], ))

                x_t, pred_x_start = sampler(timestep_rescaled_model,
                                            x_t,
                                            t,
                                            model_kwargs=model_kwargs,
                                            no_noise=step == 0)
                x_t.persistent = True

                cnt += 1
                if dump_interval > 0 and cnt % dump_interval == 0:
                    samples.append((step, x_t))
                    pred_x_starts.append((step, pred_x_start))

        assert x_t.shape == shape
        return x_t, samples, pred_x_starts

    def p_sample_loop(self, *args, channel_last=False, classifier_free_guidance_weight=None, **kwargs):
        """
        Sample data from x_T ~ N(0, I) with p(x_{t-1}|x_{t}) proposed by "Denoising Diffusion Probabilistic Models".
        See self.sample_loop for more details about sampling process.

        """

        return self.sample_loop(*args,
                                sampler=partial(
                                    self.p_sample,
                                    channel_last=channel_last,
                                    classifier_free_guidance_weight=classifier_free_guidance_weight),
                                **kwargs)

    def ddim_sample_loop(self, *args, channel_last=False, classifier_free_guidance_weight=None, **kwargs):
        """
        Sample data from x_T ~ N(0, I) with p(x_{t-1}|x_{t}, x_{0}) proposed by "Denoising Diffusion Implicit Models".
        See self.sample_loop for more details about sampling process.

        """

        return self.sample_loop(*args,
                                sampler=partial(
                                    self.ddim_sample,
                                    eta=0.,
                                    channel_last=channel_last,
                                    classifier_free_guidance_weight=classifier_free_guidance_weight),
                                **kwargs)

    def ddim_rev_sample_loop(self, *args, channel_last=False, classifier_free_guidance_weight=None, **kwargs):
        """
        Sample latent at t=T by DDIM reverse sampling.
        """

        return self.sample_loop(*args,
                                sampler=partial(
                                    self.ddim_rev_sample,
                                    eta=0.,
                                    channel_last=channel_last,
                                    classifier_free_guidance_weight=classifier_free_guidance_weight),
                                reverse=False,
                                **kwargs
                                )

    def plms_sample_loop(self, model, shape, *,
                         channel_last=False,
                         x_init=None,
                         model_kwargs=None,
                         dump_interval=-1,
                         progress=False,
                         classifier_free_guidance_weight=None):
        """
        Sample data from x_T ~ N(0, I) by "Pseudo Numerical Methods for Diffusion Models on Manifolds".
        Iteratively sample data from model from t=T to t=0.
        T is specified as the length of betas given to __init__().

        Args:
            model (collable): 
                A callable that takes x_t and t and predict noise (and sigma related parameters).
            shape (list like object): A data shape.
            channel_last (boolean): If True, the data shape is assumed to represent NHWC.
            noise (collable): A noise generator. If None, F.randn(shape) will be used.
            x_init (nn.Variable): 
                A reference image for x_0. If given, the first noisy image is created by q_sample(x_start, 0, noise=noise). 
            interval (int): 
                If > 0, all intermediate results at every `interval` step will be returned as a list.
                e.g. if interval = 10, the predicted results at {10, 20, 30, ...} will be returned.
            progress (boolean): If True, tqdm will be used to show the sampling progress.
            classifier_free_guidance_weight (float): A weight for classifier-free guidance.

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

        # update model to enable respacing if needed
        def timestep_rescaled_model(x, t, **kwargs):
            return model(x, self._rescale_timestep(t), **kwargs)

        # make sure input_cond is already computed
        self._pre_compute_model_kwargs(model_kwargs)

        with nn.auto_forward():
            # setup initial image
            if x_init is None:
                # case reverse=True
                if self.max_timesteps != self.t_start:
                    raise ValueError(
                        f"x_init must be given when the case with max_timesteps ({self.max_timesteps}) != t_start ({self.t_start})")

                # if T == t_start for the reverse process, we can rondamly sample an initial noise.
                x_init = F.randn(shape=shape)
            else:
                if isinstance(x_init, np.ndarray):
                    x_init = nn.Variable.from_numpy_array(x_init)
                else:
                    assert isinstance(x_init, nn.Variable), \
                        "noise must be an instance of np.ndarray or nn.Variable, or None."
                assert x_init.shape == shape
            x_t = x_init

            cnt = 0
            old_eps = []
            for step in indices:
                t = F.constant(step, shape=(shape[0], ))

                preds = self.p_mean_var(timestep_rescaled_model, x_t, t,
                                        model_kwargs=model_kwargs,
                                        clip_denoised=True,
                                        channel_last=channel_last,
                                        classifier_free_guidance_weight=classifier_free_guidance_weight)

                pred_noise = self.predict_noise_from_xstart(
                    x_t, t, preds.xstart)

                if len(old_eps) == 0:
                    pred_noise_prime = pred_noise
                elif len(old_eps) == 1:
                    pred_noise_prime = (3 * pred_noise - old_eps[-1]) / 2
                elif len(old_eps) == 2:
                    pred_noise_prime = (
                        23 * pred_noise - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
                elif len(old_eps) == 3:
                    pred_noise_prime = (
                        55 * pred_noise - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
                old_eps.append(pred_noise)
                if len(old_eps) > 3:
                    old_eps.pop(0)

                alpha_bar_prev = self._extract(
                    self.alphas_cumprod_prev, t, x_t.shape)
                with context_scope("float"):
                    # re-predict x_0 using pred_noise_prime
                    pred_x_start = self.predict_xstart_from_noise(
                        x_t=x_t, t=t, noise=pred_noise_prime)
                    x_t = pred_x_start * \
                        sqrt(alpha_bar_prev) + \
                        sqrt(1 - alpha_bar_prev) * pred_noise_prime

                cnt += 1
                if dump_interval > 0 and cnt % dump_interval == 0:
                    samples.append((step, x_t.d.copy()))
                    pred_x_starts.append((step, pred_x_start.d.copy()))

        assert x_t.shape == shape
        return x_t, samples, pred_x_starts

    def dpm2_sample_loop(self, model, shape, *,
                         channel_last=False,
                         x_init,
                         model_kwargs=None,
                         dump_interval=-1,
                         progress=False,
                         classifier_free_guidance_weight=None):
        """
        Sample data from x_T ~ N(0, I) by "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps".
        Iteratively sample data from model from t=T to t=0 by using DPM-solver-2.
        T is specified as the length of betas given to __init__().

        Args:
            model (collable): 
                A callable that takes x_t and t and predict noise (and sigma related parameters).
            shape (list like object): A data shape.
            channel_last (boolean): If True, the data shape is assumed to represent NHWC.
            noise (collable): A noise generator. If None, F.randn(shape) will be used.
            x_init (nn.Variable): 
                A reference image for x_0. If given, the first noisy image is created by q_sample(x_start, 0, noise=noise). 
            interval (int): 
                If > 0, all intermediate results at every `interval` step will be returned as a list.
                e.g. if interval = 10, the predicted results at {10, 20, 30, ...} will be returned.
            progress (boolean): If True, tqdm will be used to show the sampling progress.
            classifier_free_guidance_weight (float): A weight for classifier-free guidance.

        Returns:
            - x_0 (nn.Variable): the final sampled result of x_0
            - samples (a list of nn.Variable): the sampled results at every `interval`
            - pred_x_starts (a list of nn.Variable): the predicted x_0 from each x_t at every `interval`: 
        """

        def _cont_log_alpha_t(t):
            if self.beta_strategy == "linear":
                b0, b1 = 0.0001*self.max_timesteps, 0.02*self.max_timesteps
                return -(b1-b0)/4*math.pow(t, 2) - b0/2*t
            elif self.beta_strategy == "cosine":
                s = 0.008
                return math.log(math.cos(math.pi/2*(t+s)/(1+s))) - math.log(math.cos(math.pi/2*s/(1+s)))
            else:
                raise NotImplementedError()

        def _cont_log_sigma_t(t):
            return 0.5 * math.log(1 - math.exp(2*_cont_log_alpha_t(t)))

        def _cont_sigma_t(t):
            return math.sqrt(1 - math.exp(2*_cont_log_alpha_t(t)))

        def _cont_lambda_t(t):
            return _cont_log_alpha_t(t) - _cont_log_sigma_t(t)

        def _cont_t_lambda(lam):
            if self.beta_strategy == "linear":
                b0, b1 = 0.0001*self.max_timesteps, 0.02*self.max_timesteps
                return 2*math.log1p(math.exp(-2*lam)) / (math.sqrt(b0*b0+2*(b1-b0)*math.log1p(math.exp(-2*lam)))+b0)
            elif self.beta_strategy == "cosine":
                s = 0.008
                f_lambda = -0.5 * math.log1p(math.exp(-2*lam))
                return 2*(1+s)/math.pi * math.acos(math.exp(f_lambda + math.log(math.cos(math.pi*s/2/(1+s))))) - s
            else:
                raise NotImplementedError()

        def _t_cont_to_disc(t):
            # Type-1 described in the appendix of the paper
            return self.max_timesteps * max([0, t - 1/self.max_timesteps])

        def _pred_noise(model, x, t, channel_last, model_kwargs):
            pred = model(x, t, **model_kwargs)
            if self.model_var_type == ModelVarType.LEARNED_RANGE:
                pred_noise, _ = chunk(
                    pred, num_chunk=2, axis=3 if channel_last else 1)
            else:
                # Model only predicts noise
                pred_noise = pred

            # classifier-free guidance
            if classifier_free_guidance_weight is not None and classifier_free_guidance_weight > 0:
                model_kwargs_uncond = model_kwargs.copy()
                model_kwargs_uncond["cond_drop_rate"] = 1
                pred_uncond = model(x_t, t, **model_kwargs_uncond)

                if self.model_var_type == ModelVarType.LEARNED_RANGE:
                    pred_noise_uncond = pred_uncond[...,
                                                    :3] if channel_last else pred_uncond[:, :3]
                else:
                    pred_noise_uncond = pred_uncond

                # (1 + w) * eps(t, c) - w * eps(t)
                w = classifier_free_guidance_weight
                pred_noise = (1 + w) * pred_noise - w * pred_noise_uncond

            return pred_noise

        T = self.num_timesteps
        lam0 = _cont_lambda_t(1e-3)
        if self.beta_strategy == "cosine":
            lam1 = _cont_lambda_t(0.9946 * self.t_start / self.max_timesteps)
        else:
            lam1 = _cont_lambda_t(1.0 * self.t_start / self.max_timesteps)
        lam_cont_list = np.linspace(lam1, lam0, T+1, dtype=np.float64).tolist()
        t_cont_list = [_cont_t_lambda(lam_cont_list[i]) for i in range(T+1)]
        t_cont_next_list = t_cont_list[1:]
        t_cont_list = t_cont_list[:-1]

        samples = []
        pred_x_starts = []

        if progress:
            from tqdm.auto import tqdm
            t_cont_list = tqdm(t_cont_list)

        # make sure input_cond is already computed
        self._pre_compute_model_kwargs(model_kwargs)

        with nn.auto_forward():
            if x_init is None:
                # case reverse=True
                if self.max_timesteps != self.t_start:
                    raise ValueError(
                        f"x_init must be given when the case with max_timesteps ({self.max_timesteps}) != t_start ({self.t_start})")

                # if T == t_start for the reverse process, we can rondamly sample an initial noise.
                x_init = F.randn(shape=shape)
            else:
                if isinstance(x_init, np.ndarray):
                    x_init = nn.Variable.from_numpy_array(x_init)
                else:
                    assert isinstance(x_init, nn.Variable), \
                        "noise must be an instance of np.ndarray or nn.Variable, or None."
                assert x_init.shape == shape
            x_t = x_init

            if model_kwargs is None:
                model_kwargs = {}
            for i, t_cont in enumerate(t_cont_list):
                with context_scope("float"):
                    t = F.constant(_t_cont_to_disc(
                        t_cont), shape=(shape[0], ))
                    s_cont = _cont_t_lambda(
                        (_cont_lambda_t(t_cont) + _cont_lambda_t(t_cont_next_list[i])) / 2.0)
                    s = F.constant(_t_cont_to_disc(
                        s_cont), shape=(shape[0], ))
                    h = _cont_lambda_t(
                        t_cont_next_list[i]) - _cont_lambda_t(t_cont)
                    expm1_h = F.constant(math.expm1(
                        h), shape=(shape[0], 1, 1, 1))
                    expm1_h2 = F.constant(math.expm1(
                        h/2), shape=(shape[0], 1, 1, 1))

                pred_noise = _pred_noise(
                    model, x_t, t, channel_last, model_kwargs)

                with context_scope("float"):
                    u = math.exp(_cont_log_alpha_t(s_cont) -
                                 _cont_log_alpha_t(t_cont)) * x_t
                    u = u - _cont_sigma_t(s_cont) * expm1_h2 * pred_noise

                pred_noise = _pred_noise(
                    model, u, s, channel_last, model_kwargs)

                with context_scope("float"):
                    x_t = math.exp(_cont_log_alpha_t(
                        t_cont_next_list[i]) - _cont_log_alpha_t(t_cont)) * x_t
                    x_t = x_t - \
                        _cont_sigma_t(
                            t_cont_next_list[i]) * expm1_h * pred_noise

                if dump_interval > 0 and i % dump_interval == 0:
                    samples.append((t_cont, x_t.d.copy()))
                    # compute pred_x_start
                    pred_x_start = math.exp(-_cont_log_alpha_t(t_cont)) * \
                        x_t - math.exp(-_cont_lambda_t(t_cont)
                                       ) * pred_noise
                    pred_x_starts.append((t_cont, pred_x_start.d.copy()))

        assert x_t.shape == shape
        return x_t, samples, pred_x_starts
