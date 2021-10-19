# Copyright 2020,2021 Sony Corporation.
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

import numpy as np

import nnabla as nn
import nnabla.functions as F


##############################################
# classification/regression
##############################################
def sigmoid_ce(logits, value, mask=None, eps=1e-5):
    # sigmoid cross entropy and reduce_mean
    sce = F.sigmoid_cross_entropy(
        logits, F.constant(val=value, shape=logits.shape))

    if mask is not None:
        assert sce.shape[:2] == mask.shape[:2]

        sce *= F.reshape(mask, sce.shape)

        return F.sum(sce) / (F.sum(mask) + eps)

    return F.mean(sce)


def softmax_ce(logits, targets, mask=None, eps=1e-5):
    # softmax cross entropy and reduce_mean
    assert logits.shape[:-1] == targets.shape[:-1]
    assert targets.shape[-1] == 1

    sce = F.softmax_cross_entropy(logits, targets)

    if mask is not None:
        assert sce.shape[:2] == mask.shape[:2]

        sce *= F.reshape(mask, sce.shape)

        return F.sum(sce) / (F.sum(mask) + eps)

    return F.mean(sce)


def mae(x, y, mask=None, eps=1e-5):
    # l1 distance and reduce mean
    ae = F.absolute_error(x, y)

    if mask is not None:
        assert ae.shape[:2] == mask.shape[:2]

        ae *= F.reshape(mask, ae.shape)

        return F.sum(ae) / (F.sum(mask) + eps)

    return F.mean(ae)


def mse(x, y, mask=None, eps=1e-5):
    # l2 distance and reduce mean
    se = F.squared_error(x, y)

    if mask is not None:
        assert se.shape[:2] == mask.shape[:2]

        se *= F.reshape(mask, se.shape)

        return F.sum(se) / (F.sum(mask) + eps)

    return F.mean(se)


###############################################
# likelihood-based losses
###############################################

def kl_snd(mu, logvar):
    # kl divergence from standard normal gaussian to the gaussian of given parameter, mu and logvar.
    # Namely, it computes D_kl(N(mu, exp(logvar)) || N(0, 1)).

    return F.sum(F.pow_scalar(mu, 2) + F.exp(logvar) - logvar - 1) / 2


"""
Functions below are ported from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/losses.py, 
and are re-implemented with nnabla.
"""


def kl_normal(mean1, logvar1, mean2, logvar2):
    # kl divergence between two gaussians.

    return 0.5 * (((mean1 - mean2) ** 2) * F.exp(-logvar2) + F.exp(logvar1 - logvar2) + logvar2 - logvar1 - 1.0)


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distibution function of the standard normal.
    """

    return 0.5 * (1.0 + F.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * F.pow_scalar(x, 3))))


def gaussian_log_likelihood(x, mean, logstd, orig_max_val=255):
    """
    Compute the log-likelihood of a Gaussian distribution for given data `x`.

    Args:
        x (nn.Variable): Target data. It is assumed that the values are ranged [-1, 1],
                         which are originally [0, orig_max_val].
        means (nn.Variable): Gaussian mean. Must be the same shape as x.
        logstd (nn.Variable): Gaussian log standard deviation. Must be the same shape as x.
        orig_max_val (int): The maximum value that x originally has before being rescaled.

    Return:
        A log probabilies of x in nats.
    """
    assert x.shape == mean.shape == logstd.shape
    centered_x = x - mean
    inv_std = F.exp(-logstd)
    half_bin = 1.0 / orig_max_val

    def clamp(val):
        # Here we don't need to clip max
        return F.clip_by_value(val, min=1e-12, max=1e8)

    # x + 0.5 (in original scale)
    plus_in = inv_std * (centered_x + half_bin)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    log_cdf_plus = F.log(clamp(cdf_plus))

    # x - 0.5 (in original scale)
    minus_in = inv_std * (centered_x - half_bin)
    cdf_minus = approx_standard_normal_cdf(minus_in)
    log_one_minus_cdf_minus = F.log(clamp(1.0 - cdf_minus))

    log_cdf_delta = F.log(clamp(cdf_plus - cdf_minus))

    log_probs = F.where(
        F.less_scalar(x, -0.999),
        log_cdf_plus,  # Edge case for 0. It uses cdf for -inf as cdf_minus.
        F.where(F.greater_scalar(x, 0.999),
                # Edge case for orig_max_val. It uses cdf for +inf as cdf_plus.
                log_one_minus_cdf_minus,
                log_cdf_delta  # otherwise
                )
    )

    assert log_probs.shape == x.shape
    return log_probs

##############################################
# GAN loss
##############################################


def ls_gan_loss(r_out, f_out):
    # todo: set constant arbitrary
    # D
    d_gan_real = F.mean(F.squared_error(r_out,
                                        F.constant(1., shape=r_out.shape)))
    d_gan_fake = F.mean(F.squared_error(f_out,
                                        F.constant(0., shape=f_out.shape)))

    # G
    g_gan = F.mean(F.squared_error(f_out,
                                   F.constant(1., shape=f_out.shape)))

    return d_gan_real, d_gan_fake, g_gan


def hinge_gan_loss(r_out, f_out):
    # D
    d_gan_real = F.mean(F.relu(1. - r_out))
    d_gan_fake = F.mean(F.relu(1. + f_out))

    # G
    g_gan = -1 * F.mean(f_out)

    return d_gan_real, d_gan_fake, g_gan


def get_gan_loss(type):
    gan_loss_dict = {
        "ls": ls_gan_loss,
        "hinge": hinge_gan_loss
    }

    gan_loss = gan_loss_dict.get(type, None)

    if gan_loss is None:
        raise ValueError("unsupported gan loss type: {}".format(type))

    return gan_loss


def vgg16_perceptual_loss(fake, real):
    '''VGG perceptual loss based on VGG-16 network.

    Assuming the values in fake and real are in [0, 255].

    Features are obtained from all ReLU activations of the first convolution
    after each downsampling (maxpooling) layer
    (including the first convolution applied to an image).
    '''
    from nnabla.models.imagenet import VGG16

    class VisitFeatures(object):
        def __init__(self):
            self.features = []
            self.relu_counter = 0
            self.features_at = set([0, 2, 4, 7, 10])

        def __call__(self, f):
            # print(f.name, end='')
            if not f.name.startswith('ReLU'):
                # print('')
                return
            if self.relu_counter in self.features_at:
                self.features.append(f.outputs[0])
                # print('*', end='')
            # print('')
            self.relu_counter += 1

    # We use VGG16 model instead of VGG19 because VGG19
    # is not in nnabla.models.
    vgg = VGG16()

    def get_features(x):
        o = vgg(x, use_up_to='lastconv')
        f = VisitFeatures()
        o.visit(f)
        return f

    with nn.parameter_scope("vgg16_loss"):
        fake_features = get_features(fake)
        real_features = get_features(real)

    volumes = np.array([np.prod(f.shape)
                        for f in fake_features.features], dtype=np.float32)
    weights = volumes[-1] / volumes
    return sum([w * F.mean(F.absolute_error(ff, fr)) for w, ff, fr in zip(weights, fake_features.features, real_features.features)])
