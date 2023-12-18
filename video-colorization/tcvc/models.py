import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from utils import get_gan_loss


def namescope_decorator(scope):
    def wrapper1(builder_function):

        def wrapper2(*args, **kwargs):
            with nn.parameter_scope(scope):
                return builder_function(*args, **kwargs)

        return wrapper2

    return wrapper1


def get_symmetric_padwidth(pad, dims=2, channel_last=False):
    pad = (pad, ) * dims * 2

    if channel_last:
        pad += (0, 0)

    return pad


class BaseGenerator(object):
    def __init__(self, padding_type="reflect", channel_last=False):
        self.padding_type = padding_type
        self.channel_last = channel_last
        # currently deconv dose not support channel last.
        self.conv_opts = dict(w_init=I.NormalInitializer(0.02))
        # don't use adaptive parameter
        self.norm_opts = dict(no_scale=True, no_bias=True)

    def instance_norm_relu(self, x):
        # return F.relu(PF.layer_normalization(x, **self.norm_opts))
        return F.relu(PF.instance_normalization(x, **self.norm_opts))

    def residual_block(self, x, o_channels):
        pad_width = get_symmetric_padwidth(1, channel_last=self.channel_last)
        #spectral_norm_w = lambda w: PF.spectral_norm(w, dim=0)
        with nn.parameter_scope("residual_1"):
            
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            #h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts, apply_w=spectral_norm_w)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = self.instance_norm_relu(h)

        with nn.parameter_scope("residual_2"):
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            #h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts, apply_w=spectral_norm_w)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = PF.instance_normalization(h, **self.norm_opts)

        return x + h

    def residual_block_feat(self, x, feat, o_channels):
        pad_width = get_symmetric_padwidth(1, channel_last=self.channel_last)
        #spectral_norm_w = lambda w: PF.spectral_norm(w, dim=0)
        with nn.parameter_scope("residual_1"):
            
            h = F.concatenate(x, feat, axis=1)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            #h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts, apply_w=spectral_norm_w)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = self.instance_norm_relu(h)

        with nn.parameter_scope("residual_2"):
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            #h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts, apply_w=spectral_norm_w)
            h = PF.convolution(h, o_channels, (3, 3), **self.conv_opts)
            h = PF.instance_normalization(h, **self.norm_opts)

        return x + h

    def residual_loop(self, x, o_channels, num_layers):
        h = x
        for i in range(num_layers):
            with nn.parameter_scope("layer_{}".format(i)):
                h = self.residual_block(h, o_channels)

        return h

    def residual_loop_feat(self, x, feat, o_channels, num_layers):
        h = x
        for i in range(num_layers):
            with nn.parameter_scope("layer_{}".format(i)):
                h = self.residual_block_feat(h, feat, o_channels)

        return h

class TCVCGenerator(BaseGenerator):
    def __init__(self, padding_type="reflect", n_outputs=3):
        super(TCVCGenerator, self).__init__(padding_type=padding_type)
        self.n_outputs = n_outputs

    @namescope_decorator("frontend")
    def front_end(self, x, channels):
        with nn.parameter_scope("first_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, channels[0], (7, 7), **self.conv_opts)
            h = self.instance_norm_relu(h)

        for i, channel in enumerate(channels[1:]):
            with nn.parameter_scope("down_sample_layer_{}".format(i)):
                h = PF.convolution(h, channel, (4, 4), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        return h

    @namescope_decorator("residual")
    def residual(self, x, num_layers):
        return self.residual_loop(x, 256, num_layers)

    @namescope_decorator("backend")
    def back_end(self, x, channels):
        h = x
        for i, channel in enumerate(channels):
            with nn.parameter_scope("up_sample_layer_{}".format(i)):
                h = PF.deconvolution(h, channel, (4, 4), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        last_feat = h

        with nn.parameter_scope("last_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, self.n_outputs, (7, 7), **self.conv_opts)
            h = F.tanh(h)

        return h, last_feat

    def __call__(self, x, channels, downsample_input=False, n_residual_layers=8):
        if downsample_input:
            x = F.average_pooling(
                x, (3, 3), (2, 2), pad=(1, 1), including_pad=False)

        with nn.parameter_scope("generator/tcvc"):
            h = self.front_end(x, channels)
            h = self.residual(h, n_residual_layers)
            out, feat = self.back_end(h, channels[-2::-1])

        return out, feat


class TCVCGenerator_feat(BaseGenerator):
    def __init__(self, padding_type="reflect", n_outputs=3):
        super(TCVCGenerator_feat, self).__init__(padding_type=padding_type)
        self.n_outputs = n_outputs

    @namescope_decorator("frontend")
    def front_end(self, x, channels):
        with nn.parameter_scope("first_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(x, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, channels[0], (7, 7), **self.conv_opts)
            h = self.instance_norm_relu(h)

        for i, channel in enumerate(channels[1:]):
            with nn.parameter_scope("down_sample_layer_{}".format(i)):
                h = PF.convolution(h, channel, (4, 4), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        return h

    @namescope_decorator("residual")
    def residual(self, x, feat, num_layers):
        return self.residual_loop_feat(x, feat, 256, num_layers)

    @namescope_decorator("backend")
    def back_end(self, x, channels):
        h = x
        for i, channel in enumerate(channels):
            with nn.parameter_scope("up_sample_layer_{}".format(i)):
                h = PF.deconvolution(h, channel, (4, 4), stride=(
                    2, 2), pad=(1, 1), **self.conv_opts)
                h = self.instance_norm_relu(h)

        last_feat = h

        with nn.parameter_scope("last_layer"):
            pad_width = get_symmetric_padwidth(
                3, channel_last=self.channel_last)
            h = F.pad(h, pad_width=pad_width, mode=self.padding_type)
            h = PF.convolution(h, self.n_outputs, (7, 7), **self.conv_opts)
            h = F.tanh(h)

        return h, last_feat

    def __call__(self, x, feat, channels, downsample_input=False, n_residual_layers=8):
        if downsample_input:
            x = F.average_pooling(
                x, (3, 3), (2, 2), pad=(1, 1), including_pad=False)

        with nn.parameter_scope("generator/tcvc_feat"):
            h = self.front_end(x, channels)
            h = self.residual(h, feat, n_residual_layers)
            out, feat = self.back_end(h, channels[-2::-1])

        return out, feat

