import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from utils.misc import get_mask
from utils.ops import masked_fill

from .adaptor import Adaptor
from .decoder import Decoder
from .encoder import Encoder
from .module import Module


class PostNet(Module):
    r"""The PostNet implementation.

    Five 1D-convolution with 512 channels and kernel size 5.

    Args:
        n_mels (int, optional): Number of mel banks. Defaults to 80.
        n_embed (int, optional): Embedding dimension. Defaults to 512.
        kernel_size (int, optional): Kernel size. Defaults to 5.
        n_convs (int, optional): Number of convolutions. Defaults to 5.
    """

    def __init__(self, n_mels=80, n_embed=512, kernel_size=5, n_convs=5):
        self.n_mels = n_mels
        self.n_embed = n_embed
        self.kernel_size = kernel_size
        self.n_convs = n_convs

    def call(self, x, mask=None):
        """Return a mel-spectrogram.

        Args:
            x (nn.Variable): A mel-spectrogram of shape (B, L, n_mels).
            mask (nn.Variable, optional): Mask vairable of shape
                (B, max_len, 1). Defaults to None.

        Returns:
            nn.Variable: The resulting spectrogram of shape (B, L, n_mels).
        """
        in_channels = [self.n_embed] * (self.n_convs - 1) + [self.n_mels]
        x = F.transpose(x, (0, 2, 1))

        for i, channels in enumerate(in_channels):
            with nn.parameter_scope(f'filter_{i}'):
                x = PF.convolution(
                    x, channels, (self.kernel_size,),
                    pad=((self.kernel_size - 1) // 2,)
                )
                x = PF.batch_normalization(x, batch_stat=self.training)
                if i < len(in_channels) - 1:
                    x = F.tanh(x)
                if self.training:
                    x = F.dropout(x, 0.5)

        x = F.transpose(x, (0, 2, 1))

        return x


class FastSpeech2(Module):
    r"""Implemenentation of Fastspeech2.

    Args:
        hp (Hparams): Hyper-parameters.
    """

    def __init__(self, hp):
        self.hp = hp
        self.encoder = Encoder(hp)
        self.adaptor = Adaptor(hp)
        self.decoder = Decoder(hp)
        self.postnet = PostNet(hp.n_mels)

    def call(self, x, len_phone, target_pitch=None, target_energy=None,
             target_duration=None,  control_pitch=1.0,
             control_energy=1.0, control_duration=1.0):
        r"""Returns mel-spectrograms from text.

        Args:
            x (nn.Variable): Input variable of shape (B, max_len_phone).
            len_phone (nn.Variable): Length variable for phonemes of shape
                (B, 1).
            target_pitch (nn.Variable, optional): Target pitch variable of
                shape (B, max_len_phone). Defaults to None.
            target_energy (nn.Variable, optional): Target energy variable of
                shape (B, max_len_phone). Defaults to None.
            target_duration (nn.Variable, optional): Target duration variable
                of shape (B, max_len_phone). Defaults to None.
            control_pitch (float, optional): Scale controling pitch.
                Defaults to 1.0.
            control_energy (float, optional): Scale controling energy.
                Defaults to 1.0.
            control_duration (float, optional): Scale controling duration.
                Defaults to 1.0.

        Returns:
            nn.Variable: Mel-spectrogram output from decoder of
                shape (B, max_len_mel, n_mels).
            nn.Variable: Mel-spectrogram output from postnet of
                shape (B, max_len_mel, n_mels).
            nn.Variable: Log duration of shape (B, max_len_phone).
            nn.Variable: Pitch prediction of shape (B, max_len_phone).
            nn.Variable: Evergy prediction of shape (B, max_len_phone).
        """
        hp = self.hp

        if len_phone is not None:
            mask_phone = get_mask(len_phone, hp.max_len_phone)

        with nn.parameter_scope("encoder"):
            out = self.encoder(x, mask_phone)

        with nn.parameter_scope("adaptor"):
            (out, log_duration, pred_pitch,
             pred_energy, target_duration) = self.adaptor(
                out, mask_phone=mask_phone, target_pitch=target_pitch,
                target_energy=target_energy, target_duration=target_duration,
                control_pitch=control_pitch, control_energy=control_energy,
                control_duration=control_duration
            )

        len_mel = F.sum(target_duration, axis=1, keepdims=True)
        mask_mel = get_mask(len_mel, hp.max_len_mel)

        with nn.parameter_scope("decoder"):
            out = self.decoder(out, mask_mel)
            with nn.parameter_scope("linear_projection"):
                out = PF.affine(out, hp.n_mels, base_axis=2)
                out = masked_fill(out, mask_mel, hp.lower_bound)

        with nn.parameter_scope("postnet"):
            out_pos = out + self.postnet(out)
            out_pos = masked_fill(out_pos, mask_mel, hp.lower_bound)

        return out, out_pos, log_duration, pred_pitch, pred_energy
