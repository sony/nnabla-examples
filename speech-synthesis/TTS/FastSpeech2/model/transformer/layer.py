import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from utils.ops import masked_fill

from ..module import Module


class FFTBlock(Module):
    r"""Feed-forward transformer block.

    Args:
        n_head (int): Number of heads
        n_hidden (int): Hidden dimension.
        conv_filter_size (int): Size of convolutional filter.
        conv_kernel_size (int): Kernel size of convolution.
        dropout (float): Dropout probability.
    """

    def __init__(self, n_head, n_hidden, conv_filter_size,
                 conv_kernel_size, dropout):
        self.sf_attn = MultiHeadAttention(n_head, n_hidden, dropout)
        self.pw_ffw = PositionwiseFeedForward(
            conv_filter_size, conv_kernel_size, dropout)

    def call(self, x, mask=None):
        r"""Compute a feed-forward transformer.

        Args:
            x (nn.Variable): Input encoding variable of (B, seq_len, dim).
            mask (nn.Variable): Mask variable of shape (B, 1, max_len).

        Returns:
            nn.Variable: Output variable.
        """
        x = self.sf_attn(x, x, x, mask=mask)
        x = self.pw_ffw(x)
        return x


class MultiHeadAttention(Module):
    r"""Multi-head attention module.

    Args:
        n_head (int): Number of heads
        n_hidden (int): Hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(self, n_head, n_hidden, dropout):
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.dropout = dropout

    def call(self, query, key, value, mask=None):
        r"""Compute multi-head attention.

        Args:
            query (nn.Variable): Query variable of shape (B, len_q, dim).
            key (nn.Variable): Key variable of shape (B, len_k, dim).
            value (nn.Variable): Value variable of shape (B, len_v, dim_v).
            mask (nn.Variable, optional): Mask variable of shape (B, 1, len_v).
                Defaults to None.

        Returns:
            nn.Variable: Output variable.
        """
        dim = self.n_hidden // self.n_head
        output = list()

        for i in range(self.n_head):
            with nn.parameter_scope(f"head_{i}"):
                Q = PF.affine(query, dim, with_bias=False,
                              base_axis=2, name="query")
                K = PF.affine(key, dim, with_bias=False,
                              base_axis=2, name="key")
                V = PF.affine(value, dim, with_bias=False,
                              base_axis=2, name="value")
                A = F.batch_matmul(Q, K, transpose_b=True) / np.sqrt(dim)
                score = F.softmax(masked_fill(A, mask, -np.inf), axis=2)
                output.append(F.batch_matmul(score, V))

        output = F.concatenate(*output, axis=2)
        with nn.parameter_scope("projection"):
            output = PF.affine(output, self.n_hidden, base_axis=2)

        if self.training and self.dropout > 0:
            output = F.dropout(output, self.dropout)

        with nn.parameter_scope("layer_norm"):
            output = PF.layer_normalization(output + value, batch_axis=(0, 1))

        return output


class PositionwiseFeedForward(Module):
    r"""Positionwise Feed Forward module.

    Args:
        conv_filter_size (int): Size of convolutional filter.
        conv_kernel_size (int): Kernel size of convolution.
        dropout (float): Dropout probability.
    """

    def __init__(self, conv_filter_size, conv_kernel_size, dropout):
        self.conv_filter_size = conv_filter_size
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout

    def call(self, x):
        r"""Positionwise feed forward.

        Args:
            x (nn.Variable): Input encoding variable of (B, seq_len, dim).

        Returns:
            nn.Variable: Output variable.
        """
        skip = x
        x = F.transpose(x, (0, 2, 1))
        with nn.parameter_scope("first_conv"):
            x = PF.convolution(
                x, self.conv_filter_size,
                kernel=(self.conv_kernel_size,),
                pad=((self.conv_kernel_size - 1) // 2, )
            )
            x = F.relu(x)

        with nn.parameter_scope("second_conv"):
            x = PF.convolution(
                x, skip.shape[-1],
                kernel=(1,),
            )
            x = F.transpose(x, (0, 2, 1))

        if self.training and self.dropout > 0:
            x = F.dropout(x, self.dropout)

        with nn.parameter_scope("layer_norm"):
            x = PF.layer_normalization(x + skip, batch_axis=(0, 1))

        return x
