import nnabla as nn
import nnabla.functions as F
import numpy as np
from nnabla.function import PythonFunction


class Regulator(PythonFunction):
    def __init__(self, max_len, ctx):
        super(Regulator, self).__init__(ctx)
        self.max_len = max_len

    def name(self):
        return "Regulator"

    def min_outputs(self):
        return 1

    def setup_impl(self, inputs, outputs):
        o0 = outputs[0]
        b, _, d = inputs[0].shape
        o0.reset_shape((b, self.max_len, d), True)

    def forward_impl(self, inputs, outputs):
        inp = inputs[0].data
        len_tar = inputs[1].data
        y = outputs[0].data

        np_inp = inp.data.copy()
        np_len = len_tar.data.copy().astype(int)

        out = list()
        for b in range(np_inp.shape[0]):
            o = list()
            for x, l in zip(np_inp[b], np_len[b]):
                r = np.tile(x, (l, 1))
                o.append(r)
            o = np.concatenate(o)
            if o.shape[0] < self.max_len:
                o = np.pad(o, ((0, self.max_len - o.shape[0]), (0, 0)))
            o = o[:self.max_len, :]
            out.append(o)
        y.data = np.stack(out)

    def backward_impl(self, inputs, outputs, propagate_down, accum):
        # Grads of inputs and outputs
        dx0 = inputs[0].grad
        dy = outputs[0].grad

        np_len = inputs[1].data.data.copy().astype(int)
        np_gy = dy.data.copy()
        np_gx = np.zeros_like(dx0.data)

        for b in range(np_len.shape[0]):
            s = 0
            for i, l in enumerate(np_len[b]):
                np_gx[b][i] = np.sum(np_gy[b][s:s+l], axis=0)
                s += l

        grad = nn.NdArray.from_numpy_array(np_gx)
        # backward w.r.t. x0
        if propagate_down[0]:
            if accum[0]:
                dx0 += grad
            else:
                dx0.copy_from(grad)

    def grad_depends_output_data(self, i, o):
        return True

    def grad_depends_input_data(self, i, j):
        return True


def masked_fill(x, mask=None, value=0, eps=1e-6, ctx=None):
    r"""Fills elements of a variable with zero where mask is True. The shape of
    mask will be broadcastable with the shape of the underlying variable.

    Args:
        x (nn.Variable): Input variable.
        mask (nn.Variable): Mask variable. Values being different than zero
            are considered as masked.
        value (number): Value to replace masked positions.
        eps (float, optional): Toletance. Default to 1e-6.
        ctx: Context.

    Returns:
        nn.Variable: Output variable.
    """
    if mask is not None:
        mask = F.less_equal_scalar(F.abs(mask), eps)
        mask = F.broadcast(mask, x.shape)
        x = F.where(mask, x, F.constant(value, shape=x.shape))
    return x


def bucketize(x, bins, ctx=None):
    r"""Returns the indices of the bins to which each value in input belongs.

    Args:
        x (nn.Variable): Input variable.
        bins (nn.Variable): Variable containing bins.
        ctx (optional): Context. Defaults to None.

    Returns:
        nn.Variable: Output variable of indices, of same shape as x.
    """
    v = F.searchsorted(bins, x)
    v.need_grad = False

    return v


def regulate(x, len_tar, max_len=0, ctx=None):
    """Regulates the shape according to target length.

    Args:
        x (nn.Variable): Input variable of shape (B, L, D).
        len_tar (nn.Variable): Target length of shape (B, L).
        max_len (int, optional): Maximum length to pad in the output.
            Defaults to 0.
        ctx (optional): Context. Defaults to None.

    Returns:
        nn.Variable: Output variable of shape (B, max_len, D).
    """
    func = Regulator(max_len, ctx)
    return func(x, len_tar)
