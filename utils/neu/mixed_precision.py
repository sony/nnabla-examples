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

from nnabla.solvers import Solver
from nnabla import Variable
from nnabla.logger import logger


DEFAULT_INITIAL_LOG_LOSS_SCALE = 20  # 2 to the power of 20


class MixedPrecisionManager(object):
    """
    Helper class for mixed precision training.
    This class automatically scales loss value to prevent overflow during backward.
    See https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html to know the fundametal about mixed training training.

    Example:

    .. code-block:: python
        from neu.mixed_precision import MixedPrecisionManager

        mpm = MixedPrecisionManager(use_fp16=True)

        x = nn.Variable(...)
        loss = model(x, ...)
        solver = S.Sgd()
        solver.set_parameters(nn.get_parameters())

        # communicator
        comm = ...

        # data iterator
        data = ...

        for i in range(max_iter):
            solver.zero_grad()

            for accum in range(accum_cnt):
                x.d = data.next()
                loss.forward(...)
                mpm.backward(loss, ...)

            # When model parallel, do all_reduce here
            comm.all_reduce(...)

            if mpm.is_grad_overflow(solver):
                # When overflow happens, do not update parameter
                # If detecting overflow, mpm decreases loss scale.
                continue 

            # After calling solver.update, mpm increases loss scale.
            mpm.update(solver)

            # do postprocesses after parameter update like ema update.

    """

    def __init__(self,
                 use_fp16,
                 initial_log_loss_scale=DEFAULT_INITIAL_LOG_LOSS_SCALE,
                 inc_scale_factor=1e-3,
                 dec_scale_factor=1):
        self.use_fp16 = use_fp16
        self.log_loss_scale = initial_log_loss_scale
        self.inc_scale_factor = inc_scale_factor
        self.dec_scale_factor = dec_scale_factor

    @property
    def loss_scale(self):
        if not self.use_fp16:
            return 1.0

        return 2 ** self.log_loss_scale

    def is_grad_overflow(self, solver: Solver) -> bool:
        # solver.check_inf_or_nan_grad() returns True if inf or nan is detected.
        if solver.check_inf_or_nan_grad():
            # skip update and make loss scale smaller
            logger.info("[MixedPrecisionManager] Detects overflow. "
                        "Update log_loss_scale from {:.3g} to {:.3g}.".format(self.log_loss_scale, self.log_loss_scale - self.dec_scale_factor))
            self.log_loss_scale -= self.dec_scale_factor
            solver.zero_grad()
            return True

        return False

    def backward(self, loss: Variable, **kwargs):
        if not self.use_fp16:
            loss.backward(**kwargs)
            return

        loss.backward(grad=self.loss_scale, **kwargs)

    def scale_grad(self, solver):
        if not self.use_fp16:
            return

        solver.scale_grad(1. / self.loss_scale)

    def update(self, solver: Solver, *, clip_grad=None, **kwargs):
        if clip_grad is not None:
            assert isinstance(
                clip_grad, float), "clip_grad must be None or float."
            solver.clip_grad_by_norm(clip_grad)

        solver.update(**kwargs)

        if self.use_fp16:
            # increment loss scale
            self.log_loss_scale += self.inc_scale_factor
