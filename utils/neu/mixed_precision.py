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

        loss = model(...)
        solver = S.Sgd()
        solver.set_parameters(nn.get_parameters())

        for i in range(max_iter):
            loss.forward(...)

            mpm.zero_grad(solver)

            mpm.backward(loss)

            # returns false if Nan or inf occurs due to overflow.
            # In this case, solver.update() *is not* actually performed internally.
            is_updated = mpm.update(solver)  

            if is_updated:
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

        self._is_scaled = False

    def zero_grad(self, solver: Solver):
        solver.zero_grad()

    @staticmethod
    def check_grad_overflow(solver: Solver) -> bool:
        # solver.check_inf_or_nan_grad() returns True if inf or nan is detected.
        return solver.check_inf_or_nan_grad()

    def backward(self, loss: Variable, solver: Solver, **kwargs):
        """
        Return True if overflow.
        """
        if not self.use_fp16:
            loss.backward(**kwargs)
            return False

        loss.backward(grad=2 ** self.log_loss_scale, **kwargs)
        self._is_scaled = False

        if self.check_grad_overflow(solver):
            # skip update and make loss scale smaller
            logger.info("[MixedPrecisionManager] Detects overflow. "
                        "Update log_loss_scale from {:.3g} to {:.3g}.".format(self.log_loss_scale, self.log_loss_scale - self.dec_scale_factor))
            self.log_loss_scale -= self.dec_scale_factor
            solver.zero_grad()
            return True

        return False

    def update(self, solver: Solver, *, clip_grad=-1, **kwargs) -> bool:
        """
        Return True if overflow.
        """
        if not self.use_fp16:
            solver.update(**kwargs)
            return False

        solver.scale_grad(1. / (2 ** self.log_loss_scale))

        if self.check_grad_overflow(solver):
            # skip update and make loss scale smaller
            logger.info("[MixedPrecisionManager] Detects overflow. "
                        f"Update log_loss_scale from {self.log_loss_scale} to {self.log_loss_scale - self.dec_scale_factor}.")
            self.log_loss_scale -= self.dec_scale_factor
            solver.zero_grad()
            return True

        if clip_grad > 0:
            solver.clip_grad_by_norm(clip_grad)

        solver.update(**kwargs)

        self.log_loss_scale += self.inc_scale_factor
        return False
