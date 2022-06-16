'''
Forked from https://github.com/sony/nnabla-examples/blob/master/utils/neu/learning_rate_scheduler.py
'''

import nnabla


class BaseLearningRateScheduler(object):

    def __init__(self):
        self._iter = 0
        self._epoch = 0

    def set_iter_per_epoch(self, it):
        pass

    def set_epoch(self, epoch):
        '''Set current epoch number.
        '''
        self._epoch = epoch

    def get_lr_and_update(self):
        '''
        Get current learning rate and update itereation count.
        The iteration count is calculated by how many times this method is called.
        Returns: Current learning rate
        '''
        lr = self.get_lr()
        self._iter += 1
        return lr

    def get_lr(self):
        '''
        Get current learning rate according to the schedule.
        '''
        return self._get_lr(self._epoch, self._iter)

    def _get_lr(self, current_epoch, current_iter):
        '''
        Get learning rate by current iteration.
        Args:
            current_epoch(int): Epoch count.
            current_iter(int):
                Current iteration count from the beginning of training.
        Note:
            A derived class must override this method. 
        '''
        raise NotImplementedError('')


class IterCosineLearningRateScheduler(BaseLearningRateScheduler):
    '''
    ** Modified `EpochCosineLearningRateScheduler` **

    Cosine Annealing Decay with warmup.
    The learning rate gradually increases linearly towards `base_lr` during
    `warmup_iters`, then gradually decreases with cosine decay towards 0 for
    `iters - warmup_iters`.
    Args:
        base_lr (float): Base learning rate
        iters (int): See description above.
        warmup_iters (int): It performs warmup during this period.
    '''

    def __init__(self, base_lr, iters, warmup_iters=5):

        # https://github.com/sony/nnabla/blob/master/python/src/nnabla/utils/learning_rate_scheduler.py
        from nnabla.utils.learning_rate_scheduler import CosineScheduler
        super().__init__()
        self.base_lr = base_lr
        self.iters = iters  # total iterations for training
        self.warmup_iters = warmup_iters
        self.cosine = CosineScheduler(
            self.base_lr, self.iters - self.warmup_iters)  # (init_lr, max_iter)

    def _get_lr(self, current_epoch, current_iter):

        # Warmup
        if current_iter < self.warmup_iters:
            return self.base_lr * (current_iter + 1) / self.warmup_iters

        # Cosine decay
        return self.cosine.get_learning_rate(
            current_iter - self.warmup_iters)
