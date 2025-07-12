# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
from .misc import DictInterfaceFactory

# Creater function
_lr_sched_factory = DictInterfaceFactory()


def create_learning_rate_scheduler(cfg):
    '''
    Create a learning rate scheduler from config.

    Args:
        cfg (dict-like object):
            It must contain ``scheduler_type`` to specify a learning rate scheduler class.

    Returns:
        Learning rate scheduler object.

    Example:

    With the following yaml file (``example.yaml``),

    .. code-block:: yaml

        learning_rate_scheduler:
            scheduler_type: EpochStepLearningRateScheduler
            base_lr: 1e-2
            decay_at: [40, 65]
            decay_rate: 0.1
            power: 1  # Ignored in EpochStepLearningRateScheduler

    you can create a learning rate scheduler class as following.

    .. code-block:: python

        from neu.yaml_wrapper import read_yaml
        cfg = read_yaml('example.yaml)

        lr_sched = create_learning_rate_scheduler(cfg.learning_rate_scheduler)

    '''

    return _lr_sched_factory.call(cfg.scheduler_type, cfg)


class BaseLearningRateScheduler(object):
    '''
    Base class of Learning rate scheduler.

    This gives a current learning rate according to a scheduling logic
    implemented as a method `_get_lr` in a derived class. It internally
    holds the current epoch and the current iteration to calculate a
    scheduled learning rate. You can get the current learning rate by
    calling `get_lr`. You have to set the current epoch which will be
    used in `_get_lr` by manually calling  `set_epoch(self, epoch)`
    while it updates the current iteration when you call
    `get_lr_and_update`.

    Example:

    .. code-block:: python
        class EpochStepLearningRateScheduler(BaseLearningRateScheduler):
            def __init__(self, base_lr, decay_at=[30, 60, 80], decay_rate=0.1, warmup_epochs=5):
                self.base_learning_rate = base_lr
                self.decay_at = np.asarray(decay_at, dtype=np.int)
                self.decay_rate = decay_rate
                self.warmup_epochs = warmup_epochs

            def _get_lr(self, current_epoch, current_iter):
                # This scheduler warmups and decays using current_epoch
                # instead of current_iter
                lr = self.base_learning_rate
                if current_epoch < self.warmup_epochs:
                    lr = lr * (current_epoch + 1) / (self.warmup_epochs + 1)
                    return lr

                p = np.sum(self.decay_at <= current_epoch)
                return lr * (self.decay_rate ** p)

        def train(...):

            ...

            solver = Momentum()
            lr_sched = EpochStepLearningRateScheduler(1e-1)
            for epoch in range(max_epoch):
                lr_sched.set_epoch(epoch)
                for it in range(max_iter_in_epoch):
                    lr = lr_sched.get_lr_and_update()
                    solver.set_learning_rate(lr)
                    ...


    '''

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


@_lr_sched_factory.register
class EpochStepLearningRateScheduler(BaseLearningRateScheduler):
    '''
    Learning rate scheduler with step decay.

    Args:
        base_lr (float): Base learning rate
        decay_at (list of ints): It decays the lr by a factor of `decay_rate`.
        decay_rate (float): See above.
        warmup_epochs (int): It performs warmup during this period.
        legacy_warmup (bool):
            We add 1 in the denominator to be consistent with previous
            implementation.

    '''

    def __init__(self, base_lr, decay_at=[30, 60, 80], decay_rate=0.1, warmup_epochs=5, legacy_warmup=False):
        super().__init__()
        self.base_learning_rate = base_lr
        self.decay_at = np.asarray(decay_at, dtype=np.int)
        self.decay_rate = decay_rate
        self.warmup_epochs = warmup_epochs
        self.legacy_warmup_denom = 1 if legacy_warmup else 0

    def _get_lr(self, current_epoch, current_iter):
        lr = self.base_learning_rate
        # Warmup
        if current_epoch < self.warmup_epochs:
            lr = lr * (current_epoch + 1) \
                / (self.warmup_epochs + self.legacy_warmup_denom)
            return lr

        p = np.sum(self.decay_at <= current_epoch)
        return lr * (self.decay_rate ** p)


@_lr_sched_factory.register
class EpochCosineLearningRateScheduler(BaseLearningRateScheduler):
    '''
    Cosine Annealing Decay with warmup.

    The learning rate gradually increases linearly towards `base_lr` during
    `warmup_epochs`, then gradually decreases with cosine decay towards 0 for
    `epochs - warmup_epochs`.


    Args:
        base_lr (float): Base learning rate
        epochs (int): See description above.
        warmup_epochs (int): It performs warmup during this period.

    '''

    def __init__(self, base_lr, epochs, warmup_epochs=5):

        from nnabla.utils.learning_rate_scheduler import CosineScheduler
        super().__init__()
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.cosine = CosineScheduler(
            self.base_lr, self.epochs - self.warmup_epochs)

    def _get_lr(self, current_epoch, current_iter):

        # Warmup
        if current_epoch < self.warmup_epochs:
            return self.base_lr * (current_epoch + 1) / self.warmup_epochs

        # Cosine decay
        return self.cosine.get_learning_rate(
            current_epoch - self.warmup_epochs)


@_lr_sched_factory.register
class PolynomialLearningRateScheduler(BaseLearningRateScheduler):
    def __init__(self, base_lr, epochs, warmup_epochs=5, power=0.1):
        super().__init__()
        self.base_lr = base_lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.power = power
        self.iter_per_epoch = None
        self.max_iters = None
        self.warmup_iters = None
        self.poly = None

    def set_iter_per_epoch(self, it):
        from nnabla.utils.learning_rate_scheduler import PolynomialScheduler
        self.iter_per_epoch = it
        self.max_iters = it * self.epochs
        self.warmup_iters = it * self.warmup_epochs
        self.poly = PolynomialScheduler(
            self.base_lr, self.max_iters - self.warmup_iters, self.power)

    def _get_lr(self, current_epoch, current_iter):
        if self.iter_per_epoch is None:
            raise ValueError(
                'You must call a method `set_iter_per_epoch(it)` before using this scheduler.')
        if current_iter < self.warmup_iters:
            return self.base_lr * (current_iter + 1) / self.warmup_iters
        return self.poly.get_learning_rate(current_iter - self.warmup_iters)
