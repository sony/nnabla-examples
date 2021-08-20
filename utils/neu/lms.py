# Copyright 2021 Sony Corporation.
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

from contextlib import contextmanager

import nnabla.logger as logger
from nnabla.lms import SwapInOutScheduler


@contextmanager
def sechdule_scope(scheduler):
    scheduler.update_pre_hook()
    yield scheduler
    scheduler.update_post_hook()


def lms_scheduler(ctx, use_lms, gpu_memory_size=8 << 30, window_length=12 << 30):
    _check_list = [x.split(":")[0] for x in ctx.backend]
    if "cudnn" not in _check_list and "cuda" not in _check_list:
        logger.warn(
            "ctx passed to scheduler doesn't have cuda/cudnn backend. lms scheduler will not be used.")
        use_lms = False

    if use_lms:
        logger.info("[LMS] gpu_memory_limit: {}GB, prefetch_window_length: {}GB".format(float(gpu_memory_size) / (1 << 30),
                                                                                        float(window_length) / (1 << 30)))

        # Change array preference so that lms works well.
        # import nnabla_ext.cuda.init as cuda_init
        # cuda_init.prefer_cpu_pinned_array()
        # cuda_init.prefer_cuda_virtual_array()
        #
        from nnabla.ext_utils import get_extension_context
        # from nnabla import set_default_context
        be, tc = ctx.backend[0].split(":")
        # ctx = get_extension_context(be, device_id=ctx.device_id, type_config=tc)
        # set_default_context(ctx)

        cpu_ctx = get_extension_context("cpu", device_id="", type_config=tc)
        return SwapInOutScheduler(cpu_ctx, ctx, gpu_memory_size, window_length)
    else:
        class DummyScheduler(object):
            function_pre_hook = None
            function_post_hook = None
            update_pre_hook = None
            update_post_hook = None

            def start_scheduling(self):
                return None

            def end_scheduling(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyScheduler()
