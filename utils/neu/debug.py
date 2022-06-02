
from nnabla.ext_utils import import_extension_module
from nnabla.logger import logger

from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timer(device: str, tag: str, ctx: str = "cuda"):
    ext = import_extension_module(ctx)

    # perform host-device sync.
    ext.device_synchronize(device)

    start = perf_counter()

    yield

    logger.info(f"[{tag}] elapsed time: {perf_counter() - start} sec")


@contextmanager
def nvtx(name):
    from nnabla_ext.cuda.nvtx import range_push, range_pop

    range_push(name)
    yield
    range_pop()
