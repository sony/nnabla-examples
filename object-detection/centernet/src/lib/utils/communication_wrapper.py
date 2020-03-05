import nnabla as nn
import nnabla.functions as F
import nnabla.communicators as C
import random
import numpy as np
import itertools
import json
import time
import os

def create_float_context(ctx):
    from nnabla.ext_utils import get_extension_context
    ctx_float = get_extension_context(ctx.backend[0].split(':')[
                                      0], device_id=ctx.device_id)
    return ctx_float

np.random.seed(0)


class CommunicationWrapper(object):

    def __init__(self, ctx):
        try:
            comm = C.MultiProcessDataParallelCommunicator(ctx)
        except Exception as e:
            print(e)
            print("No communicator found. Running with a single process. If you run this with MPI processes, all processes will perform totally same.")
            self.n_procs = 1
            self.rank = 0
            self.ctx = ctx
            self.ctx_float = create_float_context(self.ctx)
            self.comm = None
            return

        comm.init()
        self.n_procs = comm.size
        self.rank = comm.rank
        self.ctx = ctx
        self.ctx.device_id = str(self.rank)
        self.ctx_float = create_float_context(self.ctx)
        self.comm = comm

    def all_reduce(self, params, division, inplace):
        if self.n_procs == 1:
            return
        self.comm.all_reduce(params, division=division, inplace=inplace)
