from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import nnabla as nn
import nnabla.functions as F



def _gather_feat(feat, ind, mask=None):
    ind = np.expand_dims(ind, axis=2).astype(int)
    result = np.take(feat, ind, axis=1)
    return result


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.transpose(0, 2, 3, 1)
    feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
    feat = _gather_feat(feat, ind)
    return feat
