from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.dataset.coco_config import COCO
from datasets.dataset.pascal_config import PascalVOC

data_source = {
    'coco': COCO,
    'pascal': PascalVOC,
}


def get_data_source(dataset):
    source = data_source[dataset]
    return source
