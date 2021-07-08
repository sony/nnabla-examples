from __future__ import absolute_import
import os
import sys
common_utils_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils'))
sys.path.append(common_utils_path)
from neu.yaml_wrapper import read_yaml