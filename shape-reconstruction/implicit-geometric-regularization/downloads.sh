#!/usr/bin/env bash
# Copyright 2020,2021 Sony Corporation.
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

DPATH=stanford_3d_scanning_datasets
mkdir -p $DPATH

# .tar.gz
datasets=(
    "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/drill.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz"
    "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz"
)
for dataset in "${datasets[@]}"; do
    wget ${dataset} -P ${DPATH}
done

for dataset in $(ls ${DPATH} | grep "tar.gz");do
    tar zxvf ${DPATH}/${dataset} -C ${DPATH}
done

# .gz
datasets=(
    "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_manuscript.ply.gz"
    "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz"
)
for dataset in "${datasets[@]}"; do
    wget ${dataset} -P ${DPATH}
done

for dataset in $(ls ${DPATH} | grep "ply.gz");do
    gzip -d ${DPATH}/${dataset}
done
