#!/bin/bash
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

DATAROOT=$1
NUM_TRAIN=$2

if [ -z "$DATAROOT" ]; then
    echo "You must specify the directory containing CelebV data."
    exit 1
fi

if [ -z "$NUM_TRAIN" ]; then
    echo "number of the training data not given: 30000 used."
    NUM_TRAIN=30000
fi

for PERSON in Donald_Trump  Emmanuel_Macron  Jack_Ma  Kathleen  Theresa_May; do
    split $DATAROOT/$PERSON/all_98pt.txt tmp_split_ -l $NUM_TRAIN
    mv tmp_split_aa $DATAROOT/$PERSON/train_98pt.txt
    mv tmp_split_ab $DATAROOT/$PERSON/test_98pt.txt
    echo "create train_98pt.txt and test_98pt.txt for $PERSON at $DATAROOT/$PERSON"
done
