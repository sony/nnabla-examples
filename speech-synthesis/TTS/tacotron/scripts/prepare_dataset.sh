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

DATAROOT="data"
DATADIR="LJSpeech-1.1"
BZ2ARCHIVE="${DATADIR}.tar.bz2"
ENDPOINT="http://data.keithito.com/data/speech/$BZ2ARCHIVE"
DATA_TRAIN="${DATAROOT}/${DATADIR}/metadata_train.csv"
DATA_VALID="${DATAROOT}/${DATADIR}/metadata_valid.csv"

if [ ! -d "$DATAROOT" ]; then
    mkdir "$DATAROOT"
fi

cd "$DATAROOT"

if [ ! -f "$BZ2ARCHIVE" ]; then
    echo "Data are missing, downloading ..."
    wget "$ENDPOINT"
    tar jxvf "$BZ2ARCHIVE"
fi

cd ..

echo "Preprocessing the data"

python -W ignore -m dataset