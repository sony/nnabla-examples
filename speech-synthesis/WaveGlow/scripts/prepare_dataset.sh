#!/usr/bin/env bash

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