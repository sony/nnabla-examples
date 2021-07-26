# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa as lr
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm

from hparams import hparams as hp

mel_basis = librosa_mel_fn(
    hp.sr, hp.n_fft, n_mels=hp.n_mels,
    fmin=hp.mel_fmin, fmax=hp.mel_fmax
)


def process(line):
    r"""Read audio waveform and preprocess it.

    Args:
        line (str): A line from metadata.
    """
    path = Path(hp.corpus_path) / 'wavs'
    meta = line.strip().split('|')
    wave = lr.load(path / f'{meta[0]}.wav', sr=hp.sr)[0]
    np.savez(
        Path(hp.precomputed_path) / 'data' / (meta[0] + '.npz'),
        wave=wave
    )


def run():
    path = Path(hp.corpus_path)
    save_path = Path(hp.precomputed_path)
    save_path.joinpath('data').mkdir(parents=True, exist_ok=True)

    with open(path / 'metadata.csv', encoding='utf-8') as f:
        files = f.readlines()

    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process, f) for f in files]
        for f in tqdm(as_completed(futures), total=len(futures)):
            pass

    # split data into train/test sets
    file_train = Path(hp.precomputed_path) / 'meta_train.csv'
    file_valid = Path(hp.precomputed_path) / 'meta_test.csv'
    if not (file_train.exists() and file_valid.exists()):
        random.Random(hp.seed).shuffle(files)
        with open(file_train, 'w') as f:
            f.writelines(files[:int(0.98 * len(files))])
        with open(file_valid, 'w') as f:
            f.writelines(files[int(0.98 * len(files)):])


if __name__ == "__main__":
    run()
