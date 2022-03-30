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

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa as lr
import numpy as np
import tgt
from librosa.filters import mel as librosa_mel_fn
from tqdm import tqdm

from hparams import hparams as hp
from utils import audio, misc

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
    base_path = Path(hp.precomputed_path) / 'TextGrid' / 'LJSpeech'
    textgrid = tgt.io.read_textgrid(base_path / (f'{meta[0]}' + '.TextGrid'))

    # compute phonemes and durations
    phone, duration = misc.get_alignment(
        textgrid.get_tier_by_name("phones"),
        hp.sr, hp.hop_length
    )

    # Compute fundamental frequency
    pitch = audio.compute_fundamental_frequency(wave, hp.sr, hp.hop_length)
    pitch = misc.preprocess_pitch(pitch, duration)

    # compute mel and energy
    mel, energy = audio.compute_mel_spectrogram(
        wave, mel_basis, hp.n_fft, hp.hop_length, hp.win_length)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=np.inf))
    mel = mel[:, :sum(duration)]
    energy = misc.preprocess_energy(energy, duration)

    np.savez(
        Path(hp.precomputed_path) / 'data' / (meta[0] + '.npz'),
        wave=wave, text=meta[2], phone=phone,
        mel=mel, energy=energy, duration=duration, pitch=pitch,
    )

    return (energy, pitch)


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

    # computing statistics for energy and pitch
    energy = np.concatenate([e.result()[0] for e in futures])
    pitch = np.concatenate([e.result()[1] for e in futures])
    np.savez(
        Path(hp.precomputed_path) / 'statistics.npz',
        mean_energy=np.mean(energy), std_energy=np.std(energy),
        min_energy=np.min(energy), max_energy=np.max(energy),
        mean_pitch=np.mean(pitch), std_pitch=np.std(pitch),
        min_pitch=np.min(pitch), max_pitch=np.max(pitch),
    )

    # split data into train/test sets
    file_train = Path(hp.precomputed_path) / 'meta_train.csv'
    file_valid = Path(hp.precomputed_path) / 'meta_test.csv'
    if not (file_train.exists() and file_valid.exists()):
        random.Random(hp.seed).shuffle(files)
        open(file_train, 'w').writelines(files[:int(0.98 * len(files))])
        open(file_valid, 'w').writelines(files[int(0.98 * len(files)):])


if __name__ == "__main__":
    run()
