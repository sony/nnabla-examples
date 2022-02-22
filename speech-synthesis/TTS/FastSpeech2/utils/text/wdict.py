# -*- coding: utf-8 -*-
"""This module contains standard abbreviations and ARPAbet phoneme set for
North American English.

Vowels carry a lexical stress marker:
0 - No stress
1 - Primary stress
2 - Secondary stress

References:
    - http://www.speech.cs.cmu.edu/cgi-bin/cmudict
    - https://github.com/keithito/tacotron
"""

__ABBREVIATIONS__ = [
    ('mrs', 'misess'), ('mr', 'mister'), ('dr', 'doctor'),
    ('st', 'saint'), ('co', 'company'), ('jr', 'junior'),
    ('maj', 'major'), ('gen', 'general'), ('drs', 'doctors'),
    ('rev', 'reverend'), ('lt', 'lieutenant'), ('hon', 'honorable'),
    ('sgt', 'sergeant'), ('capt', 'captain'), ('esq', 'esquire'),
    ('ltd', 'limited'), ('col', 'colonel'), ('ft', 'fort'),
]

___VALID_SYMBOLS___ = [
    'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1',
    'AH2', 'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0',
    'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0',
    'ER1', 'ER2', 'EY', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0',
    'IH1', 'IH2', 'IY', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG',
    'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W',
    'Y', 'Z', 'ZH',
    "sil", "sp", "spn",  # silence phone and unknown words
    "EOL",  # special character for end of line
]


___SYMBOL_TO_ID___ = {s: i for i, s in enumerate(___VALID_SYMBOLS___)}
___ID_TO_SYMBOL___ = {i: s for i, s in enumerate(___VALID_SYMBOLS___)}
