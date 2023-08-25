# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple
import re

event2char = {"[LAUGHTER]": '1', "[LAUGH]": '1', "[SIGH]": '2', "[LIPSMACK]": '3',"[MN]": '4', "[COUGH]": '5', "[BREATH]": '6', "[SNEEZE]": '7'}
char2event = {v: k for k, v in event2char.items()}

def get_audio_files(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                len(items) == 2
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes

def text_processing(trans, keep_events=True):
    trans = trans.strip().upper().replace("._", " ").replace("_", "").replace('[UH]', 'UH').replace('[UM]', 'UM').replace('<SPOKEN_NOISE>', '[VOCALIZED-NOISE]').replace('.PERIOD', 'PERIOD').replace('-HYPHEN', 'HYPHEN').replace("-", "").replace(".", "")
    if keep_events:
        for event in event2char:
            trans = trans.replace(event, event2char[event])
    trans = " " + re.sub('\\(\\(.*?\\)\\)|\\[.*?\\]', '', trans) + " "
    while "  " in trans:
        trans = trans.replace("  ", " ")
    trans = trans.replace(" ", "|")

    if trans != "|":
        trans = ' '.join(trans)
        if keep_events:
            for char in char2event:
                trans = trans.replace(char, char2event[char])
    return trans