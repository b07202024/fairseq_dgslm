# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os

import numpy as np

import joblib
import sys
import tqdm
import re
sys.path.append('/home/yukuanfu88/iven/fairseq_dgslm')
from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    get_audio_files,
)
from examples.textless_nlp.gslm.speech2unit.pretrained.utils import (
    get_feature_iterator,
)

def gen_kmeans(
    feature_type, checkpoint_path, layer, manifest_path, sample_pct, flatten, channel_id, kmeans_model, args
):
    generator, num_files = get_feature_iterator(
        feature_type=feature_type,
        checkpoint_path=checkpoint_path,
        layer=layer,
        manifest_path=manifest_path,
        sample_pct=sample_pct,
        channel_id=channel_id
    )
    iterator = generator()

    root, fnames, _ = get_audio_files(args.manifest_path)
    os.makedirs(os.path.dirname(args.out_quantized_file_path), exist_ok=True)
    print(f"Writing quantized predictions to {args.out_quantized_file_path}")
    channel_name = 'A' if channel_id == 1 else 'B'
    unit_path = args.out_quantized_file_path + '.unit' + channel_name
    text_path = args.out_quantized_file_path + '.text' + channel_name
    time_path = args.out_quantized_file_path + '.time' + channel_name
    with open(unit_path, "w") as unit, open(text_path, "w") as text, open(time_path, "w") as time:    
        for i, feats in enumerate(tqdm.tqdm(iterator, total=num_files)):
            pred = kmeans_model.predict(feats)
            pred_str = " ".join(str(p) for p in pred)
            trans_path = root + '/' + fnames[i].replace('flac', channel_name)
            # trans_path = root + '/' + '-'.join(fnames[i].split('-')[:2]) + ".trans.txt"
            base_fname = os.path.basename(fnames[i]).rstrip('.'+args.extension.lstrip('.'))
            with open(trans_path) as tf:
                conts = tf.readlines()
                all_trans = []
                all_dur = []
                for cont in conts:
                    dur, trans = cont.split('\t', 1)
                    start, end = dur.split(':')
                    start = int(float(start) * 16000 / 320)
                    end = int(float(end) * 16000 / 320)
                    if end > 6100:
                        break
                    if end > len(pred):
                        end = len(pred)
                    if end <= start:
                        if cont != conts[-1]:
                            print(base_fname)
                        break
                    trans = trans.strip().upper()
                    trans = re.sub('\\(\\(.*?\\)\\)|\\[.*?\\] ', '', trans)
                    while "  " in trans:
                        trans = trans.replace("  ", " ")
                    trans = trans.replace(" ", "|")
                    if trans != '':
                        all_trans.append(' '.join(trans))
                        all_dur.append(f'{start}:{end}')
                all_trans = ' <SEP> '.join(all_trans) if all_trans != [] else '|'
                all_dur = ' '.join(all_dur) if all_dur != [] else '0:1'
            # with open(trans_path) as tf:
            #     for line in tf.readlines():
            #         line = line.strip()
            #         name = line.split(' ')[0]
            #         if name == fnames[i].split('/')[-1].replace('.flac', ''):
            #             trans = line.split(' ', 1)[1]
            #             trans = trans.replace(" ", "|") + "|"
            #             trans = ' '.join(trans)
            #             break
            
            if args.channel_id is not None:
                base_fname = base_fname+f'-channel{args.channel_id}'
            if not args.hide_fname:
                unit.write(f"{base_fname}|{pred_str}\n")
                text.write(f"{base_fname}|{all_trans}\n")
                time.write(f"{all_dur}\n")
            else:
                unit.write(f"{pred_str}\n")
                text.write(f"{all_trans}\n")
                time.write(f"{all_dur}\n")
        
    del generator


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        choices=["logmel", "hubert", "w2v2", "cpc"],
        default=None,
        required=True,
        help="Acoustic feature type",
    )
    parser.add_argument(
        "--acoustic_model_path",
        type=str,
        help="Pretrained acoustic model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="The layer of the pretrained model to extract features from",
        default=-1,
    )
    parser.add_argument(
        "--kmeans_model_path",
        type=str,
        required=True,
        help="K-means model file path to use for inference",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Features file path. You don't need to enter acoustic model details if you have dumped features",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Manifest file containing the root dir and file names",
    )
    parser.add_argument(
        "--out_quantized_file_path",
        required=True,
        type=str,
        help="File path of quantized output.",
    )
    parser.add_argument(
        "--extension", type=str, default=".flac", help="Features file path"
    )
    parser.add_argument(
        "--channel_id",
        choices=['1', '2'],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    return parser


def main(args, logger):
    
    # K-means model
    logger.info(f"Loading K-means model from {args.kmeans_model_path} ...")
    kmeans_model = joblib.load(open(args.kmeans_model_path, "rb"))
    kmeans_model.verbose = False
    
    # Feature extraction
    if args.features_path is not None:
        logger.info(f"Loading acoustic features from {args.features_path}...")
        features_batch = np.load(args.features_path)
    else:
        logger.info(f"Extracting {args.feature_type} acoustic features...")
        gen_kmeans(
            feature_type=args.feature_type,
            checkpoint_path=args.acoustic_model_path,
            layer=args.layer,
            manifest_path=args.manifest_path,
            sample_pct=1.0,
            flatten=False,
            channel_id=int(args.channel_id) if args.channel_id else None,
            kmeans_model=kmeans_model,
            args=args
        )
    


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)
    main(args, logger)
