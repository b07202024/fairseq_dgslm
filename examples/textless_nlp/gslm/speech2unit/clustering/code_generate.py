import argparse
import os

# import librosa
import torch
import sys
from tqdm import tqdm
# from transformers import EncodecModel, AutoProcessor
from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

sys.path.append('/home/yukuanfu88/iven/fairseq_dgslm')
from examples.textless_nlp.gslm.speech2unit.clustering.utils import (
    text_processing
)
device = "cuda" if torch.cuda.is_available() else "cpu"

bandwidth2num = {1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32}
def quantized(manifest_path, output_path, channel_id, bandwidth, args):
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    model.set_target_bandwidth(bandwidth)

    channel_name = 'A' if channel_id == 1 else 'B'
    unit_files = [open("{}-{}.unit{}".format(output_path, id, channel_name), 'w') for id in range(bandwidth2num[bandwidth])]
    text_path = output_path + '.text' + channel_name
    time_path = output_path + '.time' + channel_name
    with torch.no_grad():
        with open(manifest_path, "r") as f, open(text_path, "w") as text, open(time_path, "w") as time:
            root_dir = f.readline().strip()
            for line in tqdm(f.readlines()[1:]):
                items = line.strip().split("\t")
                assert (
                    len(items) == 2
                ), f"File must have two columns separated by tab. Got {line}"
                name, leng = items
                base_fname = os.path.basename(name).rstrip('.'+args.extension.lstrip('.'))

                wav = torchaudio.load(f"{root_dir}/{name}")[0][channel_id - 1].unsqueeze(0).unsqueeze(0).to(device)

                codes = model.encode(wav)[0][0][0]

                code_strs = []
                for code in codes:
                    code = code[:args.max_tokens]
                    code_strs.append(" ".join(map(str, code.cpu().tolist())))
                trans_path = root_dir + '/' + name.replace(args.extension.lstrip('.'), channel_name)
                with open(trans_path) as tf:
                    conts = tf.readlines()
                    all_trans = []
                    all_dur = []
                    for cont in conts:
                        dur, trans = cont.split('\t', 1)
                        start, end = dur.split(':')
                        start = int(float(start) * model.sample_rate / 320)
                        end = int(float(end) * model.sample_rate / 320)

                        if end > codes[0].size(0):
                            if end - codes[0].size(0) > 3:
                                break # do not included the truncated text
                            end = codes[0].size(0)
                            assert end > start, "End time should be greater than start time"
                        trans = text_processing(trans)

                        if trans != "|":
                            all_trans.append(trans)
                            all_dur.append(f'{start}:{end}')
                    all_trans = '[SEP] ' + ' [SEP] '.join(all_trans) + " [SEP]"
                    all_dur = ' '.join(all_dur)
                if channel_id is not None:
                    base_fname = base_fname+f'-channel{channel_id}'
                if not args.hide_fname:
                    [unit.write(f"{base_fname}|{pred_str}\n") for unit, pred_str in zip(unit_files, code_strs)]
                    text.write(f"{base_fname}|{all_trans}\n")
                    time.write(f"{base_fname}|{all_dur}\n")
                else:
                    [unit.write(f"{pred_str}\n") for unit, pred_str in zip(unit_files, code_strs)]
                    text.write(f"{all_trans}\n")
                    time.write(f"{all_dur}\n")

def get_parser():
    parser = argparse.ArgumentParser(
        description="Quantize using K-means clustering over acoustic features."
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        choices=[1.5, 3, 6, 12, 24],
        default=6,
        required=False,
        help="Encodec bandwidth",
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
        type=int,
        choices=[1, 2],
        help="The audio channel to extract the units in case of stereo file.",
        default=None,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=6144,
        help="max length of units"
    )
    parser.add_argument(
        "--hide-fname", action='store_true',
        help="Hide file names in the output file."
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    quantized(
        manifest_path=args.manifest_path,
        output_path=args.out_quantized_file_path,
        channel_id=args.channel_id,
        bandwidth=args.bandwidth,
        args=args
    )