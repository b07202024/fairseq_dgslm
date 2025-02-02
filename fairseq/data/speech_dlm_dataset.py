# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np
import torch

from fairseq.data import FairseqDataset, MonolingualDataset, data_utils
from fairseq.data.shorten_dataset import TruncateDataset, RandomCropDataset


class SpeechDLMDataset(FairseqDataset):
    """The dataset used to train the SpeechDLM model as described in the paper:
    https://arxiv.org/pdf/2203.16502.pdf

    The input datasets is expected to be a dict over channel names with the values
    being instances of :class:`~fairseq.data.MonolingualDataset`.

    Each element of SpeechDLMDataset is a dictionary with the following keys:
        - `id` (int) : index of the item
        - `source` (OrderedDict[str, Tensor of shape (seq_len,)]) : dictionary over
            channels with the values containing the input unit tokens
        - `target_next` (OrderedDict[str, Tensor of shape (seq_len,)]) : dictionary
            over channels with the values containing the next unit tokens (input
            tokens shifted by 1).
            Its value is None if 'next' not in self.targets
        - `target_edge` (OrderedDict[str, Tensor of shape (dedup_seq_len,)]) : dictionary
            over channels with the values containing the edge unit tokens (input tokens
            deduplicated).
            Its value is None if 'edge' not in self.targets
        - `target_duration` (OrderedDict[str, Tensor of shape (dedup_seq_len,)]) :
            dictionary over channels with the values being the durations of the edge units.
            Its value is None if 'duration' not in targets.
        - `target_edge_indices` (OrderedDict[str, Tensor of shape (dedup_seq_len,)]) :
            dictionary over channels with the values being the indices of the edge units
            in the source sequence.
            Its value is None if neither 'edge' or 'duration in targets.

    Args:
        datasets (Dict[str, ~fairseq.data.MonolingualDataset]): a dictionary of
            :class:`~fairseq.data.MonolingualDataset` instances.
        targets (List[str]): list of the target types that the SpeechDLM model
            should predict.  Can be one of "next", "edge", "duration".
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self, unit_datasets, text_datasets, frame_datasets, num_code, targets=None, max_target_durations=None, shuffle=False, sep=None,
    ):
        super().__init__()
        if isinstance(unit_datasets, dict):
            unit_datasets = OrderedDict(unit_datasets)
        assert isinstance(
            unit_datasets, OrderedDict
        ), "unit_datasets is expected to be an instance of Dictionary or OrderedDict"
        assert unit_datasets, "unit_datasets is None"
        if isinstance(text_datasets, dict):
            text_datasets = OrderedDict(text_datasets)
        
        for datasets in unit_datasets.values():
            for dataset in datasets:
                assert isinstance(
                    dataset, MonolingualDataset
                ), "Each value of unit_datasets is expected to be an instance of MonolingualDataset"

        self.unit_datasets = unit_datasets
        self.text_datasets = text_datasets
        self.frame_datasets = frame_datasets
        self.targets = targets
        if max_target_durations is not None and max_target_durations > 0:
            self.max_target_durations = max_target_durations
        else:
            self.max_target_durations = float("inf")
        self.sizes = next(iter(unit_datasets.values()))[0].sizes
        self.vocab = next(iter(unit_datasets.values()))[0].vocab
        self.length = len(next(iter(unit_datasets.values()))[0])
        self.shuffle = shuffle
        self.sep = sep
        self.num_code = num_code
        self.ctc_prediction = True if self.sep else False

        for channel, datasets in unit_datasets.items():
            if self.ctc_prediction:
                assert (
                    len(datasets[0]) == len(self.text_datasets[channel]) == self.length
                ), "[{}] length mismatch ({} vs {} vs {})".format(
                    channel, len(datasets[0]), len(self.text_datasets[channel]), self.length
                )
            else:
                assert (
                    len(datasets[0]) == self.length
                ), "[{}] length mismatch ({} vs {})".format(
                    channel, len(datasets[0]), self.length
                )
            assert (datasets[0].sizes == self.sizes).all(), "[{}] sizes mismatch".format(
                channel
            )

            assert (
                datasets[0].vocab.pad() == self.vocab.pad()
            ), "pad token is expected to be the same"
            assert (
                datasets[0].vocab.eos() == self.vocab.eos()
            ), "eos token is expected to be the same"
            assert (
                datasets[0].vocab.bos() == self.vocab.bos()
            ), "bos token is expected to be the same"
            assert (
                datasets[0].vocab.unk() == self.vocab.unk()
            ), "unk token is expected to be the same"

    def __getitem__(self, index):

        def get_seg(dataset, mode):
            if mode == 'text':
                sentences = []
                sep_idx = (dataset[index] == self.sep).nonzero()
                for i in range(sep_idx[:-1].size(0)):
                    start = sep_idx[i] + 1
                    end = sep_idx[i + 1]
                    sentences.append(dataset[index][start:end])
                return sentences

            elif mode == 'frame':
                return [torch.LongTensor(list(range(start, end))) for start, end in zip(dataset[index]['start'], dataset[index]['end'])]

        source = OrderedDict(
            [

                (key, datasets[0][index]["source"])
                for (key, datasets) in self.unit_datasets.items()
            ]
        )
        text = None if not self.ctc_prediction else OrderedDict(
            [
                (key, get_seg(dataset, 'text'))
                for (key, dataset) in self.text_datasets.items()
            ]
        )
        frame = None if not self.ctc_prediction else OrderedDict(
            [
                (key, get_seg(dataset, 'frame'))
                for (key, dataset) in self.frame_datasets.items()
            ]
        )

        item = {
            "id": index,
            "source": source,
            "target_ctc": text,
            "frame_ctc": frame,
            "target_next": None,
            "target_edge": None,
            "target_duration": None,
            "target_edge_indices": None,
        }

        if self.targets is not None:
            for channel in self.unit_datasets:
                target = self._get_target(index, channel)
                for t in target:
                    if t != "ctc":
                        if item[f"target_{t}"] is None:
                            item[f"target_{t}"] = OrderedDict()
                        item[f"target_{t}"][channel] = target[t]

        return item

    def __len__(self):
        return self.length

    def _get_target(self, index, channel):
        """Get target in one of ['next', 'edge', 'duration']
        - 'next' is the future unit
        - 'edge' is the edge unit
        - 'duration' is the duration of the edge unit
        """
        if self.targets is not None:
            target = {}
            pad_idx = self.vocab.pad()
            max_dur = self.max_target_durations
            future_targets = [dataset[index]["target"] for dataset in self.unit_datasets[channel]]
            if "edge" in self.targets or "duration" in self.targets:
                edge_units = []
                edge_unit_counts = []
                edge_indices = []
                for future_target in future_targets:
                    edge_unit, edge_unit_count = torch.unique_consecutive(
                        future_target, return_counts=True
                    )
                    padding_end = edge_unit[-1] == pad_idx
                    if padding_end:
                        edge_unit = edge_unit[:-1]
                        edge_unit_count = edge_unit_count[:-1]
                    edge_index = torch.cumsum(edge_unit_count, 0)
                    edge_index = torch.cat([torch.tensor([0]), edge_index[:-1]])
                    
                    edge_units.append(edge_unit)
                    edge_unit_counts.append(edge_unit_count)
                    edge_indices.append(edge_index)
                target["edge_indices"] = edge_indices

            for t in self.targets:
                if t == "next":
                    target[t] = future_targets
                elif t == "edge":
                    target[t] = edge_units
                elif t == "duration":
                    # count the remaining duration of the last edge indices in the next sentence
                    for idx in range(len(edge_unit_counts)):
                        if not padding_end and index < len(self.unit_datasets[channel][idx]) - 1:
                            i = 0
                            next_sentence_target = self.unit_datasets[channel][idx][index + 1][
                                "target"
                            ]
                            while (
                                next_sentence_target[i] == edge_units[idx][-1]
                                and edge_unit_counts[idx][-1] + i < max_dur
                            ):
                                i += 1
                            edge_unit_counts[idx][-1] += i

                            # cut off to the maximal threshold
                            if max_dur:
                                edge_unit_counts[idx][edge_unit_counts[idx] > max_dur] = max_dur

                    target[t] = edge_unit_counts
                elif t == "ctc":
                    target[t] = None
                else:
                    raise Exception("invalid target " + t)

            return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (OrderedDict[str, LongTensor]): dictionary
                    over channel with the values being padded 2D Tensor of
                    samples `source` of shape `(bsz, src_len)`.
                    Padding will appear on the right.
                  - `src_lengths` (LongTensor): lengths of source sentences
                    in the mini-batch

                - `target` (dict): the target of the Model, containing keys:

                  - `next` (OrderedDict[str, LongTensor]): dictionary
                    over channel with the values being padded 2D Tensor of
                    batch samples' `target_next` of shape `(bsz, tgt_len)`.
                    Padding will appear on the right.
                  - `edge` (OrderedDict[str, LongTensor]): dictionary
                    over channel with the values being the concatenated
                    1D Tensor of batch samples' `target_edge` of shape
                    `(sum of dedup_tgt_len,)`
                  - `duration` (OrderedDict[str, LongTensor]): dictionary
                    over channel with the values being the concatenated
                    1D Tensor of batch samples' `target_duration` of shape
                    `(sum of dedup_tgt_len,)`
                  - `edge_indices` (OrderedDict[str, LongTensor]): dictionary
                    over channel with the values being the concatenated
                    1D Tensor of batch samples' `target_edge_indices` of
                    shape `(sum of dedup_tgt_len,)`.
                    The indices are added to multiplies of batch size
                    such that they are the actual indices in the flatten
                    `src_tokens` Tensor
        """
        if len(samples) == 0:
            return {}
        pad_idx = self.vocab.pad()
        eos_idx = self.vocab.eos()

        def merge(key, max_size=None):
            mapped_key = "target_ctc" if key == "target_ctc_length" else key
            if samples[0][mapped_key] is None:
                return None
            res = OrderedDict()
            for channel in samples[0][mapped_key]:
                if key == "source":
                    # fill batch of shape: (batch_size, max_size)
                    res[channel] = data_utils.collate_tokens(
                        [s[key][channel] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                    )
                elif key == "target_next":
                    res[channel] = []
                    for i in range(self.num_code):
                        res[channel].append(
                            data_utils.collate_tokens(
                                [s[key][channel][i] for s in samples],
                                pad_idx,
                                eos_idx,
                                left_pad=False,
                            )
                        )
                elif key == "target_ctc":
                    sentences = [sent for s in samples for sent in s[key][channel] if sent.size(0) != 0]
                    lengths = [s.size(0) > 0 for s in sentences]
                    if True not in lengths:
                        res[channel] = None
                    else:
                        res[channel] = data_utils.collate_tokens(
                            sentences,
                            pad_idx,
                            left_pad=False,
                        )
                elif key in ["target_edge", "target_duration"]:
                    # concatenate the edge units/duration
                    res[channel] = []
                    for i in range(self.num_code):            
                        res[channel].append(torch.cat([s[key][channel][i] for s in samples]))
                elif key == "target_edge_indices":
                    # increase the edge indices to the indices in the flatten batch
                    res[channel] = []
                    for i in range(self.num_code):
                        res[channel].append(torch.cat(
                            [s[key][channel][i] + j * max_size for j, s in enumerate(samples)]
                        ))
                elif key == "target_ctc_length":
                    res[channel] = torch.LongTensor([sent.numel() for s in samples for sent in s["target_ctc"][channel] if sent.size(0) != 0])
                elif key == "frame_ctc":
                    res[channel] = [s["frame_ctc"][channel] for s in samples]

            return res

        src_tokens = merge("source")
        tgt_next = merge("target_next")
        tgt_edge = merge("target_edge")
        tgt_duration = merge("target_duration")
        ctc = merge("target_ctc")
        ctc_lengths = merge("target_ctc_length")
        frame_ctc = merge("frame_ctc")
        tgt_edge_indices = merge(
            "target_edge_indices", max_size=next(iter(src_tokens.values())).size(-1)
        )
        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(len(item) for s in samples for item in s["source"].values()),
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": torch.LongTensor(
                    [next(iter(s["source"].values())).numel() for s in samples]
                ),
            },
            "target": {
                "ctc": {
                    "ctc_tokens": ctc,
                    "ctc_lengths": ctc_lengths,
                    "ctc_frames": frame_ctc
                },
                "next": tgt_next,
                "edge": tgt_edge,
                "duration": tgt_duration,
                "edge_indices": tgt_edge_indices,
            },
        }

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch(indices)
