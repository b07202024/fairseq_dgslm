# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    SpeechDLMDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class SpeechDLMConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    channels: Optional[str] = field(
        default=None,
        metadata={
            "help": 'comma-separated list of channels to load e.g., "unitA,unitB"'
            "(default: load all possible channels in the data path)"
        },
    )
    channel_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma-separated list of weights for different losses"
            "(default: None, which means all losses are treated equally)"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    # str type is a workaround to put **default=True** here
    next_unit_prediction: str = field(
        default="False",
        metadata={
            "help": "Perform Next Unit Prediction, expected str input ('True' or 'False')"
        },
    )
    edge_unit_prediction: str = field(
        default="True",
        metadata={
            "help": "Perform Edge Unit Prediction, expected str input ('True' or 'False')"
        },
    )
    duration_prediction: str = field(
        default="True",
        metadata={
            "help": "Perform Duration Prediction, expected str input ('True' or 'False')"
        },
    )
    ctc_prediction: str = field(
        default="False",
        metadata={
            "help": "Perform CTC Prediction, expected str input ('True' or 'False')"
        },
    )
    delayed_duration_target: str = field(
        default="True",
        metadata={
            "help": "Perform Delayed Duration Prediction, expected str input ('True' or 'False')"
            "(default: 'True')"
        },
    )
    max_target_durations: Optional[int] = field(
        default=256,
        metadata={"help": "max duration considered (cut off to this value)"},
    )
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")


@register_task("speech_dlm_task", dataclass=SpeechDLMConfig)
class SpeechDLMTask(LegacyFairseqTask):
    """Task for the SpeechDLM model as described in the paper:
    https://arxiv.org/pdf/2203.16502.pdf

    It create a multi-channel dataset (SpeechDLMDataset) from multiple
    dictionaries.

    Args:
        dictionaries (Dict[str, ~fairseq.data.Dictionary]): the dictionaries for
            each input channel of the SpeechDLM model
        output_dictionaries (Dict[str, ~fairseq.data.Dictionary]): the dictionaries
            for the output of each channel of the SpeechDLM model. In most cases it
            will be the same as *dictionaries*.
        targets (List[str]): list of the target types that the SpeechDLM model
            should predict.  Can be one of "next", "edge", "duration", "ctc".
            Defaults to "next".

    .. note::

        The SpeechDLM task is only compatible with
        :mod:`fairseq-train` and :mod:`fairseq-validate`.
        To generate new samples, please refer to example codes
        at examples/textless_nlp/dgslm .
    """

    def __init__(self, args, dicts, output_dicts=None, ctc_dicts=None, targets=None):
        super().__init__(args)
        self.dicts = dicts
        self.output_dicts = output_dicts or dicts
        self.ctc_dicts = ctc_dicts

        if targets is None:
            targets = ["next"]
        self.targets = targets

        self.channels = list(dicts.keys())

        if args.channel_weights is not None:
            self.channel_weights = [float(w) for w in args.channel_weights.split(",")]
        else:
            self.channel_weights = [1.0 for _ in self.channels]
        assert len(self.channel_weights) == len(
            self.channels
        ), "number of channel_weights must be the same as number of channels"

        assert str(args.next_unit_prediction).lower() in [
            "true",
            "false",
        ], f"Expected to be a string of boolean, found {args.next_unit_prediction}"
        assert str(args.edge_unit_prediction).lower() in [
            "true",
            "false",
        ], f"Expected to be a string of boolean, found {args.edge_unit_prediction}"
        assert str(args.duration_prediction).lower() in [
            "true",
            "false",
        ], f"Expected to be a string of boolean, found {args.duration_prediction}"
        assert str(args.delayed_duration_target).lower() in [
            "true",
            "false",
        ], f"Expected to be a string of boolean, found {args.delayed_duration_target}"
        assert str(args.ctc_prediction).lower() in [
            "true",
            "false",
        ], f"Expected to be a string of boolean, found {args.ctc_prediction}"
        self.next_unit_prediction = bool(
            str(args.next_unit_prediction).lower() == "true"
        )
        self.edge_unit_prediction = bool(
            str(args.edge_unit_prediction).lower() == "true"
        )
        self.duration_prediction = bool(str(args.duration_prediction).lower() == "true")
        self.delayed_duration_target = bool(
            str(args.delayed_duration_target).lower() == "true"
        )
        self.ctc_prediction = bool(str(args.ctc_prediction).lower() == "true")
        if self.ctc_prediction:
            assert ctc_dicts is not None

        self.max_target_durations = args.max_target_durations

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        """The dictionaries will be a dict over channel keys and values of type
        ~fairseq.data.Dictionary.
        """
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        data_path = paths[0]

        dicts = None
        output_dicts = None
        if args.channels is None:
            sorted_channels = sorted(
                name[5:-4]
                for name in os.listdir(data_path)
                if name[:5] == "dict." and name[-4:] == ".txt"
            )
        else:
            sorted_channels = sorted(args.channels.split(","))
        logger.info("channels: {}".format(sorted_channels))
        # load dictionaries
        dicts = OrderedDict()
        output_dicts = OrderedDict()
        if args.ctc_prediction == "True":
            ctc_dicts = OrderedDict()
        for channel in sorted_channels:
            unit_dict = Dictionary.load(
                os.path.join(data_path, "dict.{}.txt".format(channel))
            )
            logger.info("[{}] dictionary: {} types".format(channel, len(unit_dict)))

            if args.ctc_prediction == "True":
                text_dict = Dictionary.load(
                    os.path.join(data_path, "dict.{}.txt".format(channel.replace("unit", "text")))
                )
                logger.info("[{}] dictionary: {} types".format(channel.replace("unit", "text"), len(text_dict)))
            
            output_dictionary = unit_dict
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    unit_dict, args.output_dictionary_size
                )
            dicts[channel] = unit_dict
            output_dicts[channel] = output_dictionary
            if args.ctc_prediction == "True":
                ctc_dicts[channel] = text_dict
            if len(dicts) > 0:
                assert dicts[channel].pad() == dicts[sorted_channels[0]].pad()
                assert dicts[channel].bos() == dicts[sorted_channels[0]].bos()
                assert dicts[channel].eos() == dicts[sorted_channels[0]].eos()
                assert dicts[channel].unk() == dicts[sorted_channels[0]].unk()
        if args.ctc_prediction == "True":
            return (dicts, output_dicts, ctc_dicts)
        else:
            return (dicts, output_dicts, None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dicts, output_dicts, ctc_dicts = cls.setup_dictionary(args, **kwargs)

        targets = []
        if str(getattr(args, "next_unit_prediction", "false")).lower() == "true":
            targets.append("next")
        if str(getattr(args, "edge_unit_prediction", "false")).lower() == "true":
            targets.append("edge")
        if str(getattr(args, "duration_prediction", "false")).lower() == "true":
            targets.append("duration")
        if str(getattr(args, "ctc_prediction", "false")).lower() == "true":
            targets.append("ctc")
        if len(targets) == 0:
            # standard language modeling
            targets = ["next"]

        return cls(args, dicts, output_dicts, ctc_dicts=ctc_dicts, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError("Unsupported SpeechDLM target: {}".format(target))
        return model

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> SpeechDLMDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]

        channel_unit_datasets = {}
        channel_text_datasets = {}
        channel_time_datasets = {}
        for channel in self.channels:
            unit_path = os.path.join(data_path, split + "." + channel)
            text_path = os.path.join(data_path, split + "." + channel.replace("unit", "text"))
            time_path = os.path.join(data_path, split + "." + channel.replace("unit", "time"))
            in_dict = self.dicts[channel]
            text_dict = self.ctc_dicts[channel]
            
            out_dict = self.output_dicts[channel]
            unit_dataset = data_utils.load_indexed_dataset(
                unit_path, in_dict, self.args.dataset_impl, combine=combine
            )
            text_dataset = data_utils.load_indexed_dataset(
                text_path, text_dict, self.args.dataset_impl, combine=combine
            )

            time_dataset = []
            try:
                with open(time_path) as f:
                    for line in f.readlines():
                        durs = line.strip().split()
                        times = {'start': [], 'end': []}
                        for dur in durs:
                            start, end = map(int, dur.split(':'))
                            times['start'].append(start)
                            times['end'].append(end)
                        time_dataset.append(times)
            except:
                pass

            if unit_dataset is None:
                raise FileNotFoundError(
                    "[{}] Unit Dataset not found: {} ({})".format(channel, split, unit_path)
                )
            if text_dataset is None and self.ctc_prediction:
                raise FileNotFoundError(
                    "[{}] Text Dataset not found: {} ({})".format(channel, split, text_path)
                )
            if len(time_dataset) == 0 and self.ctc_prediction:
                raise FileNotFoundError(
                    "[{}] Time Dataset not found: {} ({})".format(channel, split, time_path)
                )
            
            unit_dataset = maybe_shorten_dataset(
                unit_dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )

            text_dataset = maybe_shorten_dataset(
                text_dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )

            unit_dataset = TokenBlockDataset(
                unit_dataset,
                unit_dataset.sizes,
                self.args.tokens_per_sample,
                pad=in_dict.pad(),
                eos=in_dict.eos(),
                break_mode=self.args.sample_break_mode,
                include_targets=True,
            )

            add_eos_for_other_targets = (
                self.args.sample_break_mode is not None
                and self.args.sample_break_mode != "none"
            )

            channel_unit_datasets[channel] = MonolingualDataset(
                dataset=unit_dataset,
                sizes=unit_dataset.sizes,
                src_vocab=in_dict,
                tgt_vocab=out_dict,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=False,
                targets=["future"],
                add_bos_token=self.args.add_bos_token,
            )

            channel_text_datasets[channel] = text_dataset
            channel_time_datasets[channel] = time_dataset

        self.datasets[split] = SpeechDLMDataset(
            unit_datasets=channel_unit_datasets,
            text_datasets=channel_text_datasets,
            time_datasets=channel_time_datasets,
            targets=self.targets,
            max_target_durations=self.max_target_durations,
            shuffle=True,
            sep=self.text_dictionary.indices["<SEP>"],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        src_datasets = {}
        tgt_datasets = {}
        for channel in src_tokens[0]:
            dataset = StripTokenDataset(
                TokenBlockDataset(
                    [src_tokens[i][channel] for i in range(len(src_tokens))],
                    src_lengths,
                    block_size=None,  # ignored for "eos" break mode
                    pad=self.source_dictionaries[channel].pad(),
                    eos=self.source_dictionaries[channel].eos(),
                    break_mode="eos",
                ),
                # remove eos from (end of) target sequence
                self.source_dictionaries[channel].eos(),
            )
            src_dataset = PrependTokenDataset(
                dataset,
                token=(
                    self.source_dictionaries[channel].bos()
                    if getattr(self.args, "add_bos_token", False)
                    else self.source_dictionaries[channel].eos()
                ),
            )
            tgt_dataset = AppendTokenDataset(
                dataset, token=self.source_dictionaries[channel].pad()
            )

            src_datasets[channel] = src_dataset
            tgt_datasets[channel] = tgt_dataset

        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": OrderedDict(
                        [
                            (
                                channel,
                                PadDataset(
                                    src_datasets[channel],
                                    pad_idx=self.source_dictionaries[channel].pad(),
                                    left_pad=False,
                                ),
                            )
                            for channel in src_datasets
                        ]
                    ),
                    "src_lengths": NumelDataset(
                        next(iter(src_datasets.values())), reduce=False
                    ),
                },
                "target": OrderedDict(
                    [
                        (
                            channel,
                            PadDataset(
                                tgt_datasets[channel],
                                pad_idx=self.source_dictionaries[channel].pad(),
                                left_pad=False,
                            ),
                        )
                        for channel in tgt_datasets
                    ]
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the SpeechDLM task is not supported"
                )
            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None:
                prefix_tokens = {}
                for channel in sample["net_input"]["src_tokens"]:
                    if sample["net_input"]["src_tokens"][channel].nelement():
                        prefix_tokens_channel = sample["net_input"]["src_tokens"][
                            channel
                        ]
                        if prefix_tokens_channel[:, 0].eq(bos_token).all():
                            prefix_tokens_channel = prefix_tokens_channel[:, 1:]
                        prefix_tokens[channel] = prefix_tokens_channel
                    else:
                        prefix_tokens = None
                        break
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dicts[self.channels[0]]

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dicts[self.channels[0]]
    
    @property
    def text_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.ctc_dicts[self.channels[0]]

    @property
    def source_dictionaries(self):
        """Return the dict of :class:`~fairseq.data.Dictionary` for the
        multichannel language model."""
        return self.dicts

    @property
    def target_dictionaries(self):
        """Return the dict of :class:`~fairseq.data.Dictionary` for the
        multichannel language model."""
        return self.output_dicts

    def build_generator(self, models, args, extra_gen_cls_kwargs=None):

        from fairseq.models.speech_dlm.sequence_generator import (
            multichannel_search,
            MultichannelSequenceGenerator,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        assert (
            sampling_topk < 0 or sampling
        ), "--sampling-topk requires sampling (not beam search)"
        assert (
            sampling_topp < 0 or sampling
        ), "--sampling-topp requires sampling (not beam search)"

        if sampling:
            search_strategy = multichannel_search.ContiguousMultichannelSampling(
                self.target_dictionaries, sampling_topk, sampling_topp
            )
        else:
            search_strategy = multichannel_search.ContiguousMultichannelBeamSearch(
                self.target_dictionaries
            )

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

        return MultichannelSequenceGenerator(
            models,
            self.target_dictionaries,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 500),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            duration_temperature=getattr(args, "duration_temperature", 1.0),
            **extra_gen_cls_kwargs,
        )
