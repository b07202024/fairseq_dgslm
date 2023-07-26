# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import post_process
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechDLMCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    main_and_cross_weights: Optional[str] = field(
        default="1,0",
        metadata={
            "help": "Comma-separated list of weights of Main-channel vs Cross-channel Prediction Losses"
            "(default: 1,0)"
        },
    )
    general_unit_loss_weight: float = field(
        default=0,
        metadata={
            "help": "The weight of the General Prediction Loss (Next-step Unit Prediction Loss)"
            "(default: 0)"
        },
    )
    edge_unit_loss_weight: float = field(
        default=1,
        metadata={"help": "The weight of the Edge Unit Prediction Loss" "(default: 1)"},
    )
    duration_loss_weight: float = field(
        default=1,
        metadata={
            "help": "The weight of the Edge Unit Duration Prediction Loss"
            "(default: 1)"
        },
    )
    ctc_loss_weight: float = field(
        default=0,
        metadata={
            "help": "The weight of the CTC Loss"
            "(default: 0)"
        },
    )


@register_criterion("speech_dlm_criterion", dataclass=SpeechDLMCriterionConfig)
class SpeechDLMCriterion(FairseqCriterion):
    """Criteron for the SpeechDLM model as described in the paper:
    https://arxiv.org/pdf/2203.16502.pdf

    There are 3 possible losses depending on the targets of the model:
        - general_unit_loss : The next unit prediction loss, corresponding to
            'next' target
        - edge_unit_loss : The edge unit prediction loss, corresponding to
            'edge' target
        - duration_loss : The duration prediction loss, corresponding to
            'duration' target
    """

    def __init__(
        self,
        task,
        sentence_avg,
        main_and_cross_weights,
        general_unit_loss_weight,
        edge_unit_loss_weight,
        duration_loss_weight,
        ctc_loss_weight,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        self.channels = task.channels
        self.targets = task.targets
        self.text_dictionary = task.text_dictionary
        self.delayed_duration_target = task.delayed_duration_target

        self.main_channel_weight = float(main_and_cross_weights.split(",")[0])
        self.cross_channel_weight = float(main_and_cross_weights.split(",")[1])
        assert self.main_channel_weight >= 0 and self.cross_channel_weight >= 0

        self.channel_weights = {
            channel: weight
            for channel, weight in zip(self.channels, task.channel_weights)
        }

        self.target_weights = {}
        for t in self.targets:
            if t == "next":
                self.target_weights[t] = general_unit_loss_weight
                assert (
                    general_unit_loss_weight > 0
                ), "Expect a positive --general-unit-loss-weight for next unit prediction"
            elif t == "edge":
                self.target_weights[t] = edge_unit_loss_weight
                assert (
                    edge_unit_loss_weight > 0
                ), "Expect a positive --edge-unit-loss-weight for edge unit prediction"
            elif t == "duration":
                self.target_weights[t] = duration_loss_weight
                assert (
                    duration_loss_weight > 0
                ), "Expect a positive --duration-loss-weight for duration prediction"
            elif t == "ctc":
                self.target_weights[t] = ctc_loss_weight
                assert (
                    ctc_loss_weight > 0
                ), "Expect a positive --ctc-loss-weight for ctc training"

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        tok_output = model(**sample["net_input"], full_context_alignment=True)
        ctc_output = model(**sample["net_input"], full_context_alignment=False) if "ctc" in self.targets else None
        loss_dict, stats_dict = self.compute_loss(
            model, tok_output, ctc_output, sample, reduce=reduce
        )
        nsentences = sample["net_input"]["src_tokens"][self.channels[0]].size(0)

        logging_output = {
            "nsentences": nsentences,
        }
        logging_output["nsentences"] = nsentences

        loss_all = {t: 0 for t in self.targets}
        correct_all = {t: 0 for t in self.targets}
        count_all = {t: 0 for t in self.targets}
        wer_all = wtot_all = cer_all = ctot_all = 0
        ntokens_all = 0
        sample_size_all = 0
        for channel in loss_dict:
            for pred_channel in loss_dict[channel]:
                # Get ntokens & sample_size
                ntokens = sample["net_input"]["src_tokens"][channel].numel()
                sample_size = nsentences if self.sentence_avg else ntokens
                prefix = "[{}-{}]".format(channel, pred_channel)
                log_keys = {
                    "next": "general_token",
                    "edge": "edge_token",
                    "duration": "edge_duration",
                    "ctc": "ctc"
                }

                # Log & Update the sizes
                logging_output["{}ntokens".format(prefix)] = ntokens
                logging_output["{}sample_size".format(prefix)] = sample_size
                ntokens_all += ntokens
                sample_size_all += sample_size

                for t in self.targets:
                    log_key = log_keys[t]
                    loss = loss_dict[channel][pred_channel][t]
                    
                    if t != "ctc":
                        correct, count = stats_dict[channel][pred_channel][t]
                    else:
                        correct, count, dist = stats_dict[channel][pred_channel][t]

                    # Log the statistics
                    logging_output["{}{}_loss".format(prefix, log_key)] = loss.data
                    logging_output["{}{}_correct".format(prefix, log_key)] = correct
                    logging_output["{}{}_count".format(prefix, log_key)] = count
                    if t == "ctc":
                        logging_output["{}{}_w_errors".format(prefix, log_key)] = dist["w_errors"]
                        logging_output["{}{}_w_total".format(prefix, log_key)] = dist["w_total"]
                        logging_output["{}{}_c_errors".format(prefix, log_key)] = dist["c_errors"]
                        logging_output["{}{}_c_total".format(prefix, log_key)] = dist["c_total"]

                    # Scale the training loss by weights
                    target_loss = loss * self.channel_weights[channel]
                    if pred_channel == channel:
                        target_loss = target_loss * self.main_channel_weight
                    else:
                        target_loss = target_loss * self.cross_channel_weight
                    # Normalize the losses in the training by the number of edges
                    if t in ["edge", "duration"]:
                        target_loss = target_loss / count * sample_size

                    # Update the statistics
                    loss_all[t] += target_loss
                    if t != "ctc":
                        correct_all[t] += correct
                    else:
                        wer_all += dist["w_errors"]
                        wtot_all += dist["w_total"]
                        cer_all += dist["c_errors"]
                        ctot_all += dist["c_total"]
                    count_all[t] += count

        # Logging the average statistics
        logging_output["ntokens"] = ntokens_all
        logging_output["sample_size"] = sample_size_all
        for t in self.targets:
            log_key = {
                "next": "general_token",
                "edge": "edge_token",
                "duration": "edge_duration",
                "ctc": "ctc"
            }[t]
            logging_output["{}_loss".format(log_key)] = loss_all[t].data
            if t != "ctc":
                logging_output["{}_correct".format(log_key)] = correct_all[t]
            else:
                logging_output["{}_w_errors".format(log_key)] = wer_all
                logging_output["{}_w_total".format(log_key)] = wtot_all
                logging_output["{}_c_errors".format(log_key)] = cer_all
                logging_output["{}_c_total".format(log_key)] = ctot_all
            logging_output["{}_count".format(log_key)] = count_all[t]


        # Define the training loss
        training_loss = 0
        for t in self.targets:
            training_loss += loss_all[t] * self.target_weights[t]
        logging_output["loss"] = training_loss.data

        return training_loss, sample_size_all, logging_output

    def compute_loss(self, model, tok_output, ctc_output, sample, reduce=True):
        # Get the model outputs and target
        tok_lprobs_dict = model.get_normalized_probs(tok_output, log_probs=True)
        ctc_lprobs_dict = model.get_normalized_probs(ctc_output, log_probs=True) if ctc_output else None
        target_dict = model.get_targets(sample, None)

        # Init the dictionaries
        loss_dict, stats_dict = {}, {}

        for channel in tok_lprobs_dict:
            # Init the dictionaries
            loss_dict[channel], stats_dict[channel] = {}, {}

            for pred_channel in tok_lprobs_dict[channel]:
                # Init the dictionaries
                loss_dict[channel][pred_channel] = {}
                stats_dict[channel][pred_channel] = {}

                # Get token & duration predictions
                tok_outs = tok_lprobs_dict[channel][pred_channel]
                ctc_lprobs = ctc_lprobs_dict[channel][pred_channel] if ctc_lprobs_dict else None
                if not isinstance(outputs, dict):
                    token_lprobs = tok_outs
                else:
                    token_lprobs = tok_outs["pred_token"]
                    dur_preds = tok_outs["pred_duration"]
                    dur_preds = dur_preds.view(-1)
                token_lprobs = token_lprobs.view(-1, token_lprobs.size(-1))
                token_preds = token_lprobs.argmax(dim=-1)

                if ctc_lprobs:
                    ctc_lprobs = ctc_lprobs.transpose(0, 1)
                    ctc_preds = ctc_lprobs.transpose(0, 1).argmax(dim=-1)

                # Get edge indices
                if "edge" in self.targets or "duration" in self.targets:
                    edge_indices = target_dict["edge_indices"][pred_channel]

                # Compute loss and statistics
                for t in self.targets:
                    if t in ["next", "edge"]:
                        if t == "next":
                            target = target_dict["next"][pred_channel].view(-1)
                            lprobs = token_lprobs
                            preds = token_preds
                        elif t == "edge":
                            target = target_dict["edge"][pred_channel]
                            lprobs = token_lprobs[edge_indices]
                            preds = token_preds[edge_indices]

                        loss = F.nll_loss(
                            lprobs,
                            target,
                            ignore_index=self.padding_idx,
                            reduction="sum" if reduce else "none",
                        )
                    elif t == "duration":
                        target = target_dict["duration"][pred_channel]
                        if self.delayed_duration_target:
                            duration_indices = edge_indices + 1
                            if duration_indices[-1] == len(dur_preds):
                                duration_indices = duration_indices[:-1]
                                target = target[:-1]
                        else:
                            duration_indices = edge_indices
                        preds = dur_preds[duration_indices]

                        loss = F.l1_loss(
                            preds,
                            target,
                            reduction="sum" if reduce else "none",
                        )
                        preds = preds.round()
                    elif t == "ctc":
                        c_err = c_len = w_err = w_len = 0
                        target = target_dict["ctc"]["ctc_tokens"][pred_channel]
                        target_lengths = target_dict["ctc"]["ctc_lengths"][pred_channel]
                        if "src_lengths" in sample["net_input"]:
                            input_lengths = sample["net_input"]["src_lengths"]
                        else:
                            if net_output["padding_mask"] is not None:
                                non_padding_mask = ~net_output["padding_mask"]
                                input_lengths = non_padding_mask.long().sum(-1)
                            else:
                                input_lengths = ctc_lprobs.new_full(
                                    (ctc_lprobs.size(1),), ctc_lprobs.size(0), dtype=torch.long
                                )
                        loss = F.ctc_loss(
                            ctc_lprobs,
                            target,
                            input_lengths,
                            target_lengths,
                            blank=self.text_dictionary.pad()
                        )
                    if t != "ctc":
                        correct = (preds == target).sum().float().cpu().item()
                    else:
                        if not model.training:
                            import editdistance
                            with torch.no_grad():
                                for pred, targ, inp_l in zip(
                                    ctc_preds,
                                    target,
                                    input_lengths
                                ):
                                    pred = pred[:inp_l]
                                    tarl = (targ != self.text_dictionary.pad()) & (targ != self.text_dictionary.eos())
                                    targ = targ[tarl]
                                    targ_units_arr = targ.tolist()

                                    toks = pred.unique_consecutive()
                                    pred_units_arr = toks[toks != self.text_dictionary.pad()].tolist()

                                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                                    c_len += len(targ_units_arr)

                                    targ_units = self.text_dictionary.string(targ)
                                    targ_words = post_process(targ_units, "letter").split()

                                    pred_units = self.text_dictionary.string(pred_units_arr)
                                    pred_words = post_process(pred_units, "letter").split()

                                    w_err += editdistance.eval(pred_words, targ_words)
                                    w_len += len(targ_words)

                    loss_dict[channel][pred_channel][t] = loss
                    if t != 'ctc':
                        count = float(target.size(0))
                        stats_dict[channel][pred_channel][t] = (correct, count)
                    else:
                        count = float(torch.sum(target.view(-1) != self.text_dictionary.pad()))
                        stats_dict[channel][pred_channel][t] = (0, count, 
                            {
                                "c_errors": c_err,
                                "c_total": c_len,
                                "w_errors": w_err,
                                "w_total": w_len
                            }
                        )
        return loss_dict, stats_dict

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        logging_keys = next(iter(logging_outputs)).keys()
        channels = [item[:-7] for item in logging_keys if item.endswith("ntokens")]
        target_prefixes = set(
            [
                item[:-5].split("]")[-1]
                for item in logging_keys
                if item.endswith("_loss")
            ]
        )
        for channel_prefix in channels:
            for target_prefix in target_prefixes:
                prefix = "{}{}".format(channel_prefix, target_prefix)
                count_sum = sum(
                    log.get("{}_count".format(prefix), 0) for log in logging_outputs
                )
                correct_sum = sum(
                    log.get("{}_correct".format(prefix), 0) for log in logging_outputs
                )
                loss_sum = sum(
                    log.get("{}_loss".format(prefix), 0) for log in logging_outputs
                )

                if "duration" in target_prefix:
                    # for duration we don't need to divide by log(2)
                    metrics.log_scalar(
                        "{}_loss".format(prefix),
                        loss_sum / count_sum,
                        count_sum,
                        round=3,
                    )
                elif "ctc" in target_prefix:
                    wer_sum = sum(
                        log.get("{}_w_errors".format(prefix), 0) for log in logging_outputs
                    )
                    wtot_sum = sum(
                        log.get("{}_w_total".format(prefix), 0) for log in logging_outputs
                    )
                    cer_sum = sum(
                        log.get("{}_c_errors".format(prefix), 0) for log in logging_outputs
                    )
                    ctot_sum = sum(
                        log.get("{}_c_total".format(prefix), 0) for log in logging_outputs
                    )

                    metrics.log_scalar(
                        "{}_loss".format(prefix),
                        loss_sum / count_sum,
                        count_sum,
                        round=3,
                    )
                    
                    if wtot_sum > 0 and ctot_sum > 0:
                        metrics.log_scalar(
                            "{}_wer".format(prefix),
                            wer_sum / wtot_sum,
                            wtot_sum,
                            round=3
                        )
                        metrics.log_scalar(
                            "{}_cer".format(prefix),
                            cer_sum / ctot_sum,
                            ctot_sum,
                            round=3
                        )
                else:
                    # we divide by log(2) to convert the loss from base e to base 2
                    metrics.log_scalar(
                        "{}_loss".format(prefix),
                        loss_sum / count_sum / math.log(2),
                        count_sum,
                        round=3,
                    )
                    metrics.log_derived(
                        "{}_ppl".format(prefix),
                        lambda meters, prefix=prefix: utils.get_perplexity(
                            meters["{}_loss".format(prefix)].avg
                        ),
                    )     

                accuracy = 100 * correct_sum / count_sum
                metrics.log_scalar("{}_pred_acc".format(prefix), accuracy, round=3)

        # Logging training loss
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
