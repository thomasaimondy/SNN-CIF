
import sys
import math
import editdistance
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

np.set_printoptions(threshold=10000000000)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
        :param lprobs: log probabilities with shape B x T x V
        :param target: targets with shape B x T
        :param epsilon: Epsilon
        :param ignore_index: padding index
        :param reduce: whether sum all positions loss
        :return: smoothed cross entropy loss
    """

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)   # B x T x 1
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)

    # Get final smoothed cross-entropy loss
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@dataclass
class CifCriterionConfig(FairseqDataclass):
    # General settings
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="char",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )

    # Settings of cif losses
    apply_quantity_loss: bool = field(
        default=True, 
        metadata={"help": "apply quantity loss"}
    )
    apply_ctc_loss: bool = field(
        default=True,
        metadata={"help": "apply ctc loss"}
    )
    quantity_loss_lambda: float = field(
        default=1.0,
        metadata={"help": "the interpolation weight of quantity loss"}
    )
    ctc_loss_lambda: float = field(
        default=0.25,
        metadata={"help": "the interpolation weight of ctc loss"}
    )
    apply_label_smoothing: bool = field(
        default=False,
        metadata={"help": "apply label smoothing over cross entropy loss"}
    )
    label_smoothing_type: str = field(
        default="uniform",
        metadata={"help": "specify the label smoothing type"}
    )
    label_smoothing_rate: float = field(
        default=0.1,
        metadata={"help": "the rate of label smoothing"}
    )

@register_criterion("cif", dataclass=CifCriterionConfig)
class CifCriterion(FairseqCriterion):
    def __init__(self, cfg: CifCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = task.target_dictionary.index("<ctc_blank>") \
            if "<ctc_blank>" in task.target_dictionary.indices else task.target_dictionary.bos()
        self.pad_idx = task.target_dictionary.pad() # 1
        self.eos_idx = task.target_dictionary.eos() # 2
        self.bos_idx = task.target_dictionary.bos() # 0
        self.post_process = cfg.post_process
        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        # Register losses settings
        self.apply_quantity_loss = cfg.apply_quantity_loss
        self.apply_ctc_loss = cfg.apply_ctc_loss
        self.quantity_loss_lambda = cfg.quantity_loss_lambda
        self.ctc_loss_lambda = cfg.ctc_loss_lambda

        # Register label smoothing settings
        self.label_smoothing_type = cfg.label_smoothing_type
        self.label_smoothing_rate = cfg.label_smoothing_rate
        self.apply_label_smoothing = cfg.apply_label_smoothing

    def get_loss(self, model, sample, net_output, reduce=True):
        # Get model outputs
        ctc_logits = net_output["ctc_logits"]     # B x T x V
        quantity_out = net_output["quantity_out"]   # 1
        cif_out_padding_mask = net_output["cif_out_padding_mask"]   # B x T
        decoder_out = net_output["decoder_out"][0]  # Get final decoder outputs (logits for cross-entropy loss)

        # Collect src_lengths for the calculation of ctc loss
        non_padding_mask = ~net_output["encoder_padding_mask"]
        input_lengths = non_padding_mask.int().sum(-1)

        # Collect targets and target_length for ctc loss and ce loss
        target_lengths = sample["target_lengths"]   # targets length without eos
        target_with_eos = sample["target"]
        target_with_eos_lengths = target_lengths    # targets length with eos

        # Adjust targets: move the eos token from the last location to the end of valid location
        batch_size = target_with_eos.size(0)
        target_with_eos_non_padding_mask = \
            ((target_with_eos != self.eos_idx) & (target_with_eos != self.pad_idx)).int()  # [B x T]
        add_eos_idx = \
            ((target_with_eos * target_with_eos_non_padding_mask) != 0).int().sum(dim=-1).unsqueeze(dim=-1)  # [B x 1]
        add_one_hot_tensor = torch.zeros(
            batch_size, target_with_eos_non_padding_mask.size(1)
        ).int().cuda().scatter_(1, add_eos_idx, 1) * self.eos_idx
        adjusted_target_with_eos = torch.where(
            ((target_with_eos.int() * target_with_eos_non_padding_mask) + add_one_hot_tensor) == 0,
            torch.ones_like(target_with_eos).int().cuda() * self.pad_idx,
            (target_with_eos.int() * target_with_eos_non_padding_mask) + add_one_hot_tensor,
        )

        # Calculate the ctc loss on encoder outputs
        ctc_loss = torch.tensor(0.0)
        if self.apply_ctc_loss:
            pad_mask = (adjusted_target_with_eos != self.pad_idx)
            targets_flat = adjusted_target_with_eos.masked_select(pad_mask)
            ctc_lprobs = model.get_probs_from_logits(
                ctc_logits, log_probs=True
            ).contiguous()  # (B, T, V) from the encoder
            target_lengths_for_ctc_loss = target_with_eos_lengths
            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    ctc_lprobs.transpose(0, 1), # T x B x v
                    targets_flat,
                    input_lengths,
                    target_lengths_for_ctc_loss,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity
                )

        # Calculate the quantity loss
        qtt_loss = torch.tensor(0.0)
        if self.apply_quantity_loss:
            target_lengths_for_qtt_loss = target_with_eos_lengths   # Lengths after adding eos token, [B]
            qtt_loss = torch.abs(quantity_out - target_lengths_for_qtt_loss).sum()

        # Calculate the cross-entropy loss
        cif_max_len = cif_out_padding_mask.size(1)    # Get max length of cif outputs
        tgt_max_len = target_with_eos_lengths.max()   # Get max length of targets
        reg_min_len = min(cif_max_len, tgt_max_len)   # Obtain the minimum length of cif length and target length
        ce_logprobs = model.get_probs_from_logits(
            decoder_out, log_probs=True).contiguous()  # B x T x V
        truncated_target = adjusted_target_with_eos[:, :reg_min_len]    # Truncate target to reg_min_len, B x T
        truncated_ce_logprobs = ce_logprobs[:, :reg_min_len, :]         # Truncate ce probs to reg_min_len,  B x T x V
        # Truncate target to the minimum length of original target and cif outputs,
        # because sometimes the firing number of CIF may drop <eos>.

        if not self.apply_label_smoothing:
            truncated_ce_logprobs = \
                truncated_ce_logprobs.view(-1, truncated_ce_logprobs.size(-1))
            truncated_target = \
                truncated_target.contiguous().view(-1)  # flatten targets tensor
            # print(truncated_target)
            ce_loss = F.nll_loss(
                truncated_ce_logprobs,
                truncated_target.long(),
                ignore_index=self.pad_idx,
                reduction="sum" if reduce else "none",
            )  # CE loss is the summation of all tokens, without any form of averaging
        else:
            if self.label_smoothing_type == "uniform":
                ce_loss, _ = label_smoothed_nll_loss(
                    truncated_ce_logprobs,
                    truncated_target.long(),
                    self.label_smoothing_rate,
                    self.pad_idx,
                    reduce=True if reduce else False)
            else:
                raise NotImplementedError("Invalid option: %s" % self.label_smoothing_type)

        # Calculate the total loss
        loss = ce_loss + self.quantity_loss_lambda * qtt_loss + self.ctc_loss_lambda * ctc_loss

        # Collect the number of tokens in current batch
        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item())
        ntokens_with_eos = (
            target_with_eos_lengths.sum().item())
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        # Build final logging outputs
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ce_loss": utils.item(ce_loss.data),
            "ctc_loss": utils.item(ctc_loss.data),
            "quantity_loss": utils.item(qtt_loss.data),
            "ntokens": ntokens,
            "ntokens_with_eos": ntokens_with_eos,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        # Evaluate on valid sets
        if not model.training:
            with torch.no_grad():
                lprobs_t = ce_logprobs.float().contiguous().cpu()
                cif_lengths = cif_out_padding_mask.int().sum(dim=-1)  # B x T

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0

                # Loop over all hypothesis
                for lp, t, inp_l in zip(
                        lprobs_t,
                        adjusted_target_with_eos,
                        cif_lengths
                ):
                    # print("cur length: ")
                    # print(inp_l)
                    lp = lp[:inp_l].unsqueeze(0)

                    # Process targets
                    # p = (t != self.task.target_dictionary.pad()) & (t != self.task.target_dictionary.eos())
                    p = (t != self.task.target_dictionary.pad())
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    # Process hypothesis
                    # If decoded is None, conduct greedy search decoding
                    # toks = lp.argmax(dim=-1).unique_consecutive()
                    # For ctc decoding, remove blank indices and repetitive consecutive ids

                    # Handle lp without elements
                    if min(lp.shape) == 0:
                        toks = targ
                    else:
                        toks = lp.argmax(dim=-1)

                    pred_units_arr = \
                        toks[(toks != self.blank_idx) & (toks != self.pad_idx)].tolist()

                    # Calculate character error
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()
                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    # Calculate word error
                    dist = editdistance.eval(pred_words_raw, targ_words)
                    w_errs += dist
                    wv_errs += dist
                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        net_output = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            target_lengths=sample["target_lengths"]
        )
        loss, sample_size, logging_output = self.get_loss(
            model, sample, net_output, reduce=True
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """
            Aggregate logging outputs from data parallel training.
        """

        loss_sum = utils.item(
            sum(log.get("loss", 0) for log in logging_outputs)
        )
        ce_loss_sum = utils.item(
            sum(log.get("ce_loss", 0) for log in logging_outputs)
        )
        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        quantity_loss_sum = utils.item(
            sum(log.get("quantity_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(
            sum(log.get("ntokens", 0) for log in logging_outputs)
        )
        ntokens_with_eos = utils.item(
            sum(log.get("ntokens_with_eos", 0) for log in logging_outputs)
        )
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "quantity_loss", quantity_loss_sum / sample_size / math.log(2), sample_size, round=5
        )
        metrics.log_scalar(
            "ntokens", ntokens)
        metrics.log_scalar(
            "ntokens_with_eos", ntokens_with_eos)
        metrics.log_scalar(
            "nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["c_errors"].sum * 100.0 / meters["c_total"].sum, 3
                )
                if meters["c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["w_errors"].sum * 100.0 / meters["w_total"].sum, 3
                )
                if meters["w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["wv_errors"].sum * 100.0 / meters["w_total"].sum, 3
                )
                if meters["w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
