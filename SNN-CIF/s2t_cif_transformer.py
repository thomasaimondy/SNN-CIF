#!/usr/bin/env python3

import logging
import math
from re import X
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,

)
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerConfig
from fairseq.models.speech_to_text.cif_transformer import CifMiddleware, CifMiddleware_1, CifMiddleware_2, CifMiddleware_3
from fairseq.modules import (
    FairseqDropout,
    TransformerEncoderLayer,
    AdaptiveSoftmax,
    BaseLayer,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from torch import Tensor


logger = logging.getLogger(__name__)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerDecoderBase':
        return 'TransformerDecoder'
    else:
        return module_name


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1) # GLU activation will cause dimension discount 50% in default.

        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)

        return x, self.get_out_seq_lens_tensor(src_lengths)


@register_model("s2t_cif_transformer")
class S2TCifTransformerModel(FairseqEncoderDecoderModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # input
        parser.add_argument(
            "--conv-kernel-sizes",
            type=str,
            metavar="N",
            help="kernel sizes of Conv1d subsampling layers",
        )
        parser.add_argument(
            "--conv-channels",
            type=int,
            metavar="N",
            help="# of channels in Conv1d subsampling layers",
        )
        # Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            '--encoder-freezing-updates',
            type=int,
            metavar='N',
            help='freeze encoder for first N updates'
        )
        parser.add_argument(
            "--cross-self-attention",
            type=bool,
            default=False,
        )
        parser.add_argument(
            "--no-decoder-final-norm",
            type=bool
        )

        # Cif settings
        parser.add_argument(
            "--cif-embedding-dim",
            type=int,
            help="the dimension of the inputs of cif module"
        )
        parser.add_argument(
            "--produce-weight-type",
            type=str,
            help="choose how to produce the weight for accumulation"
        )
        parser.add_argument(
            "--cif-threshold",
            type=float,
            help="the threshold of firing"
        )
        parser.add_argument(
            "--conv-cif-layer-num",
            type=int,
            help="the number of convolutional layers for cif weight generation"
        )
        parser.add_argument(
            "--conv-cif-width",
            type=int,
            help="the width of kernel of convolutional layers"
        )
        parser.add_argument(
            "--conv-cif-output-channels-num",
            type=int,
            help="the number of output channels of cif convolutional layers"
        )
        parser.add_argument(
            "--conv-cif-dropout",
            type=float,
        )
        parser.add_argument(
            "--dense-cif-units-num",
            type=int,
        )
        parser.add_argument(
            "--apply-scaling",
            type=bool,
        )
        parser.add_argument(
            "--apply-tail-handling",
            type=bool,
        )
        parser.add_argument(
            "--tail-handling-firing-threshold",
            type=float,
        )
        parser.add_argument(
            "--add-cif-ctxt-layers",
            type=bool,
        )
        parser.add_argument(
            "--cif-ctxt-layers",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-embed-dim",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-ffn-embed-dim",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-attention-heads",
            type=int,
        )
        parser.add_argument(
            "--cif-ctxt-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-activation-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-attention-dropout",
            type=float,
        )
        parser.add_argument(
            "--cif-ctxt-normalize-before",
            type=bool,
        )

        # Other settings
        parser.add_argument(
            "--calulate-ctc-logits",
            type=bool,
            default=True,
        )

    @classmethod
    def build_encoder(cls, args, task):
        encoder = S2TCifTransformerEncoder(args, task)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        return CifArTransformerDecoder(
            args, task.target_dictionary, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    def get_probs_from_logits(logits, log_probs=False):
        """
            Get normalized probabilities (or log probs) from logits.
        """

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens, target_lengths, **kwargs):
        """
            The forward method inherited from the base class has a **kwargs
            argument in its input, which is not supported in torchscript. This
            method overwrites the forward method definition without **kwargs.
        """

        encoder_out = self.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            target_lengths=target_lengths
        )
        decoder_out = self.decoder(  # wqy(4)
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )

        # return decoder_out
        final_outputs = {
            # Encoder part outputs
            "encoder_padding_mask": encoder_out["raw_encoder_padding_mask"][0],  # B x T
            "ctc_logits": encoder_out["ctc_logits"][0].transpose(0, 1),  # B x T x V

            # Cif module outputs
            "quantity_out": encoder_out["quantity_out"][0],  # Quantity out for quantity loss calculation
            "cif_out": encoder_out["encoder_out"][0],  # CIF out for decoder prediction, B x T x C
            "cif_out_padding_mask": encoder_out["encoder_padding_mask"][0],  # B x T

            # Decoder part outputs
            "decoder_out": decoder_out,  # Decoder outputs (which is final logits for ce calculation)
        }

        return final_outputs  # wqy(5)

    def get_cif_output(self, src_tokens, src_lengths, target_lengths=None):
        with torch.no_grad():
            encoder_out = self.encoder(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                target_lengths=target_lengths
            )
        return {
            "cif_out": encoder_out["encoder_out"][0],   # B x T x C
            "cif_out_padding_mask": encoder_out["encoder_padding_mask"][0],  # B x T
        }

    def step_forward_decoder(self, prev_decoded_tokens, cif_outputs):
        for k,v in cif_outputs.items():
            cif_outputs[k] = [v]
        cif_outputs["encoder_out"] = cif_outputs["cif_out"]
        cif_outputs["encoder_padding_mask"] = cif_outputs["cif_out_padding_mask"]

        with torch.no_grad():
            decoder_out = self.decoder(
                prev_output_tokens=prev_decoded_tokens,
                encoder_out=cif_outputs,
            )
        return decoder_out

class S2TCifTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args, task):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        # build cif module
        self.cif = CifMiddleware(args)

        # build ctc projection
        self.ctc_proj = None
        if args.calulate_ctc_logits:
            self.ctc_proj = Linear(args.encoder_embed_dim, len(task.target_dictionary)).cuda()

    def _forward(self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False):
        
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        
        res_net = x  # resnet

        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        ctc_logits = None
        if self.ctc_proj is not None:
            ctc_logits = self.ctc_proj(x)

        x = x + res_net
        
        encoder_outputs = {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask] if encoder_padding_mask.any() else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "conv_lengths": [input_lengths],
            "ctc_logits": [ctc_logits] if ctc_logits is not None else [],   # T x B x C
        }

        cif_out = self.cif(
            encoder_outputs=encoder_outputs,
            target_lengths=target_lengths
            if self.training else None,
            input_lengths=input_lengths
        )


        encoder_outputs["raw_encoder_out"] = [x]
        encoder_outputs["raw_encoder_padding_mask"] = [encoder_padding_mask]
        encoder_outputs["encoder_out"] = [cif_out["cif_out"]]
        encoder_outputs["encoder_padding_mask"] = [cif_out["cif_out_padding_mask"].bool()]
        # encoder_outputs["encoder_padding_mask"] = [~cif_out["cif_out_padding_mask"].bool()]
        encoder_outputs["quantity_out"] = [cif_out["quantity_out"]]

        return encoder_outputs

    def forward(self, src_tokens, src_lengths, target_lengths=None, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(src_tokens, src_lengths, target_lengths,
                                  return_all_hiddens=return_all_hiddens)
        else:
            x = self._forward(src_tokens, src_lengths, target_lengths,
                              return_all_hiddens=return_all_hiddens)
        return x  # wqy(1)

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class CifArTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=True,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder_layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        self.cif_output_dim = cfg.cif_embedding_dim
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise_pq,
                cfg.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(
                (input_embed_dim + self.cif_output_dim), embed_dim, bias=False)
            if embed_dim != (input_embed_dim + self.cif_output_dim)
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        # if cfg.layernorm_embedding:
        #     self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        # else:
        #     self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder_normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear((embed_dim + self.cif_output_dim), self.output_embed_dim, bias=False)
            if self.output_embed_dim != (embed_dim + self.cif_output_dim) else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(dictionary)

    def build_output_projection(self, dictionary):
        if not self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )   # D x V
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        else:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )   # D x V
            self.output_projection.weight = self.embed_tokens.weight

    def build_decoder_layer(self, cfg, no_encoder_attn=True):
        layer = transformer_layer.TransformerDecoderLayerBaseDirectArgs(cfg, no_encoder_attn)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None

        cif_outs = encoder_out["encoder_out"][0]

        _, cif_max_len, cif_embed_dim = cif_outs.size()
        min_reg_len = min(cif_max_len, slen)

        shifted_cif_outs = torch.cat(
            [torch.zeros(bs, 1, cif_embed_dim).cuda(), cif_outs], dim=1)[:, :cif_max_len, :]

        # regularize lengths
        cif_outs = cif_outs[:, :min_reg_len, :].cuda()
        shifted_cif_outs = shifted_cif_outs[:, :min_reg_len, :].cuda()
        prev_output_tokens = prev_output_tokens[:, :min_reg_len].cuda()

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if not self.training and incremental_state is not None:
            shifted_cif_outs = shifted_cif_outs[:, -1:, :]
            cif_outs = cif_outs[:, -1:, :]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        x = torch.cat([x, shifted_cif_outs], dim=-1)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        # if self.layernorm_embedding is not None:
        #     x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = torch.cat([x, cif_outs], dim=-1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict



class TransformerDecoderScriptable(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


# this one
@register_model_architecture(model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)  # encoder_layers
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)  # decoder_layers
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 1)  #  threshold 0.99
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(args, "conv_cif_output_channels_num", 256)
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 256)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(args, "tail_handling_firing_threshold", 0.4)
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(args, "cif_ctxt_embed_dim", args.encoder_embed_dim)
    args.cif_ctxt_ffn_embed_dim = getattr(args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.cif_ctxt_attention_heads = getattr(args, "cif_ctxt_attention_heads", args.encoder_attention_heads)
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(args, "cif_ctxt_activation_dropout", args.activation_dropout)
    args.cif_ctxt_attention_dropout = getattr(args, "cif_ctxt_attention_dropout", args.attention_dropout)
    args.cif_ctxt_normalize_before = getattr(args, "cif_ctxt_normalize_before", args.encoder_normalize_before)


@register_model_architecture(model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer_1")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 1024)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(args, "conv_cif_output_channels_num", 256)
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 256)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(args, "tail_handling_firing_threshold", 0.4)
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(args, "cif_ctxt_embed_dim", args.encoder_embed_dim)
    args.cif_ctxt_ffn_embed_dim = getattr(args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.cif_ctxt_attention_heads = getattr(args, "cif_ctxt_attention_heads", args.encoder_attention_heads)
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(args, "cif_ctxt_activation_dropout", args.activation_dropout)
    args.cif_ctxt_attention_dropout = getattr(args, "cif_ctxt_attention_dropout", args.attention_dropout)
    args.cif_ctxt_normalize_before = getattr(args, "cif_ctxt_normalize_before", args.encoder_normalize_before)


@register_model_architecture(model_name="s2t_cif_transformer", arch_name="s2t_cif_transformer_s")
def base_architecture(args):
    args.encoder_freezing_updates = getattr(args, "encoder_freezing_updates", 0)
    # Convolutional subsampler
    args.conv_kernel_sizes = getattr(args, "conv_kernel_sizes", "5,5")
    args.conv_channels = getattr(args, "conv_channels", 512)
    # Transformer
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0.0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    # Cif settings
    args.cif_embedding_dim = getattr(args, "cif_embedding_dim", args.encoder_embed_dim)
    args.produce_weight_type = getattr(args, "produce_weight_type", "conv")
    args.cif_threshold = getattr(args, "cif_threshold", 0.99)
    args.conv_cif_layer_num = getattr(args, "conv_cif_layer_num", 1)
    args.conv_cif_width = getattr(args, "conv_cif_width", 3)
    args.conv_cif_output_channels_num = getattr(args, "conv_cif_output_channels_num", 256)
    args.conv_cif_dropout = getattr(args, "conv_cif_dropout", args.dropout)
    args.dense_cif_units_num = getattr(args, "dense_cif_units_num", 256)
    args.apply_scaling = getattr(args, "conv_cif_dropout", True)
    args.apply_tail_handling = getattr(args, "apply_tail_handling", True)
    args.tail_handling_firing_threshold = getattr(args, "tail_handling_firing_threshold", 0.4)
    args.add_cif_ctxt_layers = getattr(args, "add_cif_ctxt_layers", False)
    args.cif_ctxt_layers = getattr(args, "cif_ctxt_layers", 2)
    args.cif_ctxt_embed_dim = getattr(args, "cif_ctxt_embed_dim", args.encoder_embed_dim)
    args.cif_ctxt_ffn_embed_dim = getattr(args, "cif_ctxt_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.cif_ctxt_attention_heads = getattr(args, "cif_ctxt_attention_heads", args.encoder_attention_heads)
    args.cif_ctxt_dropout = getattr(args, "cif_ctxt_dropout", args.dropout)
    args.cif_ctxt_activation_dropout = getattr(args, "cif_ctxt_activation_dropout", args.activation_dropout)
    args.cif_ctxt_attention_dropout = getattr(args, "cif_ctxt_attention_dropout", args.attention_dropout)
    args.cif_ctxt_normalize_before = getattr(args, "cif_ctxt_normalize_before", args.encoder_normalize_before)
