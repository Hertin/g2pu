"""BERT encoder definition."""

from typing import List, Optional, Tuple

import torch
import contextlib
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from .transformer_encoder import TransformerEncoder

import logging
import transformers
from transformers import BertModel, BertPreTrainedModel
from transformers import AutoTokenizer, AutoModelForMaskedLM
import fairseq
import torch.nn.functional as F
from fairseq.modules import GradMultiply
from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
from fairseq.modules import LayerNorm
import numpy as np

class BiEncoder(AbsEncoder):
    def __init__(
        self,
        acoustic_feature: str,
        unit_vocab_size: int = None,
        ctc_weight: float = 0.,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        pretrained_unit_embed: Optional[str] = None,
        freeze_finetune_updates: Optional[int] = None,
        roberta: Optional[str] = None,
        ft_mode: Optional[str] = 'full',
        data: Optional[str] = None,
        bert_model: Optional[str] = None, # set to chinese for backward compatibiity ove previous experiments
        bert_ft_mode: Optional[str] = None,
        pretrained_w2v_model: Optional[str] = None,
        conv_feature_layers: Optional[str] = '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,1)]',
        feature_grad_mult: Optional[float] = 0.1,
        **kwargs
    ):
        assert check_argument_types()
        super().__init__()

        if bert_model is None: # just for compatibility
            bert_model = 'bert-base-chinese'
        if bert_ft_mode is None:
            bert_ft_mode = 'weight'
        self.bert_ft_mode = bert_ft_mode

        self.ctc_weight = ctc_weight
        self.bert_model = bert_model
        if ctc_weight < 1.0:
            self.bert = AutoModelForMaskedLM.from_pretrained(bert_model)
            self.bert_layer_weights = None
            if self.bert_ft_mode == 'weight':
                for param in self.bert.parameters():
                    param.requires_grad = False
                num_bert_layer = len(self.bert.bert.encoder.layer)
                self.bert_layer_weights = torch.nn.Parameter(torch.ones(num_bert_layer+1).float() / (num_bert_layer+1))
            bert_hidden_size = self.bert.bert.config.hidden_size
            self.bert_proj = torch.nn.Linear(bert_hidden_size, output_size)
        
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.acoustic_feature = acoustic_feature
        self.unit_vocab_size = unit_vocab_size
        self.pretrained_w2v_model = pretrained_w2v_model
        self.freeze_finetune_updates = freeze_finetune_updates
        if acoustic_feature == 'speech':
            # from transformers import Wav2Vec2Model, Wav2Vec2Config
            # configuration = Wav2Vec2Config()
            # self.w2v = Wav2Vec2Model(configuration)
            # for param in self.w2v.parameters():
            #     param.requires_grad = False
            # num_w2v_layer = self.w2v.config.num_hidden_layers
            # # self.w2v_layer_weights = torch.nn.Parameter(torch.ones(num_w2v_layer+1).float() / (num_w2v_layer+1))
            # w2v_hidden_size = self.w2v.config.hidden_size
            # self.w2v_proj = torch.nn.Linear(w2v_hidden_size, output_size)
            
            logging.info(f'loading pretrained wav2vec from {pretrained_w2v_model}')
            models, cfg = fairseq.checkpoint_utils.load_model_ensemble([pretrained_w2v_model])
            w2v_hidden_size = cfg.model.encoder_embed_dim
            self.w2v = models[0]
            for param in self.w2v.parameters():
                param.requires_grad = False
            self.w2v_proj = torch.nn.Linear(w2v_hidden_size, output_size)

            assert freeze_finetune_updates is not None
        elif acoustic_feature == 'speech_wcnn':
            # from transformers import Wav2Vec2Model, Wav2Vec2Config
            # configuration = Wav2Vec2Config()
            # self.w2v = Wav2Vec2Model(configuration)
            # for param in self.w2v.parameters():
            #     param.requires_grad = False
            # num_w2v_layer = self.w2v.config.num_hidden_layers
            # # self.w2v_layer_weights = torch.nn.Parameter(torch.ones(num_w2v_layer+1).float() / (num_w2v_layer+1))
            # w2v_hidden_size = self.w2v.config.hidden_size
            # self.w2v_proj = torch.nn.Linear(w2v_hidden_size, output_size)
            
            logging.info(f'loading pretrained wav2vec from {pretrained_w2v_model}')
            models, cfg = fairseq.checkpoint_utils.load_model_ensemble([pretrained_w2v_model])
            w2v_hidden_size = cfg.model.encoder_embed_dim
            self.w2v = models[0]
            for param in self.w2v.parameters():
                param.requires_grad = False
            self.w2v_proj = torch.nn.Linear(w2v_hidden_size, output_size)
            w2v_fext_hidden_size = eval(cfg.model.conv_feature_layers)[0][0]
            self.w2v_fext_proj = torch.nn.Linear(w2v_fext_hidden_size, output_size)

            assert freeze_finetune_updates is not None
        elif acoustic_feature == 'speech_cnn':
            logging.info(f'loading pretrained wav2vec for cnn feature extractor from {pretrained_w2v_model}')
            models, cfg = fairseq.checkpoint_utils.load_model_ensemble([pretrained_w2v_model])
            w2v_hidden_size = eval(cfg.model.conv_feature_layers)[0][0]
            self.w2v = models[0]
            for param in self.w2v.parameters():
                param.requires_grad = False
            self.w2v_proj = torch.nn.Linear(w2v_hidden_size, output_size)
        elif acoustic_feature == 'cnn':
            logging.info('train a cnn from scratch')
            self.conv_feature_layers = conv_feature_layers
            self.feature_enc_layers = eval(conv_feature_layers)
            self.conv = ConvFeatureExtractionModel(
                conv_layers=self.feature_enc_layers,
                dropout=0.0,
                mode='default',
                conv_bias=False,
            )
            conv_hidden_size = self.feature_enc_layers[0][0]
            self.conv_proj = torch.nn.Linear(conv_hidden_size, output_size)
            self.feature_grad_mult = feature_grad_mult
            self.conv_layer_norm = LayerNorm(conv_hidden_size)
            logging.info(f'receptive field size: {self.total_receptive_field()} stride: {self.total_stride()}')

        elif acoustic_feature == 'unit':
            self.input_layer = input_layer
            self.pretrained_unit_embed = pretrained_unit_embed
            if pretrained_unit_embed is not None:
                import joblib
                logging.info(f'loading pretrained embedding from {self.pretrained_unit_embed}')
                centroids = torch.from_numpy(joblib.load(pretrained_unit_embed).cluster_centers_)
                vsize, hdim = centroids.shape
                pretrained_unit_embed = torch.cat(
                    [torch.zeros(2, hdim), centroids, torch.zeros(1, hdim)], dim=0)

            self.unit_encoder = TransformerEncoder(
                unit_vocab_size, # 503
                output_size, # vocab size 256,
                attention_heads, # 4
                linear_units, # 2048
                num_blocks, # 6
                dropout_rate, # 0.1
                positional_dropout_rate, # 0.1
                attention_dropout_rate, # 0.0
                input_layer,
                pos_enc_class,
                normalize_before,
                concat_after,
                positionwise_layer_type,
                positionwise_conv_kernel_size,
                padding_idx,
                interctc_layer_idx,
                interctc_use_conditioning,
                pretrained_unit_embed
            )
        elif acoustic_feature == 'unit_roberta':
            assert roberta is not None
            assert data is not None
            self.data = data
            model, cfg, task, = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [roberta], arg_overrides={'data': data})
            roberta = model[0]
            self.unit_encoder = roberta
            self.task = task
            
            roberta_hidden_size = roberta.args.encoder_embed_dim
            self.roberta_proj = None
            if roberta_hidden_size != output_size:
                self.roberta_proj = torch.nn.Linear(roberta_hidden_size, output_size)
            self.ft_mode = ft_mode
            if ft_mode == 'weight':
                for param in self.unit_encoder.parameters():
                    param.requires_grad = False
                num_roberta_layer = roberta.args.encoder_layers
                self.roberta_layer_weights = torch.nn.Parameter(torch.ones(num_roberta_layer+1).float() / (num_roberta_layer+1))
                
        else:
            raise ValueError(f'Unkown acoustic feature type: {acoustic_feature}')

        self._output_size = output_size

        self.num_updates = 0
        self.unfreezed = False

    def total_stride(self,):
        total_stride = np.prod([x[-1] for x in self.feature_enc_layers])
        return total_stride

    def total_receptive_field(self,):
        r = 0
        for l, (d, k, s) in enumerate(self.feature_enc_layers):
            r += (k-1) * np.prod([ss for dd, kk, ss in self.feature_enc_layers[:l]]) + 1
        return r

    def output_size(self) -> int:
        return self._output_size

    def unfreeze(self):
        if not self.unfreezed:
            for param in self.w2v.parameters():
                param.requires_grad = True
            self.unfreezed = True
            logging.info(f'start updating wav2vec')

    def wav2vec_padding_mask(self, features, padding_mask):
        input_lengths = (1 - padding_mask.long()).sum(-1)
        # apply conv formula to get real output_lengths
        output_lengths = self.w2v._get_feat_extract_output_lengths(input_lengths)

        padding_mask = torch.zeros(
            features.shape[:2], dtype=features.dtype, device=features.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return padding_mask

    def conv_padding_mask(self, features, padding_mask):
        input_lengths = (1 - padding_mask.long()).sum(-1)
        # apply conv formula to get real output_lengths
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        padding_mask = torch.zeros(
            features.shape[:2], dtype=features.dtype, device=features.device
        )

        # these two operations makes sure that all values
        # before the output lengths indices are attended to
        padding_mask[
            (
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            )
        ] = 1
        padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        return padding_mask

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        grapheme: torch.Tensor, 
        grapheme_lengths: torch.Tensor, 
        unit: Optional[torch.Tensor] = None, 
        unit_lengths: Optional[torch.Tensor] = None,
        prev_states: Optional[torch.Tensor] = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """

        # logging.info(f'unit {unit.shape} {unit_lengths}')
        grapheme_masks = (~make_pad_mask(grapheme_lengths)).to(grapheme.device)
        unit_masks = (~make_pad_mask(unit_lengths)).to(unit.device)
        
        # forward grapheme 
        grapheme_out = None
        if self.ctc_weight < 1.0:
            grapheme[~grapheme_masks] = self.tokenizer.sep_token_id
            if self.bert_ft_mode == 'weight':
                with torch.no_grad(): 
                    grapheme_out = self.bert(grapheme, attention_mask=grapheme_masks, output_hidden_states=True)
                    grapheme_out = torch.stack([h for h in grapheme_out.hidden_states]) # num_layer x B x T x H
                grapheme_out = (self.bert_layer_weights[:,None,None,None] * grapheme_out).sum(0) # B x T x H
                grapheme_out = self.bert_proj(grapheme_out)
            elif self.bert_ft_mode == 'full':
                grapheme_out = self.bert(grapheme, attention_mask=grapheme_masks, output_hidden_states=True)
                grapheme_out = grapheme_out.hidden_states[-1]
                grapheme_out = self.bert_proj(grapheme_out)
            else:
                raise ValueError(f'Unknown finetune mode {self.bert_ft_mode}')
        # forward phoneme
        if self.acoustic_feature == 'speech':
            ft = self.num_updates >= self.freeze_finetune_updates
            if ft: self.unfreeze()

            with torch.no_grad() if not ft else contextlib.ExitStack():
                unit_out = self.w2v.extract_features(source=unit, padding_mask=~unit_masks)
                if unit_out['padding_mask'] is None:
                    _b, _t, _h = unit_out['x'].shape
                    unit_lengths = torch.LongTensor([_t] * _b).to(unit_lengths)
                else:
                    unit_lengths = (~unit_out['padding_mask']).sum(dim=1)
                unit_out = unit_out['x']
                # unit_out = self.w2v(unit, attention_mask=unit_masks).last_hidden_state
                # unit_out = self.w2v(unit, attention_mask=unit_masks, output_hidden_states=True)
                # unit_out = torch.stack([h for h in unit_out.hidden_states])
            # unit_out = (self.w2v_layer_weights[:,None,None,None] * unit_out).sum(0) # B x T x H
            unit_out = self.w2v_proj(unit_out)
            # unit_lengths = self.w2v._get_feat_extract_output_lengths(unit_lengths)
            assert unit_lengths.max() == unit_out.shape[1], f'{unit_lengths} == {unit_out.shape}'
        elif self.acoustic_feature == 'speech_wcnn':
            ft = self.num_updates >= self.freeze_finetune_updates
            if ft: self.unfreeze()

            with torch.no_grad() if not ft else contextlib.ExitStack():
                source = unit
                if self.w2v.feature_grad_mult > 0:
                    features = self.w2v.feature_extractor(source)
                    if self.w2v.feature_grad_mult != 1.0:
                        features = GradMultiply.apply(features, self.w2v.feature_grad_mult)
                else:
                    with torch.no_grad():
                        features = self.w2v.feature_extractor(source)
                features = features.transpose(1, 2)
                features = self.w2v.layer_norm(features)
                
                unit_out = self.w2v.extract_features(source=unit, padding_mask=~unit_masks)
                if unit_out['padding_mask'] is None:
                    _b, _t, _h = unit_out['x'].shape
                    unit_lengths = torch.LongTensor([_t] * _b).to(unit_lengths)
                else:
                    unit_lengths = (~unit_out['padding_mask']).sum(dim=1)
                unit_out = unit_out['x']

            unit_out = self.w2v_proj(unit_out) + self.w2v_fext_proj(features)

            assert unit_lengths.max() == unit_out.shape[1], f'{unit_lengths} == {unit_out.shape}'
        elif self.acoustic_feature == 'speech_cnn':
            ft = self.num_updates >= self.freeze_finetune_updates
            if ft: self.unfreeze()

            with torch.no_grad() if not ft else contextlib.ExitStack():
                source = unit
                if self.w2v.feature_grad_mult > 0:
                    features = self.w2v.feature_extractor(source)
                    if self.w2v.feature_grad_mult != 1.0:
                        features = GradMultiply.apply(features, self.w2v.feature_grad_mult)
                else:
                    with torch.no_grad():
                        features = self.w2v.feature_extractor(source)
                features = features.transpose(1, 2)
                features = self.w2v.layer_norm(features)
            
            unit_out = self.w2v_proj(features)
            padding_mask = self.wav2vec_padding_mask(features, ~unit_masks)
            unit_lengths = (~padding_mask).sum(dim=1)
            assert unit_lengths.max() == unit_out.shape[1], f'{unit_lengths} == {unit_out.shape}'
        
        elif self.acoustic_feature == 'cnn':
            source = unit
            features = self.conv(source)
            features = GradMultiply.apply(features, self.feature_grad_mult)
            features = features.transpose(1, 2)
            features = self.conv_layer_norm(features)
            unit_out = self.conv_proj(features)
            padding_mask = self.conv_padding_mask(features, ~unit_masks)
            unit_lengths = (~padding_mask).sum(dim=1)
            assert unit_lengths.max() == unit_out.shape[1], f'{unit_lengths} == {unit_out.shape}'

        elif self.acoustic_feature == 'unit':
            unit[~unit_masks] = self.unit_vocab_size - 1
            unit_out, unit_lengths, _ = self.unit_encoder(unit, unit_lengths)
        elif self.acoustic_feature == 'unit_roberta':
            eos_ix = self.task.dictionary.eos_index            
            unit[~unit_masks] = eos_ix

            if self.ft_mode == 'weight':
                with torch.no_grad():
                    z, hidden_states = self.unit_encoder(
                        src_tokens=unit, src_lengths=unit_lengths, features_only=True, return_all_hiddens=True)
                unit_out = torch.stack([h for h in hidden_states['inner_states']])
                unit_out = (F.softmax(self.roberta_layer_weights, dim=0)[:,None,None,None] * unit_out).sum(0).transpose(0,1)
            elif self.ft_mode == 'full':
                try:
                    unit_out, hidden_states = self.unit_encoder(
                        src_tokens=unit, src_lengths=unit_lengths, features_only=True, return_all_hiddens=True)
                except Exception as e:
                    logging.info(f'unit {unit.shape} {unit}')
                    logging.info(f'unit leng {unit_lengths.shape} {unit_lengths}')
                    raise e
            if self.roberta_proj is not None:
                unit_out = self.roberta_proj(unit_out)
            else:
                raise ValueError(f'Unknown finetune mode {self.ft_mode}')
            
            assert unit_lengths.max() == unit_out.shape[1], f'{unit_lengths} == {unit_out.shape}'
        else:
            raise ValueError(f'Unknown acoustic feature: {self.acoustic_feature}')
        
        if self.training:
            self.num_updates += 1
            if self.num_updates % 1000 == 0:
                logging.info(f'[biencoder] number of updates {self.num_updates}')
            if self.num_updates == self.freeze_finetune_updates:
                logging.info(f'[biencoder] start finetuning wav2vec ...')

        return grapheme_out, grapheme_lengths, unit_out, unit_lengths
