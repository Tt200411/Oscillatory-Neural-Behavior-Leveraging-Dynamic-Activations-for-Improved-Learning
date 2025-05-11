# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from lee_oc import LeeOscillator

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='relu', 
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'),
                 # —— 新增：池化参数 —— 
                 use_pooling=True, pool_kernel=2, pool_stride=2,
                 lee_type=1):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention & Encoder / Decoder
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.encoder = Encoder(
            [ EncoderLayer(
                  AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                 d_model, n_heads, mix=False),
                  d_model, d_ff, dropout=dropout, activation=activation
              ) for _ in range(e_layers)
            ],
            [ ConvLayer(d_model) for _ in range(e_layers-1) ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [ DecoderLayer(
                  AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads, mix=mix),
                  AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads, mix=False),
                  d_model, d_ff, dropout=dropout, activation=activation
              ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

        # Lee 振荡器
        self.lee = LeeOscillator()
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'lee':
            lee_funcs = {
                1: self.lee.type1,
                2: self.lee.type2,
                3: self.lee.type3,
                4: self.lee.type4,
                5: self.lee.type5,
                6: self.lee.type6,
                7: self.lee.type7,
                8: self.lee.type8,
            }
            self.activation = lee_funcs[lee_type]

        # —— 初始化池化层 —— 
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool_enc = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
            self.pool_dec = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        else:
            self.pool_enc = self.pool_dec = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # —— Encoder Embedding + 可选池化 —— 
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, C]
        if self.pool_enc is not None:
            # [B, L, C] -> [B, C, L]
            enc_out = enc_out.transpose(1,2)
            enc_out = self.pool_enc(enc_out)
            enc_out = enc_out.transpose(1,2)            # -> [B, L', C]

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # —— Decoder Embedding + 可选池化 —— 
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, L, C]
        if self.pool_dec is not None:
            dec_out = dec_out.transpose(1,2)
            dec_out = self.pool_dec(dec_out)
            dec_out = dec_out.transpose(1,2)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, out_len, C]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        Attn = ProbAttention if attn=='prob' else FullAttention
        encoders = [
            Encoder(
                [ EncoderLayer(
                      AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                     d_model, n_heads, mix=False),
                      d_model, d_ff, dropout=dropout, activation=activation
                  ) for _ in range(el)
                ],
                [ ConvLayer(d_model) for _ in range(el-1) ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens=list(range(len(e_layers))))

        self.decoder = Decoder(
            [ DecoderLayer(
                  AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads, mix=mix),
                  AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                 d_model, n_heads, mix=False),
                  d_model, d_ff, dropout=dropout, activation=activation
              ) for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
