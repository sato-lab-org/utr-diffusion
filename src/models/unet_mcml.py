import itertools
from cProfile import label
from functools import partial
from typing import Optional
import torch
from memory_efficient_attention_pytorch import Attention as EfficientAttention
from tensorflow.python.layers.core import dropout
from src.models.layers import *
from src.models.label_emb_strategies import build_label_encoder
import torch.nn as nn

class UNet_Masked_Continuous_Multi_Labels(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4),
        channels: int = 1,
        resnet_block_groups: int = 8,
        learned_sinusoidal_dim: int = 18,
        num_label: int = 2,
        seq_len: int = 200,
        output_attention: bool = False,
        dropout: float = 0.0,
        label_emb_mode: str = 'concat'
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dropout_rate = dropout
        self.num_label = num_label

        # determine dimensions
        self.channels = channels
        # if you want to do self conditioning uncomment this
        input_channels = channels
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *(dim * m for m in dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups, dropout=self.dropout_rate)

        # time embeddings， since auther used memory_attn， query and key must be same dim
        time_dim = 4 * seq_len

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_emb_mode = label_emb_mode
        self.class_emb = build_label_encoder(
            label_emb_mode = self.label_emb_mode,
            num_label = num_label,
            emb_dim = time_dim
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out, ind==(len(in_out)- 2)) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in, ind==0, w_scale=2) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.cross_attn = EfficientAttention(
            dim=self.seq_len,
            dim_head=64,
            heads=4,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(self.seq_len*4) # when 200bp, dim * 4

    def forward(self, x: torch.Tensor, time: torch.Tensor = None, classes: torch.Tensor = None, label_is_presence = None):

        x = self.init_conv(x)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()

        if classes is not None:
            class_emb = self.class_emb(classes, label_is_presence)

            t_start += class_emb
            t_mid += class_emb
            t_end += class_emb
            t_cross += class_emb

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)

            x = block2(x, t_start)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)

        x_layer_normed = self.norm_to_cross(x.reshape(-1, self.seq_len * 4))
        crossattention_out = self.cross_attn(x_layer_normed.reshape(-1, 4, self.seq_len), t_cross.reshape(-1, 4, self.seq_len))

        x = x + crossattention_out.view(-1, 1, 4, self.seq_len)
        if self.output_attention:
            return x, crossattention_out
        return x
