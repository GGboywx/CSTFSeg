import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import build_backbone
# todo 新增：从 modules 文件中导入 TransformerDecoder, Transformer, FuzzyLayer, EdgeGuidedDecoder, 和 ASPP
from modules import TransformerDecoder, Transformer, FuzzyLayer, EdgeGuidedDecoder, ASPP
from einops import rearrange
import itertools
import numpy as np


class token_encoder(nn.Module):
    def __init__(self, in_chan=32, token_len=8, heads=8):
        super(token_encoder, self).__init__()
        self.token_len = token_len
        self.conv_a = nn.Conv2d(in_chan, token_len, kernel_size=1, padding=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, token_len, in_chan))
        self.transformer = Transformer(dim=in_chan, depth=3, heads=heads, dim_head=64, mlp_dim=64, dropout=0)

    def forward(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        tokens = torch.einsum('bln, bcn->blc', spatial_attention, x)

        tokens += self.pos_embedding
        x = self.transformer(tokens)

        return x


class CDNet(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16, img_size=512,
                 img_chan=3, chan_num=32, n_class=1, fuzzy_num=4):
        super(CDNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, img_chan,out_channels_for_pa=chan_num)


        self.aspp = ASPP(in_channels=chan_num, out_channels=chan_num, atrous_rates=[12,24,36])


        self.token_encoder = token_encoder(in_chan=chan_num, heads=8)

        self.transformer_decoder_s8 = TransformerDecoder(
            dim=chan_num, depth=2, heads=8, dim_head=64, mlp_dim=64, dropout=0
        )
        self.transformer_decoder_s4 = TransformerDecoder(
            dim=chan_num, depth=2, heads=8, dim_head=64, mlp_dim=64, dropout=0
        )

        self.decoder = EdgeGuidedDecoder(
            in_channels=chan_num,
            out_channels=chan_num,
            n_class=n_class,
            fuzzy_num=fuzzy_num
        )

    def forward(self, img1, img_size):
        # 1. 从骨干网络中提取多尺度特征
        out1_s16_raw, out1_s8, out1_s4 = self.backbone(img1)

        # **新增：将 s16 特征送入 ASPP**
        out1_s16_aspp = self.aspp(out1_s16_raw)
        # out1_s16_aspp 现在是 (B, chan_num, H/16, W/16)

        # 2. 从高层次特征 (现在是经过 ASPP 处理的 s16) 中提取全局上下文 token
        global_tokens = self.token_encoder(out1_s16_aspp)

        # 3. 将中低层特征 (s8, s4) 展平为序列，以输入 TransformerDecoder
        b, c, h8, w8 = out1_s8.shape
        b, c, h4, w4 = out1_s4.shape

        out1_s8_flat = rearrange(out1_s8, 'b c h w -> b (h w) c')
        out1_s4_flat = rearrange(out1_s4, 'b c h w -> b (h w) c')

        # 4. 使用 TransformerDecoder 将全局 tokens 融合到中低层特征中
        refined_s8_flat = self.transformer_decoder_s8(out1_s8_flat, global_tokens)
        refined_s4_flat = self.transformer_decoder_s4(out1_s4_flat, global_tokens)

        # 5. 将融合后的特征序列重新恢复为特征图
        refined_s8 = rearrange(refined_s8_flat, 'b (h w) c -> b c h w', h=h8, w=w8)
        refined_s4 = rearrange(refined_s4_flat, 'b (h w) c -> b c h w', h=h4, w=w4)

        # 6. 将经过 ASPP 处理的 s16 特征（作为 feat_s16），以及融合后的 s8, s4 特征输入解码器
        refined_feat, seg_outputs, all_edge_outputs = self.decoder(out1_s16_aspp, refined_s8, refined_s4, img_size=img_size)

        return refined_feat, seg_outputs, all_edge_outputs

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()