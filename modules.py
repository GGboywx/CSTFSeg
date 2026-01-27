import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import itertools
import numpy as np


class FuzzyLayer(nn.Module):
    def __init__(self, fuzzynum, channel):
        super(FuzzyLayer, self).__init__()
        self.n = fuzzynum
        self.channel = channel

        # 保留 conv1, conv2 但改变其作用，或者使用分组卷积
        # 优化1a: 如果你想让每个通道独立进行模糊处理
        # 此时 mu 和 sigma 应该针对每个通道有不同的值
        # 或者，你可以让模糊逻辑作用于 channel 维度上，而不是空间维度
        # 我们这里尝试一个更通用的方式：Conv1x1 降维，然后进行通道独立的模糊处理，再 Conv1x1 升维。

        # 将 in_channels 映射到一个中间维度 (例如 channel // reduction)
        # 然后在这个中间维度进行模糊计算
        # 最后再映射回 channel
        reduction = 4  # 可以调整的超参数
        hidden_dim = channel // reduction if channel // reduction > 0 else 1  # 确保不为0

        self.pre_fuzzy_conv = ConvBNReLU(channel, hidden_dim, k=1, p=0)  # 1x1 卷积降维

        # 优化1b: mu 和 sigma 应该是 (1, hidden_dim, self.n) 这样的形状，
        # 让每个隐藏维度通道都有自己的模糊参数。
        self.mu = nn.Parameter(torch.randn((1, hidden_dim, self.n, 1, 1)))  # (B, C_hidden, N, 1, 1)
        self.sigma = nn.Parameter(torch.abs(torch.randn((1, hidden_dim, self.n, 1, 1))))  # (B, C_hidden, N, 1, 1)

        # 优化1c: Fuzzy 计算后，再通过 1x1 卷积恢复到原始通道数
        self.post_fuzzy_conv = ConvBNReLU(hidden_dim, channel, k=1, p=0)

        # 最终的 BN 层可能仍然有用，但由于 ConvBNReLU 已经包含了 BN，这里可以省略，或者调整
        # self.bn_final = nn.BatchNorm2d(channel, affine=True)

    def forward(self, x):
        # x: (B, C, H, W)
        x_reduced = self.pre_fuzzy_conv(x)  # (B, hidden_dim, H, W)

        # 扩展 x_reduced 维度以便与 mu, sigma 对齐
        # (B, hidden_dim, 1, H, W)
        x_expanded = x_reduced.unsqueeze(2)  # 插入 fuzzynum 维度前

        # 计算差异，这里 mu 和 sigma 已经有 hidden_dim 维度
        # diff_div_sigma_squared: (B, hidden_dim, self.n, H, W)
        diff_div_sigma_squared = ((x_expanded - self.mu) / (self.sigma + 1e-6)).pow(2)

        # 聚合模糊值，这里可以沿 fuzzynum 维度取平均或者最大等
        # (B, hidden_dim, H, W)
        aggregated_fuzzy_value = torch.mean(diff_div_sigma_squared, dim=2)

        # 得到多通道的模糊输出
        fuzzy_output_multi_channel = torch.exp(-aggregated_fuzzy_value)  # (B, hidden_dim, H, W)

        # 将模糊输出映射回原始通道数
        fNeural = self.post_fuzzy_conv(fuzzy_output_multi_channel)  # (B, C, H, W)

        # 可以选择添加一个残差连接，或者直接返回
        return x + fNeural  # 引入残差连接，让模型选择性地使用模糊增强
        # return fNeural # 或者直接返回增强后的特征


class Residual(nn.Module):
    # ... (保持不变)
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    # ... (保持不变)
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(x, x2, **kwargs) + x


class PreNorm(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class Transformer(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Cross_Attention(nn.Module):
    # ... (保持不变，已修正 self.softmax)
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.softmax = softmax

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class PreNorm2(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x2, **kwargs):
        return self.fn(self.norm(x), self.norm(x2), **kwargs)


class TransformerDecoder(nn.Module):
    # ... (保持不变)
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads=heads,
                                                        dim_head=dim_head, dropout=dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, m, mask=None):
        """target(query), memory"""
        for attn, ff in self.layers:
            x = attn(x, m, mask=mask)
            x = ff(x)
        return x


class ConvBNReLU(nn.Sequential):
    # ... (保持不变)
    def __init__(self, in_ch, out_ch, k=3, p=1, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

# **新增：ASPP 模块**
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        self.convs = nn.ModuleList()
        # 1x1 conv branch
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        # Atrous conv branches
        for rate in atrous_rates:
            self.convs.append(ASPPConv(in_channels, out_channels, rate))
        # Image pooling branch
        self.convs.append(ASPPPooling(in_channels, out_channels))

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # 可选的 dropout
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1) # Concatenate all branches
        return self.project(res) # Project back to out_channels



class LightweightCrossEdgeFusion(nn.Module):
    # ... (保持不变)
    """
    轻量的边界->分割交互模块（cross-attn-like but conv-based）
    给定分割特征Q 和 边界特征E（空间同尺寸），输出一个增强的分割特征。
    公式（近似）：
        A = sigmoid( Conv_q(Q) + Conv_k(E) )
        V = Conv_v(E)
        out = Q + Conv_out( A * V )
    这样做比直接相乘更灵活，并且计算量小。
    """

    def __init__(self, channels):
        super().__init__()
        self.conv_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, Q, E):
        # Q, E: (B, C, H, W) ； 假设尺寸已经对齐
        q = self.conv_q(Q)
        k = self.conv_k(E)
        v = self.conv_v(E)
        attn = torch.sigmoid(q + k)  # (B, C, H, W)
        out = self.proj(attn * v)
        return Q + out  # residual 加回 Q



class EdgeGuidedDecoder(nn.Module):
    # ... (保持不变)
    """
    Edge-guided Multi-scale Decoder
    输入: feat_s16, feat_s8, feat_s4（backbone 输出）
    输出: fused_feat (decoder 中间特征), seg_out (上采样到原图大小), all_edge_outputs (多尺度边缘预测输出)
    """

    def __init__(self, in_channels=32, out_channels=32, n_class=1,
                 fuzzy_num=4):  # 只保留 FuzzyLayer 的参数
        super().__init__()
        self.out_channels = out_channels

        # lateral 将不同尺度统一到 out_channels
        self.lateral16 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral8 = nn.Conv2d(in_channels, out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels, out_channels, 1)

        # 小的 conv 用于提升浅层边界信息
        self.edge_conv_s4 = ConvBNReLU(out_channels, out_channels)
        self.edge_conv_s8 = ConvBNReLU(out_channels, out_channels)
        self.edge_conv_s16 = ConvBNReLU(out_channels, out_channels)

        # ==== 新增：Fuzzy Layer 应用于 S4 边缘特征，增强细节 ====
        self.fuzzy_edge_s4 = FuzzyLayer(fuzzy_num, out_channels)


        # ==== 修正：用于统一融合后的边缘特征通道数 ====
        # 原始代码中的 nn.Conv2d(edge_feat.shape[1], p4.shape[1], 1) 动态创建层的问题被修正
        self.edge_fusion_proj = nn.Conv2d(out_channels * 3, out_channels, 1)

        # 融合多尺度分割特征
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 边界预测头（多尺度，用于输出所有尺度的边缘损失）
        # 这里用三个独立的头来预测不同尺度的边缘
        # 这里我让 S4 基于模糊增强的特征，S8和S16基于融合后的edge_feat
        self.edge_head_s4 = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, k=1, p=0),
            nn.Conv2d(out_channels // 2, 1, 1)
        )
        self.edge_head_s8 = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, k=1, p=0),
            nn.Conv2d(out_channels // 2, 1, 1)
        )
        self.edge_head_s16 = nn.Sequential(
            ConvBNReLU(out_channels, out_channels // 2, k=1, p=0),
            nn.Conv2d(out_channels // 2, 1, 1)
        )

        # # 轻量 cross fusion（把边界信息作为 key/value 指导分割）
        self.cross_fuse = LightweightCrossEdgeFusion(out_channels)



        # refinement module：融合后再 refine
        # 输入是 fused2 (C) + edge_feat (C) = 2*C
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 最终分类器
        self.classifier = nn.Conv2d(out_channels, n_class, 1)

    def forward(self, feat_s16, feat_s8, feat_s4, img_size):
        """
        feat_s16: (B, C_in, H/16, W/16)
        feat_s8:  (B, C_in, H/8,  W/8)
        feat_s4:  (B, C_in, H/4,  W/4)
        img_size: (H, W) - 原图分辨率，用于最后上采样
        """
        out_channels = self.out_channels

        # lateral -> 同通道 out_channels
        p16 = self.lateral16(feat_s16)  # (B, out_channels, H/16, W/16)
        p8 = self.lateral8(feat_s8)  # (B, out_channels, H/8,  W/8)
        p4 = self.lateral4(feat_s4)  # (B, out_channels, H/4,  W/4)

        # 上采样到 s4 尺度
        p16_up = F.interpolate(p16, size=p4.shape[2:], mode='bilinear', align_corners=True)
        p8_up = F.interpolate(p8, size=p4.shape[2:], mode='bilinear', align_corners=True)

        # 先对浅层做小 conv 强化边界信息
        e4 = self.edge_conv_s4(p4)
        e8 = self.edge_conv_s8(p8_up)
        e16 = self.edge_conv_s16(p16_up)

        # ==== 核心修改：将 Fuzzy Layer 应用于 S4 边缘特征 ====
        e4_processed_by_fuzzy = self.fuzzy_edge_s4(e4)  # 对 S4 边缘特征进行 Fuzzy 处理


        # 边界特征融合 (multi-scale)
        # 将e16 e8 e4_processed_by_fuzzy 参与融合，而不是原始的 e
        edge_feat_concat = torch.cat([e16, e8, e4_processed_by_fuzzy], dim=1)  # (B, 3*out_channels, H/4, W/4)
        edge_feat = self.edge_fusion_proj(edge_feat_concat)  # (B, out_channels, H/4, W/4)
        edge_feat = F.relu(edge_feat)  # 增加非线性

        # ==== 多尺度边界预测 ====
        # 现在有三个边缘预测头，分别基于不同尺度的特征进行预测
        # 可以根据你的需求决定这些头各自基于哪个特征
        # 这里我让 S4 基于模糊增强的特征，S8和S16基于融合后的edge_feat
        edge_pred_logits_s4 = self.edge_head_s4(e4_processed_by_fuzzy)  # 基于模糊增强后的S4边缘预测
        edge_pred_logits_s8 = self.edge_head_s8(e8)  # 基于原始S8特征预测
        edge_pred_logits_s16 = self.edge_head_s16(e16)  # 基于原始S16特征预测

        # 上采样所有边缘预测到原始图像尺寸，用于计算多尺度边缘损失
        edge_out_s4 = F.interpolate(edge_pred_logits_s4, size=img_size, mode='bilinear', align_corners=True)
        edge_out_s8 = F.interpolate(edge_pred_logits_s8, size=img_size, mode='bilinear', align_corners=True)
        edge_out_s16 = F.interpolate(edge_pred_logits_s16, size=img_size, mode='bilinear', align_corners=True)
        all_edge_outputs = (edge_out_s4, edge_out_s8, edge_out_s16)  # 打包成元组返回

        # fused 分割特征（三尺度 concat 后 fuse）
        fused = torch.cat([p16_up, p8_up, p4], dim=1)
        fused = self.fuse(fused)  # (B, out_channels, H/4, W/4)

        # cross fusion：让分割特征主动向边界特征 query
        # 注意这里 fused2 的输入是 fused (分割特征) 和 edge_feat (融合后的边缘特征)
        fused2 = self.cross_fuse(fused, edge_feat)  # (B, out_channels, H/4, W/4)

        # 把 fused2 与 edge_feat 拼接，做 refinement
        refine_in = torch.cat([fused2, edge_feat], dim=1)
        refined = self.refine(refine_in)

        # 最终分类器
        seg_logits = self.classifier(refined)  # (B, n_class, H/4, W/4)

        # 上采样到原始分辨率
        seg_out = F.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=True)

        # 返回 refined 特征, 最终分割输出, 以及所有尺度的边缘预测输出
        return refined, seg_out, all_edge_outputs




