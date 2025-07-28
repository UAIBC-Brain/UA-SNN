import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, layer, base  # 改为单步神经元
from timm.models.layers import to_3tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

__all__ = ['spike_basd_transformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm3d(hidden_features)
        self.fc1_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

        self.fc2_conv = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm3d(out_features)
        self.fc2_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        # 单步模式不需要时间维度处理
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class spike_self_attention (nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式
        self.attn_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

    def forward(self, x, res_attn):
        # 输入形状: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.flatten(2)  # [B, C, D*H*W]
        N = D * H * W

        # Q路径
        q = self.q_conv(x)  # [B, C, N]
        q = self.q_bn(q)
        q = self.q_lif(q)
        q = q.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]

        # K路径
        k = self.k_conv(x)
        k = self.k_bn(k)
        k = self.k_lif(k)
        k = k.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 2, 3)  # [B, num_heads, head_dim, N]

        # V路径
        v = self.v_conv(x)
        v = self.v_bn(v)
        v = self.v_lif(v)
        v = v.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]

        # 注意力计算
        attn = (q @ k) * self.scale
        x = attn @ v
        x = x.permute(0, 1, 3, 2).reshape(B, C, N)  # [B, C, N]
        x = self.attn_lif(x)

        # 投影
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = x.reshape(B, C, D, H, W)  # 恢复空间维度
        return x, v


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = spike_self_attention (dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, res_attn):
        x_attn, attn = self.attn(x, res_attn)
        x = x + x_attn
        x = x + self.mlp(x)
        return x, attn


class SFP(nn.Module):
    def __init__(self, img_size_d=128, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256,
                 dropout_rate=0.1):
        super().__init__()
        self.image_size = [img_size_d, img_size_h, img_size_w]
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.D, self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1], \
                                 self.image_size[2] // patch_size[2]
        self.num_patches = self.D * self.H * self.W

        # 第一层
        self.proj_conv = nn.Conv3d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm3d(embed_dims // 8)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 第二层
        self.proj_conv1 = nn.Conv3d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm3d(embed_dims // 4)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.proj_lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式
        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 第三层
        self.proj_conv2 = nn.Conv3d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm3d(embed_dims // 2)
        self.proj_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式
        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 第四层
        self.proj_conv3 = nn.Conv3d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm3d(embed_dims)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.proj_lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # RPE
        self.rpe_conv = nn.Conv3d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm3d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True, cupy_fp32_inference=True)  # 单步模式

    def forward(self, x):
        # 输入形状: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # 第一层
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.dropout(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        # 第二层
        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.dropout1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        # 第三层
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.dropout2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        # 第四层
        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        # RPE
        x_feat = x
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = self.rpe_lif(x)
        x = x + x_feat

        return x, (D // 16, H // 16, W // 16)


class spike_basd_transformers(nn.Module):
    def __init__(self,
                 img_size_d=128, img_size_h=128, img_size_w=128, patch_size=16, in_channels=1, drop_rate=0.1, num_classes=2,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, T=4,
                 depths=[6, 8, 6],  # 确保这里是列表
                 sr_ratios=[8, 4, 2], pretrained=False, pretrained_cfg=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths if isinstance(depths, list) else [depths]  # 确保depths是列表
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]  # 使用self.depths

        patch_embed = SFP(
            img_size_d=img_size_d,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            dropout_rate=drop_rate,
        )

        block = nn.ModuleList([
            Block(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios
            ) for i in range(depths)
        ])

        setattr(self, "patch_embed", patch_embed)
        setattr(self, "block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x, (D, H, W) = getattr(self, "patch_embed")(x)
        attn = None
        for blk in getattr(self, "block"):
            x, attn = blk(x, attn)
        return x.flatten(2).mean(2)  # [B, C]

    def forward(self, x):
        # 单步模式: 输入形状 [B, C, D, H, W]
        # 在时间维度上循环
        # outputs = []
        # for t in range(self.T):
        outputs = self.forward_features(x)

        outputs = self.head(outputs)  # [B, num_classes]
        return outputs


@register_model
def spike_basd_transformer(**kwargs):
    model = spike_basd_transformers(**kwargs)
    model.default_cfg = _cfg()
    return model