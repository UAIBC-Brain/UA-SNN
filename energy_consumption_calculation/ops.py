'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, \
        MultiStepParametricLIFNode, ParametricLIFNode
except:
    from spikingjelly.activation_based.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, \
        MultiStepParametricLIFNode, ParametricLIFNode

# --- ADDED: Import your custom modules ---
try:
    from model_test import PG_LIF
except ImportError:
    pass


# -----------------------------------------

# def spike_rate(inp):
#     # 改变判断前层输入是否为脉冲的方法
#     Nspks_max = 1000
#     Nnum_max = 1000
#
#     num = inp.unique()
#     if len(num) <= Nnum_max + 1 and inp.max() <= Nspks_max and inp.min() >= 0:
#         nonzero_indices = torch.nonzero(inp)
#         nonzero_count = nonzero_indices.size(0)
#         spkhistc = None
#         spike = True
#         # 注意：这里计算的是该 batch 和 time step 下的平均发放率
#         spike_rate = torch.tensor(nonzero_count / inp.numel()).item()
#     else:
#         spkhistc = None
#         spike = False
#         spike_rate = 1
#
#     return spike, spike_rate, spkhistc
def spike_rate(inp):
    # T = inp.shape[1]
    num = inp.unique()
    if len(num) <= 2 and inp.max() <= 1 and inp.min() >= 0:
        spike = True
        spike_rate = (inp.sum() / inp.numel()).item()
    else:
        spike = False
        spike_rate = 1

    return spike, spike_rate

# import numpy as np  # 确保导入 numpy，虽然这里只用到 torch，但 hook 里可能用到
#
#
# def spike_rate(inp):
#     # --- [Debug Start] 打印输入信息 ---
#     print(f"\n>>> [Spike Rate Calculation Info]")
#     print(f"  Input Shape: {tuple(inp.shape)}")
#
#     num = inp.unique()
#
#     # 打印该层包含的唯一数值（例如 [0., 1.] 或 [0.5, -0.2, ...]）
#     # 如果数值太多（非脉冲层），只打印前5个避免刷屏
#     unique_vals = num.tolist()
#     vals_str = str(unique_vals) if len(unique_vals) <= 5 else str(unique_vals[:5]) + "..."
#     print(f"  Unique Values in Tensor: {vals_str}")
#
#     # 判断是否为脉冲层的条件
#     # 条件1: 只有不超过2种数值（通常是0和1）
#     # 条件2: 最大值不超过1
#     # 条件3: 最小值不小于0
#     is_spike = len(num) <= 2 and inp.max() <= 1 and inp.min() >= 0
#
#     if is_spike:
#         spike = True
#
#         # === 核心计算公式展示 ===
#         total_ones = inp.sum().item()  # 分子：有多少个 1 (脉冲)
#         total_elements = inp.numel()  # 分母：总共有多少个神经元/像素
#         spike_rate = total_ones / total_elements
#
#         print(f"  [Type]: SPIKE Layer (Condition Met)")
#         print(f"  [Formula]: Sum(Ones) / Total_Elements")
#         print(f"  [Calc]: {total_ones:.0f} / {total_elements}")
#         print(f"  [Result]: Rate = {spike_rate:.6f} (即 {spike_rate * 100:.4f}%)")
#
#     else:
#         spike = False
#         spike_rate = 1
#
#         print(f"  [Type]: DENSE Layer (Condition Not Met)")
#         print(f"  [Reason]: Not binary 0/1 data. (Len={len(num)}, Max={inp.max().item():.2f})")
#         print(f"  [Result]: Rate = 1 (Default 100% calculation)")
#
#     print("-" * 40)  # 分割线
#     # --- [Debug End] ---
#
#     return spike, spike_rate


def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def upsample_syops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__syops__[0] += int(output_elements_count)

    spike, rate, _ = spike_rate(output[0])

    if spike:
        module.__syops__[1] += int(output_elements_count) * rate
    else:
        module.__syops__[2] += int(output_elements_count)

    module.__syops__[3] += rate * 100


def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate = spike_rate(output[0])

    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)

    module.__syops__[3] += rate * 100


def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate = spike_rate(output)
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    # module.__spkhistc__ = spkhistc


def LIF_syops_counter_hook(module, input, output):
    # input[0] 通常是膜电位输入或电流输入
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate = spike_rate(output[0])
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    # module.__spkhistc__ = spkhistc

def LIF_syops_counter_hook(module, input, output):
    # input[0] 通常是膜电位输入或电流输入
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    spike, rate = spike_rate(output[0])
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    # module.__spkhistc__ = spkhistc

def linear_syops_counter_hook(module, input, output):
    input = input[0]
    spike, rate = spike_rate(input)

    # batch_size = input.shape[0]
    output_last_dim = output.shape[-1]

    bias_syops = output_last_dim  if module.bias is not None else 0

    module.__syops__[0] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    if spike:
        module.__syops__[1] += int(np.prod(input.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100


def pool_syops_counter_hook(module, input, output):
    input = input[0]
    spike, rate = spike_rate(input)
    module.__syops__[0] += int(np.prod(input.shape))

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100



def bn_syops_counter_hook(module, input, output):
    input = input[0]
    spike, rate = spike_rate(input)
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100


def conv_syops_counter_hook(conv_module, input, output):
    input = input[0]
    spike, rate = spike_rate(input)

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0
    if conv_module.bias is not None:
        bias_syops = out_channels * active_elements_count

    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100


def rnn_syops(syops, rnn_module, w_ih, w_hh, input_size):
    syops += w_ih.shape[0] * w_ih.shape[1]
    syops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        syops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        syops += rnn_module.hidden_size
        syops += rnn_module.hidden_size * 3
        syops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        syops += rnn_module.hidden_size * 4
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return syops


def rnn_syops_counter_hook(rnn_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        syops = rnn_syops(syops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    syops *= seq_length
    if rnn_module.bidirectional:
        syops *= 2
    rnn_module.__syops__[0] += int(syops)


def rnn_cell_syops_counter_hook(rnn_cell_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    syops = rnn_syops(syops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    rnn_cell_module.__syops__[0] += int(syops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    syops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    syops = 0

    # Q scaling
    syops += qlen * qdim

    # Initial projections
    syops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        syops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_syops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    syops += num_heads * head_syops

    # final projection, bias is always enabled
    syops += qlen * vdim * (vdim + 1)

    syops *= batch_size
    multihead_attention_module.__syops__[0] += int(syops)


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,
    # F.conv3d: conv_syops_counter_hook,
    # activations
    nn.ReLU: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,

    # Neuron IF
    MultiStepIFNode: IF_syops_counter_hook,
    IFNode: IF_syops_counter_hook,

    # Neuron LIF
    MultiStepLIFNode: LIF_syops_counter_hook,
    LIFNode: LIF_syops_counter_hook,

    # Neuron PLIF
    MultiStepParametricLIFNode: LIF_syops_counter_hook,
    ParametricLIFNode: LIF_syops_counter_hook,

    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,
    nn.GroupNorm: bn_syops_counter_hook,
    # FC
    nn.Linear: linear_syops_counter_hook,
    # Upscale
    nn.Upsample: upsample_syops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,
    # RNN
    nn.RNN: rnn_syops_counter_hook,
    nn.GRU: rnn_syops_counter_hook,
    nn.LSTM: rnn_syops_counter_hook,
    nn.RNNCell: rnn_cell_syops_counter_hook,
    nn.LSTMCell: rnn_cell_syops_counter_hook,
    nn.GRUCell: rnn_cell_syops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook

