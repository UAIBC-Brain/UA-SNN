'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
Modified for Spikformer 3D with Auto-Scaling Units.
'''

import sys
import torch.nn as nn
import numpy as np
from .engine import get_syops_pytorch
from .utils import syops_to_string, params_to_string


def format_ops(ops):
    """自动格式化 Ops 数值"""
    if ops >= 1e9:
        return f"{ops / 1e9:.3f} G"
    elif ops >= 1e6:
        return f"{ops / 1e6:.3f} M"
    elif ops >= 1e3:
        return f"{ops / 1e3:.3f} K"
    else:
        return f"{ops:.1f} "


def get_energy_cost(model, input_T=4):
    """
    Calculates energy consumption for Spikformer architecture.
    """
    print('\n' + '=' * 30)
    print('Calculating energy consumption (Detailed)')
    print('=' * 30)

    Nac = 0
    Nmac = 0

    # 1. 统计 Conv/Linear/LIF 等层的 Ops
    for name, module in model.named_modules():
        if hasattr(module, 'accumulated_syops_cost'):
            cost = module.accumulated_syops_cost
            # cost: [Total Ops, ACs, MACs, SpikeRate]
            Nac += cost[1]
            Nmac += cost[2]

    # 2. 计算 SSA (Self-Attention) 特有的 Ops
    print(f"{'Block':<6} | {'Resolution (N)':<15} | {'Q_fr':<6} {'K_fr':<6} {'V_fr':<6} | {'SSA ACs'}")
    print("-" * 65)

    if hasattr(model, 'blocks'):
        # blocks = model.block
        blocks = model.blocks
        for i, blk in enumerate(blocks):
            if hasattr(blk, 'attn'):
                attn = blk.attn

                dim = attn.dim
                num_heads = attn.num_heads
                head_dim = dim // num_heads

                # 获取发放率
                # if attn.q_lif is not None:
                q_fr = attn.q_lif.accumulated_syops_cost[3] / 100.0
                k_fr = attn.k_lif.accumulated_syops_cost[3] / 100.0
                v_fr = attn.v_lif.accumulated_syops_cost[3] / 100.0

                # 推算参与计算的元素总数 (T*B*N)
                total_elements_processed = attn.q_lif.accumulated_syops_cost[0]

                if total_elements_processed == 0:
                    print(f"  Block {i}: No data flow detected (Ops=0).")
                    continue

                tb_n = total_elements_processed / dim  # T * B * N

                # 估算 N (假设 Batch=1, T已知)
                # 这只是为了显示 debug 信息，不影响计算
                estimated_N = tb_n / input_T if input_T > 0 else 0

                # SSA MatMul 1: K^T @ V
                # Base Ops = (T*B*N) * num_heads * head_dim^2
                ops_base = tb_n * num_heads * (head_dim ** 2)

                # 估算 ACs
                ac_op1 = ops_base * min(k_fr, v_fr)  # 两个稀疏矩阵相乘
                ac_op2 = ops_base * q_fr  # 稀疏 Q * 稠密 AttnMap (近似)

                ssa_ops = ac_op1 + ac_op2
                Nac += ssa_ops

                # 使用自动格式化显示
                ops_str = format_ops(ssa_ops)
                print(f"  {i:<4} | ~{int(estimated_N):<14} | {q_fr:.2f}   {k_fr:.2f}   {v_fr:.2f}   | {ops_str}")

    print("-" * 65)

    # 汇总输出
    # 能量计算 (mJ)
    # 1 G = 10^9, 1 M = 10^6
    # E = Ops * Energy_per_op
    # 这里直接用原始数量计算，再换算单位

    E_mac = Nmac * 4.6e-12 * 1e3  # (Total Count * 4.6pJ) -> mJ
    E_ac = Nac * 0.9e-12 * 1e3  # (Total Count * 0.9pJ) -> mJ
    E_all = E_mac + E_ac

    print(f"\nTotal Statistics:")
    print(f"  MACs (Float): {format_ops(Nmac)}")
    print(f"  ACs  (Spike): {format_ops(Nac)}")
    print(f"  Energy      : {E_all:.4f} mJ")
    print('=' * 30 + '\n')
    return E_all


def get_model_complexity_info(model, input_res, dataloader=None,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              syops_units=None, param_units=None,
                              output_precision=2,
                               ):
    if backend == 'pytorch':
        syops_count, params_count, syops_model = get_syops_pytorch(model, input_res, dataloader,
                                                                   print_per_layer_stat,
                                                                   input_constructor, ost,
                                                                   verbose, ignore_modules,
                                                                   custom_modules_hooks,
                                                                   output_precision=output_precision,
                                                                   syops_units=syops_units,
                                                                   param_units=param_units)

        E_all = get_energy_cost(syops_model)
    else:
        raise ValueError('Wrong backend name')

    if as_strings:
        syops_string = syops_to_string(
            syops_count[0],
            units=syops_units,
            precision=output_precision
        )
        ac_syops_string = syops_to_string(
            syops_count[1],
            units=syops_units,
            precision=output_precision
        )
        mac_syops_string = syops_to_string(
            syops_count[2],
            units=syops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return [syops_string, ac_syops_string, mac_syops_string], params_string,E_all

    return syops_count, params_count,E_all

# '''
# Copyright (C) 2022 Guangyao Chen - All Rights Reserved
#  * You may use, distribute and modify this code under the
#  * terms of the MIT license.
#  * You should have received a copy of the MIT license with
#  * this file. If not visit https://opensource.org/licenses/MIT
# '''
#
# import sys
#
# import torch.nn as nn
#
# from .engine import get_syops_pytorch
# from .utils import syops_to_string, params_to_string
#
# # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
# # ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
# ssa_info = {'depth': 4, 'Nheads': 4, 'embSize': 384, 'patchSize': 8, 'Tsteps': 4, 'expert_num': 4}  # lifconvbn-8-768
#
#
# def get_energy_cost(model, ssa_info):
#     # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
#     print('Calculating energy consumption ...')
#     conv_linear_layers_info = []
#     Nac = 0
#     Nmac = 0
#     # print(id(model.block[3].attn.ff_list1[0].fc) == id(model.block[3].attn.ff_list1[1].fc))
#     # print(model.block[3].attn.ff_list1[0].fc.weight.data == model.block[3].attn.ff_list1[1].fc.weight.data)
#     # print("ff_list1[0]", model.block[3].attn.ff_list1[0].fc.weight.data)
#     # print("ff_list1[1]", model.block[3].attn.ff_list1[1].fc.weight.data)
#     # print("router", model.block[3].attn.router[0].weight.data)
#     for name, module in model.named_modules():
#         # print(name, module)
#         # isinstance(model.head, nn.Linear) -> True
#         # isinstance(model.head, nn.Conv1d) -> False
#         # isinstance(model.block[7].mlp.fc1_conv, nn.Conv2d) -> True
#         # isinstance(model.block.7.mlp.fc1_conv, nn.Conv1d) -> error (invalid syntax)
#         # model.patch_embed.proj_conv2 -> Conv2d(96, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         print(name, "shit")
#         # if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):  # obtain same results
#         if "conv" in name or "head" in name or "_linear" in name or "router1" in name or "0.fc" in name \
#                 or "1.fc" in name or "2.fc" in name or "3.fc" in name:
#             if 'block' in name:
#                 if 'ff_list' in name:
#                     name_split = name.split('.', 5)
#                     name = f'block[{name_split[1]}].{name_split[2]}.ff_list[{name_split[4]}].{name_split[5]}'
#                 else:
#                     name_split = name.split('.', 2)
#                     name = f'block[{name_split[1]}].{name_split[2]}'
#                 # name = f'{name_split[0]}[{int(name_split[1])}].{name_split[2]}'
#             # print(name, "shit")
#             accumulated_syops_cost = eval(f'model.{name}.accumulated_syops_cost')
#             tinfo = (name, module, accumulated_syops_cost)
#             conv_linear_layers_info.append(tinfo)
#             if abs(accumulated_syops_cost[3] - 100) < 1e-4:  # fr = 100%
#                 Nmac += accumulated_syops_cost[2]
#                 # Nmac += accumulated_syops_cost[0] * accumulated_syops_cost[3] / 100  # obtain same results
#             else:
#                 Nac += accumulated_syops_cost[1]
#                 # Nac += accumulated_syops_cost[0] * accumulated_syops_cost[3] / 100  # obtain same results
#     print('Info of Conv/Linear layers: ')
#     for tinfo in conv_linear_layers_info:
#         print(tinfo)
#
#     # calculate ops for SSA
#     print('SSA info: \n', ssa_info)
#     depth = ssa_info['depth']
#     Nheads = ssa_info['Nheads']
#     embSize = ssa_info['embSize']
#     Tsteps = ssa_info['Tsteps']
#     expert_num = ssa_info['expert_num']
#     patchSize = ssa_info['patchSize']
#     embSize_per_head = int(embSize / Nheads)
#     SSA_Nac_base = Tsteps * pow(patchSize, 2) * embSize_per_head * embSize_per_head
#     qkv_fr = []
#     for d in range(depth):
#         for e in range(expert_num):
#             q_lif_r = eval(f'model.block[{d}].attn.ff_list[{e}].unit_lif.accumulated_syops_cost[3]') / 100
#             k_lif_r = eval(f'model.block[{d}].attn.k_lif.accumulated_syops_cost[3]') / 100
#             v_lif_r = eval(f'model.block[{d}].attn.v_lif.accumulated_syops_cost[3]') / 100
#             qkv_fr.append([q_lif_r, k_lif_r, v_lif_r])
#             # calculate the number of ACs for Q*K*V matrix computation
#             tNac = SSA_Nac_base * (min(k_lif_r, v_lif_r) + q_lif_r)
#             print("dfffdf", d, tNac)
#             Nac += tNac
#     print('Firing rate of Q/K/V inputs in each block: ')
#     print(qkv_fr)
#
#     # calculate energy consumption according to E_mac = 4.6 pJ (1e-12 J) and E_ac = 0.9 pJ
#     Nmac = Nmac / 1e9  # G
#     Nac = Nac / 1e9  # G
#     E_mac = Nmac * 4.6  # mJ
#     E_ac = Nac * 0.9  # mJ
#     E_all = E_mac + E_ac
#     print(f"Number of operations: {Nmac} G MACs, {Nac} G ACs")
#     print(f"Energy consumption: {E_all} mJ")
#     return E_all
#
#
# def get_model_complexity_info(model, input_res, dataloader=None,
#                               print_per_layer_stat=True,
#                               as_strings=True,
#                               input_constructor=None, ost=sys.stdout,
#                               verbose=False, ignore_modules=[],
#                               custom_modules_hooks={}, backend='pytorch',
#                               syops_units=None, param_units=None,
#                               output_precision=2):
#     assert type(input_res) is tuple
#     assert len(input_res) >= 1
#     assert isinstance(model, nn.Module)
#
#     if backend == 'pytorch':
#         syops_count, params_count, syops_model = get_syops_pytorch(model, input_res, dataloader,
#                                                                    print_per_layer_stat,
#                                                                    input_constructor, ost,
#                                                                    verbose, ignore_modules,
#                                                                    custom_modules_hooks,
#                                                                    output_precision=output_precision,
#                                                                    syops_units=syops_units,
#                                                                    param_units=param_units)
#         # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
#         E_all = get_energy_cost(syops_model, ssa_info)
#     else:
#         raise ValueError('Wrong backend name')
#
#     if as_strings:
#         syops_string = syops_to_string(
#             syops_count[0],
#             units=syops_units,
#             precision=output_precision
#         )
#         ac_syops_string = syops_to_string(
#             syops_count[1],
#             units=syops_units,
#             precision=output_precision
#         )
#         mac_syops_string = syops_to_string(
#             syops_count[2],
#             units=syops_units,
#             precision=output_precision
#         )
#         params_string = params_to_string(
#             params_count,
#             units=param_units,
#             precision=output_precision
#         )
#         return [syops_string, ac_syops_string, mac_syops_string], params_string,E_all
#
#     return syops_count, params_count,E_all

#
#
# '''
# Copyright (C) 2022 Guangyao Chen - All Rights Reserved
# Modified for Spikformer 3D with Auto-Scaling Units.
# '''
#
# import sys
# import torch.nn as nn
# import numpy as np
# from .engine import get_syops_pytorch
# from .utils import syops_to_string, params_to_string


# def format_ops(ops):
#     """自动格式化 Ops 数值"""
#     if ops >= 1e9:
#         return f"{ops / 1e9:.3f} G"
#     elif ops >= 1e6:
#         return f"{ops / 1e6:.3f} M"
#     elif ops >= 1e3:
#         return f"{ops / 1e3:.3f} K"
#     else:
#         return f"{ops:.1f} "
#
#
# def get_energy_cost(model, input_T=4):
#     """
#     Calculates energy consumption for Spikformer 3D architecture.
#     Adapts to MS_SSA_Conv with 'STAtten' and 'SDT' modes.
#     """
#     print('\n' + '=' * 30)
#     print('Calculating energy consumption (Detailed)')
#     print('=' * 30)
#
#     Nac = 0
#     Nmac = 0
#
#     # 1. 统计 Conv/Linear/LIF 等层的 Ops (通过 spikingjelly 的 hook 自动统计)
#     for name, module in model.named_modules():
#         if hasattr(module, 'accumulated_syops_cost'):
#             cost = module.accumulated_syops_cost
#             # cost: [Total Ops, ACs, MACs, SpikeRate]
#             Nac += cost[1]
#             Nmac += cost[2]
#
#     # 2. 计算 SSA (Self-Attention) 特有的 Ops
#     # 这些是手动计算的，因为 hook 通常无法识别自定义的 sparse matmul
#     print(f"{'Block':<6} | {'Mode':<10} | {'Q_fr':<6} {'K_fr':<6} {'V_fr':<6} | {'SSA ACs'}")
#     print("-" * 75)
#
#     # 兼容 blocks (ModuleList) 命名可能的差异
#     blocks = []
#     if hasattr(model, 'blocks'):
#         blocks = model.blocks
#     elif hasattr(model, 'block'):
#         blocks = model.block
#
#     for i, blk in enumerate(blocks):
#         # 你的新代码中，attn 层是 MS_SSA_Conv
#         if hasattr(blk, 'attn'):
#             attn = blk.attn
#
#             # 获取注意力模式
#             mode = getattr(attn, 'attention_mode', 'STAtten')  # 默认为 STAtten 以防万一
#
#             dim = attn.dim
#             num_heads = attn.num_heads
#             head_dim = dim // num_heads
#
#             # 确保 LIF 节点存在且已被统计
#             if not hasattr(attn, 'q_lif') or not hasattr(attn.q_lif, 'accumulated_syops_cost'):
#                 print(f"  Block {i}: LIF statistics not found. Skipping.")
#                 continue
#
#             # 获取发放率 (Firing Rate)
#             # 假设 accumulated_syops_cost[3] 是百分比 (0-100)
#             q_fr = attn.q_lif.accumulated_syops_cost[3] / 100.0
#             k_fr = attn.k_lif.accumulated_syops_cost[3] / 100.0
#             v_fr = attn.v_lif.accumulated_syops_cost[3] / 100.0
#
#             # 推算参与计算的元素总数 (T*B*N)
#             # q_lif 输入是 (T, B, C, D, H, W)，总元素数除以 C (dim) 即为 T*B*D*H*W = T*B*N
#             total_elements_processed = attn.q_lif.accumulated_syops_cost[0]
#             tb_n = total_elements_processed / dim
#
#             ssa_ops = 0
#
#             # ==========================================
#             # 分情况讨论计算逻辑
#             # ==========================================
#
#             if mode == 'STAtten':
#                 # STAtten 本质上是分块的线性注意力 (Linear Attention)
#                 # 复杂度依然是 O(T*N * D^2)，因为 MatMul 是 (K^T @ V) 和 (Q @ Attn)
#
#                 # Base Ops = (T*B*N) * num_heads * head_dim^2
#                 ops_base = tb_n * num_heads * (head_dim ** 2)
#
#                 # 1. K^T @ V: 两个脉冲矩阵相乘
#                 # 形状: (head_dim, chunk*N) @ (chunk*N, head_dim) -> (head_dim, head_dim)
#                 ac_op1 = ops_base * min(k_fr, v_fr)
#
#                 # 2. Q @ Attn: 脉冲 Q * 浮点 AttnMap
#                 # 形状: (chunk*N, head_dim) @ (head_dim, head_dim) -> (chunk*N, head_dim)
#                 ac_op2 = ops_base * q_fr
#
#                 ssa_ops = ac_op1 + ac_op2
#
#                 ops_str = format_ops(ssa_ops)
#                 print(f"  {i:<4} | {mode:<10} | {q_fr:.2f}   {k_fr:.2f}   {v_fr:.2f}   | {ops_str}")
#
#             elif mode == 'SDT':
#                 # SDT (Spike-Driven Transformer) 是元素级运算，无矩阵乘法
#                 # 逻辑: kv = k.mul(v); x = q.mul(kv)
#                 # 这些操作其实是 Element-wise Logic (AND)，复杂度是 O(T*N*D) 而非 O(T*N*D^2)
#                 # 通常这些开销非常小，且如果是纯脉冲相乘，可以算作 Logic Ops 而非 ACs。
#                 # 但为了严谨，我们可以将其视为 "ACs" (假设硬件用加法器实现乘法) 或者忽略。
#
#                 # 这里我们计算元素级操作数: (T*B*N*C)
#                 element_ops = tb_n * dim
#
#                 # k * v
#                 op_kv = element_ops * min(k_fr, v_fr)
#
#                 # q * kv (需要知道 kv 的发放率，这里做一个近似)
#                 # SDT 中间有个 talking_heads_lif，如果有统计数据最好用那个
#                 if hasattr(attn, 'talking_heads_lif') and hasattr(attn.talking_heads_lif, 'accumulated_syops_cost'):
#                     kv_fr = attn.talking_heads_lif.accumulated_syops_cost[3] / 100.0
#                 else:
#                     kv_fr = min(k_fr, v_fr)  # 近似
#
#                 op_q_kv = element_ops * min(q_fr, kv_fr)
#
#                 ssa_ops = op_kv + op_q_kv
#
#                 ops_str = format_ops(ssa_ops) + " (Elem)"
#                 print(f"  {i:<4} | {mode:<10} | {q_fr:.2f}   {k_fr:.2f}   {v_fr:.2f}   | {ops_str}")
#
#             else:
#                 print(f"  {i:<4} | {mode:<10} | Unknown mode, skipping SSA ops.")
#
#             Nac += ssa_ops
#
#     print("-" * 75)
#
#     # 汇总输出
#     # 能量计算 (mJ) - 参数需根据实际硬件调整 (例如 45nm CMOS: 4.6pJ/MAC, 0.9pJ/AC)
#     E_mac = Nmac * 4.6e-12 * 1e3
#     E_ac = Nac * 0.9e-12 * 1e3
#     E_all = E_mac + E_ac
#
#     print(f"\nTotal Statistics:")
#     print(f"  MACs (Float): {format_ops(Nmac)}")
#     print(f"  ACs  (Spike): {format_ops(Nac)}")
#     print(f"  Energy      : {E_all:.4f} mJ")
#     print('=' * 30 + '\n')
#     return E_all
#
#
# def get_model_complexity_info(model, input_res, dataloader=None,
#                               print_per_layer_stat=True,
#                               as_strings=True,
#                               input_constructor=None, ost=sys.stdout,
#                               verbose=False, ignore_modules=[],
#                               custom_modules_hooks={}, backend='pytorch',
#                               syops_units=None, param_units=None,
#                               output_precision=2,
#                                ):
#     if backend == 'pytorch':
#         syops_count, params_count, syops_model = get_syops_pytorch(model, input_res, dataloader,
#                                                                    print_per_layer_stat,
#                                                                    input_constructor, ost,
#                                                                    verbose, ignore_modules,
#                                                                    custom_modules_hooks,
#                                                                    output_precision=output_precision,
#                                                                    syops_units=syops_units,
#                                                                    param_units=param_units)
#
#         E_all = get_energy_cost(syops_model)
#     else:
#         raise ValueError('Wrong backend name')
#
#     if as_strings:
#         syops_string = syops_to_string(
#             syops_count[0],
#             units=syops_units,
#             precision=output_precision
#         )
#         ac_syops_string = syops_to_string(
#             syops_count[1],
#             units=syops_units,
#             precision=output_precision
#         )
#         mac_syops_string = syops_to_string(
#             syops_count[2],
#             units=syops_units,
#             precision=output_precision
#         )
#         params_string = params_to_string(
#             params_count,
#             units=param_units,
#             precision=output_precision
#         )
#         return [syops_string, ac_syops_string, mac_syops_string], params_string,E_all
#
#     return syops_count, params_count,E_all