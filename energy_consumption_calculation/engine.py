'''
Copyright (C) 2022 Guangyao Chen. - All Rights Reserved
Modified for Spikformer 3D compatibility.
'''

import sys
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from progress.bar import Bar as Bar

try:
    from spikingjelly.clock_driven import surrogate, neuron, functional
except:
    from spikingjelly.activation_based import surrogate, neuron, functional

from .ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING
from .utils import syops_to_string, params_to_string

def compute_effective_event_mask(event_mask_batch, current_timestep, window_size=3, device='cuda'):
    """计算有效事件掩码：检查当前时间步及周围window_size范围内是否有事件"""
    batch_size, T, D, H, W = event_mask_batch.shape
    effective_masks = []

    for b in range(batch_size):
        start_t = max(0, current_timestep - window_size)
        end_t = min(T, current_timestep + window_size)
        window_events = event_mask_batch[b, start_t:end_t]
        has_events = torch.any(window_events > 0, dim=0)
        effective_mask = has_events.byte().to(device)
        effective_masks.append(effective_mask)

    return effective_masks
def get_syops_pytorch(model, input_res, dataloader=None,
                      print_per_layer_stat=True,
                      input_constructor=None,
                      ost=sys.stdout,
                      verbose=False, ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      syops_units='GMac',
                      param_units='M'):
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    syops_model = add_syops_counting_methods(model)
    syops_model.eval()
    syops_model.start_syops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_modules)

    device = next(syops_model.parameters()).device

    if dataloader is not None:
        syops_count = np.array([0.0, 0.0, 0.0, 0.0])
        try:
            max_len = len(dataloader)
        except:
            max_len = 100

        bar = Bar('Processing', max=max_len)
        batch_idx = 0



        # Modify loop to handle generic iterator (dict or tuple)
        for batch_data in dataloader:
          if 'fmri_data' in batch_data:
            batch_idx += 1

            torch.cuda.empty_cache()

            # Handle Dictionary Input (FusedFMRIEventDataset)
            if isinstance(batch_data, dict) and 'fmri_data' in batch_data:
                # Assuming standard keys from your dataloader
                batch = batch_data['fmri_data'].float().to(device)  # [B, T, 1, D, H, W]
                event_mask = batch_data.get('event_mask', None)  # [B, T, 1, D, H, W]

                if batch.ndim == 5:  # [B, D, H, W] -> [B, 1, D, H, W] if channel missing
                    batch = batch.unsqueeze(2)

                processed_mask = None
                if event_mask is not None:
                    processed_mask = compute_effective_event_mask(event_mask, 1 ,window_size=3)

                # Execute model for one timestep
                with torch.no_grad():
                    out_fr = 0
                    T =48
                    for t in range(48):
                        current_fmri = batch[:, t]
                        current_event_mask = compute_effective_event_mask(event_mask, t, window_size=3)
                        # out_fr += model(current_fmri, event_mask=current_event_mask, current_timestep=t)
                        out_fr += syops_model(current_fmri, event_mask=current_event_mask,current_timestep=t)

                    _ = out_fr / T
                        # _ = syops_model(batch, event_mask=processed_mask)

                functional.reset_net(syops_model)

            # Handle Standard Input (Tuple)
            elif isinstance(batch_data, (tuple, list)):
                batch = batch_data[0].float().to(next(syops_model.parameters()).device)
                with torch.no_grad():
                        _ = syops_model(batch)

          elif 'image' in batch_data:
            model_pred_all = []
            # 1. 获取 Raw Batch
            if isinstance(batch_data, dict):
                batch = batch_data['image'].unsqueeze(1)
                # batch = batch.permute(2, 0, 1, 3, 4, 5)
                batch = (batch.unsqueeze(0)).repeat(4, 1, 1, 1, 1, 1)
                # (B, T, D, H, W)
            else:
                batch = batch_data[0]

            # 2. 使用构造器处理维度 (B, T, D, H, W) -> (T, B, C, D, H, W)
            if input_constructor:
                input_tensor = input_constructor(batch)
            else:
                input_tensor = batch.float().to(next(syops_model.parameters()).device)

            with torch.no_grad():
                out_fr = 0
                for t in range(4):
                    image_x = input_tensor[t]
                    out_fr += syops_model(image_x)
                    out_fr = out_fr
                    out_fr_1 = nn.Softplus()(out_fr) + 1
                    out_fr = out_fr.float()
                    model_pred_all.append(out_fr_1.to(device))
                    id_alpha_pred_all = torch.cat(model_pred_all, dim=0)
                    p1 = id_alpha_pred_all / torch.sum(id_alpha_pred_all, dim=-1, keepdim=True)
                    scores = p1.max(-1)[0].cpu().detach().numpy()
                    p = F.softmax(out_fr, dim=1)  # 转换为概率分布
                    confidence_score = p.max(dim=1)[0].item()  # 取最大概率作为置信度
                    # print('confidence_score',confidence_score)
                    # 假设out_fr是模型输出的logits（二分类时形状为[batch_size, 1]）
                    # p = torch.sigmoid(out_fr).squeeze()  # 转换为概率分布（挤压维度为[batch_size]）
                    #
                    # # 计算置信度：正类概率和负类概率中的最大值
                    # confidence_score = torch.max(p).max().item()  # 取最大概率作为置信度
                    model_pred_all.clear()
                    if confidence_score >= 0.95:  # if  (1-uncertainty) >= 0.7 :
                        best_T = t + 1
                        break  # 满足条件，T循环
                    else:
                        best_T = t + 1
                _ = out_fr / best_T
                    # out_fr = out_fr.float()
                # _ = out_fr / 2

                # 执行前向传播，触发 Hook 计算 FLOPs
                # 模型内部 x = x[:48] 会在这里执行，因此只统计前 48 步
                # _ = syops_model(input_tensor)

            functional.reset_net(syops_model)

            bar.suffix = '({batch}/{size})'.format(batch=batch_idx, size=max_len)
            bar.next()

        # syops_count = np.array([0.0, 0.0, 0.0, 0.0])
        # bar = Bar('Processing', max=len(dataloader))
        # batch_idx = 0
        # for batch, _ in dataloader:
        #     batch_idx += 1
        #
        #     torch.cuda.empty_cache()
        #
        #     batch = batch.float().to(next(syops_model.parameters()).device)
        #
        #     with torch.no_grad():
        #         _ = syops_model(batch)
        #
        #     functional.reset_net(syops_model)
        #
        #     bar.suffix = '({batch}/{size})'.format(batch=batch_idx, size=len(dataloader))
        #     bar.next()

        bar.finish()
        syops_count, params_count = syops_model.compute_average_syops_cost()
    else:
        # 单次推理模式 (Dummy Input)
        if input_constructor:
            # 构造一个符合 input_res 的 dummy tensor
            # 假设 input_res 是 (T, D, H, W) 或者 (D, H, W)，这里还是依赖 dataloader 模式更准
            dummy_batch = torch.ones(()).new_empty((1, *input_res)).to(next(syops_model.parameters()).device)
            input_tensor = input_constructor(dummy_batch)
            _ = syops_model(input_tensor)
        else:
            try:
                batch = torch.ones(()).new_empty((1, *input_res),
                                                 dtype=next(syops_model.parameters()).dtype,
                                                 device=next(syops_model.parameters()).device)
            except StopIteration:
                batch = torch.ones(()).new_empty((1, *input_res))
            _ = syops_model(batch)

        syops_count, params_count = syops_model.compute_average_syops_cost()

    if print_per_layer_stat:
        print_model_with_syops(
            syops_model,
            syops_count,
            params_count,
            ost=ost,
            syops_units=syops_units,
            param_units=param_units,
            precision=output_precision
        )
    syops_model.stop_syops_count()
    CUSTOM_MODULES_MAPPING = {}

    return syops_count, params_count, syops_model


def accumulate_syops(self):
    if is_supported_instance(self):
        return self.__syops__
    else:
        sum = np.array([0.0, 0.0, 0.0, 0.0])
        for m in self.children():
            sum += m.accumulate_syops()
        return sum


def print_model_with_syops(model, total_syops, total_params, syops_units='GMac',
                           param_units='M', precision=3, ost=sys.stdout):
    for i in range(3):
        if total_syops[i] < 1:
            total_syops[i] = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def syops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_syops_cost = self.accumulate_syops()
        accumulated_syops_cost[0] /= model.__batch_counter__
        accumulated_syops_cost[1] /= model.__batch_counter__
        accumulated_syops_cost[2] /= model.__batch_counter__
        accumulated_syops_cost[3] /= model.__times_counter__

        self.accumulated_params_num = accumulated_params_num
        self.accumulated_syops_cost = accumulated_syops_cost

        return ', '.join([self.original_extra_repr(),
                          params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          syops_to_string(accumulated_syops_cost[0],
                                          units=syops_units, precision=precision),
                          '{:.3%} oriMACs'.format(accumulated_syops_cost[0] / total_syops[0]),
                          syops_to_string(accumulated_syops_cost[1],
                                          units=syops_units, precision=precision),
                          '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
                          syops_to_string(accumulated_syops_cost[2],
                                          units=syops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
                          '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.)])


    def syops_repr_empty(self):
        return ''

    def add_extra_repr(m):
        m.accumulate_syops = accumulate_syops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        if is_supported_instance(m):
            syops_extra_repr = syops_repr.__get__(m)
        else:
            syops_extra_repr = syops_repr_empty.__get__(m)
        if m.extra_repr != syops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = syops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_syops_counting_methods(net_main_module):
    net_main_module.start_syops_count = start_syops_count.__get__(net_main_module)
    net_main_module.stop_syops_count = stop_syops_count.__get__(net_main_module)
    net_main_module.reset_syops_count = reset_syops_count.__get__(net_main_module)
    net_main_module.compute_average_syops_cost = compute_average_syops_cost.__get__(
        net_main_module)
    net_main_module.reset_syops_count()
    return net_main_module


def compute_average_syops_cost(self):
    for m in self.modules():
        m.accumulate_syops = accumulate_syops.__get__(m)

    syops_sum = self.accumulate_syops()
    syops_sum = np.array([item / self.__batch_counter__ for item in syops_sum])

    for m in self.modules():
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    params_sum = get_model_parameters_number(self)
    return syops_sum, params_sum


def start_syops_count(self, **kwargs):
    add_batch_counter_hook_function(self)
    seen_types = set()

    def add_syops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__syops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                    CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__syops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
                    not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_syops_counter_hook_function, **kwargs))


def stop_syops_count(self):
    remove_batch_counter_hook_function(self)
    self.apply(remove_syops_counter_hook_function)
    self.apply(remove_syops_counter_variables)


def reset_syops_count(self):
    add_batch_counter_variables_or_reset(self)
    self.apply(add_syops_counter_variable_or_reset)



def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        input = input[0]
        batch_size = len(input)
    module.__batch_counter__ += batch_size
    module.__times_counter__ += 1


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0
    module.__times_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_syops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__') or hasattr(module, '__params__'):
            module.__syops_backup_syops__ = module.__syops__
            module.__syops_backup_params__ = module.__params__
        module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
        module.__params__ = get_model_parameters_number(module)
        # module.__spkhistc__ = None


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_syops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops_handle__'):
            module.__syops_handle__.remove()
            del module.__syops_handle__


def remove_syops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__'):
            del module.__syops__
            if hasattr(module, '__syops_backup_syops__'):
                module.__syops__ = module.__syops_backup_syops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__syops_backup_params__'):
                module.__params__ = module.__syops_backup_params__
        if hasattr(module, '__spkhistc__'):
            del module.__spkhistc__