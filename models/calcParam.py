import torch
import torch.nn as nn
import numpy as np

class ModelParameterCounter:
    """深度学习模型参数计算器"""
    
    @staticmethod
    def count_parameters(model, detailed=False):
        """
        计算模型总参数量
        
        Args:
            model: 深度学习模型
            detailed: 是否显示详细信息
            
        Returns:
            int: 总参数数量
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        if detailed:
            print("=" * 60)
            print("模型参数详细统计:")
            print("-" * 60)
            print(f"{'Layer Name':<30} {'Params':<15} {'Trainable'}")
            print("-" * 60)
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
                trainable_status = "Yes"
            else:
                non_trainable_params += param_count
                trainable_status = "No"
            
            if detailed:
                print(f"{name:<30} {param_count:<15,} {trainable_status}")
        
        if detailed:
            print("-" * 60)
            print(f"总参数量:     {total_params:,}")
            print(f"可训练参数:   {trainable_params:,}")
            print(f"不可训练参数: {non_trainable_params:,}")
            print("=" * 60)
        
        return total_params
    
    @staticmethod
    def count_parameters_by_layer(model):
        """
        按层分类统计参数量
        
        Args:
            model: 深度学习模型
            
        Returns:
            dict: 各层类型的参数统计
        """
        layer_stats = {}
        
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0 and not isinstance(module, nn.Sequential):
                layer_type = type(module).__name__
                params_count = sum(p.numel() for p in module.parameters())
                
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {
                        'count': 0,
                        'params': 0
                    }
                
                layer_stats[layer_type]['count'] += 1
                layer_stats[layer_type]['params'] += params_count
        
        return layer_stats
    
    @staticmethod
    def format_parameter_count(count):
        """格式化参数数量显示"""
        if count >= 1e9:
            return f"{count/1e9:.2f}B"
        elif count >= 1e6:
            return f"{count/1e6:.2f}M"
        elif count >= 1e3:
            return f"{count/1e3:.2f}K"
        else:
            return str(count)

# PyTorch专用版本
def count_pytorch_parameters(model, show_details=True):
    """
    PyTorch模型参数计数器
    
    Args:
        model: PyTorch模型
        show_details: 是否显示详细信息
        
    Returns:
        dict: 参数统计信息
    """
    stats = {
        'total_params': 0,
        'trainable_params': 0,
        'non_trainable_params': 0,
        'layers': []
    }
    
    if show_details:
        print("\n" + "="*80)
        print("PyTorch模型参数分析")
        print("="*80)
        print(f"{'模块名称':<40} {'参数量':<15} {'大小(MB)':<12} {'是否训练'}")
        print("-"*80)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_size_mb = param_count * param.element_size() / (1024 * 1024)
        
        stats['total_params'] += param_count
        if param.requires_grad:
            stats['trainable_params'] += param_count
        else:
            stats['non_trainable_params'] += param_count
            
        layer_info = {
            'name': name,
            'params': param_count,
            'size_mb': param_size_mb,
            'trainable': param.requires_grad
        }
        stats['layers'].append(layer_info)
        
        if show_details:
            trainable_str = "✓" if param.requires_grad else "✗"
            print(f"{name:<40} {param_count:<15,} {param_size_mb:<12.2f} {trainable_str}")
    
    if show_details:
        print("-"*80)
        print(f"总计参数量: {stats['total_params']:,}")
        print(f"可训练参数: {stats['trainable_params']:,}")
        print(f"冻结参数量: {stats['non_trainable_params']:,}")
        print(f"模型大小(仅参数): {stats['total_params'] * 4 / (1024**2):.2f} MB")  # 假设float32
        print("="*80)
    
    return stats

# 简单快速版本
def quick_param_count(model):
    """快速计算模型参数量"""
    return sum(p.numel() for p in model.parameters())

def quick_trainable_param_count(model):
    """快速计算可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class AdvancedModelAnalyzer:
    """高级模型分析器"""
    
    @staticmethod
    def analyze_memory_usage(model, input_shape=None):
        """
        分析模型内存使用情况
        
        Args:
            model: 模型对象
            input_shape: 输入形状 (batch_size, channels, height, width)
        """
        import gc
        
        stats = {
            'parameter_memory_mb': 0,
            'gradient_memory_mb': 0,
            'activation_memory_mb': 0,
            'total_memory_mb': 0
        }
        
        # 参数内存 (假设float32，每个参数4字节)
        param_count = sum(p.numel() for p in model.parameters())
        stats['parameter_memory_mb'] = param_count * 4 / (1024 * 1024)
        
        # 梯度内存 (与参数相同)
        stats['gradient_memory_mb'] = stats['parameter_memory_mb']
        
        # 激活内存估算 (需要输入形状)
        if input_shape is not None:
            try:
                dummy_input = torch.randn(*input_shape)
                activation_sizes = []
                
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activation_sizes.append(output.numel() * 4)  # float32
                
                hooks = []
                for module in model.modules():
                    if not isinstance(module, nn.Sequential):
                        hooks.append(module.register_forward_hook(hook_fn))
                
                with torch.no_grad():
                    model(dummy_input)
                
                for hook in hooks:
                    hook.remove()
                
                stats['activation_memory_mb'] = sum(activation_sizes) / (1024 * 1024)
                
            except Exception as e:
                print(f"激活内存计算失败: {e}")
        
        stats['total_memory_mb'] = sum(stats.values())
        
        print("\n" + "="*60)
        print("模型内存使用分析")
        print("="*60)
        print(f"参数内存占用:    {stats['parameter_memory_mb']:>8.2f} MB")
        print(f"梯度内存占用:    {stats['gradient_memory_mb']:>8.2f} MB")
        print(f"激活内存占用:    {stats['activation_memory_mb']:>8.2f} MB")
        print("-"*60)
        print(f"总内存占用估算:  {stats['total_memory_mb']:>8.2f} MB")
        print("="*60)
        
        return stats
    
    @staticmethod
    def compare_models(models_dict):
        """
        比较多个模型的参数量
        
        Args:
            models_dict: {'model_name': model_object}
        """
        results = {}
        
        print("\n" + "="*80)
        print("模型比较分析")
        print("="*80)
        print(f"{'模型名称':<20} {'参数量':<15} {'格式化':<10} {'百分比'}")
        print("-"*80)
        
        total_params_list = []
        names = []
        
        for name, model in models_dict.items():
            param_count = sum(p.numel() for p in model.parameters())
            total_params_list.append(param_count)
            names.append(name)
        
        max_params = max(total_params_list) if total_params_list else 1
        
        for name, param_count in zip(names, total_params_list):
            formatted_count = ModelParameterCounter.format_parameter_count(param_count)
            percentage = (param_count / max_params) * 100
            print(f"{name:<20} {param_count:<15,} {formatted_count:<10} {percentage:>6.1f}%")
        
        print("="*80)
        
        return dict(zip(names, total_params_list))