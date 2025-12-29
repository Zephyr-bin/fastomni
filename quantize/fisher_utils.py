import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear

class FisherPruner:
    def __init__(self, layer):
        self.layer = layer
        self.fisher_info = {} 

    def collect_fisher_info(self, input_args, device='cuda'):
        self.layer.zero_grad()
        inps, attention_mask, position_ids = input_args
        inps = inps.to(device).detach().requires_grad_(True)
        
        # [Fix] Temporarily enable gradients for WEIGHTS
        # OmniQuant freezes weights by default, so we must unfreeze them to get gradients
        original_requires_grad = {}
        for name, module in self.layer.named_modules():
            if isinstance(module, QuantLinear) and hasattr(module, 'weight'):
                original_requires_grad[name] = module.weight.requires_grad
                module.weight.requires_grad = True

        try:
            output = self.layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
            loss = output.pow(2).sum() 
            loss.backward()
        except RuntimeError as e:
            print(f"Error during Fisher collection: {e}")
            # Restore grad status even if error
            for name, module in self.layer.named_modules():
                if isinstance(module, QuantLinear) and hasattr(module, 'weight') and name in original_requires_grad:
                    module.weight.requires_grad = original_requires_grad[name]
            return
        
        for name, module in self.layer.named_modules():
            if isinstance(module, QuantLinear):
                # Now module.weight.grad should NOT be None
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    score = (module.weight.grad * module.weight).pow(2)
                    self.fisher_info[name] = score.detach().cpu()
                else:
                    print(f"Warning: No grad for {name}") # Debug print

        # Restore original requires_grad state
        for name, module in self.layer.named_modules():
            if isinstance(module, QuantLinear) and hasattr(module, 'weight') and name in original_requires_grad:
                module.weight.requires_grad = original_requires_grad[name]
                
        self.layer.zero_grad()
        del inps, output, loss
        torch.cuda.empty_cache()

    def get_outlier_mask(self, module_name, outlier_ratio=0.01):
        if module_name not in self.fisher_info:
            return None
        scores = self.fisher_info[module_name]
        k = int(scores.numel() * outlier_ratio)
        if k < 1: return None
        
        threshold = torch.kthvalue(scores.flatten(), scores.numel() - k).values
        outlier_mask = scores > threshold
        return outlier_mask
    
    def get_mixed_precision_masks(self, module_name, protect_ratio=0.05, prune_ratio=0.05):
        if module_name not in self.fisher_info:
            return None, None
            
        scores = self.fisher_info[module_name]
        numel = scores.numel()
        
        # 1. Top K% (Outliers)
        k_protect = int(numel * protect_ratio)
        flat_scores = scores.flatten()
        top_k_val = torch.kthvalue(flat_scores, numel - k_protect).values
        outlier_mask = scores > top_k_val
        
        # 2. Bottom M% (Prunable)
        k_prune = int(numel * prune_ratio)
        bottom_k_val = torch.kthvalue(flat_scores, k_prune).values
        pruning_mask = scores < bottom_k_val
        
        return outlier_mask, pruning_mask
    
    # 在 FisherPruner 类中添加此新方法
    def get_block_balanced_masks(self, module_name, block_size=32, protect_ratio=0.05, prune_ratio=0.15):
        if module_name not in self.fisher_info:
            return None, None
            
        scores = self.fisher_info[module_name] # Shape: [Out_features, In_features]
        original_shape = scores.shape
        numel = scores.numel()
        
        # 1. 计算每个 Block 中需要保护和剪枝的具体数量
        # 例如: block_size=32, protect=0.05 -> k=1.6 -> 向上取整为 2 (或者取1，取决于具体策略，这里建议取整)
        # 建议：为了硬件规整，强制转为整数数量
        k_protect = int(block_size * protect_ratio) # e.g., 32 * 0.05 = 1.6 -> 1 (保守) 或 2
        if k_protect < 1: k_protect = 1 # 至少保护 1 个
        
        k_prune = int(block_size * prune_ratio)     # e.g., 32 * 0.15 = 4.8 -> 4 (保守) 或 5
        if k_prune < 1: k_prune = 0 # 不剪枝
        
        # 2. Reshape 成 [N, block_size] 以便进行局部排序
        # 注意：如果 numel 不能被 block_size 整除，需要 padding 或 忽略最后一点（通常 LLM 维度都是 128 倍数，问题不大）
        # 这里为了安全，我们只处理能整除的部分，或者简单地 view
        if numel % block_size != 0:
            # 简单处理：如果不能整除，回退到全局筛选（或者你也可以做 padding）
            print(f"Warning: {module_name} size {numel} not divisible by {block_size}, fallback to global.")
            return self.get_mixed_precision_masks(module_name, protect_ratio, prune_ratio)

        reshaped_scores = scores.view(-1, block_size)
        
        # 3. 生成 Outlier Mask (Top K in each block)
        # topk 返回 (values, indices)
        _, top_indices = torch.topk(reshaped_scores, k=k_protect, dim=1, largest=True)
        outlier_mask = torch.zeros_like(reshaped_scores, dtype=torch.bool)
        outlier_mask.scatter_(1, top_indices, True)
        
        # 4. 生成 Pruning Mask (Bottom K in each block)
        # topk(largest=False) 找最小的
        _, bottom_indices = torch.topk(reshaped_scores, k=k_prune, dim=1, largest=False)
        pruning_mask = torch.zeros_like(reshaped_scores, dtype=torch.bool)
        pruning_mask.scatter_(1, bottom_indices, True)
        
        # 5. 确保互斥 (理论上 Top K 和 Bottom K 在 K_p + K_pr < Block 时不会重叠，但加个保险)
        # 如果既是 outlier 又是 prune (几乎不可能)，优先保 Outlier
        pruning_mask = pruning_mask & (~outlier_mask)
        
        return outlier_mask.view(original_shape), pruning_mask.view(original_shape)