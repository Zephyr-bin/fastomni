import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

# [Lite-MicroScopiQ] Import
from quantize.fisher_utils import FisherPruner
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting OmniQuant with Lite-MicroScopiQ Co-design ...")
    
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    
    # --- Model Setup ---
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    
    # --- Input Catcher ---
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # Move embedding/first layer back to CPU to save memory
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # Setup inputs for FP and Quant models
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None 
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    # ================= LAYER LOOP =================
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
        
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False 
        
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        
        # -------------------------------------------------------------------------
        # [Lite-MicroScopiQ: Mask-Aware Injection]
        # 在训练循环(LWC Optimization)开始之前注入Mask，实现Co-design。
        # -------------------------------------------------------------------------
        # if args.epochs > 0:
        #     logger.info(f"=== [Lite-MicroScopiQ] Analyzing Layer {i} for Mask-Aware Training ===")
            
        #     # 1. 初始化 FisherPruner
        #     fisher_pruner = FisherPruner(qlayer)
            
        #     # 2. 准备小批量输入 (Use subset of inps to save time/memory)
        #     # OmniQuant inps is [nsamples, seq, hidden]. We need to feed [batch, seq, hidden].
        #     # [Fix OOM] 12GB 显存建议设置为 1 或 2。虽然样本少一点，但对于权重重要性排序通常足够了。
        #     fisher_samples = 2 
        #     fisher_input = inps[:fisher_samples].to(dev)
            
        #     # Handle Masks
        #     # [Fix Dimension Mismatch]
        #     # 不能直接切片 attention_mask_batch，因为它受限于命令行参数 --batch_size (默认为1)
        #     # 我们需要根据 fisher_samples (2) 重新生成对应大小的 mask
        #     if attention_mask is not None:
        #         # attention_mask 原始形状通常是 [1, 1, seq, seq]
        #         # 我们将其复制 fisher_samples 次 -> [2, 1, seq, seq]
        #         f_attn_mask = attention_mask.repeat(fisher_samples, 1, 1, 1).to(dev)
        #         # 保持精度一致性 (OmniQuant 默认 float)
        #         if not args.deactive_amp:
        #             f_attn_mask = f_attn_mask.float()
        #     else:
        #         f_attn_mask = None
            
        #     fisher_args = (fisher_input, f_attn_mask, position_ids)
            
        #     # 3. 计算 Fisher Information
        #     # 注意：在 fisher_utils.py 中需要确保这一步能处理梯度 (requires_grad)
        #     fisher_pruner.collect_fisher_info(fisher_args, device=dev)
            
        #     # 4. 注册 Masks (使用 Block-Balanced 策略)
        #     # 策略：5% Outlier (FP16), 15% Pruning (Zero)
        #     # 这会把 qlayer 里的 QuantLinear 替换成 OutlierAwareQuantizerWrapper
        #     qlayer.register_outlier_masks(
        #         fisher_pruner, 
        #         outlier_ratio=0.05, 
        #         prune_ratio=0.05  # Block-Balanced Pruning Ratio
        #     )
            
        #     # 5. 清理显存
        #     del fisher_pruner
        #     torch.cuda.empty_cache()
        #     logger.info(f"=== [Lite-MicroScopiQ] Masks Injected. Starting Mask-Aware LWC Training... ===")
        # # -------------------------------------------------------------------------

       # -------------------------------------------------------------------------
        # [Fix] 强制在 Resume 模式下生成 Mask
       # -------------------------------------------------------------------------
        # [Fix] 强制生成 Mask
        # -------------------------------------------------------------------------
        # [Fix] 强制在 Resume 模式下生成 Mask
        if True:  
            logger.info(f"=== [Lite-MicroScopiQ] Analyzing Layer {i} for Mask-Aware Training ===")
            
            # [关键修复 1] 将当前层转为 FP32！防止梯度下溢变为0
            qlayer.float()
            
            # 1. 打开梯度
            for param in qlayer.parameters():
                param.requires_grad = True
            
            # 2. 准备 Pruner
            fisher_pruner = FisherPruner(qlayer)
            
            # [Fix OOM] 样本量设为 4 (FP32下显存占用会增加，4是安全值)
            fisher_samples = 4  
            fisher_input = inps[:fisher_samples].to(dev).float()
            
            if attention_mask is not None:
                f_attn_mask = attention_mask.repeat(fisher_samples, 1, 1, 1).to(dev).float()
            else:
                f_attn_mask = None
            
            fisher_args = (fisher_input, f_attn_mask, position_ids)
            
            # 3. 计算 Fisher (现在是 FP32，梯度不会丢了！)
            try:
                fisher_pruner.collect_fisher_info(fisher_args, device=dev)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"| WARNING: OOM during Fisher. Skipping.")
                    torch.cuda.empty_cache()
                else:
                    raise e
            
            # 4. 注册 Masks (5% 保护, 0% 剪枝)
            # [Debug] 打印一下 Mask 的非零数量，确认生效
            # (这一步是在 int_llama_layer.py 里打印的，我们这里只要调用即可)
            qlayer.register_outlier_masks(
                fisher_pruner, 
                outlier_ratio=0.125, 
                prune_ratio=0.0 
            )
            
            # 5. 清理现场
            qlayer.zero_grad() 
            
            # [关键修复 2] 恢复为 FP16，准备接下来的训练
            qlayer.half() 
            
            # 关闭梯度 (只保留LWC参数梯度)
            for name, param in qlayer.named_parameters():
                if "smooth" in name or "bound_factor" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            del fisher_pruner
            del fisher_input
            if f_attn_mask is not None: del f_attn_mask
            
            torch.cuda.empty_cache()
            logger.info(f"=== [Lite-MicroScopiQ] Masks Injected... ===")
        # -------------------------------------------------------------------------

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        # Mask-Aware Training 关键点：
                        # smooth_and_quant_temporary 会调用 Wrapper 的 forward。
                        # Wrapper 会根据 Mask 把一部分权重变为 0 (Pruned)，一部分变为 FP16 (Outlier)。
                        # LWC 只能改变剩余 INT4 部分的 scaling/clipping 参数来最小化 Loss。
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
                # ================= [Lite-MicroScopiQ: Iterative Refinement] =================
                # 在第 2 个 Epoch (index=1) 结束时，根据当前优化后的权重，重新评估重要性并更新 Mask。
                # 这体现了"Co-evolution"（协同演化）思想。
                # ================= [Lite-MicroScopiQ: Iterative Refinement] =================
                # 在第 2 个 Epoch (index=1) 结束时，根据当前优化后的权重，重新评估重要性并更新 Mask。
                # ================= [Lite-MicroScopiQ: Iterative Refinement] =================
                # ================= [Lite-MicroScopiQ: Iterative Refinement] =================
                if epochs == 1: 
                # if False:
                    logger.info(f"=== [Iterative] Re-evaluating Layer {i} Masks based on updated weights... ===")
                    
                    # 1. 清理临时变量
                    clear_temp_variable(qlayer) 
                    for name, module in qlayer.named_modules():
                        if isinstance(module, QuantLinear):
                            module.use_temporary_parameter = False
                    
                    optimizer.zero_grad()
                    
                    # [关键修复] 再次打开梯度！否则 Fisher 算出来全是 0，导致 Mask 被重置！
                    for param in qlayer.parameters():
                        param.requires_grad = True

                    # 3. 初始化 & 准备数据
                    fisher_pruner = FisherPruner(qlayer)
                    
                    # [Fix OOM] 保持 Sample=4 (和初始化一致)
                    fisher_samples = 4 
                    fisher_input = inps[:fisher_samples].to(dev).float()
                    
                    f_attn_mask = attention_mask_batch[:fisher_samples] if attention_mask_batch is not None else None
                    if f_attn_mask is not None:
                         f_attn_mask = attention_mask.repeat(fisher_samples, 1, 1, 1).to(dev).float()

                    fisher_args = (fisher_input, f_attn_mask, position_ids)
                    
                    # 4. 计算 Fisher
                    try:
                        fisher_pruner.collect_fisher_info(fisher_args, device=dev)
                    except RuntimeError as e:
                         if "out of memory" in str(e):
                            print(f"| WARNING: OOM in Iterative Step. Skipping update.")
                            torch.cuda.empty_cache()
                         else:
                            raise e
                    
                    # 5. 更新 Mask (保持 5% 保护, 0% 剪枝)
                    qlayer.register_outlier_masks(
                        fisher_pruner, 
                        outlier_ratio=0.125,
                        prune_ratio=0.0
                    )
                    
                    # [关键修复] 关闭梯度 (恢复到只训练 LWC 参数的状态)
                    qlayer.zero_grad()
                    for name, param in qlayer.named_parameters():
                        if "smooth" in name or "bound_factor" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    
                    del fisher_pruner
                    torch.cuda.empty_cache()
                    logger.info(f"=== [Iterative] Masks Updated. Continuing training... ===")
            clear_temp_variable(qlayer)
            del optimizer
        
        # Training Finish
        qlayer.half() 
        
        # Finalize Weights
        # 这里会调用 Wrapper 的 forward，将最终的混合精度权重写入 module.weight
        smooth_and_quant_inplace(qlayer, args, is_llama)
        
        # Prepare for next layer
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        
        # Real Quantization Packing (Optional / For Deployment)
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model