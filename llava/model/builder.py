from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import os


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", lora_path=None, clip_path=None):

    kwargs = {}
    
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    else:
        # Use better device mapping strategy for cuda
        kwargs['device_map'] = device_map    # 多 GPU 时
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # 可通过环境变量强制使用 FP32 以规避半精度内核 FPE
        force_fp32 = os.environ.get("LLAVA_FORCE_FP32", "0") == "1"
        if force_fp32:
            kwargs['torch_dtype'] = torch.float32
        else:
            # 优先使用 bfloat16（若支持），以降低部分 GPU/驱动上 FP16 算子触发 FPE 的风险
            try:
                prefer_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            except Exception:
                prefer_bf16 = False
            kwargs['torch_dtype'] = torch.bfloat16 if prefer_bf16 else torch.float16

    
    if 'llava' in model_name.lower():# load llava-med
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=32768)
        # Set low_cpu_mem_usage=True for quantized models to avoid meta device issues
        if load_8bit or load_4bit:
            model_kwargs = dict(low_cpu_mem_usage=True, use_flash_attention_2=False, **kwargs)
        else:
            model_kwargs = dict(low_cpu_mem_usage=True, use_flash_attention_2=False, **kwargs)
        
        model = LlavaMistralForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Load LoRA weights if specified and keep them as separate adapter
        if lora_path is not None:
            from peft import PeftModel
            print(f"Loading LoRA weights from {lora_path} without merging")
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"LoRA weights loaded successfully")
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, model_max_length=32768)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            
            # If lora_path is specified, use it instead of model_path for LoRA weights
            lora_weights_path = lora_path if lora_path is not None else model_path
            print(f"Loading LoRA weights from {lora_weights_path}")
            model = PeftModel.from_pretrained(model, lora_weights_path)
            
            # Only merge weights if lora_path is not specified (backwards compatibility)
            if lora_path is None:
                print(f"Merging weights")
                model = model.merge_and_unload()
                print('Convert to FP16...')
                model.to(torch.float16)
            else:
                print(f"Keeping LoRA weights as separate adapter")
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, model_max_length=32768)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, model_max_length=32768)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower(): # or 'mistral' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            # 直接使用传入的clip_path参数
            if clip_path and os.path.isdir(clip_path):
                print(f"Using local CLIP model from: {clip_path}")
                # 保存原始的vision_tower_name
                original_vision_tower_name = vision_tower.vision_tower_name
                # 临时修改vision_tower_name为本地路径
                vision_tower.vision_tower_name = clip_path
                try:
                    vision_tower.load_model()
                except Exception as e:
                    print(f"Failed to load local CLIP model: {e}")
                    # 如果加载失败，恢复原始的vision_tower_name并重试
                    vision_tower.vision_tower_name = original_vision_tower_name
                    vision_tower.load_model()
            else:
                vision_tower.load_model()
        
        # Handle device placement and dtype
        if load_8bit or load_4bit:
            # For quantized models, let the device_map handle placement
            pass
        else:
            # 与主模型 dtype 对齐（优先 bf16）
            target_dtype = kwargs.get('torch_dtype', torch.float16)
            vision_tower.to(device=device, dtype=target_dtype)
            if hasattr(model, 'model') and hasattr(model.model, 'mm_projector'):
                model.model.mm_projector.to(device=device, dtype=target_dtype)
            model.to(device=device, dtype=target_dtype)
        
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 32768

    # 同步生成配置的 pad/eos，避免生成时被覆盖导致不一致
    try:
        if getattr(model, 'generation_config', None) is not None:
            if model.generation_config.pad_token_id is None and getattr(tokenizer, 'pad_token_id', None) is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            if model.generation_config.eos_token_id is None and getattr(tokenizer, 'eos_token_id', None) is not None:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    return tokenizer, model, image_processor, context_len