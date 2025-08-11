import torch
from functools import partial

def replace_attn_with_checkpointed_version(module):
    """
    Thay thế các hàm attention với phiên bản hỗ trợ gradient checkpointing
    """
    orig_forward = module.forward
    
    def checkpointed_forward(*args, **kwargs):
        # Kiểm tra nếu gradient_checkpointing được bật và đang training
        if getattr(module, "gradient_checkpointing", False) and module.training:
            # Tạo function cho checkpoint
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
                
            # Sử dụng checkpoint
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(orig_forward),
                *args,
                use_reentrant=False,
                **kwargs
            )
        else:
            return orig_forward(*args, **kwargs)
    
    # Thay thế forward method
    module.forward = checkpointed_forward
    # Thêm attribute gradient_checkpointing
    module.gradient_checkpointing = False
    
    return module

def patch_internvl2_gradient_checkpointing(model):
    """
    Áp dụng patch gradient checkpointing cho InternVL2
    """
    try:
        # Lặp qua tất cả các module attention trong mô hình và patch
        from transformers.models.internvl2.modeling_internvl2 import InternVL2Attention
        
        # Patch các module attention
        for name, module in model.named_modules():
            if isinstance(module, InternVL2Attention):
                replace_attn_with_checkpointed_version(module)
        
        # Thêm method để bật/tắt gradient checkpointing
        def _set_gradient_checkpointing(self, value):
            for module in self.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = value
        
        # Gắn method vào model
        model._set_gradient_checkpointing = partial(_set_gradient_checkpointing, model)
        
        print("✅ Successfully patched InternVL2 for gradient checkpointing")
        return model
    except Exception as e:
        print(f"❌ Failed to patch InternVL2: {str(e)}")
        raise e
