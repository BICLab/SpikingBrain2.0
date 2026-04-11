

def register_model():

    from .sse_swa_moba import register_model
    register_model()
    print("vllm plugin register model successfully.")
    register_attention_backends()
    print("vllm plugin register attn_backend successfully.")


def register_attention_backends():
    
    # from .attention.moba_attn import MobaSseFlashAttentionBackend
    # Registration is handled by the @register_backend decorator on the class itself, so no additional code is needed here.
    # but this function serves as a central place to ensure the module is imported and the backend is registered when this function is called.
    from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum, _ATTN_OVERRIDES
    register_backend(
        backend=AttentionBackendEnum.FLASH_ATTN,
        class_path="sse_swa_moba_vllm.attention.moba_attn.MobaSseFlashAttentionBackend"
    )
    print(f"Registered MobaSseFlashAttentionBackend for FLASH_ATTN. Current overrides: {_ATTN_OVERRIDES[AttentionBackendEnum.FLASH_ATTN]}")