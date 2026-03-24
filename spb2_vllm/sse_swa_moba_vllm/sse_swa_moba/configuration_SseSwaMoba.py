from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger
logger = init_logger(__name__)


class SseSwaMobaConfig(PretrainedConfig):

    _supports_flash_attn=True
    model_type = "sse_swa_moba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            attn_mode: str = "chunk",
            hidden_size: int = 256, # 2048
            expand_v: float = 1.0,
            use_output_gate: bool = True,
            use_short_conv: bool = False,
            allow_neg_eigval: bool = False,
            conv_size: int = 4,
            head_dim: int = 64, # 256
            num_heads: int = 4, # 8
            num_v_heads: int | None = None,
            num_sparse_partition: int = 4,
            num_writer: int = 2,
            num_reader: int = 2,
            rope_scaling: float | None = None,
            linear_attn_type: str = "gdn",
            sse_implementation: str = "varlen",
            sse_qk_relu: bool = False,
            aux_loss_coef: float = 0.01,
            max_position_embeddings: int = 2048,
            hidden_ratio: int | None = 4,
            intermediate_size: int | None = None,
            hidden_act: str = "swish",
            num_hidden_layers: int = 3, # 24
            norm_eps: float = 1e-6,
            # mini configuration
            attn: dict | None = {
                "layers": [
                    1, 2
                ],
                "full_layers": [2],
                "num_heads": 32,
                "num_kv_heads": 8,
                "qkv_bias": False,
                "qk_norm": True,
                "window_size": 64,
                "rope_theta": 5000000,
                "moba_chunk_size": 512,
                "moba_topk": 4
            },
            swa_dropout: float = 0.5,
            use_cache: bool = True,
            pad_token_id: int | None = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            tie_word_embeddings: bool = False,
            initializer_range: float = 0.02,
            fuse_norm: bool = True,
            fuse_swiglu: bool = True,
            fuse_cross_entropy: bool = False,
            fuse_linear_cross_entropy: bool = False,
            use_l2warp: bool = False,
            vocab_size: int = 32000,
            **kwargs,
    ):
        # vllm must have
        self.num_attention_heads = num_heads
        if attn is not None:
            self.num_key_value_heads = attn['num_kv_heads']

        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.num_sparse_partition = num_sparse_partition
        self.num_writer = num_writer
        self.num_reader = num_reader
        self.rope_scaling = rope_scaling
        self.linear_attn_type = linear_attn_type
        self.sse_implementation = sse_implementation
        self.sse_qk_relu = sse_qk_relu
        self.aux_loss_coef = aux_loss_coef
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.swa_dropout = swa_dropout
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            logger.warning(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['moba_chunk_size'] = attn.get('moba_chunk_size', 1024)
            attn['moba_topk'] = attn.get('moba_topk', 4)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
