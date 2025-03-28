# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from https://github.com/huggingface/transformers/blob/52daf4ec768fb9ffe84a0c373834172a7c54aecc/src/transformers/models/llama/modeling_llama.py
#
""" MindSpore Emu3 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

from emu3.acceleration import GatherFowardSplitBackward, SplitFowardGatherBackward, get_sequence_parallel_group
from emu3.mllm.configuration_emu3 import Emu3Config
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Normal, initializer
from mindspore.communication import get_group_size

from mindway.transformers.activations import ACT2FN
from mindway.transformers.cache_utils import Cache, DynamicCache  # , get_max_length, get_seq_length, update
from mindway.transformers.mindspore_utils import ALL_LAYERNORM_LAYERS
from mindway.transformers.modeling_attn_mask_utils import (
    _MIN_FP16,
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from mindway.transformers.modeling_outputs import (  # SequenceClassifierOutputWithPast,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from mindway.transformers.modeling_utils import MSPreTrainedModel

logger = logging.get_logger(__name__)

from mindway.transformers.utils import is_flash_attn_2_available  # Ascend
from mindone.utils.version_control import check_valid_flash_attention

FLASH_IS_AVAILABLE = is_flash_attn_2_available and check_valid_flash_attention()
if FLASH_IS_AVAILABLE:
    from mindspore.ops.operations.nn_ops import FlashAttentionScore as MSFlashAttention

from mindspore.nn import CrossEntropyLoss  # BCEWithLogitsLoss, MSELoss

_CONFIG_FOR_DOC = "Emu3Config"

class Emu3RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(ops.ones(hidden_size, dtype=ms.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon) * self.weight
        return hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Emu3RMSNorm)

class Emu3MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=config.mlp_bias)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class Emu3RotaryEmbedding(nn.Cell):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))

        # Build here to make `jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, dtype=None):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        if dtype is not None:
            self.cos_cached = self.cos_cached.to(dtype)
            self.sin_cached = self.sin_cached.to(dtype)

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Emu3LinearScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class Emu3DynamicNTKScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mint.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(ms.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed





def repeat_kv(hidden_states: ms.Tensor, n_rep: int) -> ms.Tensor:
    """
    This is the equivalent of ops.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Emu3Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # self.hidden_size = config.hidden_size  # 4096
        self.num_heads = config.num_attention_heads  # 32
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)  # 4096 / 32 = 128
        self.num_key_value_heads = config.num_key_value_heads  # 8
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 32 / 8 = 4
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        # self.max_position_embeddings = config.max_position_embeddings  # 9216
        # self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Dense(config.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(config.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.v_proj = nn.Dense(config.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.attention_bias)

        # Initialize sequence parallel operator
        if (sp_group := get_sequence_parallel_group()) is not None:
            self.sp_group_size = get_group_size(sp_group)
            self.alltoall = ops.AlltoAll(self.sp_group_size, 1, 2, group=sp_group)
        else:
            self.sp_group_size = None
            self.alltoall = nn.Identity()

    def construct(
        self,
        hidden_states: ms.Tensor,
        position_embeddings: Tuple[ms.Tensor, ms.Tensor],
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        target_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view((bsz, q_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        key_states = key_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)
        value_states = value_states.view((bsz, q_len, self.num_key_value_heads, self.head_dim)).swapaxes(1, 2)

        # sequence parallel: scatter BNS'D => BN'SD
        query_states = self.alltoall(query_states.float()).to(target_dtype)
        key_states = self.alltoall(key_states.float()).to(target_dtype)
        value_states = self.alltoall(value_states.float()).to(target_dtype)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = mint.matmul(query_states, key_states.swapaxes(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        dropout=0.0 if not self.training else self.attention_dropout
        attn_weights = mint.nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)
        attn_output = attn_output.swapaxes(1, 2).contiguous()  # BN'SD => BSN'D

        # sequence parallel: gather BSN'D => BS'ND
        attn_output = self.alltoall(attn_output.float()).to(target_dtype)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Emu3FlashAttention2(Emu3Attention):
    """
    Emu3 flash attention module. This module inherits from `Emu3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_flash_attention = FLASH_IS_AVAILABLE
        dropout_rate = self.attention_dropout if self.training else 0.0

        # sequence parallel
        num_heads = self.num_heads // self.sp_group_size if self.sp_group_size is not None else self.num_heads
        self.fa_dtype = ms.float16
        if self.enable_flash_attention:
            # Q: (b s n d) -> (b n s d)  #  b - batch_size, s - seq_len, n - num_head, d - head dim
            self.flash_attention = MSFlashAttention(
                scale_value=self.head_dim**-0.5,
                head_num=num_heads,
                keep_prob=1 - dropout_rate,
                input_layout="BNSD",  # BSH or BNSD
            )
        else:
            self.flash_attention = None

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        # Emu3FlashAttention2 attention does not support output_attentions

        bsz, q_len, _ = hidden_states.shape
        target_dtype = hidden_states.dtype

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x head_dim x seq_length x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        # sequence parallel: scatter BNS'D => BN'SD
        query_states = self.alltoall(query_states.float()).to(target_dtype)
        key_states = self.alltoall(key_states.float()).to(target_dtype)
        value_states = self.alltoall(value_states.float()).to(target_dtype)
        # print(f"query_states {query_states.shape}, key_states {key_states.shape}, value_states {value_states.shape}")

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # NOTE: MSFlashAttention needs shape of BNSD ==> [batch_size,  num_heads, sequence_length, head_dim].

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if attention_mask is not None:
            attention_mask = self.convert_mask_to_fa_format(attention_mask)
        attn_output = self.flash_attention(
            query_states.to(self.fa_dtype),
            key_states.to(self.fa_dtype),
            value_states.to(self.fa_dtype),
            None,
            None,
            None,
            attention_mask,
        )[3]
        attn_output = attn_output.to(target_dtype)

        attn_output = attn_output.swapaxes(1, 2)  # b h n d -> b n h d (bsz, q_len, num_heads, head_dim)

        # sequence parallel: gather BSN'D => BS'ND
        attn_output = self.alltoall(attn_output.float()).to(target_dtype)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_weights = None

        return attn_output, attn_weights, past_key_value

    def convert_mask_to_fa_format(self, attention_mask):
        if attention_mask is not None:
            if attention_mask.dtype == ms.bool_:
                # flip mask, since ms FA treats 1 as discard, 0 as retain.
                attention_mask = 1 - attention_mask
                attention_mask = attention_mask.to(ms.uint8)
            else:
                # attention_mask has beed inverted before in _prepare_4d_causal_mask: 0: retain, -inf: discard
                attention_mask = attention_mask.to(ms.float16)
                attention_mask = ops.select(
                    ops.equal(attention_mask, _MIN_FP16),
                    ops.ones((), ms.uint8),
                    ops.zeros((), ms.uint8),
                )

        return attention_mask


class Emu3SdpaAttention(Emu3Attention):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


EMU3_ATTENTION_CLASSES = {
    "eager": Emu3Attention,
    "flash_attention_2": Emu3FlashAttention2,
    "sdpa": Emu3SdpaAttention,  # Not supported
}


class Emu3DecoderLayer(nn.Cell):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = EMU3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(p=config.attention_dropout)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
            cache_position (`ms.Tensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if not output_attentions:
            self_attn_weights = None
        outputs += (self_attn_weights,)

        return outputs


class Emu3VQVAEVectorQuantizer(nn.Cell):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to te one described in
    the VQ-VAE (Vector Quantized Variational AutoEncoder) paper. It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.
    Current implementation improves over previous ones by avoiding costly matrix multiplications
    and allowing for post-hoc remapping of indices.
    """
    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        self.embedding.embedding_table.set_data(
            initializer(
                Uniform(scale=1.0 / config.codebook_size),
                self.embedding.embedding_table.shape,
                self.embedding.embedding_table.dtype,
            )
        )

    def construct(self, hidden_state: ms.Tensor):
        # b t c h w -> b t h w c
        batch_size, temporal, channels, height, width = hidden_state.shape
        hidden_state = hidden_state.permute(0, 1, 3, 4, 2).contiguous()  # (b, t, h, w, c)
        hidden_state_flattened = hidden_state.view(-1, channels)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_state_sum = ops.sum(hidden_state_flattened**2, dim=1, keepdim=True)
        embedding_sum = ops.sum(self.embedding.embedding_table**2, dim=1)

        # "bd,dn->bn"
        distances = 2 * ops.matmul(hidden_state_flattened, self.embedding.embedding_table.permute(0, 1))
        distances = hidden_state_sum + embedding_sum - distances

        min_encoding_indices = ops.argmin(distances, axis=1)
        min_encoding_indices = min_encoding_indices.view(batch_size, temporal, height, width)
        return min_encoding_indices



class Emu3VisionVQActivation(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x: ms.Tensor):
        return x * ops.sigmoid(x)




class Emu3VQVAEEncoderConvDownsample(nn.Cell):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, pad_mode="pad", padding=0, has_bias=True
        )
        padding = ((0, 0), (0, 0), (0, 1), (0, 1))
        self.pad_op = ops.Pad(padding)

    def construct(self, x: ms.Tensor):
        # pad = (0, 1, 0, 1)
        # x = ops.pad(x, pad, mode="constant", value=0) # Graph mode not support bfloat16
        x = self.pad_op(x)
        x = self.conv(x)
        return x


class Emu3VisionVQUpsample(nn.Cell):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor):
        x = ops.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        x = self.conv(x)
        return x

class Emu3VisionVQCausalConv3d(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int, ...]] = (3, 1, 1),
        stride: Union[int, Tuple[int, ...]] = (1, 1, 1),
        conv3d_dtype=ms.float16,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        hw_pad = [k - s for k, s in zip(kernel_size[1:], stride[1:])]
        self.padding = tuple()
        padding = ((0, 0), (0, 0), (2, 0))
        for p in hw_pad:
            padding += ((p // 2 + p % 2, p // 2),)  # NOTE: add in ((a,b),) format, make sure padding in (N,2) shape
        self.pad_op = ops.Pad(padding)
        # for p in hw_pad[::-1]:
        #     self.padding += (p // 2 + p % 2, p // 2),
        # self.padding += (2, 0)

        if ms.__version__ >= "2.5":
            self.conv = mint.nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ).to_float(conv3d_dtype)
        else:
            self.conv = nn.Conv3d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                pad_mode="valid",
                has_bias=True,
            ).to_float(conv3d_dtype)

    def construct(self, x: ms.Tensor):
        origin_dtype = x.dtype
        # x = ops.pad(x, self.padding, mode="constant", value=0)
        x = self.pad_op(x)
        x = self.conv(x)
        x = x.to(origin_dtype)
        return x


class Emu3VisionVQResnetTemporalBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        conv3d_dtype=ms.float16,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        stride = (1, 1, 1)
        kernel_size = (3, 3, 3)

        self.norm1 = nn.BatchNorm3d(in_channels)
        self.conv1 = Emu3VisionVQCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = Emu3VisionVQCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Emu3VisionVQCausalConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            else:
                if ms.__version__ >= "2.5":
                    self.nin_shortcut = mint.nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
                    ).to_float(conv3d_dtype)
                else:
                    self.nin_shortcut = nn.Conv3d(
                        in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
                    ).to_float(conv3d_dtype)

    def construct(self, x: ms.Tensor):
        origin_dtype = x.dtype
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
            x = x.to(origin_dtype)

        return x + h


class Emu3VisionVQSpatialNorm(nn.Cell):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        norm_layer: nn.Cell = nn.GroupNorm,
        add_conv: bool = False,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        self.norm_layer = norm_layer(
            num_channels=f_channels,
            num_groups=num_groups,
            eps=eps,
            affine=affine,
        )

        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(
                zq_channels, zq_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
            )

        self.conv_y = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )
        self.conv_b = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor, zq: ms.Tensor):
        zq = ops.interpolate(zq, size=x.shape[-2:], mode="nearest")

        if self.add_conv:
            zq = self.conv(zq)

        x = self.norm_layer(x)
        x = x * self.conv_y(zq) + self.conv_b(zq)
        return x


class Emu3VisionVQResnetBlock(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm1 = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, **norm_kwargs)
        else:
            self.norm1 = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
            self.norm2 = Emu3VisionVQSpatialNorm(out_channels, zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
                )

    def construct(self, x: ms.Tensor, zq: Optional[ms.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        h = self.norm1(x, *norm_args)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, *norm_args)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Emu3VisionVQAttnBlock(nn.Cell):
    def __init__(self, in_channels: int, zq_ch: Optional[int] = None, add_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
        else:
            self.norm = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, pad_mode="pad", has_bias=True
        )

    def construct(self, x: ms.Tensor, zq: Optional[ms.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        nx = self.norm(x, *norm_args)
        q = self.q(nx)
        k = self.k(nx)
        v = self.v(nx)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        score = ops.bmm(q.permute(0, 2, 1), k)
        score = score / (c**0.5)
        score = ops.softmax(score, axis=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        v = ops.bmm(v, score.permute(0, 2, 1))
        v = v.reshape(b, c, h, w)

        v = self.proj_out(v)

        return x + v


class Emu3VisionVQTemporalUpsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (3, 3, 3),
        stride: Tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def construct(self, x: ms.Tensor):
        b, c, t, h, w = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous().view(b, c * h * w, t)  # (b, c, h, w, t) => (b, c*h*w, t)
        x = ops.interpolate(x, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        x = x.view(b, c, h, w, x.shape[-1]).permute(0, 1, 4, 2, 3).contiguous()
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalDownsample(nn.Cell):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (4, 3, 3),
        stride: Tuple[int, ...] = (2, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
        )

    def construct(self, x: ms.Tensor):
        x = self.conv(x)
        return x




class Emu3VisionVQEncoder(nn.Cell):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels, self.ch, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.CellList()
        for i_level in range(self.num_resolutions):
            block = nn.CellList()
            attn = nn.CellList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in))

            down = nn.Cell()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VisionVQDownsample(block_in)

            self.down.append(down)

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # end
        self.norm_out = nn.GroupNorm(num_channels=block_in, num_groups=32, eps=1e-6, affine=True)

        out_z_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = nn.Conv2d(
            block_in, out_z_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()

        for i in range(temporal_down_blocks):
            conv = Emu3VisionVQTemporalDownsample(out_z_channels, out_z_channels)
            self.time_conv.append(conv)

        self.time_res_stack = nn.SequentialCell(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=out_z_channels,
                    out_channels=out_z_channels,
                    dropout=config.dropout,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        self.act = Emu3VisionVQActivation()

    def construct(self, x: ms.Tensor):
        t = x.shape[1]
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.act(h)

        h = self.conv_out(h)

        h_n = h.shape[0] // t
        h = h.reshape(h_n, t, *h.shape[1:])
        h = h.permute(0, 2, 1, 3, 4)

        for conv in self.time_conv:
            h = self.act(conv(h))

        h = self.time_res_stack(h)
        h = h.permute(0, 2, 1, 3, 4)

        return h


class Emu3VisionVQDecoder(nn.Cell):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        # in_ch_mult = (1,) + tuple(config.ch_mult)
        zq_ch = config.embed_dim

        block_in = config.ch * config.ch_mult[-1]
        self.time_res_stack = nn.SequentialCell(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=config.z_channels,
                    out_channels=config.z_channels,
                    dropout=config.dropout,
                )
                for _ in range(config.num_res_blocks)
            ]
        )

        tempo_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.CellList()
        for i in range(tempo_upsample_block_num):
            conv = Emu3VisionVQTemporalUpsample(config.z_channels, config.z_channels)
            self.time_conv.append(conv)

        self.conv_in = nn.Conv2d(
            config.z_channels, block_in, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

        # middle
        self.mid = nn.Cell()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in, zq_ch)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )

        # upsampling
        self.up = nn.CellList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.CellList()
            attn = nn.CellList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                        zq_ch=zq_ch,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in, zq_ch))

            up = nn.Cell()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Emu3VisionVQUpsample(block_in)

            self.up.insert(0, up)

        self.act = Emu3VisionVQActivation()

        self.norm_out = Emu3VisionVQSpatialNorm(block_in, zq_ch)
        self.conv_out = nn.Conv2d(
            block_in, config.out_channels, kernel_size=3, stride=1, padding=1, pad_mode="pad", has_bias=True
        )

    def construct(self, z: ms.Tensor, zq: ms.Tensor):
        z_zq = mint.cat((z, zq), dim=0)
        z_zq = z_zq.permute(0, 2, 1, 3, 4)
        z_zq = self.time_res_stack(z_zq)

        for conv in self.time_conv:
            z_zq = self.act(conv(z_zq))

        z_zq = z_zq.permute(0, 2, 1, 3, 4)

        h, zq = mint.chunk(z_zq, 2, dim=0)

        h = h.reshape(h.shape[0] * h.shape[1], *h.shape[2:])
        zq = zq.reshape(zq.shape[0] * zq.shape[1], *zq.shape[2:])

        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, zq)
        h = self.act(h)
        h = self.conv_out(h)

        return h


class Emu3VisionVQPretrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Emu3VisionVQConfig
    base_model_prefix = "emuvideovq"
    main_input_name = "pixel_values"
    _no_split_modules = ["Emu3VisionVQResnetBlock", "Emu3VisionVQAttnBlock", "Emu3VisionVQResnetTemporalBlock"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)) or (
            ms.__version__ >= "2.5" and isinstance(module, mint.nn.Conv3d)
        ):
            module.weight.set_data(
                initializer(HeNormal(mode="fan_out", nonlinearity="relu"), module.weight.shape, module.weight.dtype)
            )
        elif isinstance(module, nn.Linear):
            weight = initializer(HeNormal(negative_slope=math.sqrt(5)), module.weight.shape, module.weight.dtype)
            module.weight.set_data(weight)
            if module.bias is not None:
                fan_in, _ = initializer._calculate_fan_in_and_fan_out(module.weight.shape)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                bias_weight = initializer(
                    Uniform(scale=bound), module.embedding_table.shape, module.embedding_table.dtype
                )
                module.bias.set_data(bias_weight)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            module.gamma.set_data(initializer(Constant(1), module.gamma.shape, module.gamma.dtype))
            module.beta.set_data(initializer(Constant(0), module.beta.shape, module.beta.dtype))
        elif isinstance(module, nn.BatchNorm3d):
            module.bn2d.gamma.set_data(initializer(Constant(1), module.bn2d.gamma.shape, module.bn2d.gamma.dtype))
            module.bn2d.beta.set_data(initializer(Constant(0), module.bn2d.beta.shape, module.bn2d.beta.dtype))


class Emu3VisionVQModel(Emu3VisionVQPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = Emu3VisionVQEncoder(config)
        self.decoder = Emu3VisionVQDecoder(config)
        self.quantize = Emu3VisionVQVectorQuantizer(config)

        self.quant_conv = Emu3VisionVQCausalConv3d(config.z_channels, config.embed_dim)
        self.post_quant_conv = Emu3VisionVQCausalConv3d(config.embed_dim, config.z_channels)

        self.spatial_scale_factor = 2 ** (len(config.ch_mult) - 1)

        self.post_init()

    def encode(self, x: ms.Tensor):
        ndim = x.ndim
        if ndim == 4:
            t = self.config.temporal_downsample_factor
            b, c, h, w = x.shape
            x = x.unsqueeze(1).tile((1, t, 1, 1, 1))
        elif ndim == 5:
            b, t, c, h, w = x.shape

        h = self.encoder(x)

        # b t c h w -> b c t h w
        h = h.permute(0, 2, 1, 3, 4)
        h = self.quant_conv(h)
        # b c t h w -> b t c h w
        h = h.permute(0, 2, 1, 3, 4)

        codes = self.quantize(h)

        if ndim == 4:
            codes = codes.squeeze(1)

        return codes

    def decode(self, x: ms.Tensor):
        ndim = x.ndim
        if ndim == 3:
            x = x.unsqueeze(1)

        b, t, h, w = x.shape
        quant = self.quantize.embedding(x.flatten(start_dim=0))
        c = quant.shape[-1]
        quant = quant.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        quant2 = self.post_quant_conv(quant)

        quant = quant.permute(0, 2, 1, 3, 4)
        quant2 = quant2.permute(0, 2, 1, 3, 4)

        video = self.decoder(quant2, quant)
        video = video.reshape(
            b,
            t * self.config.temporal_downsample_factor,
            self.config.out_channels,
            h * self.spatial_scale_factor,
            w * self.spatial_scale_factor,
        )
        if ndim == 3:
            return video[:, 0]
        return video






EMU3_START_DOCSTRING = r"""
    This model inherits from [`MSPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a mindspore.nn.Cell subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3PreTrainedModel(MSPreTrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Emu3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True  # support Cache Classes
    # True: use DynamicCache by default, if cache_implementation=="static", use StaticCache;
    # False: use default Tuple static cache
    # Emu3 only supports DynamicCache for now

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Dense):
            module.weight.set_data(initializer(Normal(sigma=std, mean=0.0), module.weight.shape, module.weight.dtype))
            if module.bias is not None:
                module.bias.set_data(initializer("zeros", module.bias.shape, module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(
                initializer(Normal(sigma=std, mean=0.0), module.embedding_table.shape, module.embedding_table.dtype)
            )
            if module.padding_idx is not None:
                module.embedding_table[module.padding_idx] = 0


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(ms.Tensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(ms.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3Model(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3DecoderLayer`]

    Args:
        config: Emu3Config
    """

    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize sequence parallel
        if (sp_group := get_sequence_parallel_group()) is not None:
            logger.info(f"Initialize Emu3 model with sequence parallel group `{sp_group}`.")
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=1, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=1, grad_scale="up", group=sp_group)
        else:
            self.split_forward_gather_backward = nn.Identity()
            self.gather_forward_split_backward = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def recompute(self, cell, **recompute_kwargs):
        if isinstance(cell, nn.Dropout):
            return
        if not cell._has_config_recompute:
            cell.recompute(**recompute_kwargs)
        if isinstance(cell, nn.CellList):
            self.recompute(cell[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            cell.add_flags(output_no_recompute=True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True
        if gradient_checkpointing_kwargs is None:
            # gradient_checkpointing_kwargs = {"mp_comm_recompute": True, "parallel_optimizer_comm_recompute": True}
            gradient_checkpointing_kwargs = {}

        # llama layers
        for decoder_layer in self.layers:
            assert isinstance(decoder_layer, Emu3DecoderLayer)
            for name, cell in decoder_layer.name_cells().items():
                self.recompute(cell, **gradient_checkpointing_kwargs)
        self.recompute(self.embed_tokens, **gradient_checkpointing_kwargs)
        self.recompute(self.norm, **gradient_checkpointing_kwargs)

        logger.info(f"{self.__class__.__name__}: enable recompute.")

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        use_legacy_cache = False
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache) and self._supports_cache_class
            if use_legacy_cache:  # CFG logit processor use
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            if isinstance(past_key_values, Cache):
                past_key_values_length = past_key_values.get_usable_length(seq_length)
            else:  # tuple static cache
                pass
                # max_length = get_max_length(past_key_values)
                # previous_seq_length = get_seq_length(past_key_values)
                # if max_length is not None and previous_seq_length + seq_length > max_length:
                #     past_key_values_length = max_length - seq_length
                # else:
                #     past_key_values_length = previous_seq_length
        if position_ids is None:
            position_ids = ops.arange(past_key_values_length, seq_length + past_key_values_length, dtype=ms.int32)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds.to(ms.float16), past_key_values_length
        )
        # eager or FA, dtype fixed to fp16 (for FA accuracy)
        # inverted 4D mask (0 retain, -inf discard)
        # use cache: shape = [B, 1, q_seq, kv_seq=q_seq+past_kv_seq]
        # no cache:  shape = [B, 1, q_seq, q_seq]

        # sequence parallel start: BxSxD => BxS'xD (S'=S/M)
        inputs_embeds = self.split_forward_gather_backward(inputs_embeds)  # BSD => BS'D

        # embed positions
        hidden_states = self.dropout(inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # sequence parallel end
        hidden_states = self.gather_forward_split_backward(hidden_states)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            )  # CFG logit processor use
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Emu3ForCausalLM(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.loss_fct = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from emu3.mllm import Emu3ForCausalLM, Emu3Processor, Emu3Tokenizer
        >>> from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
        >>> from transformers.generation.configuration_utils import GenerationConfig
        >>> from mindway.transformers.generation.logits_process import (
        >>>     LogitsProcessorList,
        >>>     PrefixConstrainedLogitsProcessor,
        >>>     UnbatchedClassifierFreeGuidanceLogitsProcessor
        >>> )
        >>> from PIL import Image
        >>> from mindspore import Tensor

        >>> model = Emu3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)
        >>> tokenizer = Emu3Tokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> image_processor = Emu3VisionVQImageProcessor.from_pretrained(PATH_TO_CONVERTED_IMAGE_PROCESSER)
        >>> image_tokenizer = Emu3VisionVQModel.from_pretrained(PATH_TO_CONVERTED_TOKENIZER_WEIGHTS).set_train(False)
        >>> processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        >>> # Generation
        >>> prompt = "An Emu in cartoon style, it is wearing sunglasses."

        >>> pos_inputs = processor(text=prompt, mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="np")
        >>> neg_inputs = processor(text="", mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="np")

        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     use_cache=True,
        >>>     eos_token_id=model.config.eos_token_id,
        >>>     pad_token_id=model.config.pad_token_id,
        >>>     max_new_tokens=40960,
        >>>     do_sample=True,
        >>>     top_k=2048,
        >>> )

        >>> h, w = pos_inputs.image_size[0]
        >>> constrained_fn = processor.build_prefix_constrained_fn(h, w)
        >>> logits_processor = LogitsProcessorList([
        >>>     UnbatchedClassifierFreeGuidanceLogitsProcessor(
        >>>         classifier_free_guidance,
        >>>         model,
        >>>         unconditional_ids=Tensor(neg_inputs.input_ids),
        >>>     ),
        >>>     PrefixConstrainedLogitsProcessor(
        >>>         constrained_fn,
        >>>         num_beams=1,
        >>>     ),
        >>> ])

        >>> outputs = model.generate(Tensor(pos_inputs.input_ids), GENERATION_CONFIG, logits_processor=logits_processor)
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> mm_list = processor.decode(outputs[0])

        >>> # Understanding
        >>> prompt = "Provide a one-sentence caption for the provided image."
        >>> image = Image.open(TEST_IMAGE_PATH)

        >>> inputs = processor(text=text, image=image, mode='U', padding_side="left", padding="longest", return_tensors="np")
        >>> input_ids = Tensor(inputs.input_ids)
        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     pad_token_id=tokenizer.pad_token_id,
        >>>     bos_token_id=tokenizer.bos_token_id,
        >>>     eos_token_id=tokenizer.eos_token_id,
        >>> )

        >>> outputs = model.generate(input_ids, GENERATION_CONFIG, max_new_tokens=100)
        >>> answer = processor.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [ops.dense(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = mint.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:  # training pipeline
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = self.loss_fct(shift_logits, shift_labels)
            return loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.int().cumsum(-1) - 1  # cumsum support int32/int8/uint8 not support int64
            position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
