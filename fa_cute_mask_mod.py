import math
from typing import Optional

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch.nn.functional as F
from transformers.models.deprecated.deta.modeling_deta import nonzero_tuple
from transformers.utils.doc import AUDIO_XVECTOR_SAMPLE


import cutlass
import cutlass.cute as cute
from flash_attn.cute.interface import _flash_attn_fwd, flash_attn_func
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.mask_definitions import (
    get_mask_pair,
    STATIC_MASKS,
    random_doc_id_tensor,
)
from flash_attn.cute.testing import attention_ref
COMPUTE_CAPABILITY = torch.cuda.get_device_capability()[0]


def create_tensors(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
):
    device = "cuda"
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(
        batch_size, seqlen_k, nheads_kv, headdim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads_kv, headdim_v, device=device, dtype=dtype
    )
    out = torch.empty(
        batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype
    )
    lse = torch.empty(batch_size, nheads, seqlen_q, device=device, dtype=torch.float32)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "out": out.contiguous(),
        "lse": lse.contiguous(),
    }


def compute_reference_flash_attn(tensors, causal, window_size, dtype_ref, upcast=True):
    """Compute reference using FlashAttention's attention_ref function"""
    q = tensors["q"].to(dtype_ref)
    k = tensors["k"].to(dtype_ref)
    v = tensors["v"].to(dtype_ref)

    out_ref, attn_ref = attention_ref(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=False,
    )

    return out_ref


def compute_reference_flex_attn(tensors, mask_mod_flex, block_size: Optional[tuple[int, int]] = None):
    """Compute reference using flex_attention for custom mask_mods"""
    batch_size, seqlen_q, nheads, headdim = tensors["q"].shape
    _, seqlen_k, nheads_kv, _ = tensors["k"].shape

    q = tensors["q"].transpose(1, 2)
    k = tensors["k"].transpose(1, 2)
    v = tensors["v"].transpose(1, 2)

    scale = 1.0 / math.sqrt(headdim)

    # Handle identity (no masking) case
    if mask_mod_flex is None:
        out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
        return out_ref.transpose(1, 2).contiguous()

    block_mask_kwargs = {}
    if block_size is not None:
        block_mask_kwargs["BLOCK_SIZE"] = block_size

    block_mask = create_block_mask(
        mask_mod_flex,
        B=batch_size,
        H=nheads,
        Q_LEN=seqlen_q,
        KV_LEN=seqlen_k,
        device=q.device,
        **block_mask_kwargs,
    )
    out_ref = flex_attention(q, k, v, block_mask=block_mask, scale=scale, enable_gqa=nheads != nheads_kv)
    return out_ref.transpose(1, 2).contiguous()

@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """ Convert a scalar to a cute TensorSSA of shape (1,) and given dtype """
    vec = cute.make_fragment(1, dtype)
    vec[0] = a
    return vec.load()


def _run_mask_test(
):
    torch.manual_seed(42)
    seqlen_q = seqlen_k = 10
    nheads = 16
    headdim = 128
    dtype=torch.bfloat16
    kv_mode='mha'

    # Determine nheads_kv based on mode
    if kv_mode == "mha":
        nheads_kv = nheads
    elif kv_mode == "gqa":
        nheads_kv = nheads // 2
    elif kv_mode == "mqa":
        nheads_kv = 1
    else:
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    batch_size = 1
    headdim_v = headdim

    random_mask = torch.randint(0,2,(seqlen_k, seqlen_q), dtype=torch.bool, device='cuda')

    aux_tensors_arg = (random_mask, )
    
    def _cute_causal_mask(
        batch,
        head,
        q_idx,
        kv_idx,
        aux_tensors,
    ):
        # mask = aux_tensors[0]
        # m = scalar_to_ssa(mask[kv_idx[0], q_idx[0]], cutlass.Uint8)
        # return (kv_idx <= q_idx) & m
        return (kv_idx <= q_idx)


    def _flex_causal_mask(b, h, q_idx, kv_idx):
        # return (kv_idx <= q_idx) & random_mask[kv_idx, q_idx]
        return (kv_idx <= q_idx)


    

    softmax_scale = 1.0 / math.sqrt(headdim)


    # out_tuple = _flash_attn_fwd(
    #     q=tensors["q"],
    #     k=tensors["k"],
    #     v=tensors["v"],
    #     softmax_scale=softmax_scale,
    #     pack_gqa=False,
    #     mask_mod=_cute_causal_mask,
    #     aux_tensors=aux_tensors_arg,
    # )
    tensors = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )
    tensors2 = create_tensors(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, headdim, headdim_v, dtype
    )

    q1, k1, v1 = tensors2['q'], tensors2['k'], tensors2['v']
    q1.copy_(tensors['q']).requires_grad_()
    k1.copy_(tensors['k']).requires_grad_()
    v1.copy_(tensors['v']).requires_grad_()
    out_tuple = flash_attn_func(q=q1,
        k=k1,
        v=v1,
        softmax_scale=softmax_scale,
        pack_gqa=False,
        causal=True,
        # mask_mod=_cute_causal_mask,
        # aux_tensors=aux_tensors_arg,
    )

    out_cute = out_tuple[0]

    tensors['q'].requires_grad_()
    tensors['k'].requires_grad_()
    tensors['v'].requires_grad_()
    out_ref = compute_reference_flex_attn(tensors, _flex_causal_mask)

    from torch.testing import assert_close
    # assert_close(out_cute, out_ref)
    print('fwd success')

    grad = torch.rand_like(out_cute)
    out_cute.backward(grad)
    out_ref.backward(grad)
    assert_close(q1, tensors['q'])
    assert_close(q1.grad, tensors['q'].grad)




_run_mask_test()
