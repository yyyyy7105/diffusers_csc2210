# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Triton block-sparse attention kernel for the Flux 2 transformer.

This module implements a block-sparse attention forward pass using Triton.
The kernel skips tiles whose block-pair connectivity bit is False, issuing no
memory loads or compute for those tiles (real FLOP reduction).

Usage::

    from diffusers.models.triton_block_sparse_attn import triton_block_sparse_attention

    # query / key / value : [B, H, S, D]  fp16 or bf16, contiguous, on CUDA
    # conn                : [NB, NB]  bool  (NB = S // block_size)
    out = triton_block_sparse_attention(query, key, value, conn, block_size=64)

The kernel follows the Flash-Attention 2 online-softmax tile structure:
  * Outer loop: query tiles (row-blocks), one Triton program per tile.
  * Inner loop: key/value tiles (column-blocks), **guarded by the connectivity
    bit** — inactive tiles are skipped entirely.
  * Numerics: online-softmax accumulators in fp32 for numerical stability;
    QK^T and PV dot products cast to fp32.

Fallback: when Triton or CUDA is unavailable, ``triton_block_sparse_attention``
falls back to a pure-PyTorch masked-softmax implementation that is correct but
does not reduce FLOPs.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _block_sparse_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        Conn_ptr,  # [NB_q, NB_k]  int32 connectivity flags (0 = skip, 1 = compute)
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        stride_cn0,
        stride_cn1,
        NB_K,  # runtime int — loop bound (number of key-block columns)
        BLOCK_SIZE: tl.constexpr,  # tile size in sequence dimension
        HEAD_DIM: tl.constexpr,  # head dimension (must be power-of-two ≤ 256)
        SCALE: tl.constexpr,  # attention scale = 1 / sqrt(HEAD_DIM)
    ):
        """
        One Triton program handles one (batch, head, query-block) triple.

        Input tensors are [B, H, S, D] in *row-major* (BHSD) layout.
        The connectivity map ``Conn`` is [NB_q, NB_k] int32.
        """
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        qb_id = tl.program_id(2)  # which query-block row this program processes

        offs_q = tl.arange(0, BLOCK_SIZE) + qb_id * BLOCK_SIZE  # token indices
        offs_d = tl.arange(0, HEAD_DIM)  # head-dim indices

        # Load the query tile  [BLOCK_SIZE, HEAD_DIM]
        Q_base = batch_id * stride_qb + head_id * stride_qh
        q = tl.load(Q_ptr + Q_base + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd)

        # Online-softmax running accumulators (in fp32 for numerical stability)
        m_i = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)  # row max
        l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)  # normalisation sum
        acc = tl.zeros([BLOCK_SIZE, HEAD_DIM], dtype=tl.float32)  # output accumulator

        K_base = batch_id * stride_kb + head_id * stride_kh
        V_base = batch_id * stride_vb + head_id * stride_vh
        Conn_base = qb_id * stride_cn0

        # Iterate over key-block columns; skip inactive tiles
        for kb_id in range(NB_K):
            conn_flag = tl.load(Conn_ptr + Conn_base + kb_id * stride_cn1)
            if conn_flag != 0:
                offs_k = tl.arange(0, BLOCK_SIZE) + kb_id * BLOCK_SIZE

                # Load K tile as [HEAD_DIM, BLOCK_SIZE] (transposed) for dot product
                k = tl.load(
                    K_ptr + K_base + offs_k[None, :] * stride_ks + offs_d[:, None] * stride_kd
                )  # [HD, BS]

                # Load V tile as [BLOCK_SIZE, HEAD_DIM]
                v = tl.load(
                    V_ptr + V_base + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
                )  # [BS, HD]

                # QK^T  [BLOCK_SIZE, BLOCK_SIZE] — cast to fp32 for stability
                s = tl.dot(q.to(tl.float32), k.to(tl.float32)) * SCALE

                # Online-softmax update
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])  # [BS, BS]
                l_i = alpha * l_i + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v.to(tl.float32))
                m_i = m_new

        # Normalise and write output  [BLOCK_SIZE, HEAD_DIM]
        acc = acc / l_i[:, None]

        O_base = batch_id * stride_ob + head_id * stride_oh
        tl.store(
            Out_ptr + O_base + offs_q[:, None] * stride_os + offs_d[None, :] * stride_od,
            acc.to(q.dtype),  # cast back to the input dtype (fp16 or bf16)
        )


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback (correct, but no FLOP savings)
# ---------------------------------------------------------------------------


def _masked_attention_pytorch(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    conn: Optional[torch.BoolTensor],  # [NB, NB]
    block_size: int = 64,
) -> torch.Tensor:
    """Masked softmax attention — PyTorch reference / CPU fallback."""
    if conn is None:
        return F.scaled_dot_product_attention(query, key, value)

    B, H, S, D = query.shape
    NB = S // block_size
    # Expand connectivity to token-level boolean mask  [S, S]
    # True  = attend, False = mask out
    expanded = conn.unsqueeze(1).unsqueeze(3).expand(NB, block_size, NB, block_size)
    token_mask = expanded.reshape(NB * block_size, NB * block_size)[:S, :S]  # trim any padding
    zeros = torch.zeros(1, device=query.device, dtype=query.dtype)
    neg_inf = torch.full((1,), float("-inf"), device=query.device, dtype=query.dtype)
    attn_mask = torch.where(token_mask, zeros, neg_inf)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def triton_block_sparse_attention(
    query: torch.Tensor,  # [B, H, S, D]  fp16 or bf16, CUDA, contiguous
    key: torch.Tensor,
    value: torch.Tensor,
    conn: torch.BoolTensor,  # [NB, NB]  block-pair connectivity
    block_size: int = 64,
) -> torch.Tensor:
    """Block-sparse attention using the Triton kernel.

    Only block-pairs whose connectivity bit is ``True`` are computed; the rest
    are skipped entirely, reducing both memory traffic and compute.

    Args:
        query:     ``[B, H, S, D]`` query tensor in fp16 or bf16 on CUDA.
        key:       ``[B, H, S, D]`` key tensor, same dtype/device.
        value:     ``[B, H, S, D]`` value tensor, same dtype/device.
        conn:      ``[NB, NB]`` boolean connectivity matrix where
                   ``NB = S // block_size``.  ``conn[r, c] = True`` means the
                   query-block *r* attends to key-block *c*.
        block_size: Number of tokens per block tile.  Must divide ``S``.
                    Defaults to 64.

    Returns:
        Output tensor ``[B, H, S, D]`` in the same dtype as ``query``.

    Note:
        * ``S`` must be divisible by ``block_size``.
        * ``D`` (head dim) must be a power-of-two ≤ 256.
        * Falls back to a masked PyTorch implementation when Triton or CUDA
          is unavailable, or when the shape constraints are not met.
    """
    if not _HAS_TRITON or not query.is_cuda:
        return _masked_attention_pytorch(query, key, value, conn, block_size)

    B, H, S, D = query.shape

    if S % block_size != 0:
        # Sequence length not divisible — fall back gracefully
        return _masked_attention_pytorch(query, key, value, conn, block_size)

    _valid_head_dims = {16, 32, 64, 128, 256}
    if D not in _valid_head_dims:
        return _masked_attention_pytorch(query, key, value, conn, block_size)

    NB = S // block_size

    # Ensure connectivity map has the right shape and lives on GPU as int32
    nb_stored = conn.shape[0]
    if nb_stored != NB:
        # Nearest-neighbour resize of the connectivity map
        idx = torch.linspace(0, nb_stored - 1, NB, dtype=torch.long)
        conn_resized = conn[idx][:, idx]
    else:
        conn_resized = conn
    conn_gpu = conn_resized.to(dtype=torch.int32, device=query.device).contiguous()

    # Ensure inputs are contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    out = torch.empty_like(query)
    scale = 1.0 / math.sqrt(D)

    grid = (B, H, NB)
    _block_sparse_attn_fwd_kernel[grid](
        query,
        key,
        value,
        out,
        conn_gpu,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        conn_gpu.stride(0),
        conn_gpu.stride(1),
        NB_K=NB,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        SCALE=scale,
    )
    return out
