import math
import torch
import torch.nn.functional as F

MEMORY_LAYOUT = {
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len
    device = text_mask.device

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=device)

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

def chunked_attention(q, k, v, chunk_size=1024):
    """
    Memory-efficient attention computation by processing in chunks
    """
    batch_size, n_heads, seq_len, head_dim = q.shape
    value_len = k.shape[2]
    
    scale = 1 / math.sqrt(head_dim)
    q = q * scale
    
    # Process attention in chunks to save memory
    out = torch.zeros_like(q)
    for i in range(0, seq_len, chunk_size):
        chunk_end = min(i + chunk_size, seq_len)
        
        # Compute attention scores for this chunk
        scores = torch.matmul(q[:, :, i:chunk_end], k.transpose(-2, -1))  # [B, H, chunk, V]
        scores = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        chunk_out = torch.matmul(scores, v)  # [B, H, chunk, D]
        out[:, :, i:chunk_end] = chunk_out
        
    return out

def attention(
    q,
    k,
    v,
    mode="vanilla",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention with memory-efficient implementation.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Only 'torch' and 'vanilla' supported on MPS
        drop_rate (float): Dropout rate in attention map
        attn_mask (torch.Tensor): Attention mask with shape [b, a, s, s1]
        causal (bool): Whether to use causal attention
        cu_seqlens_q (torch.Tensor): Not used on MPS
        cu_seqlens_kv (torch.Tensor): Not used on MPS
        max_seqlen_q (int): Not used on MPS
        max_seqlen_kv (int): Not used on MPS
        batch_size (int): Batch size

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "vanilla":
        # Use chunked attention to save memory
        x = chunked_attention(q, k, v)
        
        if causal:
            # Apply causal mask after attention
            causal_mask = torch.triu(torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=q.device), diagonal=1)
            x = x.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0)
            
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                x = x.masked_fill(~attn_mask, 0)
            else:
                x = x + attn_mask
                
        if drop_rate > 0:
            x = F.dropout(x, p=drop_rate, training=True)
    else:
        raise NotImplementedError(f"Unsupported attention mode on MPS: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

def parallel_attention(
    q,
    k,
    v,
    mode="vanilla",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Memory-efficient parallel attention implementation for MPS.
    Uses the same interface as regular attention for compatibility.
    """
    return attention(
        q, k, v,
        mode=mode,
        drop_rate=drop_rate,
        attn_mask=attn_mask,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        batch_size=batch_size
    )
