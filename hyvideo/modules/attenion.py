import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = True,
    needs_weights: bool = False,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Implements the scaled dot product attention with softmax.
    Args:
        query: Query tensor of shape (B, H, L, E)
        key: Key tensor of shape (B, H, S, E)
        value: Value tensor of shape (B, H, S, V)
        mask: Optional mask tensor of shape (L, S)
        causal: If True, applies causal attention masking
        dropout_p: Dropout probability
        training: Whether in training mode
        needs_weights: If True, returns attention weights
        key_padding_mask: Optional mask tensor of shape (B, S)
    Returns:
        output: Attention output tensor
        weights: Optional attention weights if needs_weights is True
    """
    B, H, L, E = query.shape
    _, _, S, D = value.shape
    
    scale = math.sqrt(E)
    
    query = query / scale
    
    # (B, H, L, E) x (B, H, E, S) -> (B, H, L, S)
    attn = torch.matmul(query, key.transpose(-2, -1))
    
    if mask is not None:
        attn = attn + mask
        
    if causal:
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
    if key_padding_mask is not None:
        attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
        
    # (B, H, L, S) x (B, H, S, D) -> (B, H, L, D)
    output = torch.matmul(attn, value)
    
    if needs_weights:
        return output, attn
    return output


def parallel_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    training: bool = True,
    needs_weights: bool = False,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parallel implementation of scaled dot product attention.
    Uses the same interface as the regular attention function.
    """
    B, H, L, E = query.shape
    _, _, S, D = value.shape
    
    scale = math.sqrt(E)
    query = query / scale
    
    # Parallel attention computation
    attn = torch.einsum('bhle,bhse->bhls', query, key)
    
    if mask is not None:
        attn = attn + mask
        
    if causal:
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
    if key_padding_mask is not None:
        attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p)
        
    output = torch.einsum('bhls,bhsd->bhld', attn, value)
    
    if needs_weights:
        return output, attn
    return output


def get_cu_seqlens(text_mask, img_seq_len):
    """
    Compute cumulative sequence lengths for batched attention.
    Args:
        text_mask: Text attention mask of shape (B, L)
        img_seq_len: Length of image sequence
    Returns:
        cu_seqlens: Cumulative sequence lengths tensor
    """
    batch_size = text_mask.shape[0]
    device = text_mask.device
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens[2 * i + 1] = cu_seqlens[2 * i] + text_mask[i].sum()
        cu_seqlens[2 * i + 2] = cu_seqlens[2 * i + 1] + img_seq_len
    return cu_seqlens
