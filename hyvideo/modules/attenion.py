def get_cu_seqlens(text_mask, img_seq_len):
    batch_size = text_mask.shape[0]
    device = text_mask.device
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=device)
    for i in range(batch_size):
        cu_seqlens[2 * i + 1] = cu_seqlens[2 * i] + text_mask[i].sum()
        cu_seqlens[2 * i + 2] = cu_seqlens[2 * i + 1] + img_seq_len
    return cu_seqlens
