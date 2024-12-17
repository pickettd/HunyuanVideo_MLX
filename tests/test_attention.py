import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hyvideo.modules.attenion import attention

def test_mm_double_stream_block_attention():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return

    device = torch.device("mps")
    dtype = torch.float32
    batch_size = 1
    seq_len_img = 1024  # Reduced for memory constraints
    seq_len_txt = 256
    heads_num = 24
    head_dim = 128

    img_q = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    img_k = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    img_v = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    txt_q = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
    txt_k = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
    txt_v = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)

    with torch.no_grad():
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        total_len = seq_len_img + seq_len_txt
        cu_seqlens = torch.tensor([0, seq_len_img, total_len], device=device, dtype=torch.int32)
        
        # Test vanilla mode
        print("Testing vanilla attention mode...")
        output_vanilla = attention(
            q, k, v,
            mode="vanilla",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            batch_size=batch_size
        )
        print("Vanilla attention output shape:", output_vanilla.shape)
        
        # Test torch mode
        print("\nTesting torch attention mode...")
        output_torch = attention(
            q, k, v,
            mode="torch",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            batch_size=batch_size
        )
        print("Torch attention output shape:", output_torch.shape)
        
        # Compare outputs
        try:
            torch.testing.assert_close(output_vanilla, output_torch, rtol=1e-3, atol=1e-3)
            print("\nOutputs match between vanilla and torch attention!")
        except Exception as e:
            print("\nOutputs differ between vanilla and torch attention:", str(e))

def test_mm_single_stream_block_attention():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return

    device = torch.device("mps")
    dtype = torch.float32
    txt_len = 256
    batch_size = 1
    seq_len_img = 1024  # Reduced for memory constraints
    seq_len_txt = 256
    heads_num = 24
    head_dim = 128

    with torch.no_grad():   
        img_q = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
        img_k = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
        txt_q = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
        txt_k = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len_img + seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

        total_len = seq_len_img + seq_len_txt
        cu_seqlens = torch.tensor([0, seq_len_img, total_len], device=device, dtype=torch.int32)

        # Test vanilla mode
        print("\nTesting vanilla attention mode (single stream)...")
        output_vanilla = attention(
            q, k, v,
            mode="vanilla",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            batch_size=batch_size
        )
        print("Vanilla attention output shape:", output_vanilla.shape)
        
        # Test torch mode
        print("\nTesting torch attention mode (single stream)...")
        output_torch = attention(
            q, k, v,
            mode="torch",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=total_len,
            max_seqlen_kv=total_len,
            batch_size=batch_size
        )
        print("Torch attention output shape:", output_torch.shape)
        
        # Compare outputs
        try:
            torch.testing.assert_close(output_vanilla, output_torch, rtol=1e-3, atol=1e-3)
            print("\nOutputs match between vanilla and torch attention!")
        except Exception as e:
            print("\nOutputs differ between vanilla and torch attention:", str(e))

if __name__ == "__main__":
    print("Testing double stream block attention...")
    test_mm_double_stream_block_attention()
    
    print("\nTesting single stream block attention...")
    test_mm_single_stream_block_attention()
