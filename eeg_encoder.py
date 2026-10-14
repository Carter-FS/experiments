import torch
from typing import Literal
import torch.nn as nn
import numpy as np
import sys, os


def sinusoidal_positional_encoding(seq_len: int, dim: int, device=None):
    """
    Returns:
        pe: (seq_len, dim)
    """
    device = device or torch.device("cpu")
    pe = torch.zeros(seq_len, dim, device=device)

    position = torch.arange(0, seq_len, device=device).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-np.log(10000.0) / dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def fixed_k_subsample(x: torch.Tensor,
                      pad_mask: torch.Tensor,
                      k: int,
                      mode: str = "random",
                      generator: torch.Generator | None = None):
    """
    Fixed-K subsampling for variable-length sequences.

    Args
    ----
    x:        [B, T, D]
    pad_mask: [B, T]  True = PAD, False = valid
    k:        number of valid tokens to keep per sample
    mode:     "random" (default) or "uniform" (evenly spaced)
    generator: optional torch.Generator for reproducibility

    Returns
    -------
    x_k:        [B, K, D]
    pad_mask_k: [B, K]  (should be all False unless a sample had <K valid tokens)
    idx:        [B, K]  indices selected from original T
    """
    assert x.dim() == 3
    B, T, D = x.shape
    device = x.device

    valid = ~pad_mask  # True for valid
    lengths = valid.sum(dim=1)  # [B]

    idx_out = torch.empty((B, k), dtype=torch.long, device=device)
    pad_out = torch.zeros((B, k), dtype=torch.bool, device=device)  # False by default

    for b in range(B):
        vb = torch.nonzero(valid[b], as_tuple=False).squeeze(1)  # valid indices in [0..T-1]
        L = vb.numel()
        if L == 0:
            # degenerate: no valid tokens -> return zeros + all pad
            idx_out[b] = 0
            pad_out[b] = True
            continue

        if L >= k:
            if mode == "random":
                perm = torch.randperm(L, device=device, generator=generator)[:k]
                sel = vb[perm]
            elif mode == "uniform":
                # evenly spaced selection across the valid indices
                pos = torch.linspace(0, L - 1, steps=k, device=device)
                sel = vb[pos.round().long().clamp(0, L - 1)]
            else:
                raise ValueError("mode must be 'random' or 'uniform'")
        else:
            # Not enough valid tokens: sample with replacement to reach K
            if mode == "random":
                rep = torch.randint(0, L, (k,), device=device, generator=generator)
                sel = vb[rep]
            elif mode == "uniform":
                pos = torch.linspace(0, L - 1, steps=k, device=device)
                sel = vb[pos.round().long().clamp(0, L - 1)]
            # (optional) if you prefer to mark the extra as PAD instead of replacement:
            # pad_out[b, L:] = True

        idx_out[b] = sel

    x_k = x.gather(dim=1, index=idx_out.unsqueeze(-1).expand(-1, -1, D))
    pad_mask_k = pad_out  # usually all False (or True if L==0 or you chose PAD strategy)
    return x_k
    

class LearnedQueryPooling(nn.Module):
    def __init__(self, dim, out_dim=512, num_queries=4, num_heads=4,):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True,  # <--- IMPORTANT: cleaner interface
        )
        self.attn_norm = nn.RMSNorm(dim)
        self.out_linear = nn.Sequential(
            nn.Linear(dim, dim * 2), 
            nn.GELU(),
            nn.Linear(dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, D)
        key_padding_mask: (B, T) bool, True = ignore token
        return: (B, out_dim)
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(
                    f"key_padding_mask must be shape (B,T)=({B},{T}), "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)

        # Expand learned queries
        q = self.query.to(device=device, dtype=dtype)[None, :, :].expand(
            B, self.num_queries, D
        )  # (B, Q, D)

        # Cross-attention: queries attend over sequence x
        out, _ = self.attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        out = self.attn_norm(out)                 # (B, Q, D)
        pooled = self.out_linear(out.mean(dim=1))  # (B, out_dim)

        return pooled


class EEGEmbedder(torch.nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super(EEGEmbedder, self).__init__()
        self.__dict__.update(kwargs)

        kwargs.pop("out_dim")

        self.channel_drop = nn.Dropout1d(0.1)

        self.embedder = nn.Linear(self.d_model, self.d_model)

        self.enc =  nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_model * 2,
                dropout=self.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            )
        self.model = nn.TransformerEncoder(self.enc, num_layers=self.num_layers, enable_nested_tensor=False)
        self.post_norm = nn.LayerNorm(self.d_model)
        self.learned_query = LearnedQueryPooling(self.d_model, out_dim=self.out_dim, num_queries=2, num_heads=self.nhead)
        
        self.missing_pooled_embedding = nn.Parameter(torch.randn(self.out_dim) * 0.02)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape        
        
        if key_padding_mask is None:
            key_padding_mask = torch.zeros((B, T), device=x.device, dtype=torch.bool)
        else:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)

        # Missing iff all tokens are masked for that sample
        is_missing = key_padding_mask.all(dim=1)  # (B,)
        missing = self.missing_pooled_embedding.to(device=x.device, dtype=x.dtype)[None, :]

        global_emb = missing.expand(B, -1).clone()
        keep = ~is_missing
        if keep.any():
            xk = x[keep]
            xk = self.embedder(xk)
            
            positional_encoding = sinusoidal_positional_encoding(xk.shape[1], xk.shape[2])
            xk = xk + positional_encoding.unsqueeze(0).to(xk.device)
    
            xk = torch.swapaxes(xk, 1, 2)
            xk = self.channel_drop(xk)
            xk = torch.swapaxes(xk, 1, 2)
            
            outs = self.model(xk, src_key_padding_mask=key_padding_mask[keep])
            outs = self.post_norm(outs)
    
            if self.training:
                mode = "random"
            else:
                mode = "uniform"
                
            outs = fixed_k_subsample(outs, pad_mask=key_padding_mask,
                          k=32,
                          mode=mode
            )
                                     
            _global_emb = self.learned_query(outs, key_padding_mask=None)

            global_emb[keep] = _global_emb
            
        return global_emb
