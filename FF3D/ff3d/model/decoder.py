"""Transformer query decoder with iterative mask prediction.

Batched/padded implementation: the per-sample Python loops are replaced by
padded batched attention so K sliding-window regions run in one forward pass
(fills the GPU instead of many tiny batch-1 calls). The public interface is
unchanged — forward(x_list, queries_list) still takes/returns per-sample lists;
padding + masking are internal and numerically equivalent to per-sample calls.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_softmax_eager(scores, mask):
    """scores (B,H,Q,M), mask (B,1,Q,*) bool True=block -> softmax over M."""
    return scores.masked_fill(mask, float('-inf')).softmax(dim=-1)


_masked_softmax_compiled = None


def _ensure_triton_libcuda():
    """Point Triton at the real driver so torch.compile can JIT. This container's
    Triton 2.1 finds libcuda only via `ldconfig -p`, which doesn't list the
    --nv-injected /.singularity.d/libs/libcuda.so.1; monkeypatch its lookup to a
    tmp dir holding a libcuda.so symlink. No-op if Triton/driver aren't present."""
    try:
        import triton.common.build as _tb
        for cand in ('/.singularity.d/libs/libcuda.so.1',
                     '/usr/lib/x86_64-linux-gnu/libcuda.so.1'):
            if os.path.exists(cand):
                d = '/tmp/_ff_libcuda'
                os.makedirs(d, exist_ok=True)
                link = os.path.join(d, 'libcuda.so')
                if not os.path.exists(link):
                    os.symlink(cand, link)
                _tb.libcuda_dirs = lambda: [d]
                return
    except Exception:
        pass


def masked_softmax(scores, mask):
    """Fuse masked_fill + softmax. With FF_FUSED_SOFTMAX, torch.compile lets
    inductor fold the masked_fill into the softmax reduction (one fewer full
    read/write of the (B,H,Q,M) score matrix). dynamic=True → compile once for
    the varying (Q,M) region sizes. Numerically identical to the eager path."""
    global _masked_softmax_compiled
    if os.environ.get('FF_FUSED_SOFTMAX', '0') not in ('0', 'false', 'False'):
        if _masked_softmax_compiled is None:
            _ensure_triton_libcuda()
            _masked_softmax_compiled = torch.compile(_masked_softmax_eager, dynamic=True)
        return _masked_softmax_compiled(scores, mask)
    return _masked_softmax_eager(scores, mask)


def _pad_stack(tensors):
    """List[(L_i, D)] -> (B, Lmax, D), pad_mask (B, Lmax) bool (True=pad), lengths."""
    B = len(tensors)
    lengths = [int(t.shape[0]) for t in tensors]
    Lmax = max(lengths) if lengths else 0
    out = tensors[0].new_zeros((B, Lmax, tensors[0].shape[-1]))
    pad = torch.ones((B, Lmax), dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        L = lengths[i]
        if L:
            out[i, :L] = t
            pad[i, :L] = False
    return out, pad, lengths


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, fix=False):
        super().__init__()
        self.fix = fix
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sources, src_pad, queries, attn_mask=None):
        """sources (B,M,D); src_pad (B,M) True=pad; queries (B,Q,D);
        attn_mask (B,Q,M) bool True=block (source padding already folded in).

        Manual attention reusing the MultiheadAttention module's weights, but the
        mask stays (B,1,Q,M) and broadcasts over heads — avoiding the
        expand->reshape to (B*H,Q,M) that materialised a large contiguous copy
        every layer. Numerically equivalent to nn.MultiheadAttention.
        """
        mha = self.attn
        E = mha.embed_dim
        H = self.num_heads
        Dh = E // H
        B, Q, _ = queries.shape
        M = sources.shape[1]

        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(queries, w[:E], b[:E]).view(B, Q, H, Dh).transpose(1, 2)   # (B,H,Q,Dh)
        k = F.linear(sources, w[E:2 * E], b[E:2 * E]).view(B, M, H, Dh).transpose(1, 2)
        v = F.linear(sources, w[2 * E:], b[2 * E:]).view(B, M, H, Dh).transpose(1, 2)

        if os.environ.get('FF_SDPA', '0') not in ('0', 'false', 'False'):
            # Fused attention: scaled_dot_product_attention computes scores+mask+
            # softmax+AV in one tiled kernel (mem-efficient backend for arbitrary
            # masks) — no full (B,H,Q,M) score/masked_fill/softmax round-trip.
            # SDPA bool mask is True=ATTEND (opposite of our True=block); the
            # (B,1,Q,M) shape broadcasts over heads.
            if attn_mask is not None:
                m = (~attn_mask).unsqueeze(1)              # (B,1,Q,M)
            else:
                m = (~src_pad)[:, None, None, :]           # (B,1,1,M)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)  # (B,H,Q,Dh)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (Dh ** -0.5)  # (B,H,Q,M)
            m = attn_mask.unsqueeze(1) if attn_mask is not None else src_pad[:, None, None, :]
            out = torch.matmul(masked_softmax(scores, m), v)  # (B,H,Q,Dh)
        out = out.transpose(1, 2).reshape(B, Q, E)
        out = mha.out_proj(out)                        # (B,Q,E)

        if self.fix:
            out = self.dropout(out)
        out = out + queries
        if self.fix:
            out = self.norm(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, qry_pad):
        """x (B,Q,D); qry_pad (B,Q) True=pad (masked as keys)."""
        z, _ = self.attn(x, x, x, key_padding_mask=qry_pad)
        z = self.dropout(z) + x
        z = self.norm(z)
        return z


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation='gelu'):
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), act, nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x (B,Q,D)."""
        return self.norm(self.net(x) + x)


class QueryDecoder(nn.Module):
    """Transformer decoder with iterative mask-guided cross-attention.

    Args:
        num_layers: Number of transformer layers (default 6).
        num_semantic_queries: Learnable semantic queries (default 3).
        in_channels: Backbone output channels (default 32).
        d_model: Hidden dimension (default 256).
        num_heads: Attention heads (default 8).
        hidden_dim: FFN hidden dimension (default 1024).
    """

    def __init__(self, num_layers=6, num_semantic_queries=3, in_channels=32,
                 d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0,
                 activation='gelu', fix_attention=True, objectness_flag=True,
                 attn_mask=True, **kwargs):
        super().__init__()
        self.num_semantic_queries = num_semantic_queries
        self.objectness_flag = objectness_flag
        self.attn_mask = attn_mask

        # Projections
        self.input_proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.x_mask = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.query_proj = nn.Sequential(nn.Linear(in_channels, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        # Learnable semantic queries — in in_channels space (32), projected via query_proj
        self.num_semantic_queries = num_semantic_queries
        if num_semantic_queries > 0:
            self.semantic_queries = nn.Embedding(num_semantic_queries, in_channels)

        # Transformer layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout, fix_attention) for _ in range(num_layers)])
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([
            FFN(d_model, hidden_dim, dropout, activation) for _ in range(num_layers)])

        # Output heads (out_cls is NOT used in this model variant)
        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def _get_queries(self, instance_queries, batch_size):
        """Combine instance queries with learnable semantic queries, then project.

        Instance queries (Q, in_channels) + semantic queries (S, in_channels)
        → concat → query_proj → (Q+S, d_model). Returns a per-sample list.
        """
        device = None
        for q in instance_queries:
            if len(q) > 0:
                device = q.device
                break
        if device is None:
            device = self.semantic_queries.weight.device

        result = []
        for i in range(batch_size):
            sem_q = self.semantic_queries.weight.to(device)  # (S, in_channels)
            inst_q = instance_queries[i]
            if len(inst_q) == 0:
                inst_q = torch.empty(0, sem_q.shape[1], device=device)
            concat = torch.cat([inst_q, sem_q], dim=0)  # (Q+S, in_channels)
            result.append(self.query_proj(concat))       # (Q+S, d_model)
        return result

    def _forward_head(self, queries, mask_feats, src_pad, qry_pad):
        """Batched mask/score/attn-mask prediction.

        queries (B,Q,D), mask_feats (B,M,D), src_pad (B,M), qry_pad (B,Q).
        Returns pred_scores (B,Q,1) | None, pred_masks (B,Q,M), attn_masks (B,Q,M) | None.
        """
        norm_q = self.out_norm(queries)                              # (B,Q,D)
        pred_scores = self.out_score(norm_q) if self.objectness_flag else None
        pred_masks = torch.einsum('bqd,bmd->bqm', norm_q, mask_feats)  # (B,Q,M)

        attn_masks = None
        if self.attn_mask:
            amask = (pred_masks.sigmoid() < 0.5)                     # True=block
            amask = amask | src_pad.unsqueeze(1)                     # never attend padded sources
            real = (~src_pad)                                        # (B,M) real keys
            n_real = real.sum(-1, keepdim=True)                     # (B,1)
            blocked_real = (amask & real.unsqueeze(1)).sum(-1)       # (B,Q)
            full = blocked_real >= n_real                           # rows blocking all real keys
            amask = amask & ~(full.unsqueeze(-1) & real.unsqueeze(1))  # unblock real for those
            amask = amask & ~qry_pad.unsqueeze(-1)                   # padded queries attend all (no NaN)
            attn_masks = amask.detach()

        return pred_scores, pred_masks, attn_masks

    def forward(self, x, instance_queries):
        """Forward pass with iterative mask prediction.

        Args:
            x: List[(M_i, C)] backbone features per sample.
            instance_queries: List[(Q_i, C)] query features per sample.

        Returns:
            Dict with 'masks' (per-sample list of (Q_i+S, M_i)), 'scores'
            (per-sample list of (Q_i+S, 1)), and 'aux_outputs'.
        """
        B = len(x)
        inst_feats_l = [self.input_proj(y) for y in x]
        mask_feats_l = [self.x_mask(y) for y in x]
        queries_l = self._get_queries(instance_queries, B)

        inst_feats, src_pad, src_len = _pad_stack(inst_feats_l)   # (B,M,D)
        mask_feats, _, _ = _pad_stack(mask_feats_l)               # (B,M,D)
        queries, qry_pad, qry_len = _pad_stack(queries_l)         # (B,Q,D)

        def _unpad(masks_bt, scores_bt):
            m = [masks_bt[i, :qry_len[i], :src_len[i]] for i in range(B)]
            if scores_bt is None:
                s = [None] * B
            else:
                s = [scores_bt[i, :qry_len[i]] for i in range(B)]
            return m, s

        scores_all, masks_all = [], []
        pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats, src_pad, qry_pad)
        scores_all.append(pred_score)
        masks_all.append(pred_mask)

        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, src_pad, queries, attn_mask)
            queries = self.self_attn_layers[i](queries, qry_pad)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats, src_pad, qry_pad)
            scores_all.append(pred_score)
            masks_all.append(pred_mask)

        final_masks, final_scores = _unpad(masks_all[-1], scores_all[-1])
        aux_outputs = []
        for j in range(len(masks_all) - 1):
            m, s = _unpad(masks_all[j], scores_all[j])
            aux_outputs.append({'masks': m, 'scores': s})

        return dict(masks=final_masks, scores=final_scores, aux_outputs=aux_outputs)
