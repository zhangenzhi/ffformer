"""Transformer query decoder with iterative mask prediction."""
import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sources, queries, attn_masks=None):
        """sources: List[(M_i, D)], queries: List[(Q_i, D)], attn_masks: List[(Q_i, M_i)]."""
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            mask = attn_masks[i] if attn_masks is not None else None
            out, _ = self.attn(queries[i], k, v, attn_mask=mask)
            if self.fix:
                out = self.dropout(out)
            out = out + queries[i]
            if self.fix:
                out = self.norm(out)
            outputs.append(out)
        return outputs


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation='gelu'):
        super().__init__()
        act = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), act, nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return [self.norm(self.net(y) + y) for y in x]


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

        # Output heads (out_cls is NOT used in this model variant — commented out in original)
        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def _get_queries(self, instance_queries, batch_size):
        """Combine instance queries with learnable semantic queries, then project.

        Instance queries (Q, in_channels=32) + semantic queries (S, in_channels=32)
        → concat → query_proj → (Q+S, d_model=256)
        """
        # Find device from first non-empty query
        device = None
        for q in instance_queries:
            if len(q) > 0:
                device = q.device
                break

        result = []
        for i in range(batch_size):
            sem_q = self.semantic_queries.weight.to(device)  # (S, in_channels)
            inst_q = instance_queries[i]  # (Q, in_channels)
            if len(inst_q) == 0:
                inst_q = torch.empty(0, sem_q.shape[1], device=device)
            concat = torch.cat([inst_q, sem_q], dim=0)  # (Q+S, in_channels)
            result.append(self.query_proj(concat))  # (Q+S, d_model)
        return result

    def _forward_head(self, queries, mask_feats):
        """Predict masks, scores, and generate attention masks."""
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_q = self.out_norm(queries[i])
            pred_scores.append(self.out_score(norm_q) if self.objectness_flag else None)

            # Mask prediction: dot product between queries and voxel features
            pred_mask = torch.einsum('qd,md->qm', norm_q, mask_feats[i])

            if self.attn_mask:
                amask = (pred_mask.sigmoid() < 0.5).bool()
                # Reset rows that are fully masked (attend to nothing)
                amask[amask.sum(-1) == amask.shape[-1]] = False
                attn_masks.append(amask.detach())

            pred_masks.append(pred_mask)

        return pred_scores, pred_masks, attn_masks if self.attn_mask else None

    def forward(self, x, instance_queries):
        """Forward pass with iterative mask prediction.

        Args:
            x: List[(M_i, C)] backbone features per sample.
            instance_queries: List[(Q_i, C)] query features from ISA/grid selection.

        Returns:
            Dict with 'cls_preds', 'masks', 'scores', 'aux_outputs'.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(instance_queries, len(x))

        # Initial prediction
        scores_all, masks_all = [], []
        pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats)
        scores_all.append(pred_score)
        masks_all.append(pred_mask)

        # Iterative refinement
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask = self._forward_head(queries, mask_feats)
            scores_all.append(pred_score)
            masks_all.append(pred_mask)

        aux_outputs = [
            {'masks': m, 'scores': s}
            for s, m in zip(scores_all[:-1], masks_all[:-1])]

        return dict(
            masks=masks_all[-1],
            scores=scores_all[-1],
            aux_outputs=aux_outputs)
