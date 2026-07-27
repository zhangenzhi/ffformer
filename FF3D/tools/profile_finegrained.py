"""Fine-grained inference profiler — instrumentation lives HERE, not in the model.

Monkeypatches record_function-wrapped copies of the hot methods onto the model
classes at runtime, so ff3d/model/*.py stays clean. The patched bodies mirror
the committed source (broadcast-mask attention) — keep in sync if the model
changes. Reports per-sub-step CUDA time:

  9a qkv_proj | 9b attn_core | 9c out_proj | 9_self_attn | 9_ffn
  9d mask_logits | 9e sigmoid_thresh
  10_voxel2point | 10_sigmoid | 10_nms | 10_zfilter  + CPU/GPU sync

Usage: python FF3D/tools/profile_finegrained.py --input area1.las --chunk 20 [--fp16-decoder]
"""
import argparse, os, sys, time
import numpy as np, torch, laspy
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function as rf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ff3d.model import ForestFormer3D
from ff3d.model.forestformer3d import load_pretrained
import ff3d.model.forestformer3d as FM
import ff3d.model.decoder as DEC
from ff3d.utils.nms import mask_matrix_nms


# ---- instrumented copies (mirror committed source) ----
def cross_forward(self, sources, src_pad, queries, attn_mask=None):
    mha = self.attn; E = mha.embed_dim; H = self.num_heads; Dh = E // H
    B, Q, _ = queries.shape; M = sources.shape[1]
    with rf("9a_qkv_proj"):
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(queries, w[:E], b[:E]); k = F.linear(sources, w[E:2*E], b[E:2*E]); v = F.linear(sources, w[2*E:], b[2*E:])
        q = q.view(B, Q, H, Dh).transpose(1, 2); k = k.view(B, M, H, Dh).transpose(1, 2); v = v.view(B, M, H, Dh).transpose(1, 2)
    with rf("9b_attn_core"):
        scores = torch.matmul(q, k.transpose(-2, -1)) * (Dh ** -0.5)
        scores = scores.masked_fill(attn_mask.unsqueeze(1) if attn_mask is not None else src_pad[:, None, None, :], float('-inf'))
        out = torch.matmul(scores.softmax(dim=-1), v).transpose(1, 2).reshape(B, Q, E)
    with rf("9c_out_proj"):
        out = mha.out_proj(out)
    if self.fix: out = self.dropout(out)
    out = out + queries
    if self.fix: out = self.norm(out)
    return out


def head(self, queries, mask_feats, src_pad, qry_pad):
    with rf("9d_mask_logits"):
        norm_q = self.out_norm(queries)
        pred_scores = self.out_score(norm_q) if self.objectness_flag else None
        pred_masks = torch.einsum('bqd,bmd->bqm', norm_q, mask_feats)
    attn_masks = None
    if self.attn_mask:
        with rf("9e_sigmoid_thresh"):
            amask = (pred_masks.sigmoid() < 0.5) | src_pad.unsqueeze(1)
            real = ~src_pad; n_real = real.sum(-1, keepdim=True)
            full = (amask & real.unsqueeze(1)).sum(-1) >= n_real
            amask = amask & ~(full.unsqueeze(-1) & real.unsqueeze(1)) & ~qry_pad.unsqueeze(-1)
            attn_masks = amask.detach()
    return pred_scores, pred_masks, attn_masks


def pbf(self, out, superpoints, coordinates, query_indices, idx=0):
    pred_masks = out['masks'][idx]; pred_scores = out['scores'][idx]; n_sem = self.num_classes
    with rf("10_voxel2point"):
        sem_res = pred_masks[-n_sem:, :].sigmoid()[:, superpoints].argmax(0)
    inst_masks = pred_masks[:-n_sem, :]; inst_scores = pred_scores[:-n_sem, :]
    labels = torch.zeros(inst_scores.shape[0], dtype=torch.long, device=inst_scores.device)
    scores = inst_scores.squeeze(-1)
    if scores.numel() > 300:
        scores, ti = scores.topk(300, sorted=False); labels = labels[ti]; inst_masks = inst_masks[ti]
    with rf("10_sigmoid"):
        mask_sigmoid = inst_masks.sigmoid()
    if mask_sigmoid.shape[0] > 0:
        thr = mask_sigmoid > 0.01
        scores = scores * ((mask_sigmoid * thr).sum(1) / (thr.sum(1) + 1e-6))
        with rf("10_nms"):
            scores, labels, mask_sigmoid, _ = mask_matrix_nms(mask_sigmoid, labels, scores, kernel='linear')
    with rf("10_voxel2point"):
        mask_pred = (mask_sigmoid > 0.15)[:, superpoints]
    is_stuff = (sem_res == 0).float()
    scores[(mask_pred.float() @ is_stuff) / mask_pred.sum(1).clamp(min=1) > 0.5] = 0
    with rf("10_zfilter"):
        gp = coordinates[sem_res == 0]
        gzmax = gp[:, 2].max().item() if gp.size(0) > 0 else float('inf')
        if mask_pred.shape[0] > 0:
            z = coordinates[:, 2]
            min_z = torch.where(mask_pred, z.unsqueeze(0), z.new_full((1,), float('inf'))).min(dim=1).values
            scores.masked_fill_((mask_pred.sum(1) == 0) | (min_z > gzmax + 5), 0)
    keep = (scores > 0) & (mask_pred.sum(1) > 10)
    return mask_pred[keep], scores[keep], sem_res


def wrap(cls, name, label):
    orig = getattr(cls, name)
    def w(self, *a, **k):
        with rf(label):
            return orig(self, *a, **k)
    setattr(cls, name, w)


def patch():
    DEC.CrossAttentionLayer.forward = cross_forward
    DEC.QueryDecoder._forward_head = head
    FM.ForestFormer3D._predict_by_feat = pbf
    wrap(DEC.SelfAttentionLayer, 'forward', '9_self_attn')
    wrap(DEC.FFN, 'forward', '9_ffn')


def load_points(path, half):
    las = laspy.read(path)
    x = np.asarray(las.x, np.float64); y = np.asarray(las.y, np.float64); z = np.asarray(las.z, np.float64)
    if half:
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        m = (np.abs(x-cx) <= half) & (np.abs(y-cy) <= half); x, y, z = x[m], y[m], z[m]
    xyz = np.column_stack([x, y, z]).astype(np.float64)
    xyz -= [xyz[:,0].mean(), xyz[:,1].mean(), xyz[:,2].min()]
    return torch.from_numpy(xyz.astype(np.float32)).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--checkpoint', default='work_dirs/clean_forestformer/epoch_3000_fix.pth')
    ap.add_argument('--chunk', type=float, default=None)
    ap.add_argument('--fp16-decoder', action='store_true')
    args = ap.parse_args()
    if args.fp16_decoder: os.environ['FF_FP16_DECODER'] = '1'
    patch()
    pts = load_points(args.input, args.chunk)
    print(f'{pts.shape[0]:,} pts, fp16_decoder={args.fp16_decoder}', flush=True)
    model = ForestFormer3D(decoder_cfg=dict(num_layers=6, num_semantic_queries=3, in_channels=32,
        d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0, activation='gelu',
        fix_attention=True, objectness_flag=True, attn_mask=True))
    load_pretrained(model, args.checkpoint); model = model.cuda().eval()
    with torch.no_grad(): model.predict(pts)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad(): out = model.predict(pts)
        torch.cuda.synchronize()
    ins = out['instance_pred']
    print(f'trees {len(np.unique(ins[ins>=0]))}\n', flush=True)
    labels = ['9a_qkv_proj','9b_attn_core','9c_out_proj','9_self_attn','9_ffn',
              '9d_mask_logits','9e_sigmoid_thresh','10_voxel2point','10_sigmoid','10_nms','10_zfilter']
    vals = {e.key: e.cuda_time_total/1000.0 for e in prof.key_averages() if e.key in labels}
    sync = sum(e.self_cuda_time_total for e in prof.key_averages()
               if any(s in e.key for s in ['Memcpy DtoH','cudaStreamSynchronize','cudaMemcpyAsync']))/1000.0
    print('===== FINE-GRAINED (CUDA total ms) =====', flush=True)
    for L in labels:
        print(f'  {L:<20}{vals.get(L,0):>9.1f} ms', flush=True)
    print(f'  {"CPU/GPU sync":<20}{sync:>9.1f} ms', flush=True)


if __name__ == '__main__':
    main()
