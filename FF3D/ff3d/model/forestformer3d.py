"""ForestFormer3D: main model class.

Pure PyTorch implementation — no mm* dependencies.
Handles both training (loss) and inference (predict) paths.
"""
import os
import time

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn

from .backbone import SpConvUNet
from .decoder import QueryDecoder
from ..utils.grid_utils import grid_sample, grid_subsample_queries
from ..utils.nms import mask_matrix_nms
from ..utils.ply_io import save_ply_predictions


class _BNWrapper(nn.Module):
    """Wraps BatchNorm1d with a 'batch_norm' submodule name to match checkpoint keys."""
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1)

    def forward(self, x):
        return self.batch_norm(x)


class ForestFormer3D(nn.Module):
    """ForestFormer3D for forest point cloud instance + semantic segmentation.

    Args:
        in_channels: Input feature dim (3 for xyz-only).
        num_channels: Backbone output channels (default 32).
        voxel_size: Voxelization resolution in meters (default 0.2).
        num_classes: Number of semantic classes (default 3: ground/wood/leaf).
        min_spatial_shape: Minimum spatial shape for SpConv (default 128).
        query_point_num: Max number of instance queries (default 300).
        radius: Cylinder radius for inference (default 16).
        score_th: Score threshold for instance filtering (default 0.4).
        prepare_epoch: Epochs before activating decoder (default 1000).
        decoder_cfg: Dict of decoder kwargs.
    """

    def __init__(self, in_channels=3, num_channels=32, voxel_size=0.2,
                 num_classes=3, min_spatial_shape=128, query_point_num=300,
                 radius=16, score_th=0.4, chunk=20000, prepare_epoch=1000,
                 decoder_cfg=None):
        super().__init__()
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_point_num = query_point_num
        self.radius = radius
        self.score_th = score_th
        self.chunk = chunk
        self.prepare_epoch = prepare_epoch

        # Backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, num_channels, kernel_size=3, padding=1,
                              bias=False, indice_key='subm1'))
        self.unet = SpConvUNet(
            num_planes=[num_channels * (i + 1) for i in range(5)],
            return_blocks=True)
        self.output_layer = spconv.SparseSequential(
            nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1), nn.ReLU(inplace=True))

        # Auxiliary heads — structure must match checkpoint key layout:
        # Embed.0.0.0 = Linear, Embed.0.0.1.batch_norm = BN, Embed.1 = Linear
        # (matches torch_points3d Seq(MLP([C,C])).append(Linear(C,5)))
        self.Embed = nn.Sequential(
            nn.Sequential(nn.Sequential(nn.Linear(num_channels, num_channels, bias=False),
                                        _BNWrapper(num_channels)), nn.ReLU()),
            nn.Linear(num_channels, 5))
        self.BiSemantic = nn.Sequential(
            nn.Sequential(nn.Sequential(nn.Linear(num_channels, num_channels, bias=False),
                                        _BNWrapper(num_channels)), nn.ReLU()),
            nn.Linear(num_channels, 2),
            nn.LogSoftmax(dim=-1))

        # Decoder
        decoder_cfg = decoder_cfg or {}
        decoder_cfg.setdefault('in_channels', num_channels)
        self.decoder = QueryDecoder(**decoder_cfg)

    # ──────────────────── Backbone ────────────────────

    def extract_feat(self, x_sparse):
        """Run sparse U-Net backbone. Returns List[(M_i, C)] per sample."""
        x = self.input_conv(x_sparse)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        # Split by batch index
        return [x.features[x.indices[:, 0] == i] for i in x.indices[:, 0].unique()]

    def collate(self, points_list):
        """Convert list of point tensors to SpConv sparse tensor.

        Pure PyTorch implementation — no MinkowskiEngine dependency.
        Handles batching, voxelization (floor + unique), and feature averaging.

        Args:
            points_list: List of (N_i, 3) tensors (GPU or CPU).

        Returns:
            coordinates: (M, 4) int32 — [batch_idx, x, y, z] voxel coordinates.
            features: (M, 3) float32 — mean-centered xyz features per voxel.
            inverse_mapping: (N_total,) int64 — maps each input point to its voxel.
            spatial_shape: (3,) int — max voxel extent.
        """
        all_coords = []   # (N_i, 4) with batch_idx
        all_feats = []    # (N_i, 3) centered xyz
        offsets = [0]      # cumulative point count per sample

        for batch_idx, p in enumerate(points_list):
            # Normalize to voxel coordinates
            coords = torch.floor((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size).int()
            # Add batch index as first column
            batch_col = torch.full((len(p), 1), batch_idx, dtype=torch.int32, device=p.device)
            all_coords.append(torch.cat([batch_col, coords], dim=1))
            # Features = centered xyz
            all_feats.append(p[:, :3] - p[:, :3].mean(0))
            offsets.append(offsets[-1] + len(p))

        coords_cat = torch.cat(all_coords, dim=0)  # (N_total, 4)
        feats_cat = torch.cat(all_feats, dim=0)     # (N_total, 3)

        # Voxelization: unique voxels, averaging features, inverse mapping
        uniq_coords, inverse = torch.unique(coords_cat, dim=0, return_inverse=True)
        M = uniq_coords.shape[0]

        # Average features per voxel
        feat_sum = torch.zeros(M, feats_cat.shape[1], device=feats_cat.device, dtype=feats_cat.dtype)
        feat_cnt = torch.zeros(M, device=feats_cat.device, dtype=feats_cat.dtype)
        feat_sum.index_add_(0, inverse, feats_cat)
        feat_cnt.index_add_(0, inverse, torch.ones(len(feats_cat), device=feats_cat.device))
        features = feat_sum / feat_cnt.unsqueeze(1)

        spatial_shape = torch.clip(uniq_coords[:, 1:].max(0)[0] + 1, self.min_spatial_shape)

        return uniq_coords, features, inverse, spatial_shape

    # ──────────────────── Inference ────────────────────

    @torch.no_grad()
    def predict(self, points, output_path=None, pts_semantic_gt=None, pts_instance_gt=None,
                progress_cb=None, use_amp=True):
        """Full-scene inference with sliding window.

        Args:
            points: (N, 3) tensor on GPU — full scene point cloud.
            output_path: Optional path to save PLY result.
            pts_semantic_gt: Optional (N,) GT semantic labels for evaluation.
            pts_instance_gt: Optional (N,) GT instance labels for evaluation.

        Returns:
            dict with 'semantic_pred', 'instance_pred', 'instance_scores'.
        """
        self.eval()
        device = points.device
        N = points.shape[0]

        step_size = self.radius / 4
        grid_size = 0.2
        num_points = 640000
        score_th1 = self.score_th
        score_th2 = 0.3

        # Generate sliding window regions
        regions = self._generate_regions(points, self.radius, step_size)

        # Global accumulators
        votes_counter = np.zeros((N, self.num_classes), dtype=np.int16)
        all_pre_ins = np.full(N, -1, dtype=np.int64)
        global_instance_scores = np.zeros(N, dtype=np.float64)
        best_masks = []
        max_instance = 0

        from tqdm import tqdm
        for region_idx, region in enumerate(tqdm(regions, desc="Inference")):
            # ① Cylinder crop
            region_mask = ((points[:, 0] - region[0]) ** 2 +
                           (points[:, 1] - region[1]) ** 2) <= self.radius ** 2
            pc1 = points[region_mask]
            pc1_indices = torch.where(region_mask)[0]
            if len(pc1) == 0:
                continue

            # ② Grid sample (voxelization)
            pc2, pc2_indices, grid_inverse = grid_sample(pc1, pc1_indices, grid_size)

            # ③ Point sample
            if len(pc2) <= num_points:
                pc3, pc3_indices = pc2, pc2_indices
                nn_idx_pc1 = grid_inverse
            else:
                choices = np.random.choice(len(pc2), num_points, replace=False)
                pc3 = pc2[choices]
                pc3_indices = pc2_indices[choices]
                nn_idx_pc1 = None  # fallback to cdist below

            # ④ Collate → sparse tensor
            coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
            x_sparse = spconv.SparseConvTensor(features, coordinates, spatial_shape, 1)

            # ⑤–⑪ Neural forward in mixed precision (fp16 on CUDA). autocast only
            # affects float matmul/conv; int/index ops and numpy accumulation are
            # untouched. Decoder outputs are cast back to fp32 before thresholding.
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                # ⑤ Backbone
                x = self.extract_feat(x_sparse)

                # ⑥ Aux heads
                embed_logits = self.Embed(x[0])
                bi_semantic_logits = self.BiSemantic(x[0])
                semantic_predictions_bi = torch.argmax(bi_semantic_logits, dim=1)
                tree_indices = torch.where(semantic_predictions_bi == 1)[0]

                # ⑦ NN mapping (optimized: grid inverse or cdist fallback)
                if nn_idx_pc1 is None:
                    nn_idx_pc1_parts = []
                    for ss in range(0, pc1.shape[0], self.chunk):
                        ee = min(ss + self.chunk, pc1.shape[0])
                        nn_idx_pc1_parts.append(
                            torch.cdist(pc1[ss:ee].float(), pc3.float()).argmin(1))
                    nn_idx_pc1 = torch.cat(nn_idx_pc1_parts)

                if tree_indices.numel() > 1:
                    # ⑧ Grid-based query selection
                    tree_xyz = pc3[tree_indices]
                    selected_local = grid_subsample_queries(tree_xyz, self.query_point_num)
                    selected = tree_indices[selected_local]
                    queries = [x[0][selected]]

                    # ⑨ Decoder
                    x_dec = self.decoder(x, queries)

                    # ⑩ Predict masks + scores
                    masks, scores, sem_res = self._predict_by_feat(
                        x_dec, inverse_mapping2, pc3, selected)
                    # scores back to fp32 for stable thresholding / accumulation.
                    # NOTE: masks stays as-is — it is used as a boolean index.
                    scores = scores.float()

                    # ⑪ Post-process: score filter + edge filter + merge
                    self._process_cylinder_results(
                        masks, scores, sem_res, pc1, pc3, pc1_indices, nn_idx_pc1,
                        region, score_th1, all_pre_ins, global_instance_scores,
                        best_masks, max_instance, votes_counter, region_mask)
                    max_instance += int(masks.shape[0]) if masks is not None and len(masks) > 0 else 0
                else:
                    # No trees: just vote semantic labels
                    proj_logits = bi_semantic_logits[inverse_mapping2]
                    sem_pred = torch.argmax(proj_logits, dim=1)
                    cyl_sem = sem_pred[nn_idx_pc1]
                    ids_np = torch.where(region_mask)[0].cpu().numpy()
                    sem_np = cyl_sem.cpu().numpy().astype(int)
                    np.add.at(votes_counter, (ids_np, sem_np), 1)

            del pc1, pc1_indices, pc2, pc2_indices, pc3, pc3_indices
            del embed_logits, bi_semantic_logits
            if region_idx % 50 == 49:
                torch.cuda.empty_cache()
            if progress_cb is not None:
                progress_cb(region_idx + 1, len(regions))

        # Global post-processing
        final_semantic = votes_counter.argmax(1)
        final_semantic[votes_counter.sum(1) == 0] = -1
        all_pre_ins[final_semantic == 0] = -1  # ground → no instance

        # Remove small instances
        uniq, cnt = np.unique(all_pre_ins, return_counts=True)
        all_pre_ins[np.isin(all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])] = -1

        # Filter and merge overlapping instances
        unique_best = [(pts, iid, sc) for pts, iid, sc in best_masks
                       if np.any(all_pre_ins[pts] == iid)]
        clean_ins, _, merged_scores = self._merge_instances(
            all_pre_ins, unique_best, overlap_threshold=score_th2)

        # Relabel to continuous IDs
        unique_labels = np.unique(clean_ins)
        unique_labels = unique_labels[unique_labels >= 0]
        relabel = {old: new for new, old in enumerate(unique_labels)}
        relabel[-1] = -1
        clean_ins = np.vectorize(relabel.get)(clean_ins)

        # Save PLY if requested
        if output_path:
            save_ply_predictions(
                points.cpu().numpy(), final_semantic, clean_ins, merged_scores,
                output_path, pts_semantic_gt, pts_instance_gt)

        return {
            'semantic_pred': final_semantic,
            'instance_pred': clean_ins,
            'instance_scores': merged_scores,
        }

    # ──────────────────── Inference helpers ────────────────────

    @staticmethod
    def _generate_regions(points, radius, step_size):
        x_min, x_max = points[:, 0].min().item(), points[:, 0].max().item()
        y_min, y_max = points[:, 1].min().item(), points[:, 1].max().item()
        regions = []
        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                regions.append((x, y))
                y += step_size
            x += step_size
        return regions

    def _predict_by_feat(self, out, superpoints, coordinates, query_indices):
        """Extract instance masks + scores + semantic from decoder output.

        Returns:
            masks: (K, N_raw) bool instance masks in raw point space.
            scores: (K,) confidence scores.
            sem_res: (N_raw,) semantic labels in raw point space (0/1/2).
        """
        pred_masks = out['masks'][0]  # (Q+S, M_voxel)
        pred_scores = out['scores'][0]  # (Q+S, 1)
        n_sem = self.num_classes

        # Semantic: argmax of last 3 (semantic query) masks, mapped to raw points via superpoints
        sem_masks = pred_masks[-n_sem:, :]  # (3, M_voxel)
        sem_res = sem_masks.sigmoid()[:, superpoints].argmax(0)  # (N_raw,)

        # Instance masks + scores (exclude semantic queries)
        inst_masks = pred_masks[:-n_sem, :]  # (Q, M_voxel)
        inst_scores = pred_scores[:-n_sem, :]  # (Q, 1)

        labels = torch.zeros(inst_scores.shape[0], dtype=torch.long, device=inst_scores.device)
        scores = inst_scores.squeeze(-1)

        # topk_insts=300 (matches original test_cfg)
        topk_insts = 300
        if scores.numel() > topk_insts:
            scores, topk_idx = scores.topk(topk_insts, sorted=False)
            labels = labels[topk_idx]
            inst_masks = inst_masks[topk_idx]

        # Object normalization
        mask_sigmoid = inst_masks.sigmoid()
        if mask_sigmoid.shape[0] > 0:
            mask_pred_thr = mask_sigmoid > 0.01
            mask_scores = (mask_sigmoid * mask_pred_thr).sum(1) / (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

            # Matrix NMS
            scores, labels, mask_sigmoid, keep_inds = mask_matrix_nms(
                mask_sigmoid, labels, scores, kernel='linear')

        # Map to raw points via superpoints, threshold
        mask_pred = (mask_sigmoid > 0.15)[:, superpoints]  # (K, N_raw)

        # Suppress masks that are mostly ground
        is_stuff = (sem_res == 0).float()
        ground_ratio = (mask_pred.float() @ is_stuff) / mask_pred.sum(1).clamp(min=1)
        scores[ground_ratio > 0.5] = 0

        # Filter by z values (same as original)
        ground_pts = coordinates[sem_res == 0]
        ground_z_max = ground_pts[:, 2].max().item() if ground_pts.size(0) > 0 else float('inf')
        for i in range(mask_pred.size(0)):
            if mask_pred[i].sum() == 0:
                scores[i] = 0
                continue
            z_vals = coordinates[mask_pred[i], 2]
            if z_vals.numel() > 0 and z_vals.min().item() > ground_z_max + 5:
                scores[i] = 0

        # Keep valid
        keep = (scores > 0) & (mask_pred.sum(1) > 10)
        return mask_pred[keep], scores[keep], sem_res

    def _process_cylinder_results(self, masks, scores, sem_res, pc1, pc3,
                                   pc1_indices, nn_idx_pc1, region, score_th,
                                   all_pre_ins, global_instance_scores,
                                   best_masks, max_instance, votes_counter, region_mask):
        """Process per-cylinder predictions: edge filter, score merge, vote."""
        if masks is None or len(masks) == 0:
            # Semantic voting only
            ids_np = torch.where(region_mask)[0].cpu().numpy()
            sem_np = sem_res[nn_idx_pc1].cpu().numpy().astype(int)
            np.add.at(votes_counter, (ids_np, sem_np), 1)
            return

        valid_scores = scores > score_th
        edge_threshold = self.radius - 0.5
        pc3_dist = torch.sqrt((pc3[:, 0] - region[0]) ** 2 + (pc3[:, 1] - region[1]) ** 2)

        valid_edge = np.ones(len(masks), dtype=bool)
        for i in range(len(masks)):
            if valid_scores[i] and torch.any(pc3_dist[masks[i]] > edge_threshold):
                valid_edge[i] = False

        keep = torch.where(
            torch.tensor(valid_scores.cpu().numpy() & valid_edge, device=pc3.device))[0]

        if keep.numel():
            masks_kept = masks[keep]
            scores_kept = scores[keep]

            # Map masks from pc3 to pc1 space
            N1 = pc1.shape[0]
            best_score = torch.full((N1,), -1., device=pc3.device)
            best_mid = torch.full((N1,), -1, dtype=torch.long, device=pc3.device)

            # Expand masks via nn_idx
            max_chunk = max(1, (torch.iinfo(torch.int32).max // masks_kept.shape[0]) - 1_000_000)
            chunk_size = min(nn_idx_pc1.shape[0], max_chunk)
            rows_list, cols_list = [], []
            for start in range(0, nn_idx_pc1.shape[0], chunk_size):
                end = min(start + chunk_size, nn_idx_pc1.shape[0])
                mk = masks_kept[:, nn_idx_pc1[start:end]]
                r, c = mk.nonzero(as_tuple=True)
                rows_list.append(r)
                cols_list.append(c + start)
            rows = torch.cat(rows_list)
            cols = torch.cat(cols_list)
            score_per_hit = scores_kept[rows]

            best_score.index_reduce_(0, cols, score_per_hit, reduce='amax')
            improved = score_per_hit == best_score[cols]
            best_mid.index_put_((cols[improved],), rows[improved], accumulate=False)

            pts_glob = pc1_indices.cpu().numpy()
            new_scores = best_score.cpu().numpy()
            better = new_scores > global_instance_scores[pts_glob]

            if better.any():
                global_instance_scores[pts_glob[better]] = new_scores[better]
                all_pre_ins[pts_glob[better]] = max_instance + best_mid.cpu().numpy()[better]
                for mid in np.unique(best_mid.cpu().numpy()[better]):
                    sel_pts = cols[rows == mid].cpu().numpy()
                    best_masks.append((pts_glob[sel_pts], max_instance + int(mid),
                                       float(scores_kept[int(mid)])))

        # Semantic voting
        cyl_sem = sem_res[nn_idx_pc1]
        ids_np = torch.where(region_mask)[0].cpu().numpy()
        sem_np = cyl_sem.cpu().numpy().astype(int)
        np.add.at(votes_counter, (ids_np, sem_np), 1)

    @staticmethod
    def _merge_instances(all_pre_ins, best_masks, overlap_threshold=0.3):
        """Merge overlapping instances by score priority."""
        N = len(all_pre_ins)
        if not best_masks:
            return all_pre_ins, [], np.zeros(N, dtype=np.float64)

        best_masks = sorted(best_masks, key=lambda x: -x[2])
        taken = np.zeros(N, dtype=bool)
        merged_ins = np.full(N, -1, dtype=np.int64)
        merged_scores = np.zeros(N, dtype=np.float64)
        kept = []

        for pts_idx, inst_id, score in best_masks:
            overlap = taken[pts_idx].mean()
            if overlap > overlap_threshold:
                continue
            merged_ins[pts_idx] = inst_id
            merged_scores[pts_idx] = score
            taken[pts_idx] = True
            kept.append((pts_idx, inst_id, score))

        return merged_ins, kept, merged_scores


def load_pretrained(model, checkpoint_path, fix_spconv=True):
    """Load pretrained weights, optionally fixing spconv weight permutation.

    Args:
        model: ForestFormer3D instance.
        checkpoint_path: Path to .pth file.
        fix_spconv: If True and checkpoint doesn't have '_fix' in name,
                    apply weight permutation for spconv compatibility.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Fix spconv weight layout.
    # The original fix_spconv_checkpoint.py applies permute(1,2,3,4,0) to get (D,H,W,Cin,Cout).
    # Current spconv expects (Cout,D,H,W,Cin), so we apply permute(4,0,1,2,3) to undo the fix.
    # For unfixed checkpoints, no permutation is needed.
    is_fixed = '_fix' in os.path.basename(checkpoint_path)
    if fix_spconv:
        for key in list(state_dict.keys()):
            if (key.startswith('unet') or key.startswith('input_conv')) \
                    and key.endswith('weight') and len(state_dict[key].shape) == 5:
                if is_fixed:
                    state_dict[key] = state_dict[key].permute(4, 0, 1, 2, 3)
                else:
                    state_dict[key] = state_dict[key].permute(1, 2, 3, 4, 0)

    # Load with flexible key matching
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (may be expected for criterion/optimizer)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model
