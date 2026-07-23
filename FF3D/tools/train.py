"""ForestFormer3D training script — pure PyTorch, no mm* dependencies.

Usage:
    python tools/train.py \
        --data-root /path/to/ForAINetV2 \
        --work-dir work_dirs/train_v1

    # Resume from checkpoint
    python tools/train.py \
        --data-root /path/to/ForAINetV2 \
        --work-dir work_dirs/train_v1 \
        --resume work_dirs/train_v1/epoch_1500.pth
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ff3d.model import ForestFormer3D
from ff3d.data import ForAINetV2Dataset, collate_fn
from ff3d.loss import discriminative_loss, instance_loss_layer, semantic_loss_with_aux
from ff3d.utils.grid_utils import grid_subsample_queries

import spconv.pytorch as spconv
from torch_scatter import scatter_add


def compute_loss(model, batch, epoch, prepare_epoch=1000, device='cuda'):
    """Compute all losses for one training step.

    Phase 1 (epoch ≤ prepare_epoch): discriminative + binary semantic loss only.
    Phase 2 (epoch > prepare_epoch): + decoder instance + semantic loss.
    """
    points_list = [p.to(device) for p in batch['points']]
    batch_size = len(points_list)

    # Forward: collate → backbone → aux heads
    coordinates, features, inverse_mapping, spatial_shape = model.collate(points_list)
    x_sparse = spconv.SparseConvTensor(features, coordinates, spatial_shape, batch_size)
    x = model.extract_feat(x_sparse)

    embed_logits = [model.Embed(y) for y in x]
    bi_semantic_logits = [model.BiSemantic(y) for y in x]

    # Phase 1 losses
    total_disc_loss = 0
    total_sem_bi_loss = 0

    for i in range(batch_size):
        # Voxel superpoints
        voxel_sp = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
        voxel_sp = torch.unique(voxel_sp, return_inverse=True)[1]

        pts_inst = batch['pts_instance_mask'][i].to(device)
        inst_mask_bool = batch['instance_mask'][i].to(device)

        # Discriminative loss on foreground voxels
        valid = inst_mask_bool
        voxel_inst = _get_voxel_instance_labels(pts_inst[valid], voxel_sp[valid])
        valid_voxel_idx = torch.unique(voxel_sp[valid])
        filtered_embed = embed_logits[i][valid_voxel_idx]
        batch_idx = torch.full_like(voxel_inst, i)
        disc_losses = discriminative_loss(filtered_embed, voxel_inst, batch_idx, 5)
        total_disc_loss += disc_losses['ins_loss']

        # Binary semantic loss
        n_vox = voxel_sp.max().item() + 1
        voxel_counts = torch.bincount(voxel_sp, minlength=n_vox).float()
        fg_counts = torch.bincount(voxel_sp, weights=inst_mask_bool.float(), minlength=n_vox)
        bi_y = (fg_counts / voxel_counts.clamp(min=1) > 0.5).long()
        total_sem_bi_loss += F.nll_loss(bi_semantic_logits[i], bi_y)

    total_disc_loss /= batch_size
    total_sem_bi_loss /= batch_size

    loss_dict = {
        'discriminative_loss': total_disc_loss,
        'semantic_loss_bi': total_sem_bi_loss,
    }

    # Phase 2: decoder + instance loss
    if epoch > prepare_epoch:
        queries_list = []
        queries_inslabel_list = []

        for i in range(batch_size):
            voxel_sp = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_sp = torch.unique(voxel_sp, return_inverse=True)[1]
            pts_inst = batch['pts_instance_mask'][i].to(device)
            inst_mask_bool = batch['instance_mask'][i].to(device)
            valid_voxel_idx = torch.unique(voxel_sp[inst_mask_bool])

            if valid_voxel_idx.numel() < 10:
                queries_list.append(torch.empty(0, x[i].shape[1], device=device))
                queries_inslabel_list.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            with torch.no_grad():
                voxel_inst = _get_voxel_instance_labels(pts_inst[inst_mask_bool], voxel_sp[inst_mask_bool])
                sem_pred = torch.argmax(bi_semantic_logits[i], dim=1)
                tree_idx = torch.where(sem_pred == 1)[0]

                if tree_idx.numel() < 2:
                    queries_list.append(torch.empty(0, x[i].shape[1], device=device))
                    queries_inslabel_list.append(torch.empty(0, dtype=torch.long, device=device))
                    continue

                # Grid subsampling for query selection (optimized)
                sample_coords = coordinates[coordinates[:, 0] == i, 1:].float()
                tree_xyz = sample_coords[tree_idx]
                sel_local = grid_subsample_queries(tree_xyz, model.query_point_num)
                selected = tree_idx[sel_local]

                queries_list.append(x[i][selected])

                # Instance labels for queries
                all_voxel_inst = _get_voxel_instance_labels(pts_inst, voxel_sp)
                queries_inslabel_list.append(all_voxel_inst[selected])

        # Check if any valid queries
        if all(len(q) == 0 for q in queries_list):
            return loss_dict

        # Filter empty queries
        valid_indices = [i for i in range(batch_size) if len(queries_list[i]) > 0]
        x_valid = [x[i] for i in valid_indices]
        queries_valid = [queries_list[i] for i in valid_indices]
        inslabel_valid = [queries_inslabel_list[i] for i in valid_indices]

        # Decoder forward
        decoder_out = model.decoder(x_valid, queries_valid)

        # Prepare GT masks for instance loss
        n_sem = model.num_classes
        gt_inst_masks_list = []
        gt_sem_masks_list = []
        ratio_list = []

        for idx, i in enumerate(valid_indices):
            voxel_sp = inverse_mapping[coordinates[:, 0][inverse_mapping] == i]
            voxel_sp = torch.unique(voxel_sp, return_inverse=True)[1]
            pts_inst = batch['pts_instance_mask'][i].to(device)
            pts_sem = batch['pts_semantic_mask'][i].to(device)

            # GT semantic masks (one-hot, per voxel)
            gt_sem = _get_gt_semantic_masks(pts_sem, voxel_sp, n_sem)
            gt_sem_masks_list.append(gt_sem)

            # GT instance masks (per voxel)
            gt_inst = _get_gt_inst_masks(pts_inst, voxel_sp)

            # Filter stuff (ground) instances
            inst_labels = batch['gt_labels_3d'][i].to(device)
            stuff_classes = [0]
            keep = torch.tensor([l not in stuff_classes for l in inst_labels], device=device)
            gt_inst = gt_inst[keep]
            inst_labels = inst_labels[keep]

            gt_inst_masks_list.append(gt_inst)
            ratio_list.append(batch['ratio_inspoint'][i])

        # Instance loss (with deep supervision)
        inst_pred_masks = [m[:-n_sem] for m in decoder_out['masks']]
        inst_pred_scores = [s[:-n_sem] if s is not None else None for s in decoder_out['scores']]

        inst_loss = instance_loss_layer(
            inst_pred_masks, inst_pred_scores,
            gt_inst_masks_list, None, inslabel_valid, ratio_list)

        # Aux layers
        if 'aux_outputs' in decoder_out:
            for aux in decoder_out['aux_outputs']:
                aux_masks = [m[:-n_sem] for m in aux['masks']]
                aux_scores = [s[:-n_sem] if s is not None else None for s in aux['scores']]
                inst_loss += instance_loss_layer(
                    aux_masks, aux_scores,
                    gt_inst_masks_list, None, inslabel_valid, ratio_list)

        loss_dict['inst_loss'] = inst_loss

        # Semantic loss (with deep supervision)
        sem_losses = semantic_loss_with_aux(decoder_out, gt_sem_masks_list, 0.2, n_sem)
        loss_dict.update(sem_losses)

    return loss_dict


def _get_voxel_instance_labels(pts_inst, voxel_sp):
    """Majority-vote instance label per voxel."""
    inst = pts_inst.clone()
    bg_label = None
    if torch.any(inst == -1):
        bg_label = inst.max().item() + 1
        inst[inst == -1] = bg_label

    _, inst = torch.unique(inst, return_inverse=True, sorted=True)
    _, vox = torch.unique(voxel_sp, return_inverse=True, sorted=True)

    n_vox = vox.max().item() + 1
    n_inst = inst.max().item() + 1
    combined = vox * n_inst + inst
    counts = torch.bincount(combined, minlength=n_vox * n_inst).reshape(n_vox, n_inst)
    labels = counts.argmax(dim=-1)

    if bg_label is not None:
        labels[labels == bg_label] = -1
    return labels


def _get_gt_semantic_masks(pts_sem, voxel_sp, num_classes):
    """GT semantic masks: (num_classes, M) one-hot per voxel."""
    n_vox = voxel_sp.max().item() + 1
    combined = voxel_sp * num_classes + pts_sem
    counts = torch.bincount(combined, minlength=n_vox * num_classes).reshape(n_vox, num_classes)
    labels = counts.argmax(dim=-1)
    masks = F.one_hot(labels, num_classes).float().T
    return masks


def _get_gt_inst_masks(pts_inst, voxel_sp):
    """GT instance masks: (K, M) binary per voxel."""
    n_vox = voxel_sp.max().item() + 1
    inst_ids = torch.unique(pts_inst)
    inst_ids = inst_ids[inst_ids >= 0]

    masks = []
    for iid in inst_ids:
        point_mask = (pts_inst == iid).float()
        voxel_mask = torch.zeros(n_vox, device=pts_inst.device)
        voxel_mask.scatter_add_(0, voxel_sp, point_mask)
        voxel_count = torch.zeros(n_vox, device=pts_inst.device)
        voxel_count.scatter_add_(0, voxel_sp, torch.ones_like(point_mask))
        masks.append((voxel_mask / voxel_count.clamp(min=1)) > 0.5)
    if masks:
        return torch.stack(masks).float()
    return torch.zeros(0, n_vox, device=pts_inst.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--work-dir', default='work_dirs/train')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--prepare-epoch', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    device = 'cuda'

    # Model
    model = ForestFormer3D(
        prepare_epoch=args.prepare_epoch,
        decoder_cfg=dict(
            num_layers=6, num_semantic_queries=3, in_channels=32,
            d_model=256, num_heads=8, hidden_dim=1024, dropout=0.0,
            activation='gelu', fix_attention=True, objectness_flag=True, attn_mask=True)
    ).to(device)

    # Data
    train_set = ForAINetV2Dataset(
        args.data_root,
        os.path.join(args.data_root, 'meta_data/train_list.txt'),
        augment=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, prefetch_factor=10, persistent_workers=True)

    # Optimizer + PolyLR scheduler (matching original)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_iters = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_iters, power=0.9)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training: {len(train_set)} samples, {len(train_loader)} iters/epoch, "
          f"batch_size={args.batch_size}")
    print(f"Phase 1: epoch 1-{args.prepare_epoch}, Phase 2: epoch {args.prepare_epoch+1}-{args.epochs}")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            loss_dict = compute_loss(model, batch, epoch, args.prepare_epoch, device)

            total_loss = sum(loss_dict.values())
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            scheduler.step()

            # Accumulate for logging
            for k, v in loss_dict.items():
                val = v.item() if hasattr(v, 'item') else float(v)
                epoch_loss[k] = epoch_loss.get(k, 0) + val

        # Log
        dt = time.time() - t0
        n_iters = len(train_loader)
        loss_str = '  '.join(f'{k}: {v/n_iters:.4f}' for k, v in epoch_loss.items())
        total = sum(epoch_loss.values()) / n_iters
        print(f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s  loss: {total:.4f}  {loss_str}")

        # Save checkpoint
        if epoch % 100 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.work_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, ckpt_path)
            # Keep only last 3 checkpoints
            for old_epoch in range(1, epoch - 2):
                old_path = os.path.join(args.work_dir, f'epoch_{old_epoch}.pth')
                if os.path.exists(old_path):
                    os.remove(old_path)

    print("Training complete!")


if __name__ == '__main__':
    main()
