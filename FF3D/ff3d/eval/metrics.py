"""Evaluation metrics for forest point cloud segmentation.

Computes F1, Precision, Recall, mIoU, PQ, SQ, RQ, MUCov, MWCov
using the same protocol as the original ForestFormer3D UnifiedSegMetric.
"""
import numpy as np
from scipy import stats


def _compact_instances(ins_f, sem_f, skip_val, ins_class):
    """Assign compact 0..K-1 labels to instances whose majority semantic label is
    ins_class (others → -1). Returns (labels[N], K, sizes[K]). Loops over the few
    hundred unique instance ids only — never over points in Python."""
    labels = np.full(ins_f.shape[0], -1, dtype=np.int64)
    sizes = []
    next_id = 0
    for g in np.unique(ins_f):
        if g == skip_val:
            continue
        tmp = ins_f == g
        seg_i = int(stats.mode(sem_f[tmp], keepdims=True)[0][0])
        if seg_i == ins_class:
            labels[tmp] = next_id
            sizes.append(int(tmp.sum()))
            next_id += 1
    return labels, next_id, np.asarray(sizes, dtype=np.int64)


def evaluate_scene(sem_pred, ins_pred, sem_gt, ins_gt,
                   stuff_class_inds=(0,), thing_class_inds=(1, 2)):
    """Evaluate one scene. Returns raw counts for global aggregation.

    Fully vectorized (numpy) — mathematically identical to the original per-point /
    nested-loop protocol, but the per-point semantic confusion is done with
    bincount and the pred×gt instance IoU with a single co-occurrence matrix
    (np.add.at) instead of an O(n_pred·n_gt·N) double loop over full arrays.
    """
    NUM_SEM = 4    # 0:unclassified, 1:ground, 2:wood, 3:leaf
    NUM_BIN = 3    # 0:unclassified, 1:stuff, 2:thing

    # Shift by 1 (0 = unclassified)
    sem_gt_i = (sem_gt + 1).astype(np.int64)
    sem_pre_i = (sem_pred + 1).astype(np.int64)

    # Semantic confusion (bincount == per-point tally; slice guards stray labels)
    gt_sem = np.bincount(sem_gt_i, minlength=NUM_SEM)[:NUM_SEM].astype(float)
    pos_sem = np.bincount(sem_pre_i, minlength=NUM_SEM)[:NUM_SEM].astype(float)
    sem_match = sem_gt_i == sem_pre_i
    true_pos_sem = np.bincount(sem_gt_i[sem_match], minlength=NUM_SEM)[:NUM_SEM].astype(float)

    # Binary mapping: stuff → 1, thing → 2
    sem_gt_bi = np.copy(sem_gt_i)
    sem_pre_bi = np.copy(sem_pre_i)
    for sc in stuff_class_inds:
        sem_gt_bi[sem_gt_i == sc + 1] = 1
        sem_pre_bi[sem_pre_i == sc + 1] = 1
    for tc in thing_class_inds:
        sem_gt_bi[sem_gt_i == tc + 1] = 2
        sem_pre_bi[sem_pre_i == tc + 1] = 2

    gt_bi = np.bincount(sem_gt_bi, minlength=NUM_BIN)[:NUM_BIN].astype(float)
    pos_bi = np.bincount(sem_pre_bi, minlength=NUM_BIN)[:NUM_BIN].astype(float)
    bi_match = sem_gt_bi == sem_pre_bi
    true_pos_bi = np.bincount(sem_gt_bi[bi_match], minlength=NUM_BIN)[:NUM_BIN].astype(float)

    # Instance evaluation (filter ground)
    idxc = (sem_gt_bi != 1) | (sem_pre_bi != 1)
    pred_ins_f = ins_pred[idxc]
    gt_ins_f = ins_gt[idxc]
    pred_sem_f = sem_pre_bi[idxc]
    gt_sem_f = sem_gt_bi[idxc]

    INS_CLASS = 2  # tree

    # Compact tree-instance labels (skip pred -1 / gt 0, as in the original).
    P, n_pred, psize = _compact_instances(pred_ins_f, pred_sem_f, -1, INS_CLASS)
    G, n_gt, gsize = _compact_instances(gt_ins_f, gt_sem_f, 0, INS_CLASS)

    # IoU matrix (n_pred, n_gt) via a single co-occurrence pass.
    if n_pred and n_gt:
        both = (P >= 0) & (G >= 0)
        inter = np.zeros((n_pred, n_gt), dtype=np.int64)
        np.add.at(inter, (P[both], G[both]), 1)
        union = psize[:, None] + gsize[None, :] - inter   # >=1 (both sides non-empty)
        iou = inter / union
    else:
        iou = np.zeros((n_pred, n_gt))

    # PQ metrics: each pred matched to its best-IoU gt (>=0.5 → TP, else FP).
    tp_list, fp_list = [], []
    iou_tp = 0.0
    if n_pred:
        if n_gt == 0:
            fp_list = [1.0] * n_pred                      # no gt → every pred is FP
        else:
            ovmax = iou.max(axis=1)                       # best gt per pred
            tp_mask = ovmax >= 0.5
            n_tp = int(tp_mask.sum())
            tp_list = [1.0] * n_tp
            fp_list = [1.0] * (n_pred - n_tp)
            iou_tp = float(ovmax[tp_mask].sum())

    # Coverage: each gt covered by its best-IoU pred.
    sum_cov = 0.0
    num_gt_pt = 0
    mean_weighted_cov = 0.0
    if n_gt and n_pred:
        ovmax_gt = iou.max(axis=0)                        # best pred per gt (>=0)
        sum_cov = float(ovmax_gt.sum())
        num_gt_pt = int(gsize.sum())
        mean_weighted_cov = float((ovmax_gt * gsize).sum())

    return {
        'true_pos_sem': true_pos_sem,
        'pos_sem': pos_sem,
        'gt_sem': gt_sem,
        'true_pos_bi': true_pos_bi,
        'pos_bi': pos_bi,
        'gt_bi': gt_bi,
        'n_gt_ins': n_gt,
        'tp': tp_list,
        'fp': fp_list,
        'iou_tp': iou_tp,
        'sum_cov': sum_cov / max(n_gt, 1) if n_gt else 0,
        'mean_wcov': mean_weighted_cov / max(num_gt_pt, 1) if n_gt else 0,
        'has_gt': n_gt > 0,
        'has_pred': n_pred > 0,
    }


def aggregate_metrics(scene_results):
    """Aggregate per-scene results into global metrics.

    Args:
        scene_results: List of dicts from evaluate_scene().

    Returns:
        Dict with mIoU, mIoU_binary, F1, Precision, Recall, mPQ, mSQ, mRQ, MUCov, MWCov.
    """
    NUM_SEM = 4
    NUM_BIN = 3

    true_pos_sem = np.zeros(NUM_SEM)
    pos_sem = np.zeros(NUM_SEM)
    gt_sem = np.zeros(NUM_SEM)
    true_pos_bi = np.zeros(NUM_BIN)
    pos_bi = np.zeros(NUM_BIN)
    gt_bi = np.zeros(NUM_BIN)
    total_gt_ins = 0
    all_tp = []
    all_fp = []
    iou_tp_total = 0.0
    all_cov = []
    all_wcov = []

    for r in scene_results:
        true_pos_sem += r['true_pos_sem']
        pos_sem += r['pos_sem']
        gt_sem += r['gt_sem']
        true_pos_bi += r['true_pos_bi']
        pos_bi += r['pos_bi']
        gt_bi += r['gt_bi']
        total_gt_ins += r['n_gt_ins']
        all_tp.extend(r['tp'])
        all_fp.extend(r['fp'])
        iou_tp_total += r['iou_tp']
        if r['has_gt'] and r['has_pred']:
            all_cov.append(r['sum_cov'])
            all_wcov.append(r['mean_wcov'])

    # Semantic mIoU
    valid_sem = [i for i in range(1, NUM_SEM) if gt_sem[i] > 0]
    iou_list = []
    for i in range(1, NUM_SEM):
        iou = true_pos_sem[i] / (gt_sem[i] + pos_sem[i] - true_pos_sem[i] + 1e-8)
        iou_list.append(iou)
    mIoU = np.mean([iou_list[i - 1] for i in valid_sem]) if valid_sem else 0

    # Binary mIoU
    valid_bi = [i for i in range(1, NUM_BIN) if gt_bi[i] > 0]
    iou_bi = []
    for i in range(1, NUM_BIN):
        iou = true_pos_bi[i] / (gt_bi[i] + pos_bi[i] - true_pos_bi[i] + 1e-8)
        iou_bi.append(iou)
    mIoU_bi = np.mean([iou_bi[i - 1] for i in valid_bi]) if valid_bi else 0

    # Instance metrics
    tp_sum = sum(all_tp)
    fp_sum = sum(all_fp)
    prec = tp_sum / (tp_sum + fp_sum + 1e-8)
    rec = tp_sum / (total_gt_ins + 1e-8)
    F1 = 2 * prec * rec / (prec + rec + 1e-8)
    RQ = F1
    SQ = iou_tp_total / (tp_sum + 1e-8)
    PQ = SQ * RQ

    MUCov = np.mean(all_cov) if all_cov else 0
    MWCov = np.mean(all_wcov) if all_wcov else 0

    return {
        'mIoU': round(mIoU, 4),
        'mIoU_binary': round(mIoU_bi, 4),
        'mPQ': round(PQ, 4),
        'mSQ': round(SQ, 4),
        'mRQ': round(RQ, 4),
        'mPrecision': round(prec, 4),
        'mRecall': round(rec, 4),
        'F1': round(F1, 4),
        'mMUCov': round(MUCov, 4),
        'mMWCov': round(MWCov, 4),
    }
