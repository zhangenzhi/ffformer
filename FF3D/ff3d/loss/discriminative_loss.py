"""Discriminative loss for instance embedding clustering."""
import torch
from torch_scatter import scatter


def discriminative_loss_single(prediction, correct_label, feature_dim,
                                delta_v=0.5, delta_d=1.5):
    """Discriminative loss for one sample.

    Args:
        prediction: (N, feature_dim) embedding vectors.
        correct_label: (N,) instance IDs.
        feature_dim: embedding dimension.
    """
    reshaped_pred = prediction.reshape(-1, feature_dim)
    unique_labels, unique_id, counts = torch.unique(
        correct_label, return_inverse=True, return_counts=True)
    num_instances = unique_labels.size()

    if num_instances[0] == 0:
        zero = torch.tensor(0.0, device=prediction.device)
        return zero, zero, zero

    # Instance centers
    segmented_sum = scatter(reshaped_pred, unique_id, dim=0, reduce="sum")
    mu = segmented_sum / (counts.reshape(-1, 1).float() + 1e-8)

    # l_var: pull same-instance points toward center
    mu_expand = torch.gather(mu, 0, unique_id.unsqueeze(1).expand(-1, mu.size(-1)))
    distance = torch.norm(reshaped_pred - mu_expand, p=1, dim=1)
    distance = torch.clamp(distance - delta_v, min=0.) ** 2
    l_var = scatter(distance, unique_id, dim=0, reduce="sum")
    l_var = (l_var / (counts.float() + 1e-8)).sum() / float(num_instances[0])

    # l_dist: push different instance centers apart
    if num_instances[0] > 1:
        mu_a = mu.repeat(num_instances[0], 1)
        mu_b = mu.repeat(1, num_instances[0]).reshape(num_instances[0] ** 2, feature_dim)
        mu_diff = mu_b - mu_a
        eye = torch.eye(num_instances[0], device=prediction.device)
        diff_mask = (eye == 0).reshape(-1)
        mu_norm = torch.norm(mu_diff[diff_mask], p=1, dim=1)
        mu_norm = torch.clamp(2 * delta_d - mu_norm, min=0.) ** 2
        l_dist = mu_norm.mean()
    else:
        l_dist = torch.tensor(0.0, device=prediction.device)

    # l_reg: regularization
    l_reg = torch.norm(mu, p=1, dim=1).mean()

    return l_var, l_dist, 0.001 * l_reg


def discriminative_loss(prediction, correct_label, batch_idx, feature_dim):
    """Compute discriminative loss over a batch.

    Returns dict with 'ins_loss'.
    """
    l_var, l_dist, l_reg = discriminative_loss_single(
        prediction, correct_label, feature_dim)
    total = l_var + l_dist + l_reg
    return {'ins_loss': total}
