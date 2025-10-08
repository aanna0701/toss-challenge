"""
MixUp Data Augmentation for Binary Classification with Sample Weights

Reference:
- MixUp: Beyond Empirical Risk Minimization (Zhang et al., 2017)
- https://arxiv.org/abs/1710.09412
"""

import numpy as np


def mixup_with_weights(X, y, base_weight=None, alpha=0.3, ratio=1.0, rng=None):
    """
    MixUp augmentation with per-sample weights for binary classification.
    
    Args:
        X (np.ndarray): Input features (N, D) float32
        y (np.ndarray): Binary labels (N,) in {0, 1}
        base_weight (np.ndarray, optional): Per-sample weights (N,). 
            If None, uses 1.0 for all samples.
        alpha (float): Beta distribution parameter. Default: 0.3
            - alpha=1.0: uniform mixing
            - alpha<1.0: prefer original samples
            - alpha>1.0: prefer more balanced mixing
        ratio (float): Ratio of MixUp samples to add. Default: 1.0
            - 1.0: add N MixUp samples (doubles dataset size)
            - 0.5: add N/2 MixUp samples
        rng (np.random.Generator, optional): Random number generator
    
    Returns:
        X_mix (np.ndarray): Mixed features (m, D)
        y_mix (np.ndarray): Mixed labels (m,) - continuous values
        w_mix (np.ndarray): Mixed sample weights (m,)
        
    where m = int(N * ratio)
    
    Example:
        >>> X = np.random.randn(1000, 10).astype('float32')
        >>> y = np.random.randint(0, 2, 1000)
        >>> base_weight = np.where(y == 1, 10.0, 1.0)  # class weights
        >>> X_mix, y_mix, w_mix = mixup_with_weights(X, y, base_weight, alpha=0.3, ratio=1.0)
        >>> # Combine with original data
        >>> X_aug = np.vstack([X, X_mix])
        >>> y_aug = np.hstack([y, y_mix])
        >>> w_aug = np.hstack([base_weight, w_mix])
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(y)
    m = int(n * ratio)
    
    if m == 0:
        # Return empty arrays if no MixUp samples requested
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty(0, dtype=y.dtype),
            np.empty(0, dtype=X.dtype)
        )
    
    # Sample pairs for mixing
    idx1 = rng.integers(0, n, size=m)
    idx2 = rng.integers(0, n, size=m)
    
    # Sample mixing coefficients from Beta distribution
    lam = rng.beta(alpha, alpha, size=m).astype(X.dtype)  # (m,)
    
    # Mix features and labels
    X_mix = lam[:, None] * X[idx1] + (1 - lam)[:, None] * X[idx2]
    y_mix = lam * y[idx1] + (1 - lam) * y[idx2]
    
    # Mix weights
    if base_weight is None:
        w_mix = np.ones(m, dtype=X.dtype)
    else:
        w1 = base_weight[idx1].astype(X.dtype)
        w2 = base_weight[idx2].astype(X.dtype)
        w_mix = lam * w1 + (1 - lam) * w2
    
    return X_mix, y_mix, w_mix


def apply_mixup_to_dataset(X, y, class_weight=None, alpha=0.3, ratio=1.0, rng=None):
    """
    Apply MixUp augmentation and combine with original dataset.
    
    Args:
        X (np.ndarray): Input features (N, D)
        y (np.ndarray): Binary labels (N,) in {0, 1}
        class_weight (tuple, optional): (weight_class_0, weight_class_1)
            If None, uses balanced weights based on class distribution
        alpha (float): Beta distribution parameter. Default: 0.3
        ratio (float): Ratio of MixUp samples to add. Default: 1.0
        rng (np.random.Generator, optional): Random number generator
    
    Returns:
        X_aug (np.ndarray): Augmented features (N + m, D)
        y_aug (np.ndarray): Augmented labels (N + m,)
        w_aug (np.ndarray): Augmented sample weights (N + m,)
    
    Example:
        >>> X = np.random.randn(1000, 10).astype('float32')
        >>> y = np.random.randint(0, 2, 1000)
        >>> X_aug, y_aug, w_aug = apply_mixup_to_dataset(X, y, alpha=0.3, ratio=0.5)
        >>> print(f"Original: {len(X)}, Augmented: {len(X_aug)}")
    """
    # Calculate base weights from class_weight
    if class_weight is None:
        # Auto-balanced weights
        pos_ratio = y.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else 1.0
        base_weight = np.where(y == 1, scale_pos_weight, 1.0).astype(X.dtype)
    else:
        weight_0, weight_1 = class_weight
        base_weight = np.where(y == 1, weight_1, weight_0).astype(X.dtype)
    
    # Generate MixUp samples
    X_mix, y_mix, w_mix = mixup_with_weights(X, y, base_weight, alpha, ratio, rng)
    
    # Combine with original data
    if len(X_mix) > 0:
        X_aug = np.vstack([X, X_mix])
        y_aug = np.hstack([y, y_mix])
        w_aug = np.hstack([base_weight, w_mix])
    else:
        X_aug = X
        y_aug = y
        w_aug = base_weight
    
    return X_aug, y_aug, w_aug


def mixup_batch_torch(batch_x, batch_y, alpha=0.3, device=None):
    """
    Apply MixUp to a PyTorch batch (for online augmentation during training).
    
    Args:
        batch_x (torch.Tensor): Input features (B, ...)
        batch_y (torch.Tensor): Labels (B,)
        alpha (float): Beta distribution parameter
        device: torch device
    
    Returns:
        mixed_x (torch.Tensor): Mixed features
        mixed_y (torch.Tensor): Mixed labels (continuous)
        lam (torch.Tensor): Mixing coefficients
    
    Example:
        >>> import torch
        >>> batch_x = torch.randn(32, 10)
        >>> batch_y = torch.randint(0, 2, (32,)).float()
        >>> mixed_x, mixed_y, lam = mixup_batch_torch(batch_x, batch_y, alpha=0.3)
    """
    import torch
    
    if device is None:
        device = batch_x.device
    
    batch_size = batch_x.size(0)
    
    # Sample lambda from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=batch_size)
        lam = torch.from_numpy(lam).float().to(device)
    else:
        lam = torch.ones(batch_size).to(device)
    
    # Random permutation
    index = torch.randperm(batch_size).to(device)
    
    # Mix inputs and targets
    # Handle different input dimensions
    lam_expanded = lam.view(-1, *([1] * (batch_x.dim() - 1)))
    mixed_x = lam_expanded * batch_x + (1 - lam_expanded) * batch_x[index]
    
    # Mix labels (for binary classification, labels should be float)
    mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
    
    return mixed_x, mixed_y, lam

