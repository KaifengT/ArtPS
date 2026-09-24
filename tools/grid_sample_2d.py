import torch
from typing import Optional

def custom_grid_sample_2d(
    input: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    """
    A custom implementation of grid_sample that supports double backward (second-order derivatives).
    Fixed to mode='bilinear', padding_mode='zeros', align_corners=False.
    """
    assert mode == "bilinear", "Only bilinear mode is supported"
    assert padding_mode == "zeros", "Only zeros padding mode is supported"
    assert align_corners is False, "Only align_corners=False is supported"

    if input.dim() == 4:
        return _grid_sample_2d(input, grid)
    elif input.dim() == 5:
        return _grid_sample_3d(input, grid)
    else:
        raise ValueError(f"Input must be 4D or 5D, got {input.dim()}D")

def _grid_sample_2d(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # input: (N, C, H, W)
    # grid: (N, H_out, W_out, 2)
    
    N, C, H, W = input.shape
    N, H_out, W_out, _ = grid.shape
    
    # Extract coordinates
    x = grid[..., 0]
    y = grid[..., 1]
    
    # Denormalize coordinates (align_corners=False)
    # [-1, 1] -> [-0.5, size-0.5]
    # formula: ((grid + 1) * size - 1) / 2
    ix = ((x + 1) * W - 1) / 2
    iy = ((y + 1) * H - 1) / 2
    
    # Get corner coordinates
    ix0 = torch.floor(ix)
    ix1 = ix0 + 1
    iy0 = torch.floor(iy)
    iy1 = iy0 + 1
    
    # Clamp to be inside image boundaries for gathering
    # We will mask out the values later
    ix0_clamped = torch.clamp(ix0, 0, W - 1).long()
    ix1_clamped = torch.clamp(ix1, 0, W - 1).long()
    iy0_clamped = torch.clamp(iy0, 0, H - 1).long()
    iy1_clamped = torch.clamp(iy1, 0, H - 1).long()
    
    # Calculate weights
    # wa = (ix1 - ix) * (iy1 - iy)
    # wb = (ix - ix0) * (iy1 - iy)
    # wc = (ix1 - ix) * (iy - iy0)
    # wd = (ix - ix0) * (iy - iy0)
    
    # Note: ix1 = ix0 + 1, so ix1 - ix = 1 - (ix - ix0)
    # Let's define alpha, beta
    alpha = ix - ix0
    beta = iy - iy0
    
    wa = (1 - alpha) * (1 - beta)
    wb = alpha * (1 - beta)
    wc = (1 - alpha) * beta
    wd = alpha * beta
    
    # Prepare batch indices for advanced indexing
    # We want output (N, C, H_out, W_out)
    # We use permute to put C at the end for easier indexing: (N, H, W, C)
    input_perm = input.permute(0, 2, 3, 1)
    
    # Create batch indices
    # shape (N, H_out, W_out)
    batch_idx = torch.arange(N, device=input.device).view(N, 1, 1).expand(N, H_out, W_out)
    
    # Gather values
    # input_perm[batch_idx, iy, ix] -> (N, H_out, W_out, C)
    Ia = input_perm[batch_idx, iy0_clamped, ix0_clamped]
    Ib = input_perm[batch_idx, iy0_clamped, ix1_clamped]
    Ic = input_perm[batch_idx, iy1_clamped, ix0_clamped]
    Id = input_perm[batch_idx, iy1_clamped, ix1_clamped]
    
    # Calculate masks for zero padding
    # Check if original coordinates are within bounds
    def in_bounds(xx, yy, w, h):
        return (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    
    mask_a = in_bounds(ix0, iy0, W, H).float().unsqueeze(-1)
    mask_b = in_bounds(ix1, iy0, W, H).float().unsqueeze(-1)
    mask_c = in_bounds(ix0, iy1, W, H).float().unsqueeze(-1)
    mask_d = in_bounds(ix1, iy1, W, H).float().unsqueeze(-1)
    
    # Expand weights to match C dimension
    wa = wa.unsqueeze(-1)
    wb = wb.unsqueeze(-1)
    wc = wc.unsqueeze(-1)
    wd = wd.unsqueeze(-1)
    
    # Interpolate
    out = (wa * mask_a * Ia +
           wb * mask_b * Ib +
           wc * mask_c * Ic +
           wd * mask_d * Id)
           
    # Permute back to (N, C, H_out, W_out)
    return out.permute(0, 3, 1, 2)

def _grid_sample_3d(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    # input: (N, C, D, H, W)
    # grid: (N, D_out, H_out, W_out, 3)
    
    N, C, D, H, W = input.shape
    N, D_out, H_out, W_out, _ = grid.shape
    
    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]
    
    ix = ((x + 1) * W - 1) / 2
    iy = ((y + 1) * H - 1) / 2
    iz = ((z + 1) * D - 1) / 2
    
    ix0 = torch.floor(ix)
    ix1 = ix0 + 1
    iy0 = torch.floor(iy)
    iy1 = iy0 + 1
    iz0 = torch.floor(iz)
    iz1 = iz0 + 1
    
    ix0_c = torch.clamp(ix0, 0, W - 1).long()
    ix1_c = torch.clamp(ix1, 0, W - 1).long()
    iy0_c = torch.clamp(iy0, 0, H - 1).long()
    iy1_c = torch.clamp(iy1, 0, H - 1).long()
    iz0_c = torch.clamp(iz0, 0, D - 1).long()
    iz1_c = torch.clamp(iz1, 0, D - 1).long()
    
    alpha = ix - ix0
    beta = iy - iy0
    gamma = iz - iz0
    
    # 8 corners
    # 000, 100, 010, 110, 001, 101, 011, 111 (x, y, z order)
    # Weights
    w000 = (1 - alpha) * (1 - beta) * (1 - gamma)
    w100 = alpha * (1 - beta) * (1 - gamma)
    w010 = (1 - alpha) * beta * (1 - gamma)
    w110 = alpha * beta * (1 - gamma)
    w001 = (1 - alpha) * (1 - beta) * gamma
    w101 = alpha * (1 - beta) * gamma
    w011 = (1 - alpha) * beta * gamma
    w111 = alpha * beta * gamma
    
    input_perm = input.permute(0, 2, 3, 4, 1) # N, D, H, W, C
    
    batch_idx = torch.arange(N, device=input.device).view(N, 1, 1, 1).expand(N, D_out, H_out, W_out)
    
    # Gather
    I000 = input_perm[batch_idx, iz0_c, iy0_c, ix0_c]
    I100 = input_perm[batch_idx, iz0_c, iy0_c, ix1_c]
    I010 = input_perm[batch_idx, iz0_c, iy1_c, ix0_c]
    I110 = input_perm[batch_idx, iz0_c, iy1_c, ix1_c]
    I001 = input_perm[batch_idx, iz1_c, iy0_c, ix0_c]
    I101 = input_perm[batch_idx, iz1_c, iy0_c, ix1_c]
    I011 = input_perm[batch_idx, iz1_c, iy1_c, ix0_c]
    I111 = input_perm[batch_idx, iz1_c, iy1_c, ix1_c]
    
    def in_bounds(xx, yy, zz, w, h, d):
        return (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h) & (zz >= 0) & (zz < d)
        
    m000 = in_bounds(ix0, iy0, iz0, W, H, D).float().unsqueeze(-1)
    m100 = in_bounds(ix1, iy0, iz0, W, H, D).float().unsqueeze(-1)
    m010 = in_bounds(ix0, iy1, iz0, W, H, D).float().unsqueeze(-1)
    m110 = in_bounds(ix1, iy1, iz0, W, H, D).float().unsqueeze(-1)
    m001 = in_bounds(ix0, iy0, iz1, W, H, D).float().unsqueeze(-1)
    m101 = in_bounds(ix1, iy0, iz1, W, H, D).float().unsqueeze(-1)
    m011 = in_bounds(ix0, iy1, iz1, W, H, D).float().unsqueeze(-1)
    m111 = in_bounds(ix1, iy1, iz1, W, H, D).float().unsqueeze(-1)
    
    w000 = w000.unsqueeze(-1)
    w100 = w100.unsqueeze(-1)
    w010 = w010.unsqueeze(-1)
    w110 = w110.unsqueeze(-1)
    w001 = w001.unsqueeze(-1)
    w101 = w101.unsqueeze(-1)
    w011 = w011.unsqueeze(-1)
    w111 = w111.unsqueeze(-1)
    
    out = (w000 * m000 * I000 +
           w100 * m100 * I100 +
           w010 * m010 * I010 +
           w110 * m110 * I110 +
           w001 * m001 * I001 +
           w101 * m101 * I101 +
           w011 * m011 * I011 +
           w111 * m111 * I111)
           
    return out.permute(0, 4, 1, 2, 3)
