"""
Tensor-based Medium Reconstruction Module
This module implements tensor decomposition for medium reconstruction,
replacing the simple MLP-based approach in traditional splatting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def positional_encoding(positions: torch.Tensor, freqs: int) -> torch.Tensor:
    """Positional encoding similar to tensor implementation."""
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class MLPRender_Medium(torch.nn.Module):
    """MLP for rendering medium properties from tensor features."""
    
    def __init__(self, in_channel: int, view_pe: int = 6, feature_c: int = 128):
        super(MLPRender_Medium, self).__init__()
        
        self.in_mlp_c = (3 + 2 * view_pe * 3) + in_channel
        self.view_pe = view_pe
        
        layer1 = torch.nn.Linear(self.in_mlp_c, feature_c)
        layer2 = torch.nn.Linear(feature_c, feature_c)
        layer3 = torch.nn.Linear(feature_c, 9)  # 3 for rgb, 3 for bs, 3 for attn
        
        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace=True), 
            layer2, torch.nn.ReLU(inplace=True), 
            layer3
        )
        torch.nn.init.constant_(self.mlp[-1].bias, 0)
    
    def forward(self, pts: torch.Tensor, viewdirs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: sample points [N, 3]
            viewdirs: view directions [N, 3]  
            features: tensor features [N, feature_dim]
        
        Returns:
            medium outputs [N, 9] (rgb + bs + attn)
        """
        indata = [features, viewdirs]
        if self.view_pe > 0:
            indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)
        medium_out = self.mlp(mlp_in)
        return medium_out


class MediumTensorBase(torch.nn.Module):
    """Base class for tensor-based medium reconstruction."""
    
    def __init__(
        self, 
        aabb: torch.Tensor,
        grid_size: Tuple[int, int, int],
        device: torch.device,
        density_n_comp: int = 8,
        appearance_n_comp: int = 24,
        app_dim: int = 27,
        view_pe: int = 6,
        feature_c: int = 128,
        step_ratio: float = 0.5
    ):
        super(MediumTensorBase, self).__init__()
        
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb.to(device)
        self.device = device
        self.step_ratio = step_ratio
        
        # Compute AABB properties
        self.aabb_size = self.aabb[1] - self.aabb[0]
        self.inv_aabb_size = 2.0 / self.aabb_size
        self.grid_size = torch.LongTensor(grid_size).to(device)
        self.units = self.aabb_size / (self.grid_size - 1)
        self.step_size = torch.mean(self.units) * self.step_ratio
        
        # Tensor decomposition modes (similar to TensoRF)
        self.mat_mode = [[0, 1], [0, 2], [1, 2]]
        self.vec_mode = [2, 1, 0]
        
        # Initialize tensor volumes
        self.init_tensor_volumes(grid_size, device)
        
        # Initialize rendering module
        self.render_module = MLPRender_Medium(self.app_dim, view_pe, feature_c).to(device)
    
    def init_tensor_volumes(self, grid_size: Tuple[int, int, int], device: torch.device):
        """Initialize tensor decomposition volumes. To be implemented by subclasses."""
        pass
    
    def normalize_coord(self, xyz_sampled: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [-1, 1] range."""
        return (xyz_sampled - self.aabb[0]) * self.inv_aabb_size - 1
    
    def compute_medium_features(self, xyz_sampled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute density and appearance features. To be implemented by subclasses."""
        pass
    
    def forward(
        self, 
        xyz_sampled: torch.Tensor, 
        viewdirs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for medium reconstruction.
        
        Args:
            xyz_sampled: Sample points [N, 3]
            viewdirs: View directions [N, 3]
            
        Returns:
            medium_rgb: Medium RGB values [N, 3]
            medium_bs: Medium backscatter values [N, 3] 
            medium_attn: Medium attenuation values [N, 3]
        """
        # Normalize coordinates
        xyz_normalized = self.normalize_coord(xyz_sampled)
        
        # Compute tensor features
        density_features, app_features = self.compute_medium_features(xyz_normalized)
        
        # Render medium properties
        medium_out = self.render_module(xyz_sampled, viewdirs, app_features)
        
        # Split outputs
        medium_rgb = torch.sigmoid(medium_out[..., :3])
        medium_bs = F.softplus(medium_out[..., 3:6])
        medium_attn = F.softplus(medium_out[..., 6:])
        
        return medium_rgb, medium_bs, medium_attn


class MediumTensorVM(MediumTensorBase):
    """VM decomposition for medium reconstruction."""
    
    def init_tensor_volumes(self, grid_size: Tuple[int, int, int], device: torch.device):
        """Initialize VM decomposition tensors."""
        res = grid_size[0]  # Assume cubic grid
        
        # Combined plane and line coefficients (similar to TensoRF)
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device)
        )
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device)
        )
        
        # Basis matrix for appearance features
        self.basis_mat = torch.nn.Linear(
            self.app_n_comp * 3, self.app_dim, bias=False, device=device
        )
    
    def compute_medium_features(self, xyz_sampled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute density and appearance features using VM decomposition."""
        # Prepare coordinates for grid sampling
        coordinate_plane = torch.stack((
            xyz_sampled[..., self.mat_mode[0]], 
            xyz_sampled[..., self.mat_mode[1]], 
            xyz_sampled[..., self.mat_mode[2]]
        )).detach()
        
        coordinate_line = torch.stack((
            xyz_sampled[..., self.vec_mode[0]], 
            xyz_sampled[..., self.vec_mode[1]], 
            xyz_sampled[..., self.vec_mode[2]]
        ))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).detach()
        
        # Sample density features
        plane_feats_density = F.grid_sample(
            self.plane_coef[:, -self.density_n_comp:], 
            coordinate_plane, align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        
        line_feats_density = F.grid_sample(
            self.line_coef[:, -self.density_n_comp:], 
            coordinate_line, align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        
        density_feature = torch.sum(plane_feats_density * line_feats_density, dim=0)
        
        # Sample appearance features
        plane_feats_app = F.grid_sample(
            self.plane_coef[:, :self.app_n_comp], 
            coordinate_plane, align_corners=True
        ).view(3 * self.app_n_comp, -1)
        
        line_feats_app = F.grid_sample(
            self.line_coef[:, :self.app_n_comp], 
            coordinate_line, align_corners=True
        ).view(3 * self.app_n_comp, -1)
        
        app_features = self.basis_mat((plane_feats_app * line_feats_app).T)
        
        return density_feature, app_features
    
    def get_optparam_groups(self, lr_init_spatial: float = 0.02, lr_init_network: float = 0.001):
        """Get parameter groups for optimization."""
        grad_vars = [
            {'params': self.line_coef, 'lr': lr_init_spatial},
            {'params': self.plane_coef, 'lr': lr_init_spatial},
            {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
            {'params': self.render_module.parameters(), 'lr': lr_init_network}
        ]
        return grad_vars
    
    def vector_comp_diffs(self) -> torch.Tensor:
        """Orthogonality regularization for vector components."""
        total = 0
        for idx in range(len(self.vec_mode)):
            n_comp, n_size = self.line_coef[idx].shape[1:-1]
            dotp = torch.matmul(
                self.line_coef[idx].view(n_comp, n_size), 
                self.line_coef[idx].view(n_comp, n_size).transpose(-1, -2)
            )
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[..., :-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total
    
    def upsample_volume_grid(self, res_target: Tuple[int, int, int]):
        """Upsample tensor volumes to target resolution."""
        scale = res_target[0] / self.line_coef.shape[2]  # Assuming xyz have same scale
        
        plane_coef = F.interpolate(
            self.plane_coef.detach().data, 
            scale_factor=scale, 
            mode='bilinear', 
            align_corners=True
        )
        line_coef = F.interpolate(
            self.line_coef.detach().data, 
            size=(res_target[0], 1), 
            mode='bilinear', 
            align_corners=True
        )
        
        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.line_coef = torch.nn.Parameter(line_coef)
        
        # Update grid properties
        self.grid_size = torch.LongTensor(res_target).to(self.device)
        self.units = self.aabb_size / (self.grid_size - 1)
        self.step_size = torch.mean(self.units) * self.step_ratio
        
        print(f'Medium tensor upsampled to {res_target}')


class MediumTensorVMSplit(MediumTensorBase):
    """VM Split decomposition for medium reconstruction."""
    
    def init_tensor_volumes(self, grid_size: Tuple[int, int, int], device: torch.device):
        """Initialize VM split decomposition tensors."""
        self.density_plane, self.density_line = self.init_one_tensor(
            [self.density_n_comp] * 3, grid_size, 0.1, device
        )
        self.app_plane, self.app_line = self.init_one_tensor(
            [self.app_n_comp] * 3, grid_size, 0.1, device
        )
        
        self.basis_mat = torch.nn.Linear(
            sum([self.app_n_comp] * 3), self.app_dim, bias=False
        ).to(device)
    
    def init_one_tensor(
        self, 
        n_component: list, 
        grid_size: Tuple[int, int, int], 
        scale: float, 
        device: torch.device
    ) -> Tuple[torch.nn.ParameterList, torch.nn.ParameterList]:
        """Initialize one set of tensor decomposition."""
        plane_coef, line_coef = [], []
        
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            mat_id_0, mat_id_1 = self.mat_mode[i]
            
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0]))
            ))
            line_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))
            ))
        
        return (
            torch.nn.ParameterList(plane_coef).to(device), 
            torch.nn.ParameterList(line_coef).to(device)
        )
    
    def compute_medium_features(self, xyz_sampled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute features using VM split decomposition."""
        # Prepare coordinates
        coordinate_plane = torch.stack((
            xyz_sampled[..., self.mat_mode[0]],
            xyz_sampled[..., self.mat_mode[1]], 
            xyz_sampled[..., self.mat_mode[2]]
        )).detach().view(3, -1, 1, 2)
        
        coordinate_line = torch.stack((
            xyz_sampled[..., self.vec_mode[0]],
            xyz_sampled[..., self.vec_mode[1]],
            xyz_sampled[..., self.vec_mode[2]]
        ))
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).detach().view(3, -1, 1, 2)
        
        # Compute density features
        density_feature = 0
        for idx in range(len(self.vec_mode)):
            plane_feat = F.grid_sample(
                self.density_plane[idx], 
                coordinate_plane[[idx]], 
                align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            
            line_feat = F.grid_sample(
                self.density_line[idx],
                coordinate_line[[idx]],
                align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            
            density_feature = density_feature + torch.sum(plane_feat * line_feat, dim=0)
        
        # Compute appearance features
        app_features = []
        for idx in range(len(self.vec_mode)):
            plane_feat = F.grid_sample(
                self.app_plane[idx],
                coordinate_plane[[idx]],
                align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            
            line_feat = F.grid_sample(
                self.app_line[idx],
                coordinate_line[[idx]], 
                align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            
            app_features.append(torch.sum(plane_feat * line_feat, dim=0))
        
        app_features = self.basis_mat(torch.stack(app_features, dim=-1))
        
        return density_feature, app_features
    
    def get_optparam_groups(self, lr_init_spatial: float = 0.02, lr_init_network: float = 0.001):
        """Get parameter groups for optimization."""
        grad_vars = [
            {'params': self.density_line, 'lr': lr_init_spatial},
            {'params': self.density_plane, 'lr': lr_init_spatial},
            {'params': self.app_line, 'lr': lr_init_spatial},
            {'params': self.app_plane, 'lr': lr_init_spatial},
            {'params': self.basis_mat.parameters(), 'lr': lr_init_network},
            {'params': self.render_module.parameters(), 'lr': lr_init_network}
        ]
        return grad_vars