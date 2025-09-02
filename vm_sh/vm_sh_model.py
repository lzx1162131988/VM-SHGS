"""
Python package for combining 3DGS with volume rendering to enable medium modeling using VM-SH framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from vm_sh._torch_impl import quat_to_rotmat
from vm_sh.project_gaussians import project_gaussians
from vm_sh.rasterize import rasterize_gaussians
from vm_sh.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from vm_sh.medium_vm import MediumTensorVM, MediumTensorVMSplit


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@dataclass
class VMSHModelConfig(ModelConfig):
    """VM-SH Model Config"""

    _target: Type = field(default_factory=lambda: VMSHModel)
    num_steps: int = 15000
    """Number of steps to train the model"""
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    reset_alpha_thresh: float = 0.5
    """threshold of opacity for resetting alpha"""
    cull_scale_thresh: float = 10.
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    zero_medium: bool = False
    """If True, zero out the medium field"""
    reset_alpha_every: int = 5
    """Every this many refinement steps, reset the alpha"""
    abs_grad_densification: bool = True
    """If True, use absolute gradient for densification"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians (0.0004, 0.0008)"""
    densify_size_thresh: float = 0.001
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    clip_thresh: float = 0.01
    """minimum depth threshold"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    main_loss: Literal["l1", "reg_l1", "reg_l2"] = "reg_l1"
    """main loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim"] = "reg_ssim"
    """ssim loss to use"""
    
    # 复合损失函数权重配置
    depth_loss_weight: float = 0.1
    """深度监督损失权重"""
    pose_consistency_weight: float = 0.05
    """相对位姿一致性损失权重"""
    tv_loss_weight: float = 0.05
    """总变差损失权重"""
    recon_loss_weight: float = 0.8
    """重建损失权重"""
    stop_split_at: int = 10000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    
    # Tensor medium reconstruction options
    use_tensor_medium: bool = True
    """Whether to use tensor decomposition for medium reconstruction."""
    medium_tensor_type: Literal["VM", "VMSplit"] = "VM"
    """Type of tensor decomposition for medium reconstruction."""
    medium_grid_size: Tuple[int, int, int] = (128, 128, 128)
    """Grid size for medium tensor decomposition."""
    medium_density_n_comp: int = 8
    """Number of density components for medium tensor."""
    medium_app_n_comp: int = 24
    """Number of appearance components for medium tensor."""
    medium_app_dim: int = 27
    """Dimension of appearance features for medium tensor."""
    medium_view_pe: int = 6
    """View direction positional encoding levels for medium tensor."""
    medium_feature_c: int = 128
    """Feature dimension for medium tensor MLP."""


class VMSHModel(Model):
    """
    Args:
        config: VM-SH configuration to instantiate model
    """

    config: VMSHModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # Initialize tensor-based medium reconstruction
        CONSOLE.log(f"Using {self.config.medium_tensor_type} tensor medium reconstruction")
        
        # Compute scene bbox for tensor initialization
        if self.seed_points is not None and self.seed_points[0].shape[0] > 0:
            points = self.seed_points[0]
            aabb_min = points.min(dim=0)[0] - 1.0
            aabb_max = points.max(dim=0)[0] + 1.0
            aabb = torch.stack([aabb_min, aabb_max])
        else:
            # Default AABB for random initialization
            aabb = torch.tensor([[-self.config.random_scale/2] * 3, 
                               [self.config.random_scale/2] * 3], dtype=torch.float32)
        
        if self.config.medium_tensor_type == "VM":
            self.medium_tensor = MediumTensorVM(
                aabb=aabb,
                grid_size=self.config.medium_grid_size,
                device=self.device,
                density_n_comp=self.config.medium_density_n_comp,
                appearance_n_comp=self.config.medium_app_n_comp,
                app_dim=self.config.medium_app_dim,
                view_pe=self.config.medium_view_pe,
                feature_c=self.config.medium_feature_c
            )
        else:  # VMSplit
            self.medium_tensor = MediumTensorVMSplit(
                aabb=aabb,
                grid_size=self.config.medium_grid_size,
                device=self.device,
                density_n_comp=self.config.medium_density_n_comp,
                appearance_n_comp=self.config.medium_app_n_comp,
                app_dim=self.config.medium_app_dim,
                view_pe=self.config.medium_view_pe,
                feature_c=self.config.medium_feature_c
            )

        if self.seed_points is not None and not self.config.random_init:
            # 检查seed_points是否有效
            if self.seed_points[0].shape[0] > 0:
                means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
                CONSOLE.log(f"Using {self.seed_points[0].shape[0]} COLMAP 3D points for initialization")
            else:
                CONSOLE.log("Warning: COLMAP 3D points are empty! Falling back to random initialization")
                means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
            CONSOLE.log(f"Using random initialization with {self.config.num_random} points")
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = self.config.num_steps
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        # if self.step >= self.config.stop_split_at:
        #     return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.abs_grad_densification:
                assert self.xys_grad_abs is not None
                grads = self.xys_grad_abs.detach().norm(dim=-1)
            else:
                assert self.xys.grad is not None
                grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.depths_accum = self.depths
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]
                self.depths_accum[visible_mask] = self.depths[visible_mask] + self.depths_accum[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % reset_interval > self.num_train_data + self.config.refine_every)
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])

                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()

                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads

                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads

                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # if self.step < self.config.stop_screen_size_at:
                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )                
                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
    
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

                
            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:                
                # Reset value is set to be reset_alpha_thresh
                reset_value = self.config.reset_alpha_thresh
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            
            self.xys_grad_norm = None
            self.vis_counts = None
            self.depths_accum = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        if self.step < self.config.stop_split_at:
            cull_alpha_thresh = self.config.cull_alpha_thresh
        else:
            cull_alpha_thresh = self.config.cull_alpha_thresh_post
        culls = (torch.sigmoid(self.opacities) < cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        
        # Add tensor medium parameters
        medium_params = self.medium_tensor.get_optparam_groups()
        for i, param_group in enumerate(medium_params):
            group_name = f"medium_group_{i}"
            gps[group_name] = param_group["params"]
        
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        self.last_fx = camera.fx.item()
        self.last_fy = camera.fy.item()

        # Medium computation using tensor decomposition
        # Create 3D grid coordinates for the entire image
        y = torch.linspace(0., H, H, device=self.device)
        x = torch.linspace(0., W, W, device=self.device)
        yy, xx = torch.meshgrid(y, x)
        yy = (yy - cy) / camera.fy.item()
        xx = (xx - cx) / camera.fx.item()
        directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
        directions = directions / norms
        directions = directions @ R.T
        
        # Sample points along rays for medium reconstruction
        # Use simplified sampling - can be made more sophisticated
        num_samples = 64
        near, far = 0.1, 10.0  # Reasonable bounds for medium scenes
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=self.device)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        
        # Broadcast to match image shape
        z_vals = z_vals[None, None, :].expand(H, W, -1)  # [H, W, num_samples]
        
        # Convert to world coordinates
        rays_o = T.squeeze(-1)[None, None, :].expand(H, W, -1)  # [H, W, 3]
        rays_d = directions  # [H, W, 3]
        
        # Sample points: rays_o + z_vals * rays_d
        pts = rays_o[..., None, :] + z_vals[..., :, None] * rays_d[..., None, :]  # [H, W, num_samples, 3]
        
        # Flatten for tensor processing
        pts_flat = pts.reshape(-1, 3)  # [H*W*num_samples, 3]
        dirs_flat = directions.reshape(-1, 3)  # [H*W, 3]
        dirs_flat = dirs_flat[:, None, :].expand(-1, num_samples, -1).reshape(-1, 3)  # [H*W*num_samples, 3]
        
        # Forward through tensor medium reconstruction
        with torch.no_grad():  # Reduce memory for now
            medium_rgb_flat, medium_bs_flat, medium_attn_flat = self.medium_tensor(
                pts_flat, dirs_flat
            )
        
        # Reshape back to image space and integrate along rays
        medium_rgb_vol = medium_rgb_flat.reshape(H, W, num_samples, 3)  # [H, W, num_samples, 3]
        medium_bs_vol = medium_bs_flat.reshape(H, W, num_samples, 3)
        medium_attn_vol = medium_attn_flat.reshape(H, W, num_samples, 3)
        
        # Simple integration - take weighted average or use volume rendering
        # For now, use distance-weighted average
        weights = torch.softmax(-z_vals, dim=-1)  # Closer samples get higher weight
        medium_rgb = torch.sum(weights[..., None] * medium_rgb_vol, dim=2)  # [H, W, 3]
        medium_bs = torch.sum(weights[..., None] * medium_bs_vol, dim=2)
        medium_attn = torch.sum(weights[..., None] * medium_attn_vol, dim=2)
            
        if self.config.zero_medium:
            medium_rgb = torch.zeros_like(medium_rgb)
            medium_bs = torch.zeros_like(medium_bs)
            medium_attn = torch.zeros_like(medium_attn)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = medium_rgb
                depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                        "rgb_object": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                        "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}
        else:
            crop_ids = None

        if crop_ids is not None and crop_ids.sum() != 0:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
            clip_thresh=self.config.clip_thresh,
        )  # type: ignore

        self.depths = depths.detach()
        
        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = medium_rgb
            depth = medium_rgb.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = medium_rgb.new_zeros(*rgb.shape[:2], 1)
            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": medium_rgb, 
                    "rgb_object": torch.zeros_like(rgb), "rgb_clear": torch.zeros_like(rgb), "rgb_clear_clamp": torch.zeros_like(rgb), "rgb_medium": medium_rgb, "pred_image": rgb,
                    "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}

        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)
        
        self.xys_grad_abs = torch.zeros_like(self.xys)

        rgb_object, rgb_clear, rgb_medium, depth_im, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            self.xys_grad_abs,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            medium_rgb,
            medium_bs,
            medium_attn,
            H,
            W,
            BLOCK_WIDTH,
            background=medium_rgb,
            return_alpha=True,
            step=self.step,
        )  # type: ignore
        
        rgb = rgb_object + rgb_medium
        rgb_clear_clamp = torch.clamp(rgb_clear, 0., 1.)
        rgb_clear = rgb_clear / (rgb_clear + 1.)
        
        depth_im = depth_im[..., None]
        alpha = alpha[..., None]
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())  
                 
        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": medium_rgb, 
                "rgb_object": rgb_object, "rgb_clear": rgb_clear, "rgb_clear_clamp": rgb_clear_clamp, "rgb_medium": rgb_medium, "pred_image": rgb,
                "medium_rgb": medium_rgb, "medium_bs": medium_bs, "medium_attn": medium_attn}  # type: ignore
        
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            # alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return image[..., :3]
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        for i in range(3):
            # 3 channels
            metrics_dict[f"medium_attn_{i}"] = outputs["medium_attn"][:, :, i].mean()
            metrics_dict[f"medium_bs_{i}"] = outputs["medium_bs"][:, :, i].mean()
            metrics_dict[f"medium_rgb_{i}"] = outputs["medium_rgb"][:, :, i].mean()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """计算和返回损失字典。
        
        实现复合损失函数，包含四个主要组成部分：
        1. 重建损失 (Reconstruction Loss)
        2. 深度监督损失 (Depth Supervision Loss)
        3. 相对位姿一致性损失 (Relative Pose Consistency Loss)
        4. 总变差损失 (Total Variation Loss)

        Args:
            outputs: 模型输出结果
            batch: 对应的真值数据批次
            metrics_dict: 指标字典，其中某些可用于损失计算
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["pred_image"]

        # 设置遮罩部分为黑色（如果存在遮罩）
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # ========== 1. 重建损失 (Reconstruction Loss) ==========
        # 使用L2损失进行图像重建
        recon_loss = torch.mean((gt_img - pred_img) ** 2)
        
        # SSIM损失作为重建损失的一部分
        if self.config.ssim_loss != "ssim":
            simloss = 1 - self.ssim((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...], (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
        else:
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        
        # 组合构成完整的重建损失
        reconstruction_loss = (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss
        
        # ========== 2. 深度监督损失 (Depth Supervision Loss) ==========
        depth_loss = torch.tensor(0.0, device=self.device)
        if "depth" in batch and "depth" in outputs:
            gt_depth = batch["depth"].to(self.device)
            pred_depth = outputs["depth"]
            
            # 确保尺寸匹配
            if gt_depth.shape != pred_depth.shape:
                gt_depth = self._downscale_if_required(gt_depth)
            
            # 使用有效深度掩码（排除无效深度值）
            valid_mask = (gt_depth > 0) & (gt_depth < 100)  # 假设有效深度范围
            if valid_mask.sum() > 0:
                # 使用 L1 损失或相对损失
                depth_loss = torch.abs(gt_depth[valid_mask] - pred_depth[valid_mask]).mean()
                
        # ========== 3. 相对位姿一致性损失 (Relative Pose Consistency Loss) ==========
        pose_consistency_loss = torch.tensor(0.0, device=self.device)
        # 这里可以实现相对位姿一致性约束
        # 例如：相邻视角间的几何一致性、深度一致性等
        if "camera_indices" in batch and len(batch["camera_indices"]) > 1:
            # 这里是一个简化的示例，可以根据具体需求实现
            # 例如：计算相邻视角间渲染结果的一致性
            pass
        
        # ========== 4. 总变差损失 (Total Variation Loss) ==========
        # 根据公式: L_TV = (1/|V|) * Σ sqrt(Δx² + Δy² + Δz²)
        tv_loss = torch.tensor(0.0, device=self.device)
        
        # 对渲染图像应用2D总变差正则化
        if pred_img.dim() == 3:  # [H, W, C]
            # 计算水平和垂直方向的梯度
            delta_h = pred_img[1:, :, :] - pred_img[:-1, :, :]  # Δy
            delta_w = pred_img[:, 1:, :] - pred_img[:, :-1, :]  # Δx
            
            # 对于2D图像，使用2D总变差: sqrt(Δx² + Δy²)
            # 需要处理尺寸匹配问题
            min_h, min_w = min(delta_h.shape[0], delta_w.shape[0]), min(delta_h.shape[1], delta_w.shape[1])
            delta_h_crop = delta_h[:min_h, :min_w, :]
            delta_w_crop = delta_w[:min_h, :min_w, :]
            
            tv_2d = torch.sqrt(delta_h_crop**2 + delta_w_crop**2 + 1e-8).mean()
            tv_loss += tv_2d
        
        # 对3D密度场应用3D总变差正则化（按您的公式）
        if hasattr(self, 'medium_tensor') and hasattr(self.medium_tensor, 'density_tensor'):
            # 获取3D密度张量
            density_field = self.medium_tensor.density_tensor  # 假设形状为 [X, Y, Z] 或 [X, Y, Z, 1]
            
            if density_field.dim() >= 3:
                # 确保是3D张量 [X, Y, Z]
                if density_field.dim() == 4:
                    density_field = density_field.squeeze(-1)
                
                X, Y, Z = density_field.shape
                
                # 计算三个方向的梯度
                # Δx方向梯度
                delta_x = density_field[1:, :, :] - density_field[:-1, :, :]  # [X-1, Y, Z]
                # Δy方向梯度  
                delta_y = density_field[:, 1:, :] - density_field[:, :-1, :]  # [X, Y-1, Z]
                # Δz方向梯度
                delta_z = density_field[:, :, 1:] - density_field[:, :, :-1]  # [X, Y, Z-1]
                
                # 找到公共维度以计算3D梯度范数
                min_x = min(delta_x.shape[0], delta_y.shape[0], delta_z.shape[0])
                min_y = min(delta_x.shape[1], delta_y.shape[1], delta_z.shape[1])
                min_z = min(delta_x.shape[2], delta_y.shape[2], delta_z.shape[2])
                
                # 裁剪到相同尺寸
                delta_x_crop = delta_x[:min_x, :min_y, :min_z]
                delta_y_crop = delta_y[:min_x, :min_y, :min_z]
                delta_z_crop = delta_z[:min_x, :min_y, :min_z]
                
                # 计算3D总变差: sqrt(Δx² + Δy² + Δz²)
                tv_3d = torch.sqrt(delta_x_crop**2 + delta_y_crop**2 + delta_z_crop**2 + 1e-8)
                
                # 归一化：除以体素总数 |V|
                volume_size = min_x * min_y * min_z
                tv_3d_normalized = tv_3d.sum() / volume_size
                
                tv_loss += tv_3d_normalized
        
        # 备选方案：如果从 outputs 中获取密度场
        elif "medium_density" in outputs:
            medium_density = outputs["medium_density"]
            if medium_density.dim() >= 3:
                # 确保是3D张量
                if medium_density.dim() == 4:
                    medium_density = medium_density.squeeze(-1)
                
                X, Y, Z = medium_density.shape
                
                # 计算三个方向的梯度 (按您的公式)
                delta_x = medium_density[1:, :, :] - medium_density[:-1, :, :]
                delta_y = medium_density[:, 1:, :] - medium_density[:, :-1, :]
                delta_z = medium_density[:, :, 1:] - medium_density[:, :, :-1]
                
                # 找到公共维度
                min_x = min(delta_x.shape[0], delta_y.shape[0], delta_z.shape[0])
                min_y = min(delta_x.shape[1], delta_y.shape[1], delta_z.shape[1])
                min_z = min(delta_x.shape[2], delta_y.shape[2], delta_z.shape[2])
                
                # 裁剪到相同尺寸
                delta_x_crop = delta_x[:min_x, :min_y, :min_z]
                delta_y_crop = delta_y[:min_x, :min_y, :min_z]
                delta_z_crop = delta_z[:min_x, :min_y, :min_z]
                
                # 计算3D总变差: L_TV = (1/|V|) * Σ sqrt(Δx² + Δy² + Δz²)
                tv_3d = torch.sqrt(delta_x_crop**2 + delta_y_crop**2 + delta_z_crop**2 + 1e-8)
                volume_size = min_x * min_y * min_z
                tv_3d_normalized = tv_3d.sum() / volume_size
                
                tv_loss += tv_3d_normalized
        
        # ========== 组合所有损失 ==========
        total_loss = (
            self.config.recon_loss_weight * reconstruction_loss + 
            self.config.depth_loss_weight * depth_loss +
            self.config.pose_consistency_weight * pose_consistency_loss +
            self.config.tv_loss_weight * tv_loss
        )
        
        # 返回各个损失组成部分以便监控
        return {
            "main_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "depth_loss": depth_loss,
            "pose_consistency_loss": pose_consistency_loss,
            "tv_loss": tv_loss,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device), obb_box=obb_box)
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])

        predicted_rgb = outputs["pred_image"]
        predicted_rgb = torch.clamp(predicted_rgb, 0.0, 1.0)

        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(predicted_rgb.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = predicted_rgb

        output_gt_rgb = gt_rgb.cpu()

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"gt": output_gt_rgb, "rgb_medium": outputs["rgb_medium"], "rgb_object": outputs["rgb_object"], "depth": outputs["depth"], "rgb": outputs["rgb"], "rgb_clear": outputs["rgb_clear"]}
        return metrics_dict, images_dict
