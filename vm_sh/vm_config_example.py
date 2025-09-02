"""
Configuration for VM-SH with Tensor Medium Reconstruction
This demonstrates the tensor decomposition configuration for medium reconstruction.
"""

from vm_sh.vm_sh_config import VMSHModelConfig

# Example configuration with tensor medium reconstruction
tensor_vm_config = VMSHModelConfig(
    # Tensor medium parameters
    medium_tensor_type="VM",  # or "VMSplit"
    medium_grid_size=(64, 64, 64),  # Start with smaller grid for memory
    medium_density_n_comp=8,
    medium_app_n_comp=24,
    medium_app_dim=27,
    medium_view_pe=6,
    medium_feature_c=128,
    
    # Standard VM-SH parameters
    num_steps=15000,
    warmup_length=500,
    refine_every=100,
    resolution_schedule=3000,
    background_color="black",
    num_downscales=2,
    
    # Gaussian parameters
    cull_alpha_thresh=0.5,
    densify_grad_thresh=0.0008,
    densify_size_thresh=0.001,
    
    # Rendering parameters
    sh_degree=3,
    rasterize_mode="classic",
    
    # Loss parameters
    ssim_lambda=0.2,
    main_loss="reg_l1",
    ssim_loss="reg_ssim",
    
    # 复合损失函数权重配置
    recon_loss_weight=0.8,       # 重建损失权重
    depth_loss_weight=0.1,       # 深度监督损失权重  
    pose_consistency_weight=0.05, # 相对位姿一致性损失权重
    tv_loss_weight=0.05,         # 总变差损失权重
)