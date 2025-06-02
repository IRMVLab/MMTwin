"""
Repository: https://github.com/IRMVLab/MMTwin
Paper: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction
Authors: Ma et.al.

This file contains how to build diffusion models, encoders, and decoders for MMTwin.
"""

from models import gaussian_diffusion as gd
from models.gaussian_diffusion import SpacedDiffusion, space_timesteps
from models.denoising_model import HOIMamba_homo, HOIMambaTransformer
from models.pre_encoder import PreEncoder, MotionEncoder, LocEncoder, GLIPEncoder, VoxelEncoder, VoxelFeatEncoder
from models.post_decoder import PostDecoder


def create_model_and_diffusion(
    vl_dim,
    patch_dim,
    loc_feat_dim,
    glip_feat_dim,
    target_dim,
    n_layers,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    **kwargs,
):

    pre_encoder =  PreEncoder(input_dims=(glip_feat_dim+loc_feat_dim), output_dims=target_dim, encoder_hidden_dims=64)
    post_decoder =  PostDecoder(input_dims=target_dim, output_dims=3, encoder_hidden_dims1=256, encoder_hidden_dims2=64)
    motion_encoder =  MotionEncoder(input_dims=3*3, output_dims=target_dim, encoder_hidden_dims=64)
    loc_encoder =  LocEncoder(3, hidden_features=256, out_features=target_dim)
    # hard coding 7x12
    glip_encoder =  GLIPEncoder(input_dims_conv=vl_dim, output_dims_conv=target_dim, input_dims=7*12, output_dims=1)
    voxel_encoder =  VoxelEncoder(input_dims=1, output_dims=patch_dim)
    occ_feat_encoder =  VoxelFeatEncoder(input_dims=patch_dim, output_dims=target_dim, encoder_hidden_dims=64) 
    denoising_model = HOIMambaTransformer(d_model=2*target_dim, input_dims=target_dim,
                                          output_dims=target_dim, hidden_t_dim=target_dim, n_layers=n_layers)

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return denoising_model, pre_encoder, diffusion, post_decoder, motion_encoder, loc_encoder, glip_encoder,voxel_encoder, occ_feat_encoder


def homo_create_model_and_diffusion(
    feat_dim,
    n_layers,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    **kwargs,
):

    denoising_model = HOIMamba_homo(d_model=feat_dim, input_dims=feat_dim,
                                    output_dims=feat_dim, hidden_t_dim=feat_dim,
                                    n_layers=n_layers,)

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return denoising_model, diffusion