def add_exp_opts(parser):
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true",
                        help="evaluate model")
    parser.add_argument("--snapshot", default=10, type=int, metavar="N",
                        help="How often to take a snapshot of the model")
    parser.add_argument("--use_cuda", default=1, type=int, 
                        help="use GPU (default: True)")
    parser.add_argument("--test_novel", default=True, type=bool, 
                        help="test on novel scenes")
    parser.add_argument("--test_space", default="2d", type=str, 
                        help="test on 2D/3D spaces")
    parser.add_argument("--log_dir", default="./log", type=str, 
                        help="test on novel scenes")


def add_data_opts(parser):
    parser.add_argument('--data_path', type=str, default="/data/HTPdata")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_frames', type=int, default=40)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--target', type=str, default="3d")
    parser.add_argument('--scenes', default=None)
    parser.add_argument('--modalities', default=['loc'])
    parser.add_argument('--tinyset', type=bool, default=False)
    parser.add_argument('--load_all', type=bool, default=True)
    parser.add_argument('--use_odom', type=bool, default=True)
    parser.add_argument('--centralize', type=bool, default=True)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--glip_feats_path', type=str, default='/data/HTPdata/EgoPAT3D-postproc/glip_feats')
    parser.add_argument('--motion_feats_path', type=str, default='/data/HTPdata/EgoPAT3D-postproc/motion_feats')
    parser.add_argument('--voxel_path', type=str, default='/data/HTPdata/EgoPAT3D-postproc/egopat_voxel_filtered')
    parser.add_argument('--voxel_res', type=float, default=0.05)
    parser.add_argument('--grid_size', type=int, default=20)
    parser.add_argument('--origin_xyz', default=[-0.5, -0.5, 0.0])


def add_nets_opts(parser):
    # for HTP diffusion
    parser.add_argument('--vl_dim', type=int, default=256)
    parser.add_argument('--patch_dim', type=int, default=64)
    parser.add_argument('--loc_feat_dim', type=int, default=1024)
    parser.add_argument('--glip_feat_dim', type=int, default=1024)
    parser.add_argument('--target_dim', type=int, default=1024)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--infer_gap', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default="sqrt")
    parser.add_argument('--learn_sigma', type=bool, default=False)
    parser.add_argument('--timestep_respacing', type=str, default="")
    parser.add_argument('--predict_xstart', type=bool, default=True)
    parser.add_argument('--rescale_timesteps', type=bool, default=True)
    parser.add_argument('--sigma_small', type=bool, default=False)
    parser.add_argument('--rescale_learned_sigmas', type=bool, default=False)
    parser.add_argument('--use_kl', type=bool, default=False)

    # for Egomotion diffusion
    parser.add_argument('--em_feat_dim', type=int, default=1024)
    parser.add_argument('--em_n_layers', type=int, default=6)
    parser.add_argument('--em_diffusion_steps', type=int, default=1000)
    parser.add_argument('--em_infer_gap', type=int, default=1000)
    parser.add_argument('--em_noise_schedule', type=str, default="sqrt")
    parser.add_argument('--em_learn_sigma', type=bool, default=False)
    parser.add_argument('--em_timestep_respacing', type=str, default="")
    parser.add_argument('--em_predict_xstart', type=bool, default=True)
    parser.add_argument('--em_rescale_timesteps', type=bool, default=True)
    parser.add_argument('--em_sigma_small', type=bool, default=False)
    parser.add_argument('--em_rescale_learned_sigmas', type=bool, default=False)
    parser.add_argument('--em_use_kl', type=bool, default=False)

def add_traineval_opts(parser):
    parser.add_argument("--manual_seed", default=1, type=int)
    parser.add_argument("-j", "--workers", default=16, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--optimizer", default="adamw", choices=["rms", "adam", "sgd", "adamw"])
    parser.add_argument("--lr", "--learning-rate", default=0.5e-4, type=float, metavar="LR")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--scheduler", default="cosine", choices=['cosine', 'step', 'multistep'])
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--lr_decay_step", nargs="+", default=10, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--schedule_sampler", default="lossaware", type=str)
    parser.add_argument("--losses_homo_w", default=5e-4, type=float)
    parser.add_argument("--losses_reg_w", default=0.2, type=float)
    parser.add_argument("--losses_angle_w", default=0.01, type=float)
