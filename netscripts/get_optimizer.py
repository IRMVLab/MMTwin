import torch

class Warmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            init_lr_ratio: float = 0.0,
            num_epochs: int = 5,
            last_epoch: int = -1,
            iters_per_epoch: int = None,
):
        self.base_scheduler = scheduler
        self.warmup_iters = max(num_epochs * iters_per_epoch, 1)
        if self.warmup_iters > 1:
            self.init_lr_ratio = init_lr_ratio
        else:
            self.init_lr_ratio = 1.0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        assert self.last_epoch < self.warmup_iters
        return [
            el * (self.init_lr_ratio + (1 - self.init_lr_ratio) *
                  (float(self.last_epoch) / self.warmup_iters))
            for el in self.base_lrs
        ]

    def step(self, *args, **kwargs):
        if self.last_epoch < (self.warmup_iters - 1):
            super().step(*args, **kwargs)
        else:
            self.base_scheduler.step(*args, **kwargs)


def get_optimizer(args, pre_encoder, model_denoise, homo_denoise, post_decoder, train_loader, motion_encoder=None, loc_encoder=None, glip_encoder=None, voxel_encoder=None, occ_feat_encoder=None):
    assert train_loader is not None, "train_loader is None, " \
                                     "warmup or cosine learning rate need number of iterations in dataloader"
    iters_per_epoch = len(train_loader)
    pre_encoder_params = [p for p_name, p in pre_encoder.named_parameters() if  p.requires_grad]
    model_denoise_params = [p for p_name, p in model_denoise.named_parameters() if  p.requires_grad]
    homo_denoise_params = [p for p_name, p in homo_denoise.named_parameters() if  p.requires_grad]
    post_decoder_params = [p for p_name, p in post_decoder.named_parameters() if  p.requires_grad]
    motion_encoder_params = [p for p_name, p in motion_encoder.named_parameters() if  p.requires_grad]
    glip_encoder_params = [p for p_name, p in glip_encoder.named_parameters() if  p.requires_grad]
    voxel_encoder_params = [p for p_name, p in voxel_encoder.named_parameters() if  p.requires_grad]
    occ_feat_encoder_params = [p for p_name, p in occ_feat_encoder.named_parameters() if  p.requires_grad]
    loc_encoder_params = [p for p_name, p in loc_encoder.named_parameters() if  p.requires_grad]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([{'params': model_denoise_params, 'weight_decay': 0.0}, {'params': pre_encoder_params}, {'params': post_decoder_params}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "rms":
        optimizer = torch.optim.RMSprop([{'params': model_denoise_params, 'weight_decay': 0.0}, {'params': pre_encoder_params}, {'params': post_decoder_params}],
                                        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD([{'params': model_denoise_params, 'weight_decay': 0.0}, {'params': pre_encoder_params}, {'params': post_decoder_params}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params': model_denoise_params, 'weight_decay': 0.0}, 
                                       {'params': homo_denoise_params, 'weight_decay': 0.0},
                                       {'params': pre_encoder_params}, 
                                       {'params': post_decoder_params}, 
                                       {'params': motion_encoder_params}, 
                                       {'params': loc_encoder_params}, 
                                       {'params': glip_encoder_params}, 
                                       {'params': voxel_encoder_params}, 
                                       {'params': occ_feat_encoder_params},], 
                                       lr=args.lr)
    else:
        raise ValueError("unsupported optimizer type")

    for group in optimizer.param_groups:
        group["lr"] = args.lr
        group["initial_lr"] = args.lr

    print("args.scheduler -----------> ", args.scheduler)
    if args.scheduler == "step":
        assert isinstance(args.lr_decay_step, int), "learning rate scheduler need integar lr_decay_step"
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.scheduler == "multistep":
        if isinstance(args.lr_decay_step, list):
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay_step, gamma=args.lr_decay_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.epochs // 2, gamma=0.1)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*iters_per_epoch,
                                                               last_epoch=-1, eta_min=0)
    else:
        raise ValueError("Unrecognized learning rate scheduler {}".format(args.scheduler))

    main_scheduler = Warmup(optimizer, scheduler, init_lr_ratio=0., num_epochs=args.warmup_epochs,
                            iters_per_epoch=iters_per_epoch)
    return optimizer, main_scheduler




