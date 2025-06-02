TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=13314 --use_env run_train.py \
