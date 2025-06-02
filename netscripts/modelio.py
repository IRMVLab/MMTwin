import os
import shutil
import traceback
import warnings
import torch


def load_checkpoint(model, resume_path, strict=True, device=None):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path)
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = {
                "module.{}".format(key): item
                for key, item in checkpoint["state_dict"].items()}
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint["epoch"]))
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    return checkpoint["epoch"]


def load_checkpoint_by_name(model, resume_path, state_dict_name, strict=True, device=None):
    if os.path.isfile(resume_path):
        print("=> loading "+state_dict_name+" from checkpoint '{}'".format(resume_path))
        if device is not None:
            checkpoint = torch.load(resume_path, map_location=device)
        else:
            checkpoint = torch.load(resume_path)
        if "epoch" not in checkpoint.keys():
            checkpoint["epoch"] = 1000
        if "module" in list(checkpoint[state_dict_name].keys())[0]:
            state_dict = checkpoint[state_dict_name]
        else:
            state_dict = {
                "module.{}".format(key): item
                for key, item in checkpoint[state_dict_name].items()}
            print("=> loaded "+state_dict_name+" checkpoint '{}' (epoch {})".format(resume_path, checkpoint["epoch"]))
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    return checkpoint["epoch"]

def load_optimizer(optimizer, resume_path, state_dict_name='optimizer', device=None):
    if os.path.isfile(resume_path):
        print("=> loading "+state_dict_name+" from checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        optimizer.load_state_dict(checkpoint)
    else:
        raise ValueError("=> no optimizer checkpoint found at '{}'".format(resume_path))

def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
