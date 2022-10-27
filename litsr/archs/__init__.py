import importlib
import os
from os import path as osp
from collections import OrderedDict
import torch
from litsr.utils.registry import ArchRegistry
from litsr.utils.logger import logger

__all__ = ["create_net", "load_net", "load_or_create_net", "freeze", "unfreeze"]

# Import all archs
arch_folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(arch_folder):
    for f in files:
        if f.endswith("_arch.py"):
            importlib.import_module(f"litsr.archs.{osp.splitext(f)[0]}")


def create_net(opt):
    """Create Network from option"""
    name = opt["name"]
    args = opt["args"]

    Net = ArchRegistry.get(name)
    net = Net(**args)

    return net


def default_state_dict_getter(state_dict):
    state_dict = state_dict["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("sr_net"):
            k = k.replace("sr_net.", "")
            new_state_dict[k] = v
        if k.startswith("net_G"):
            k = k.replace("net_G.", "")
            new_state_dict[k] = v
        if k.startswith("net_D"):
            k = k.replace("net_D.", "")
            new_state_dict[k] = v
    return new_state_dict


def load_net(opt, state_dict_getter=default_state_dict_getter):
    """Load network parameters from chekcpoint"""
    if not opt.get("pretrained_path"):
        raise Exception("pretrained path not given")

    state_dict = torch.load(opt.pretrained_path, map_location=torch.device("cpu"))
    state_dict = state_dict_getter(state_dict)
    net = create_net(opt)
    net.load_state_dict(state_dict, strict=True)
    logger.info(
        "Model {0} loaded from {1}".format(net.__class__.__name__, opt.pretrained_path)
    )
    return net


def load_or_create_net(opt, state_dict_getter=default_state_dict_getter):
    if not opt.get("pretrained_path"):
        return create_net(opt)
    else:
        return load_net(opt, state_dict_getter)


def freeze(module):
    if not isinstance(module, torch.nn.Module):
        raise ValueError("input must be torch.nn.Module")
    for p in module.parameters():
        p.requires_grad = False
    logger.info("Model {0} has been frozen. ".format(module.__class__.__name__))


def unfreeze(module):
    if not isinstance(module, torch.nn.Module):
        raise ValueError("input must be torch.nn.Module")
    for p in module.parameters():
        p.requires_grad = True
    logger.info("Model {0} has been unfrozen. ".format(module.__class__.__name__))
