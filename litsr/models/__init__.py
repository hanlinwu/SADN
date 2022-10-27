import os
import importlib
from os import path as osp
from litsr.utils.registry import ModelRegistry

__all__ = ["create_model", "load_model"]

# Import all models
model_folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(model_folder):
    for f in files:
        if f.endswith("_model.py"):
            importlib.import_module(f"litsr.models.{osp.splitext(f)[0]}")


def create_model(opt):
    """create model from option"""
    model_name = opt.lit_model.get("name")
    if not model_name:
        return
    model = ModelRegistry.get(model_name)
    return model(opt)


def load_model(opt, ckpt_path=None, strict=True, overwrite_hparams=True):
    """load model from checkpoint path"""
    model_name = opt.lit_model.get("name")
    if not model_name:
        return
    model = ModelRegistry.get(model_name)
    if overwrite_hparams:
        model = model.load_from_checkpoint(ckpt_path, opt=opt, strict=strict)
    else:
        model = model.load_from_checkpoint(ckpt_path, strict=strict)
    print("Model loaded from {0}".format(ckpt_path))
    return model
