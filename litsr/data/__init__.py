# import all data module

import importlib
import os
from os import path as osp

from litsr.utils.registry import DataModuleRegistry, DatasetRegistry

from .downsampled_dataset import (
    DownsampledDataset,
    DownsampledDatasetMS,
)
from .image_folder import (
    HSImageFolder,
    ImageFolder,
    PairedHSImageFolder,
    PairedImageFolder,
)
from .paired_image_dataset import PairedImageDataset
from .single_image_dataset import SingleImageDataset
from .sr_datamodule import SRDataModule

__all__ = ["create_data_module"]


def create_data_module(opt):
    """create model from option"""
    data_module_name = opt.get("name")
    if not data_module_name:
        return
    data_module = DataModuleRegistry.get(data_module_name)
    return data_module(opt.args)


# Import all datamodules
folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(folder):
    for f in files:
        if f.endswith("_datamodule.py"):
            importlib.import_module(f"litsr.data.{osp.splitext(f)[0]}")
