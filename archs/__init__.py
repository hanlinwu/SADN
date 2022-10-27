import importlib
import os
from os import path as osp

from litsr.archs import (
    load_or_create_net,
    load_net,
    create_net,
    freeze,
    unfreeze,
    ArchRegistry,
)


# Import all archs
arch_folder = osp.dirname(osp.abspath(__file__))
for root, dirs, files in os.walk(arch_folder):
    for f in files:
        if f.endswith("_arch.py"):
            importlib.import_module(f"archs.{osp.splitext(f)[0]}")
