import json
import os
import re
import shutil
import sys
from multiprocessing import Pool
from os import path as osp
from pathlib import Path
from typing import Iterable, Optional

import cv2
import lmdb
import numpy as np
import yaml
from easydict import EasyDict as Dict
from litsr.transforms import uint2single
from tqdm import tqdm


def read_json(fname):
    """Read json from path"""
    fname = Path(fname)
    with fname.open("rt") as handle:
        return Dict(json.load(handle))


def write_json(content, fname):
    """Write json to path"""
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    """Real yaml from path

    Args:
        fname (str): path

    Returns:
        Easydict: Easydict
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return Dict(yaml.load(handle, Loader=yaml.Loader))


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir_clean(path):
    """Create a directory that is guaranteed to be empty"""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            os.makedirs(path)
        except:
            print("warning! Cannot remove {0}".format(path))
    else:
        os.makedirs(path)


def load_file_list(path: str, regx: str) -> list:
    file_list = os.listdir(path)
    return_list = []
    for _, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(os.path.join(path, f))
    return sorted(return_list)


def read_image(
    path: str, mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:
    """Read image to a 3 dimentional numpy by OpenCV"""
    img = cv2.imread(path)
    assert mode in ("RGB", "BGR")
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_float:
        img = uint2single(img)
    return img


def read_images(
    path_list: Iterable[str], mode: Optional[str] = "RGB", to_float: bool = False
) -> np.ndarray:
    """Read images to a 4 dimentional numpy array by OpenCV"""
    rslt = []
    for path in tqdm(path_list):
        rslt.append(read_image(path, mode, to_float))
    return np.array(rslt)


def make_lmdb_from_imgs(
    data_path,
    lmdb_path,
    img_path_list,
    keys,
    batch=5000,
    compress_level=1,
    multiprocessing_read=False,
    n_thread=40,
    map_size=None,
):
    """Make lmdb from images.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), (
        "img_path_list and keys should have the same length, "
        f"but got {len(img_path_list)} and {len(keys)}"
    )
    print(f"Create lmdb for {data_path}, save to {lmdb_path}...")
    print(f"Totoal images: {len(img_path_list)}")
    if not lmdb_path.endswith(".lmdb"):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f"Folder {lmdb_path} already exists. Exit.")
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f"Read images with multiprocessing, #thread: {n_thread} ...")
        pbar = tqdm(total=len(img_path_list), unit="image")

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f"Read {key}")

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback,
            )
        pool.close()
        pool.join()
        pbar.close()
        print(f"Finish reading {len(img_path_list)} images.")

    # create lmdb environment
    # if map_size is None:
    #     # obtain data size for one image
    #     img = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
    #     print(osp.join(data_path, img_path_list[0]))
    #     _, img_byte = cv2.imencode(
    #         ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    #     )
    #     data_size_per_img = img_byte.nbytes
    #     print("Data size per image is: ", data_size_per_img)
    #     data_size = data_size_per_img * len(img_path_list)
    #     map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=1024 ** 4)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit="chunk")
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, "meta_info.txt"), "w")
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f"Write {key}")
        key_byte = key.encode("ascii")
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level
            )
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f"{key}.png ({h},{w},{c}) {compress_level}\n")
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print("\nFinish writing lmdb.")


def read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode(
        ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]
    )
    return (key, img_byte, (h, w, c))


def imfrombytes(content, mode="RGB", to_float=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if to_float:
        img = uint2single(img)
    return img
