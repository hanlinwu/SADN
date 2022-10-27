import os
import pickle
import re
from os import path as osp

import torch.utils.data as data
from litsr.utils import load_file_list, read_image
from tqdm import tqdm
from scipy import io as scio
from litsr.utils.io_utils import make_lmdb_from_imgs, imfrombytes


def load_mat(path):
    mat = scio.loadmat(path, verify_compressed_data_integrity=False)
    return mat["t"]


class ImageFolder(data.Dataset):
    """
    Construct a dataset from a folder
    """

    def __init__(self, datapath, repeat=1, cache=None, first_k=None, data_length=None):
        self.datapath = datapath
        self.repeat = repeat or 1
        self.cache = cache
        self.data_length = data_length

        if cache == "bin":
            [self.filenames, self.files] = self.load_data_from_bin(first_k)
        elif cache == "memory":
            [self.filenames, self.files] = self.load_data_from_memory(first_k)
        elif cache == "lmdb":
            [self.filenames, self.files] = self.load_data_from_lmdb(first_k)
        else:
            [self.filenames, self.files] = self.load_data()

        if first_k:
            self.filenames = self.filenames[:first_k]
            self.files = self.files[:first_k]

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "memory":
            return x
        elif self.cache == "bin":
            with open(x, "rb") as f:
                x = pickle.load(f)
            return x
        elif self.cache == "lmdb":
            with self.lmdb_client.begin(write=False) as txn:
                img_bytes = txn.get(x.encode("ascii"))
            return imfrombytes(img_bytes)
        else:
            return read_image(x)

    def load_data_from_lmdb(self, first_k=None):
        try:
            import lmdb
        except ImportError:
            raise ImportError("Please install lmdb to enable LmdbBackend.")

        lmdb_path = osp.join(
            osp.dirname(self.datapath), osp.basename(self.datapath) + ".lmdb"
        )
        file_names = load_file_list(self.datapath, regx=".*.png")
        file_names = [osp.basename(n) for n in file_names]
        keys = [osp.splitext(f)[0] for f in file_names]
        if not osp.exists(lmdb_path):
            make_lmdb_from_imgs(
                self.datapath, lmdb_path, file_names, keys, multiprocessing_read=True
            )

        self.lmdb_client = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False
        )

        return file_names, keys

    def load_data_from_bin(self, first_k=None):
        file_names = load_file_list(self.datapath, regx=".*.png")
        file_names = [osp.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        bin_root = osp.join(
            osp.dirname(self.datapath), "_bin_" + osp.basename(self.datapath)
        )
        if not osp.exists(bin_root):
            os.mkdir(bin_root)
            print("mkdir", bin_root)
        files = []

        for f in file_names:
            bin_file = osp.join(bin_root, f.split(".")[0] + ".pkl")
            if not osp.exists(bin_file):
                with open(bin_file, "wb") as bin_f:
                    pickle.dump(
                        read_image(osp.join(self.datapath, f)),
                        bin_f,
                    )
                print("dump", bin_file)
            files.append(bin_file)
        return file_names, files

    def load_data_from_memory(self, first_k=None, from_pickle=True):
        if from_pickle:
            file_names, filenames = self.load_data_from_bin(first_k)
            files = []
            pbar = tqdm(filenames)
            pbar.set_description("load data (from pickle)")
            for f in pbar:
                with open(f, "rb") as f_:
                    files.append(pickle.load(f_))

            return file_names, files

        file_names = load_file_list(self.datapath, regx=".*.png")
        file_names = [osp.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        files = []
        pbar = tqdm(file_names)
        pbar.set_description("load data")
        for f in pbar:
            files.append(read_image(osp.join(self.datapath, f)))

        return file_names, files

    def load_data(self):
        files = load_file_list(self.datapath, regx=".*.png")
        file_names = [osp.basename(n) for n in files]
        return file_names, files

    def __len__(self):
        if self.data_length:
            return self.data_length
        return int(len(self.files) * self.repeat)


class PairedImageFolder(data.Dataset):
    def __init__(
        self, path1, path2, repeat=1, cache=None, first_k=None, data_length=None
    ):
        self.dataset_1 = ImageFolder(
            path1, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )
        self.dataset_2 = ImageFolder(
            path2, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )

        self.filenames = self.dataset_2.filenames

        assert len(self.dataset_1) == len(self.dataset_2)

    def __getitem__(self, i):
        return tuple([self.dataset_1[i], self.dataset_2[i]])

    def __len__(self):
        return len(self.dataset_1)


class ConcatDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class HSImageFolder(data.Dataset):
    """
    Construct a hyper-spectral image dataset from a folder
    """

    def __init__(self, datapath, repeat=1, cache=None, first_k=None, data_length=None):
        self.datapath = datapath
        self.repeat = repeat or 1
        self.cache = cache
        self.data_length = data_length

        if cache == "bin":
            [self.filenames, self.files] = self.load_data_from_bin(first_k)
        elif cache == "memory":
            [self.filenames, self.files] = self.load_data_from_memory(first_k)
        else:
            [self.filenames, self.files] = self.load_data()

        if first_k:
            self.filenames = self.filenames[:first_k]
            self.files = self.files[:first_k]

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == "memory":
            return x
        elif self.cache == "bin":
            with open(x, "rb") as f:
                x = pickle.load(f)
            return x
        else:
            return load_mat(x)

    def load_data_from_bin(self, first_k=None):
        file_names = load_file_list(self.datapath, regx=".*.mat")
        file_names = [osp.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        bin_root = osp.join(
            osp.dirname(self.datapath), "_bin_" + osp.basename(self.datapath)
        )
        if not osp.exists(bin_root):
            os.mkdir(bin_root)
            print("mkdir", bin_root)
        files = []

        for f in file_names:
            bin_file = osp.join(bin_root, f.split(".")[0] + ".pkl")
            if not osp.exists(bin_file):
                with open(bin_file, "wb") as bin_f:
                    pickle.dump(
                        load_mat(osp.join(self.datapath, f)),
                        bin_f,
                    )
                print("dump", bin_file)
            files.append(bin_file)
        return file_names, files

    def load_data_from_memory(self, first_k=None, from_pickle=True):
        if from_pickle:
            file_names, filenames = self.load_data_from_bin(first_k)
            files = []
            pbar = tqdm(filenames)
            pbar.set_description("load data (from pickle)")
            for f in pbar:
                with open(f, "rb") as f_:
                    files.append(pickle.load(f_))

            return file_names, files

        file_names = load_file_list(self.datapath, regx=".*.mat")
        file_names = [osp.basename(n) for n in file_names]

        if first_k:
            file_names = file_names[:first_k]

        files = []
        pbar = tqdm(file_names)
        pbar.set_description("load data")
        for f in pbar:
            files.append(load_mat(osp.join(self.datapath, f)))

        return file_names, files

    def load_data(self):
        files = load_file_list(self.datapath, regx=".*.mat")
        file_names = [osp.basename(n) for n in files]
        return file_names, files

    def __len__(self):
        if self.data_length:
            return self.data_length
        return int(len(self.files) * self.repeat)


class PairedHSImageFolder(data.Dataset):
    def __init__(
        self, path1, path2, repeat=1, cache=None, first_k=None, data_length=None
    ):
        self.dataset_1 = HSImageFolder(
            path1, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )
        self.dataset_2 = HSImageFolder(
            path2, repeat=repeat, cache=cache, first_k=first_k, data_length=data_length
        )

        self.filenames = self.dataset_2.filenames

        assert len(self.dataset_1) == len(self.dataset_2)

    def __getitem__(self, i):
        return tuple([self.dataset_1[i], self.dataset_2[i]])

    def __len__(self):
        return len(self.dataset_1)
