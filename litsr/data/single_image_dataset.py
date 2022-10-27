import torch.utils.data as data
from litsr import transforms
from litsr.data.image_folder import ImageFolder
from litsr.utils.registry import DatasetRegistry


@DatasetRegistry.register()
class SingleImageDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        mean=None,
        std=None,
        return_img_name=False,
    ):

        self.dataset = ImageFolder(
            img_path, repeat=repeat, cache=cache, first_k=first_k
        )
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        img = self.dataset[idx]

        img = transforms.uint2single(img)
        img = transforms.single2tensor(img) * self.rgb_range

        if self.mean and self.std:
            transforms.normalize(img, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return img, file_name
        else:
            return img

    def __len__(self):
        return len(self.dataset)
