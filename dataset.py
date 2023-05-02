import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

from utils import crop_sample, pad_sample, resize_sample, normalize_volume


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "val", "test", "demo"]

        # read images
        volumes = []
        masks = []
        for (dirpath, dirnames, filenames) in os.walk(
            os.path.join(images_dir, subset, "images")
        ):

            filenames = sorted(filter(lambda x: ".jpeg" in x, filenames))
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                volumes.append(filepath)

                maskpath = "mask_images".join(filepath.rsplit("images", 1)).replace(
                    ".jpeg", ".png"
                )
                masks.append(maskpath)

        # create list of tuples (volume, mask)
        self.data = list(zip(volumes, masks))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        self.random_sampling = random_sampling

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len((self.data))

    def __getitem__(self, idx):
        imagepath, maskpath = self.data[idx]

        image = np.array(imread(imagepath))
        mask = np.array(imread(maskpath, as_gray=True))

        # if len(mask.shape) < 3:
        #     mask = np.stack((mask,mask,mask), axis=-1)

        # crop to smallest enclosing volume
        # image, mask = crop_sample((image, mask))

        # # pad to square
        (image, mask) = pad_sample((image, mask))

        # resize
        (image, mask) = resize_sample((image, mask), size=self.image_size)

        # normalize channel-wise
        (image) = normalize_volume(image)

        # add channel dimension to masks
        mask = mask[..., np.newaxis]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor
