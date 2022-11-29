import os

import cv2
import numpy as np
import torch
import torch.utils.data

from random import sample

from joint_semantic_segmentation.utils import rotate

__all__ = ['SemanticSegmentationDataset']


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, mask_flags, img_dir, mask_dir, img_ext='.jpg', mask_ext='.jpg', num_classes=5, contrast_learn=False):
        """
        Args:
            img_ids (list): Image ids/files.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (function, optional): Rotation transform function
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.png
            │   ├── 0aab0a.png
            │   ├── 0b1761.png
            │   ├── ...
            |
            └── masks
            |   ├── 0a7e06.png
            |   ├── 0aab0a.png
            |   ├── 0b1761.png
            |   ├── ...
            |
        """
        self.img_ids = img_ids
        self.mask_flags = mask_flags
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.contrast_learn = contrast_learn

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, 'image' + img_id + self.img_ext),cv2.IMREAD_GRAYSCALE)
        img = img.astype('uint8')

        mask = None
        if self.mask_flags[idx]:
            mask = cv2.imread(os.path.join(self.mask_dir, 'mask' + img_id + self.mask_ext),cv2.IMREAD_GRAYSCALE)
            mask = mask.astype('uint8')
            #mask = mask.transpose(2, 0, 1)
            class_masks = [np.where(mask == clss, 1, 0) for clss in [0, 100, 150, 200, 250]]
            mask = np.array(class_masks)

        if self.contrast_learn:
            rotated_imgs = [rotate(img, rot) for rot in [0, 90, 180, 270]]
            img = np.array(rotated_imgs)
        else:
            img = img.transpose(2, 0, 1)
        
        return img, mask