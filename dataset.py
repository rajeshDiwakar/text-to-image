from __future__ import print_function, division
import os
import glob
import json
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class VideoDataset(Dataset):
    """text to video dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        video_paths = glob.glob(os.path.join(root_dir,'*'))
        self.meta = []
        for p in video_paths:
            vid = os.path.basename(p)
            with open(p+'/chunks.json') as f:
                caption = json.load(f)
                for cap in caption:
                    self.meta.append({'vid':vid,"text":cap['context'],'frames':cap['frames']})
        print('Number of samples:',len(self.meta))


    def __len__(self):
        # raise NotImplementedError
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('type idx:',type(idx))
        images = []
        meta = self.meta[idx]
        vid = meta['vid']
        for frame in meta['frames'][:5]:
            img_name = os.path.join(self.root_dir,vid,'images',
                                    '%d.jpg'%frame)
            image = io.imread(img_name)
            images.append(image)
        if len(images)<5:
            images = images+[images[-1] for _ in range(5-len(images))]
        sample = (np.array(images),meta['text'])

        if self.transform:
            raise NotImplementedError
            sample = self.transform(sample)

        return sample

if __name__== '__main__':

    dataset = VideoDataset(root_dir='/home/rajesh/work/limbo/text-to-image/dataset',
                                               transform=None)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1, shuffle=True,
                                                 num_workers=1)
    for i,batch in enumerate(dataset_loader):
        if i>0: break
        print(type(batch[0]),type(batch[1]))
        print(batch[0].shape)
        print(batch[1])
