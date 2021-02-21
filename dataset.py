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
from collections import defaultdict
from einops import rearrange
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import spacy
nlp = spacy.load('en_core_web_sm')

from torchvision import transforms, datasets

'''
mean = torch.tensor([1, 2, 3], dtype=torch.float32)
std = torch.tensor([2, 2, 2], dtype=torch.float32)

normalize = T.Normalize(mean.tolist(), std.tolist())

unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
'''



class ImageDataset(Dataset):
    """text to video dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.meta =[None]*100
        # self.vocab = [None]*100
        # return
        self.root_dir = root_dir
        self.transform = transform
        self.images = glob.glob(os.path.join(root_dir,'*/images/*'))
        # self.meta = []
        # self.vocab = defaultdict(int)
        # _ = self.vocab['<BOS>'] # should be 0
        # self.pad = self.vocab['<PAD>']
        #
        # for p in video_paths:
        #     vid = os.path.basename(p)
        #     with open(p+'/chunks.json') as f:
        #         caption = json.load(f)
        #         for cap in caption:
        #             text = cap['context']
        #             text_ids = [self.vocab[tok.text] for tok in nlp.tokenizer(text)]
        #             self.meta.append({'vid':vid,"text":text,'text_ids':text_ids,'frames':cap['frames']})
        #
        # self.meta = self.meta[:100]
        print('Number of images:',len(self.images))

        # if os.path.isfile('vocab.json'):
        #     input('overwriting vocab.json. Please back up')
        # with open('vocab.json','w') as f:
        #     json.dump(self.vocab,f)


    def __len__(self):
        # raise NotImplementedError
        return len(self.images)

    def __getitem__(self, idx):
        # text = torch.randint(0, 100, (30,))
        # images = torch.randn(3, 64, 64)
        # return images,text
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('type idx:',type(idx))
        # images = []
        # meta = self.meta[idx]
        # vid = meta['vid']
        # for frame in meta['frames'][:5]:
        #     img_name = os.path.join(self.root_dir,vid,'images',
        #                             '%d.jpg'%frame)
        #     image = io.imread(img_name)
        #     images.append(image)
        # if len(images)<5:
        #     images = images+[images[-1] for _ in range(5-len(images))]
        images = io.imread(self.images[idx])
        images = np.array(images,dtype=np.float32)
        images = images/128.0 - 1
        images = rearrange(images,'h w c -> c h w')
        # text_ids = meta['text_ids'][:30]
        # if len(text_ids) < 30:
        #     text_ids += [self.pad for _ in range(30-len(text_ids))]
        sample = (torch.from_numpy(images),[0])

        if self.transform:
            raise NotImplementedError
            sample = self.transform(sample)

        return sample


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
        # self.meta =[None]*100
        # self.vocab = [None]*100
        # return
        self.root_dir = root_dir
        self.transform = transform
        video_paths = glob.glob(os.path.join(root_dir,'*'))
        self.meta = []
        self.vocab = defaultdict(int)
        _ = self.vocab['<BOS>'] # should be 0
        self.pad = self.vocab['<PAD>']

        for p in video_paths:
            vid = os.path.basename(p)
            with open(p+'/chunks.json') as f:
                caption = json.load(f)
                for cap in caption:
                    text = cap['context']
                    text_ids = [self.vocab[tok.text] for tok in nlp.tokenizer(text)]
                    self.meta.append({'vid':vid,"text":text,'text_ids':text_ids,'frames':cap['frames']})

        self.meta = self.meta
        print('Number of samples:',len(self.meta))
        print('Vocab:',len(self.vocab))

        # if 0 and  os.path.isfile('vocab.json'):
        #     input('overwriting vocab.json. Please back up')
        with open('vocab.json','w') as f:
            json.dump(self.vocab,f)


    def __len__(self):
        # raise NotImplementedError
        return len(self.meta)

    def __getitem__(self, idx):
        # text = torch.randint(0, 100, (30,))
        # images = torch.randn(5,3 ,64, 64)
        # return images,text
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

        images = np.array(images,dtype=np.float32)
        images = images/128.0 - 1
        images = rearrange(images,'t h w c -> t c h w')

        text_ids = meta['text_ids'][:30]
        if len(text_ids) < 30:
            text_ids += [self.pad for _ in range(30-len(text_ids))]
        sample = (torch.from_numpy(images),torch.tensor(text_ids).long())

        if self.transform:
            raise NotImplementedError
            sample = self.transform(sample)

        return sample

def test_videodataset():
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

def test_defaultdataset():
    data_transform = transforms.Compose([
            transforms.Scale((64,64)),
            # transforms.RandomSizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    dataset = datasets.ImageFolder(root='/home/rajesh/work/limbo/text-to-image/dataset/7IoF9IrZnXU',
                                               transform=data_transform) # SHAPE IS bchw
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=16, shuffle=True,
                                                 num_workers=1)

    for i,batch in enumerate(dataset_loader):
        if i>0: break
        print(type(batch[0]),type(batch[1]))
        print(batch[0].shape) # shape is B C H W
        print(batch[1])
        print(batch[0])

if __name__== '__main__':

    # test_defaultdataset()
    test_videodataset()
