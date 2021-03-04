#!/usr/bin/env python
# coding: utf-8
'''
mkdir dall_e/data/
wget -O dall_e/data/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl
wget -O dall_e/data/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl
'''


import io
import os, sys
import requests
import PIL
# sys.path.append('..')
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dall_e import map_pixels, unmap_pixels, load_model
# from IPython.display import display, display_markdown
import torch.nn.functional as F
import gc
import numpy

class Dalle(object):
    def __init__(self,target_image_size=256,proc_image_size=256,enc=None,dec=None,device='cpu'):
        assert proc_image_size in [128,256,64]
        self.target_image_size = target_image_size
        self.proc_image_size = proc_image_size

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc, self.dec = None, None
        self.device = torch.device(device)
        if enc:
            self.enc = load_model(enc, dev)
            self.enc.to(self.device)
        if dec:
            self.dec = load_model(dec, dev)
            self.dec.to(self.device)

# def download_image(url):
#     resp = requests.get(url)
#     resp.raise_for_status()
#     return PIL.Image.open(io.BytesIO(resp.content))

# def preprocess(img):
#     s = min(img.size)
#
#     if s < target_image_size:
#         raise ValueError(f'min dim for image {s} < {target_image_size}')
#
#     r = target_image_size / s
#     s = (round(r * img.size[1]), round(r * img.size[0]))
#     img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
#     img = TF.center_crop(img, output_size=2 * [target_image_size])
#     img = torch.unsqueeze(T.ToTensor()(img), 0)
#     return map_pixels(img)

    def preprocess(self,img):
        s = min(img.size)

        if s < self.proc_image_size:
            raise ValueError(f'min dim for image {s} < {self.proc_image_size}')

        r = self.proc_image_size / s
        s = (round(r * img.size[1]), round(r * img.size[0]))
        img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
        img = TF.center_crop(img, output_size=2 * [self.proc_image_size])
        if self.target_image_size != self.proc_image_size:
            num_pad = int(self.target_image_size-self.proc_image_size)//2
            img = TF.pad(img,(num_pad,num_pad,num_pad,num_pad),padding_mode='constant',fill=0)
        img = torch.unsqueeze(T.ToTensor()(img), 0)
        return map_pixels(img)

    # add batch size
    def encode(self,img):
        '''
        img is pil image
        '''
        if not self.enc:
            raise Exception('Encoder not intialised')

        if type(img) == str:
            img = PIL.Image.open(img)
        elif type(img) == numpy.ndarray:
            img = T.ToPILImage()(img)
        x = self.preprocess(img) #[b c h w]
        x.to(self.device)
        with torch.no_grad():
            z_logits = self.enc(x) #[enc(x) for x in xs]
        z=torch.argmax(z_logits,axis=1)

        if self.target_image_size == self.proc_image_size:
            del x,z_logits
            gc.collect()
            return z.numpy()

        # z_ = torch.zeros_like(z)
        # z_[:] = 4522
        num_pad = (self.target_image_size-self.proc_image_size)//16 #=2*8 , 2 for padding, 8 for num_layers=3
        # z_[:,num_pad:-num_pad,num_pad:-num_pad] = z[:,num_pad:-num_pad,num_pad:-num_pad]
        ret = z[:,num_pad:-num_pad,num_pad:-num_pad].numpy()
        del x,z_logits, z
        gc.collect()
        return ret

    def decode(self,codes,pad=4522):
        '''
        codes is [bXnXn] n can be 8,16,32. It is a tensor
        '''
        if type(codes) == list:
            codes = torch.tensor(codes)
        if type(codes) == numpy.ndarray:
            codes = torch.from_numpy(codes)

        if len(codes.shape)==2:
            codes = torch.unsqueeze(0)
        b,n = codes.shape[:2]
        if n==8 or n==16:
            l = int((32-n)/2)
            codes_pad = torch.zeros(b,32,32)
            codes_pad[:] = pad
            codes_pad[:,l:-l,l:-l] = codes
        elif n != 32:
            raise ValueError
        # print(codes_pad)
        codes_pad = codes_pad.long()
        z = F.one_hot(codes_pad, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float().to(self.device)
        with torch.no_grad():
            x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))[0]
        x_rec = x_rec.permute(1,2,0)
        print(x_rec.shape)
        ret = x_rec.detach().numpy()
        if n==16:
            ret = ret[64:-64,64:-64,:]
        elif n==8:
            ret = ret[32:-32,32:-32,:]
        del x_rec,x_stats
        gc.collect()
        # x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
        return ret

    def to(self,device):
        self.device = device
        if self.enc:
            self.enc.to(device)
        if self.dec:
            self.dec.to(device)






if __name__ == '__main__':

    img = 'dall_e/data/test.png'
    enc = 'dall_e/data/encoder.pkl'
    dec = 'dall_e/data/decoder.pkl'

    import cv2
    img = cv2.imread(img)
    de = Dalle(enc=enc,dec=dec,proc_image_size=128)
    img_enc = de.encode(img)
    print(str(img_enc)[:200])
    print(img_enc.shape)

    img_rec = de.decode(img_enc)
    print(str(img_rec)[:200])
    print(img_rec.shape)
    cv2.imshow('Reconstructed',img_rec)
    cv2.waitKey(0)







# # x = preprocess(download_image('https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg'))
# # img = PIL.Image.open('/home/rajesh/work/data/storiesgan/data/49.png')
# # x = preprocess_small(img)
# # display_markdown('Original image:')
# # display(T.ToPILImage(mode='RGB')(x[0]))
#
#
#
# files = ['/home/rajesh/work/data/storiesgan/data/%s.png'%i for i in ['35','36','37','49']][:1]
# imgs =[ PIL.Image.open(f) for f in files]
# xs =[ preprocess_small(img) for img in imgs]
# xs = torch.cat(xs,dim=0)
# z_logits = enc(xs) #[enc(x) for x in xs]
# z=torch.argmax(z_logits,axis=1)
# z_ = torch.zeros_like(z)
# z_[:] = 4522
# z_[:,8:-8,8:-8] = z[:,8:-8,8:-8]
# print('z',z.shape)
# # diff = z_logits
#
#
#
# # torch.set_printoptions(edgeitems=8)
#
#
#
# import torch.nn.functional as F
#
# z_logits = enc(x)
# z = torch.argmax(z_logits, axis=1)
#
# z_ = torch.zeros_like(z)
# z_[:] = 4522
# z_[:,8:-8,8:-8] = z[:,8:-8,8:-8]
# z = z_
#
# z = F.one_hot(z, num_classes=enc.vocab_size).permute(0, 3, 1, 2).float()
#
# x_stats = dec(z).float()
# x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
# x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
#
# display_markdown('Reconstructed image:')
# display(x_rec)
#
#
# del z_logits
# del z
# del z_
# del x_rec
# del x_stats
# del x
# del img
# gc.collect()
#
#
# # In[ ]:
