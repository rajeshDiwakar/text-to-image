'''
python main_vae.py --train --session test3 --epochs 10 --save_every 10 --test_every 5 --data data/images --drive_id $drive_parent_id --batch_size 256
python main_vae.py --train --session test3 --epochs 10 --save_every 10 --test_every 5 --data data/images --drive_id $drive_parent_id --batch_size 16
python main_vae.py --train --session test3 --epochs 10 --save_every 10 --test_every 5 --data /home/rajesh/work/limbo/text-to-image/dataset  --batch_size 16
'''


import os,sys,time, glob
import torch, torchvision
from dalle_pytorch import DiscreteVAE

from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import argparse

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter("ignore")

from drive import MyDrive
from dataset import ImageDataset

# def upload_to_drive(*args,**kwargs):
#     pass
mdrive = MyDrive()
upload_to_drive = mdrive=upload_to_drive

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data_transform = transforms.Compose([
        transforms.Scale((64,64)),
        # transforms.RandomSizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

def train(args):

    batch_size = args.batch_size
    # dataset = datasets.ImageFolder(root=args.data,
    #                                            transform=data_transform)
    dataset = ImageDataset(args.data,transform=None)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=4)
    dataset_loader_test = torch.utils.data.DataLoader(dataset,
                                                 batch_size=min(batch_size,32), shuffle=True,
                                                 num_workers=1)

    assert len(dataset_loader)>0, 'no image found in data dir'

    sess_dir = os.path.join('sessions',args.session)
    output_image_dir = os.path.join(sess_dir,'output')
    os.makedirs(sess_dir,exist_ok=True)
    os.makedirs(output_image_dir,exist_ok=True)
    weight_vae = os.path.join(sess_dir,args.weight_vae)

    summary_dir = os.path.join(sess_dir,'summary')
    writer = SummaryWriter(summary_dir)
    vae = DiscreteVAE(
        image_size = args.image_size,
        num_layers = args.num_layers,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens = args.num_tokens,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim = args.codebook_dim,      # codebook dimension
        hidden_dim = args.hidden_dim,         # hidden dimension
        num_resnet_blocks = 1,   # number of resnet blocks
        temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
    )
    vae.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)
    # dataset_size = len(dataset_loader)
    # assert dataset_size>= batch_size, 'dataset (%d) < batch_size(%d) '%(len(dataset_loader), batch_size)
    dataset_size = len(dataset_loader)
    print("Batch per iteration: ",dataset_size)
    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*dataset_size,20*dataset_size], gamma=0.1)
    image_list = []
    for epoch in range(args.epochs):
        running_loss = 0.0

        for it,batch in enumerate(dataset_loader):
            # images = torch.randn(4, 3, 256, 256)
            it = epoch*dataset_size + it
            images = batch[0].to(device)
            optimizer.zero_grad()
            loss = vae(images, return_recon_loss = True)
            loss.backward()
            optimizer.step()
            sys.stdout.write('\r[%s] %6d/%6d: loss: %f'%(time.asctime(),it,args.epochs*dataset_size,loss.item()))
            running_loss += loss.item()
            writer.add_scalar('training loss',running_loss / (it+1),epoch * dataset_size+ it)
            if it%args.save_every==(args.save_every-1):
                torch.save(vae.state_dict(), weight_vae)
                try:
                    files = glob.glob(os.path.join(summary_dir,'*'))
                    upload_to_drive(files,args.drive_id)
                    # upload_to_drive(files+image_list,args.drive_id)
                    # image_list = []
                except Exception as e:
                    print('Error while uploading to drive\n',str(e))


            if it % args.test_every == (args.test_every-1):

                with torch.no_grad():
                    for i, batch in enumerate(dataset_loader_test):
                        batch = batch[0].to(device)
                        batch_pred = vae.forward(batch)

                        batch = torchvision.utils.make_grid(batch)
                        batch_pred = torchvision.utils.make_grid(batch_pred)

                        batch = torch.cat((batch,batch_pred),-1)
                        img = batch
                        img = img / 2 + 0.5     # unnormalize
                        npimg = img.cpu().numpy()
                        fig = plt.figure(figsize=(12,9))
                        plt.imshow(np.transpose(npimg, (1, 2, 0)))
                        # impath = os.path.join(output_image_dir,'iter-%d_img-%d.jpg'%(epoch*dataset_size+it,i))
                        # plt.savefig( impath )
                        # image_list.append(impath)
                        writer.add_figure('generated images',
                                fig,
                                global_step=epoch * dataset_size + it)

                        if i>= 0:
                            break
        sys.stdout.write('\n[%s]Epoch:%d loss: %f\n'%(' '.join(time.asctime().split(' ')[1:-1]),epoch,running_loss/dataset_size)) # some samples are going for testing??
        scheduler.step()




def test(args):

    vae = DiscreteVAE(
        image_size = args.image_size,
        num_layers = args.num_layers,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens = args.num_tokens,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim = args.codebook_dim,      # codebook dimension
        hidden_dim = args.hidden_dim,         # hidden dimension
        num_resnet_blocks = 1,   # number of resnet blocks
        temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
    )
    vae.to(device)
    sess_dir = os.path.join('sessions',args.session)
    weight_vae = os.path.join(sess_dir,args.weight_vae)

    vae.load_state_dict(torch.load(weight_vae, map_location=device))

    # load dataset
    # dataset = datasets.ImageFolder(root='data/images',
    #                                            transform=data_transform)
    dataset = ImageDataset(args.data,transform=None)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4, shuffle=False,
                                                 num_workers=1)

    with torch.no_grad():
        for batch in dataset_loader:
            batch = batch[0].to(device)
            batch_pred = vae.forward(batch)

            batch = torchvision.utils.make_grid(batch)
            batch_pred = torchvision.utils.make_grid(batch_pred)

            batch = torch.cat((batch,batch_pred),1)
            imshow(batch)
            # imshow(torchvision.utils.make_grid(batch))
            # imshow(torchvision.utils.make_grid(batch_pred))
            y = input('quite [y/n]?')
            if y.lower() == 'y':
                break

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true',default=False,help='For training use --train')
    parser.add_argument('--session',default='sess1')
    parser.add_argument('--remark',default='this is a test')
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--save_every',default=100,type=int)
    parser.add_argument('--test_every',default=100,type=int)
    parser.add_argument('--resume',default=0,type=int)
    parser.add_argument('--data',default='data',help='data dir. data/classes/img.jpg')
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--drive_id',default='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A')

    parser.add_argument('--image_size',type=int,default=64)
    parser.add_argument('--num_layers',type=int,default=5)
    parser.add_argument('--num_tokens',type=int,default=1024)
    parser.add_argument('--codebook_dim',type=int,default=256)
    parser.add_argument('--hidden_dim',type=int,default=64)

    parser.add_argument('--test',action='store_true',default=False,help='for testing use --test')  #not really required
    parser.add_argument('--weight_vae',default='vae.pth')

    args = parser.parse_args()
    if not args.test and not args.train:
        print('Either --test or --train must be specified')
        exit(1)

    if args.train:
        train(args)
    else:
        test(args)