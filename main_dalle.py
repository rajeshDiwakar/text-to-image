'''
python main_dalle.py --train --data '/home/rajesh/work/limbo/text-to-image/dataset' --weight sessions/test2/weights.pth
'''


import os,sys,time, glob
import torch, torchvision
from dalle_pytorch import DiscreteVAE, DALLE

from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import argparse

from torch.utils.tensorboard import SummaryWriter

import warnings

from dataset import VideoDataset
warnings.simplefilter("ignore")

from drive import upload_to_drive

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
    dataset = VideoDataset(args.data)
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=1)
    dataset_loader_test = torch.utils.data.DataLoader(dataset,
                                                 batch_size=32, shuffle=True,
                                                 num_workers=1)

    assert len(dataset_loader)>0, 'no image found in data dir'

    sess_dir = os.path.join('sessions',args.session)
    output_image_dir = os.path.join(sess_dir,'output')
    os.makedirs(sess_dir,exist_ok=True)
    os.makedirs(output_image_dir,exist_ok=True)
    weight_path = os.path.join(sess_dir,args.weight)

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
    vae.load_state_dict(torch.load(args.weight, map_location=device))
    # vae = DiscreteVAE(
    # image_size = 256,
    # num_layers = 3,
    # num_tokens = 8192,
    # codebook_dim = 1024,
    # hidden_dim = 64,
    # num_resnet_blocks = 1,
    # temperature = 0.9
    # )

    dalle = DALLE(
        dim = 256,#1024,
        vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
        num_text_tokens = len(dataset.vocab), #10000,    # vocab size for text
        text_seq_len = 256,         # text sequence length
        depth = 2,#12,                 # should aim to be 64
        heads = 16,                 # attention heads
        dim_head = 64,              # attention head dimension
        attn_dropout = 0.1,         # attention dropout
        ff_dropout = 0.1            # feedforward dropout
    )

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)
    # dataset_size = len(dataset_loader)
    # assert dataset_size>= batch_size, 'dataset (%d) < batch_size(%d) '%(len(dataset_loader), batch_size)
    dataset_size = len(dataset_loader) # number of batches in one epoch
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
            text = batch[1].to(device)
            mask = torch.ones_like(text).bool()

            optimizer.zero_grad()
            # text = torch.randint(0, 10000, (4, 256))
            # images = torch.randn(4, 3, 256, 256)
            # loss = vae(images, return_recon_loss = True)
            loss = dalle(text, images, mask = mask, return_loss = True)
            loss.backward()
            optimizer.step()
            sys.stdout.write('\r[%s] %6d/%6d: loss: %f'%(time.asctime(),it,args.epochs*dataset_size,loss.item()))
            running_loss += loss.item()
            writer.add_scalar('training loss',running_loss / (it+1),epoch * dataset_size+ it)
            if it%args.save_every==(args.save_every-1):
                torch.save(vae.state_dict(), weight_path)
                try:
                    if args.drive_id:
                        files = glob.glob(os.path.join(summary_dir,'*'))
                        upload_to_drive(files,args.drive_id)
                    # upload_to_drive(files+image_list,args.drive_id)
                    # image_list = []
                except Exception as e:
                    print('Error while uploading to drive\n',str(e))


            if it % args.test_every == (args.test_every-1):

                with torch.no_grad():
                    for i, batch in enumerate(dataset_loader_test):
                        # images = batch[0].to(device)
                        text = batch[1].to(device)
                        # images_pred = vae.forward(images)
                        images_pred = dalle.forward(text)

                        images = torchvision.utils.make_grid(images)
                        images_pred = torchvision.utils.make_grid(images_pred)

                        images = torch.cat((imags,images_pred),-1)
                        img = images
                        img = img / 2 + 0.5     # unnormalize
                        npimg = img.cpu().numpy()
                        fig = plt.figure(figsize=(12,9))
                        plt.imshow(np.transpose(npimg, (1, 2, 0)))

                        # impath = os.path.join(output_image_dir,'iter-%d_img-%d.jpg'%(epoch*dataset_size+it,i))
                        # plt.savefig( impath )
                        # image_list.append(impath)

                        # writer.add_figure('generated images',
                        #         fig,
                        #         global_step=epoch * dataset_size + it)
                        writer.add_video('generated images',
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
    vae.load_state_dict(torch.load(args.weight, map_location=device))

    # load dataset
    dataset = datasets.ImageFolder(root='data/images',
                                               transform=data_transform)
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
    # parser.add_argument('--drive_id',default='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A')
    parser.add_argument('--drive_id',default=None)

    parser.add_argument('--image_size',type=int,default=64)
    parser.add_argument('--num_layers',type=int,default=1)
    parser.add_argument('--num_tokens',type=int,default=1024)
    parser.add_argument('--codebook_dim',type=int,default=256)
    parser.add_argument('--hidden_dim',type=int,default=64)

    parser.add_argument('--test',action='store_true',default=False,help='for testing use --test')  #not really required
    parser.add_argument('--weight',default='weights.pth')

    args = parser.parse_args()
    if not args.test and not args.train:
        print('Either --test or --train must be specified')
        exit(1)

    if args.train:
        train(args)
    else:
        test(args)
