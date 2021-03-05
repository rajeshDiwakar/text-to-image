'''
python main_dalle.py --train --data '/home/rajesh/work/limbo/text-to-image/dataset' --weight sessions/test2/weights.pth
timelimit -t 600 -T 600 python main_dalle.py --train --data '/home/rajesh/work/limbo/text-to-image/dataset' --sess test3 --test_every 4 --epochs 2 --save_every 4 --batch_size 32

timelimit -t 600 -T 600 python main_dalle.py --train --data '/home/rajesh/work/limbo/text-to-image/dataset' --sess test11_l1 --test_every 1 --epochs 2 --save_every 2 --batch_size 8 --codebook_dim 512 --num_tokens 2048 --num_layers 3
timelimit -t 600 -T 600 python main_dalle.py --train --data '/home/rajesh/work/limbo/text-to-image/dataset' --sess test1 --test_every 1 --epochs 2 --save_every 2 --batch_size 8
'''


import os,sys,time, glob,json
import torch, torchvision
# from dalle_pytorch import DiscreteVAE, DALLE
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
import argparse
import warnings

from dataset import VideoDataset, GPTDataset, GPTDatasetSplit
warnings.simplefilter("ignore")

from drive import MyDrive

mdrive = MyDrive()
upload_to_drive = mdrive.upload_to_drive
# def upload_to_drive(*args,**kwargs):
#     pass

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
    dataset = GPTDataset(args.data,context_size=args.context_size)
    dataset.shuffle()
    # dataset_size = len(dataset)
    # split into train test
    train_size = int(len(dataset)*0.85)
    test_size = len(dataset)-train_size

    train_dataset = GPTDatasetSplit(dataset,offset=0,len=train_size)
    test_dataset = GPTDatasetSplit(dataset,offset=train_size,len=test_size)
    del train_size,test_size
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size, shuffle=True,collate_fn=GPTDataset.collate_fn,
                                                 num_workers=2)
    # test_dataset = GPTDataset(args.test_data,context_size=args.context_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=min(batch_size,32), shuffle=True,collate_fn=GPTDataset.collate_fn,
                                                 num_workers=1)

    assert len(train_loader)>0, 'no image found in data dir'

    sess_dir = os.path.join('sessions',args.session)
    # output_image_dir = os.path.join(sess_dir,'output')
    os.makedirs(sess_dir,exist_ok=True)
    # os.makedirs(output_image_dir,exist_ok=True)
    # weight_vae = os.path.join(sess_dir,args.weight_vae)
    weight_dalle = os.path.join(sess_dir,args.weight_dalle)

    summary_dir = os.path.join(sess_dir,'summary')
    writer = SummaryWriter(summary_dir)
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # img_vocab_size = 8192
    # tokenizer.add_tokens(['img%d'%i for i in range(img_vocab_size)])
    config = GPT2Config(n_positions=args.gpt_n_ctx,n_ctx=args.gpt_n_ctx,vocab_size=dataset.vocab_size,gradient_checkpointing=args.grad_check) #.vocab_size))
    model = GPT2LMHeadModel(config).from_pretrained('gpt2').to(device)
    emb = model.resize_token_embeddings(dataset.vocab_size)
    model.train()
    # print('weight shape',emb.weight.shape)
    # print(dataset.img_vocab_size+dataset.tokenizer.vocab_size)
    # print('embedding shape:',emb.shape)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)
    # dataset_size = len(train_loader)
    # assert dataset_size>= batch_size, 'dataset (%d) < batch_size(%d) '%(len(train_loader), batch_size)
    dataset_size = len(train_loader) # number of batches in one epoch
    print("Batch per iteration: ",dataset_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*dataset_size,20*dataset_size], gamma=0.1)
    image_list = []
    for epoch in range(args.epochs):
        running_loss = 0.0

        for it,batch in enumerate(train_loader):
            # images = torch.randn(4, 3, 256, 256)
            it = epoch*dataset_size + it
            # images = batch[0].to(device)
            # text = batch[1].to(device)
            # mask = torch.ones_like(text).bool()
            inputs = batch[0] # rajesh check
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)
            # labels = batch[1].to(device)
            outputs = model(**inputs, labels=inputs['input_ids']) # rajesh check
            loss = outputs.loss
            # logits = outputs.logits
            # loss.backward()
            optimizer.zero_grad()
            # text = torch.randint(0, 10000, (4, 256))
            # images = torch.randn(4, 3, 256, 256)
            # loss = vae(images, return_recon_loss = True)
            # loss = dalle(text, images, mask = mask, return_loss = True)

            loss.backward()
            optimizer.step()
            sys.stdout.write('\r[%s] %6d/%6d: loss: %f'%(time.asctime(),it,args.epochs*dataset_size,loss.item()))
            running_loss += loss.item()
            writer.add_scalar('training loss',running_loss / (it+1),epoch * dataset_size+ it)
            if it%args.save_every==(args.save_every-1):
                torch.save(model.state_dict(), weight_dalle)
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
                    running_loss = 0
                    for i, batch in enumerate(test_loader):
                        inputs = batch[0]
                        inputs['input_ids'] = inputs['input_ids'].to(device)
                        inputs['attention_mask'] = inputs['attention_mask'].to(device)
                        # labels = batch[1].to(device)
                        outputs = model(**inputs, labels=inputs['input_ids']) # rajesh check
                        running_loss += outputs.loss
                        if i>9:break
                    running_loss = running_loss/i if i>0 else 0
                    writer.add_scalar('validation loss',running_loss,epoch * dataset_size+ it)


        sys.stdout.write('\n[%s]Epoch:%d loss: %f'%(' '.join(time.asctime().split(' ')[1:-1]),epoch,running_loss/dataset_size)) # some samples are going for testing??

        # sys.stdout.write('\n[%s]Epoch:%d loss: %f\033[F'%(' '.join(time.asctime().split(' ')[1:-1]),epoch,running_loss/dataset_size)) # some samples are going for testing??
        # scheduler.step()




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
    # parser.add_argument('--train_data',required=True)
    # parser.add_argument('--test_data',required=True)
    parser.add_argument('--batch_size',default=128,type=int)
    # parser.add_argument('--drive_id',default='15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A')
    parser.add_argument('--drive_id',default=None)

    parser.add_argument('--dim',type=int,default=512)
    parser.add_argument('--text_seq_len',type=int,default=30)
    parser.add_argument('--video_seq_len',type=int,default=5)
    parser.add_argument('--depth',type=int,default=2)
    # parser.add_argument('--hidden_dim',type=int,default=64)

    parser.add_argument('--test',action='store_true',default=False,help='for testing use --test')  #not really required
    parser.add_argument('--weight_vae',default='vae.pth')
    parser.add_argument('--weight_dalle',default='dalle.pth')

    parser.add_argument('--grad_check',default=False,action='store_true')
    parser.add_argument('--context_size',default=10,type=int)
    parser.add_argument('--gpt_n_ctx',default=512,type=int)

    args = parser.parse_args()
    if not args.test and not args.train:
        print('Either --test or --train must be specified')
        exit(1)

    if args.train:
        train(args)
    else:
        test(args)
