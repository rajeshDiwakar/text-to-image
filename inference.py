'''
should we leave space after arpa braces?
'''


import matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')
#import matplotlib.pylab as plt

#import IPython.display as ipd
# /home/rajesh/work/limbo/tacotron2/waveglow
import sys, os, time
sys.path.append('../tacotron2/waveglow/')
import numpy as np
import torch
import random
from random import randint
import librosa

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence, cmudict
from denoiser import Denoiser
import re
import argparse
import json
import gc
import pickle as pkl

from einops import rearrange
from utils import to_gpu, get_mask_from_lengths

import warnings
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cmu',default=True, action='store_false',help='true/false cmu pronounciation json file')
args = parser.parse_args()


hparams = create_hparams()
hparams.sampling_rate = 22050

#
cmu = '/home/rajesh/work/limbo/tacotron2/models/cmudict-0.7b' if args.cmu else None
# if cmu is not None:
#     # with open(cmu) as f:
#     #     cmu = json.load(f)
#     cmudict = cmudict.CMUDict(cmu,True)

if hparams.cmudict_path is not None:
    cmudict = cmudict.CMUDict(hparams.cmudict_path)
if hparams.aligndict_path is not None:
    with open(hparams.aligndict_path,'rb') as f:
        aligndict = pkl.load(f)

def seed_torch(seed=1234):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

#seed_torch(hparams.seed)

# In[3]:


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')





# #### Load model from checkpoint

# In[9]:
audio_dir='output'
os.makedirs(audio_dir,exist_ok=True)

prefix='aug_'
checkpoint_path = 'outdir/checkpoint_1500' #"models/tacotron2_statedict.pt"
#prefix='rajdeep_17000_'
#checkpoint_path='outdir/checkpoint_17000'
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path,map_location='cpu')['state_dict'])
# _ = model.eval().half()


# #### Load WaveGlow for mel2audio synthesis and denoiser

# In[2]:


waveglow_path = '../tacotron2/models/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path,map_location='cpu')['model']
#waveglow.eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

def get_arpabet(text):
    if not len(text):return ''
    words = text.lower()
    # words = re.split('[^a-z]',text)
    matches = re.finditer('[^a-z]+',words)
    arpa = []
    pos = 0
    for m in matches:
        if m.start()>pos:
            w=text[pos:m.start()]
            pron = cmudict.lookup(w)
            if pron: pron=pron[0]
            arpa.append('{%s}'%pron if pron else w)
            arpa.append(m.group())
            pos=m.end()
            if not pron: print('No spell for: %s'%w)
    if pos < len(text):
        w=text[pos:]
        pron = cmudict.lookup(w)
        if pron: pron=pron[0]
        arpa.append('{%s}'%pron if pron else w)

    return ''.join(arpa)

def run_cmd(cmd):
    cmd = cmd.split(':')
    if cmd[0]=='set_seed':
        try:
            hparams.seed = int(cmd[1])
        except Exception as e:
            print(e)
    else:
        print('Invalid command: %s'%':'.join(cmd))
testtexts = ["All day the dust sifted down from the sky, and the next day it sifted down.",
            "He rolled his cigarette slowly and perfectly, studied it, smoothed it.",
            'And she replied, "Guy took the jackpot not two hours ago.',
            'Now and then the flies roared softly at the screen door.'
            ][:1]
testtexts=["Nanin, a Japanese master during the Meiji era received a university professor who came to inquire about Zen."]
testtexts=['adhiyajna','adhibhoot','adhidaiva','duryodhan']
#testtexts= ["TO THE RED COUNTRY and part of the gray country of Oklahoma, the last rains came gently, and they did not cut the scarred earth. The plows crossed and recrossed the rivulet marks. The last rains lifted the corn quickly and scattered weed colonies and grass along the sides of the roads so that the gray country and the dark red country began to disappear under a green cover."]
# testtexts = ["Well, it is not very fascinating, but should you waste time in something else?",
#         "silence is the best prayer, silence is the best updesha.",
#         "All of your cognition is construction of your mind. Find out who you are and be free.",
#         "Where are you going? Into what? Nothing is different from you. Don't get mislead by the body thought."
#         ]
testtext = ''
# seeds = [randint(0,10000) for _ in range(10)]
seeds = [1234]
while True:
# for testtext in testtexts:

    # for i in range(5)
    for seed in seeds[:1]:
        hparams.seed = seed
        seed_torch(hparams.seed)

        if testtext:
            text = testtext
        else:
            text = input('\n> ').strip().split(':')[-1].strip('"')
            testtext = ''
        text_org = text
        if text == 'q': break
        elif text[0] == '@':
            run_cmd(text[1:])
            continue
        elif text == '':
                print('Enter q to exit.\n> ')
                continue
        # if cmu is not None:
        #     text = get_arpabet(text)
        print('synthesising',text)
        start=time.time()
        sequence,alignments = text_to_sequence(text, ['english_cleaners'],cmudict, 0.5,aligndict,1,hparams.dim_align)
        sequence = np.array(sequence)[None, :]
        alignments = alignments[None,:]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).long()
        alignments = torch.from_numpy(alignments).float()
        # alignments_ref = None
        # print('testing alignment injection with dhritrashtra')
        # # alignments_ref = torch.from_numpy(np.load('output/nv_dhritrashtra_seed=4545_align_0.npy'))
        # alignments_ref = torch.from_numpy(np.load('/home/rajesh/dhrit22050_mello_align.npy'))
        # alignments_ref = rearrange(alignments_ref,'t b c -> b t c')

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference((sequence,alignments))
        audios = []
        if 1:
            print('using original audio generation. requires more memory for longer sequence')
            with torch.no_grad():
                audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
                audios.append(audio[0].data.cpu().numpy())

                #ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
                # audio=audio[0].data.cpu().numpy()

                # audio_denoised = denoiser(audio, strength=0.01)[:, 0]
                # ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

        else:
            # audio_full=audio[0].data.cpu().numpy()
            print('mel_outputs_postnet: ',mel_outputs_postnet.shape)
            # print('audio_full: ',audio_full.shape)

            _,row,col = mel_outputs_postnet.shape
            with torch.no_grad():
                for c in range(col//100):
                    audio = waveglow.infer(mel_outputs_postnet.narrow(2,c*100,min(col-c*100,100)), sigma=0.666)
                    # audio = denoiser(audio, strength=0.01)[:, 0] #audio_denoised
                    audios.append(audio[0].data.cpu().numpy())

        audio_parts = np.concatenate(audios)
        audio = audio_parts
        print('audio parts: ',audio_parts.shape)
        audio_path = os.path.join(audio_dir,prefix+re.sub('[^0-9a-zA-Z]','_',text_org[:100])) + '_seed=%d'%hparams.seed
        for i in range(100):
                if not os.path.isfile(audio_path+'_%d.wav'%i):
                        audio_path = audio_path+'_%d.wav'%i
                        break
        # np.save(audio_path.replace('.wav','.npy'),alignments.detach().numpy())
        print('Saving audio in: %s'%audio_path)
        # librosa.output.write_wav(audio_path, audio_full, hparams.sampling_rate)
        librosa.output.write_wav(audio_path.replace('.wav','_parts.wav'), audio, hparams.sampling_rate)
        print('Time taken: %.2fs'%(time.time()-start))
        gc.collect()
