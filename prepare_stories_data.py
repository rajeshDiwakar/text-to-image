'''
bert-serving-start -model_dir /home/rajesh/work/limbo/bert/uncased_L-12_H-768_A-12/ -num_worker=1 -cpu
pip install bert-serving-server
pip install bert-serving-client

python prepare_stories_data.py -d ../storiesgan/data/vids/ --ss 10s
python prepare_stories_data.py --ss 10s --overwrite --context 60 -j '/home/rajesh/work/data/storiesgan/data/vids/7IoF9IrZnXU.en.json'
'''

'''
adjust 500ms
handle case of empty frames
'''

import os,sys,time,glob,json
import re
import logging
#from bert_serving.client import BertClient
import argparse
from tqdm import tqdm
from colorama import Fore
import cv2
import traceback


logging.basicConfig(level=logging.DEBUG)

# emb_cache = {}
update_cache = False
def ms2str(milliseconds):
    hh = milliseconds//3600000
    milliseconds %= 3600000
    mm = milliseconds//60000
    milliseconds %= 60000
    ss = milliseconds // 1000
    ms = int(milliseconds % 1000)
    return '%02d:%02d:%02d.%d'%(hh,mm,ss,ms)

def str2ms(time_str):
    m = re.match('([0-9]+):([0-9]+):([0-9]+).([0-9]+)',time_str)
    h,m,s,ms = m.group(1,2,3,4)
    return (int(h)*3600 + int(m)*60 + int(s))*1000+ int(ms)

#bc = BertClient(check_length=False)

def get_text_embedding(texts):
    if type(texts)!=list:
        texts = [texts]
    return bc.encode(texts)

def chunk_caption(caption,win_size=8,hop_length=5,context_size=30,mode='word',fps=25,frame_seq_size=3,frame_hop_len=5,ss=0):
    if not len(caption): return []
    chunks = []
    global emb_cache
    if mode =='word':
        # words =[i for cap in caption for i in cap['words']]
        words = []
        for cap in caption:
            for w in cap['words']:
                if re.match('\[.+\]',w['text']):
                    logging.debug('discarding: %s'%w['text'])
                    continue
                words.append(w)

        for i in tqdm(range(win_size,len(words),hop_length),bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),desc='chunking captions...'):
            mwords = [w for w in words[i-win_size:i]]
            start = str2ms(mwords[0]['start'])
            end = str2ms(mwords[-1]['end'])
            if start < ss:
                continue
            text=' '.join([w['text'] for w in mwords])
            context = ' '.join([w['text'] for w in words[max(0,i-context_size):i]])
            if not context.strip():
                logging.warning('Found empty context: %s'%text)

            try:
                embedding_context = emb_cache[context]
                embedding_text = emb_cache[text]
            except KeyError:
                embedding_context = []#get_text_embedding([context])[0].tolist()
                embedding_text = []#get_text_embedding([text])[0].tolist() # todo: make it single call
                emb_cache[context] = embedding_context
                emb_cache[text] = embedding_text
                update_cache = True

            frames =list(range(int(start*fps/1000.),int(end*fps/1000.),fps//2)) # taking 1 fps #check
            # for the duration of text there will be many frames.
            # We can sample a few continuous frames at different time in this duration and each such group can represent this text
            mframes = [frames[i:i+frame_seq_size] for i in range(0,len(frames)-frame_seq_size,frame_hop_len)]
            if not len(mframes):
                logging.warning('frames len %d  '%len(frames)+' \n>> '+str(frames))

            chunks.append({'text':text,
                           'context':context,
                           'start':ms2str(start),
                           'end':ms2str(end),
                           'embedding_context':embedding_context,
                           'embedding_text':embedding_text,
                           'mframes':mframes,
                           'frames':frames}
                           )
    else:
        raise NotImplementedError
    return chunks

def parse_ms(text):
    text = text.lower().strip("' ")
    if type(text) == int:
        return text*1000
    if text.endswith('s'):
        return int(text[:-1])*1000
    if text.endswith('ms'):
        return int(text[:-2])
    raise ValueError

def extract_frames(vid,chunks,outdir):

    os.makedirs(outdir,exist_ok=True)
    # frames = [frame for c in chunks for frames in c['mframes'] for frame in frames]
    frames =[frame for c in chunks for frame in c['frames']]

    frames =list(set(frames))
    frames.sort()
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened(): raise ValueError('Unable to open file %s'%vid)

    for frame in tqdm(frames,bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),desc='Extracting frames'):
        img_path = os.path.join(outdir,'%d.jpg'%frame)
        if os.path.isfile(img_path):
            sys.stdout.write('\r%s already exists'%img_path)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ret, img_cv = cap.read()
        if not ret:
            raise ValueError('Unexpected end of frames')
        img_cv = cv2.resize(img_cv,(64,64))
        cv2.imwrite(img_path,img_cv)

    cap.release()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--json',help='input json file')
    parser.add_argument('-d','--dir',help='input directory')
    parser.add_argument('--output',default='dataset',help='output path for json input and output dir for dir input')
    parser.add_argument('--fps',default=12,type=int,help='fps of input video')
    parser.add_argument('--context',default=30,type=int,help='context size')
    parser.add_argument('--win',default=8,type=int,help='no of words in sent')
    parser.add_argument('--hop',default=5,type=int,help='hop length between two sents')
    parser.add_argument('--ss',default='0',type=str,help='seek. use it to skip some duration')
    parser.add_argument('--indent',default=None,type=int,help='indent while writing json')
    parser.add_argument('--overwrite',action='store_true',default=False,help='skips already processed files')
    parser.add_argument('--emb_cache',default='emb_cache.json',help='path to embedding cache')
    args = parser.parse_args()
    args.ss = parse_ms(args.ss)
    global emb_cache
    if os.path.isfile(args.emb_cache):
        with open(args.emb_cache) as f:
            try:
                emb_cache = json.load(f)
            except ValueError:
                print('Failed to decode cache json')
                emb_cache = {}
    else:
        emb_cache = {}
    try:
        if args.json:
            with open(args.json) as f:
                captions = json.load(f)

            chunks = chunk_caption(captions,args.win,args.hop,args.context,'word',args.fps,ss=args.ss)
            fname = os.path.basename(args.json).replace('.en.json','')
            outdir = os.path.join(args.output,fname)
            json_path = os.path.join(outdir,'chunks.json')
            if not args.overwrite and os.path.isfile(json_path):
                print('Skipping %s. File exists'%fname)
            else:
                img_dir = os.path.join(outdir,'images')
                os.makedirs(outdir,exist_ok=True)
                os.makedirs(img_dir,exist_ok=True)
                with open(json_path,'w') as f:
                    json.dump(chunks,f,indent=args.indent)
                vid = args.json.replace('.en.json','.mp4')
                extract_frames(vid,chunks,img_dir)

        elif args.dir:
            files = glob.glob(os.path.join(args.dir,'*.en.json'))
            logging.info('Found %d json files'%len(files))
            for file in tqdm(files):
                with open(file) as f:
                    captions = json.load(f)

                chunks = chunk_caption(captions,args.win,args.hop,args.context,'word',args.fps,ss=args.ss)
                fname = os.path.basename(file).replace('.en.json','')
                outdir = os.path.join(args.output,fname)
                json_path = os.path.join(outdir,'chunks.json')
                img_dir = os.path.join(outdir,'images')
                if not args.overwrite and os.path.isfile(json_path):
                    print('Skipping %s. File exists'%fname)
                    continue
                os.makedirs(outdir,exist_ok=True)
                os.makedirs(img_dir,exist_ok=True)
                with open(json_path,'w') as f:
                    json.dump(chunks,f,indent=args.indent)
                vid = file.replace('.en.json','.mp4')
                extract_frames(vid,chunks,img_dir)

        else:
            raise ValueError('Input atleast one of json or dir')
    except Exception as e:
        traceback.print_exc()
    finally:
        if update_cache:
            with open(args.emb_cache,'w') as f:
                json.dump(emb_cache,f)
