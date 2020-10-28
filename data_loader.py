
import os,sys,glob,json
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *
import argparse
import joblib as pickle
from skimage.color import rgb2gray
from tqdm import tqdm
import imageio
import random



def load_flowers_dataset():

    dataset = '102flowers' #
    need_256 = True # set to True for stackGAN
    if 1:
        """
        images.shape = [8000, 64, 64, 3]
        captions_ids = [80000, any]
        """
        cwd = os.getcwd()
        img_dir = os.path.join(cwd, '102flowers')
        caption_dir = os.path.join(cwd, 'text_c10')
        VOC_FIR = cwd + '/vocab.txt'

        ## load captions
        caption_sub_dir = load_folder_list( caption_dir )
        captions_dict = {}
        processed_capts = []
        for sub_dir in caption_sub_dir: # get caption file list
            with tl.ops.suppress_stdout():
                files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
                for i, f in enumerate(files):
                    file_dir = os.path.join(sub_dir, f)
                    key = int(re.findall('\d+', f)[0])
                    t = open(file_dir,'r')
                    lines = []
                    for line in t:
                        line = preprocess_caption(line)
                        lines.append(line)
                        processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
                    assert len(lines) == 10, "Every flower image have 10 captions"
                    captions_dict[key] = lines
        print(" * %d x %d captions found " % (len(captions_dict), len(lines)))

        ## build vocab
        if not os.path.isfile('vocab.txt'):
            _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
        else:
            print("WARNING: vocab.txt already exists")
        vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

        ## store all captions ids in list
        captions_ids = []
        try: # python3
            tmp = captions_dict.items()
        except: # python3
            tmp = captions_dict.iteritems()
        for key, value in tmp:
            for v in value:
                captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
                # print(v)              # prominent purple stigma,petals are white inc olor
                # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
                # exit()
        captions_ids = np.asarray(captions_ids)
        print(" * tokenized %d captions" % len(captions_ids))

        ## check
        img_capt = captions_dict[1][1]
        print("img_capt: %s" % img_capt)
        print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
        img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
        print("img_capt_ids: %s" % img_capt_ids)
        print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])

        ## load images
        with tl.ops.suppress_stdout():  # get image files list
            imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
        print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
        s = time.time()

        images = []
        images_256 = []
        for name in imgs_title_list:
            # print(name)
            img_raw = imageio.imread( os.path.join(img_dir, name) )
            img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
            img = img.astype(np.float32)
            images.append(img)
            if need_256:
                img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
                img = img.astype(np.float32)

                images_256.append(img)
        # images = np.array(images)
        # images_256 = np.array(images_256)
        print(" * loading and resizing took %ss" % (time.time()-s))

        n_images = len(captions_dict)
        n_captions = len(captions_ids)
        n_captions_per_image = len(lines) # 10

        print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

        captions_ids_train, captions_ids_test = captions_ids[: 8000*n_captions_per_image], captions_ids[8000*n_captions_per_image :]
        images_train, images_test = images[:8000], images[8000:]
        if need_256:
            images_train_256, images_test_256 = images_256[:8000], images_256[8000:]
        n_images_train = len(images_train)
        n_images_test = len(images_test)
        n_captions_train = len(captions_ids_train)
        n_captions_test = len(captions_ids_test)
        print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
        print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

        save_all(vocab, '_vocab.pickle')
        save_all((images_train_256, images_train), '_image_train.pickle')
        save_all((images_test_256, images_test), '_image_test.pickle')
        save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
        save_all((captions_ids_train, captions_ids_test), '_caption.pickle')

def load_stories_dataset(args):
        cwd = os.getcwd()
        need_256 = False
        # img_dir = os.path.join(cwd, 'stories')
        # caption_dir = os.path.join(cwd, 'text_c10')
        # VOC_FIR = cwd + '/vocab.txt'
        # captions_dict = {}
        # processed_capts = []
        text_embeds = []
        images = []
        textid2imageids = {}
        img_dirs = glob.glob(os.path.join(args.data_dir,'*'))
        print('Found %d image dirs'%len(img_dirs))
        id2img = {}
        text_count = 0
        for path in tqdm(img_dirs):
            img_paths = glob.glob(path+'/images/*')
            # id2img256 ={}
            for img_path in img_paths:
                img_id = '/'.join(img_path.split('/')[-3:])
                # img_id = os.path.basename(img_path)
                img_raw = imageio.imread( img_path )

                img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
                if args.img_channels==1:
                    img = rgb2gray(img)
                    # print(img.shape)
                    # exit(0)
                img = img.astype(np.float32)
                # images.append(img)
                id2img[img_id] = img
                # if need_256:
                #     img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
                #     img = img.astype(np.float32)
                #     # images_256.append(img)
                #     id2img256[img_id] = img
            with open(path+'/chunks.json') as f:
                chunks = json.load(f)
                # id_base ='%d.jpg'
                id_base = os.path.basename(path)+'/images/%d.jpg'
                for chunk in chunks:
                    if len(chunk['frames'])==0: continue
                    for frames in chunk['frames']:
                        # img_seq = [id2img[id_base%frame] for frame in frames]
                        # images.append(img_seq)
                        text_embeds.append(chunk['embedding'])
                        textid2imageids[text_count] = [id_base%frame for frame in frames]
                        text_count+=1
        # ## load captions
        # # caption_sub_dir = load_folder_list( caption_dir )
        #
        # for sub_dir in caption_sub_dir: # get caption file list
        #     with tl.ops.suppress_stdout():
        #         files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt')
        #         for i, f in enumerate(files):
        #             file_dir = os.path.join(sub_dir, f)
        #             key = int(re.findall('\d+', f)[0])
        #             t = open(file_dir,'r')
        #             lines = []
        #             for line in t:
        #                 line = preprocess_caption(line)
        #                 lines.append(line)
        #                 processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>"))
        #             assert len(lines) == 10, "Every flower image have 10 captions"
        #             captions_dict[key] = lines
        # print(" * %d x %d captions found " % (len(captions_dict), len(lines)))
        #
        # ## build vocab
        # # if not os.path.isfile('vocab.txt'):
        # #     _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1)
        # # else:
        # #     print("WARNING: vocab.txt already exists")
        # # vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")
        #
        # ## store all captions ids in list
        # captions_ids = []
        # try: # python3
        #     tmp = captions_dict.items()
        # except: # python3
        #     tmp = captions_dict.iteritems()
        # for key, value in tmp:
        #     for v in value:
        #         captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
        #         # print(v)              # prominent purple stigma,petals are white inc olor
        #         # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
        #         # exit()
        # captions_ids = np.asarray(captions_ids)
        # print(" * tokenized %d captions" % len(captions_ids))
        #
        # ## check
        # img_capt = captions_dict[1][1]
        # print("img_capt: %s" % img_capt)
        # print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
        # img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
        # print("img_capt_ids: %s" % img_capt_ids)
        # print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])
        #
        # ## load images
        # with tl.ops.suppress_stdout():  # get image files list
        #     imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
        # print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))
        # s = time.time()
        #
        # images = []
        # images_256 = []
        # for name in imgs_title_list:
        #     # print(name)
        #     img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
        #     img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
        #     img = img.astype(np.float32)
        #     images.append(img)
        #     if need_256:
        #         img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
        #         img = img.astype(np.float32)
        #
        #         images_256.append(img)
        # # images = np.array(images)
        # # images_256 = np.array(images_256)
        # print(" * loading and resizing took %ss" % (time.time()-s))
        captions_ids = text_embeds
        n_images = len(id2img)
        n_captions = len(captions_ids)
        n_captions_per_image = 1 # len(lines) # 10

        print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))
        # captions_ids_train, captions_ids_test = captions_ids[: int(n_captions*0.8)], captions_ids[int(n_captions*0.8) :]
        # images_train, images_test = images[:int(n_images*0.8)], images[int(n_images*0.8):]
        # # if need_256:
        # #     images_train_256, images_test_256 = images_256[:8000], images_256[8000:]
        # n_images_train = len(images)*8//10
        # n_images_test = len(images)*2//10
        # n_captions_train = len(captions_ids_train)
        # n_captions_test = len(captions_ids_test)
        # print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
        # print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))
        #
        # # save_all(vocab, '_vocab.pickle')
        # save_all((images_train_256, images_train), '_image_train.pickle')
        # save_all((images_test_256, images_test), '_image_test.pickle')
        # save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
        # save_all((captions_ids_train, captions_ids_test), '_caption.pickle')

        save_all(captions_ids,'_caption.pickle')
        # save_all(images,'_image.pickel')
        save_all(id2img,'_id2img.pickle')
        save_all(textid2imageids,'_textid2imageids.pickle')

def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',required=True,help='datset root path')
    parser.add_argument('--data_name',required=True,choices=['stories','102flowers'],help='dataset name')
    parser.add_argument('--img_channels',default=1,type=int,help='1(default) for bnw, 3 for color')
    args = parser.parse_args()
    if args.data_name == '102flowers':
        load_flowers_dataset()
    elif args.data_name == 'stories':
        load_stories_dataset(args)
