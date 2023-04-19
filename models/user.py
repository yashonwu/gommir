# import os, sys
# from sys import path
# sys.path.insert(0, os.getcwd())
import torch
import random
import numpy as np
import pickle
from torch.autograd import Variable
import json
# from transformers import AutoTokenizer, AutoModel
import clip

from captioning import captioner
from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet
from .data_utils import get_data_splits

# two functions are tested in this script:
# fc, att = captioner.compute_img_feat_batch(images)
# fc: N x 2048, att: N x 7 x 7 x2048
# and
# seq_id, sentences = captioner.gen_caption_from_feat(feat_tuple_target, feat_tuple_target)
# seq_id: N x 8, sentences: N string sentences

class UserSim:
    def __init__(self, data_type='', caption_model_dir=''):
        self.data_type = data_type
        
        # load trained captioner model
        params = {}
        params['model'] = 'resnet101'
        params['model_root'] = 'imagenet_weights'
        params['att_size'] = 7
        params['beam_size'] = 1
        self.captioner_relative = captioner.Captioner(is_relative= True, model_path= caption_model_dir, image_feat_params= params)
        self.captioner_relative.opt['use_att'] = True
        self.vocabSize = self.captioner_relative.get_vocab_size()
        
        random.seed(42)
        
        # # load pretrained textual encoder: bert-base-uncased
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        
        self.clip_model, preprocess = clip.load("RN101")
        self.clip_model.cuda().eval()
        
        from .data_utils import get_fc_att_embeddings, get_absolute_feature_embeddings
        self.train_fc_input, self.train_att_input, self.val_fc_input, self.val_att_input, self.test_fc_input, self.test_att_input = get_fc_att_embeddings(data_type)
        self.train_feature, self.val_feature, self.test_feature = get_absolute_feature_embeddings(data_type)
        
        self.train_imgs, self.val_imgs, self.test_imgs = get_data_splits(data_type)
        
        self.train_size = len(self.train_imgs)
        self.val_size = len(self.val_imgs)
        self.test_size = len(self.test_imgs)

        print('init. done!\n#img: {}/ {} / {}'.format(self.train_size, self.val_size, self.test_size))
        print('use cuda:', torch.cuda.is_available())

        return

    def sample_idx(self, img_idx, split):
        if split == 'train':
            split_size = self.train_size
        elif split == 'val':
            split_size = self.val_size
        elif split == 'test':
            split_size = self.test_size

        for i in range(img_idx.size(0)):
            img_idx[i] = random.randint(0, split_size - 1)
        return
    
    def sample_k_idx(self, img_idx, split, top_k):
        if split == 'train':
            split_size = self.train_size
        elif split == 'val':
            split_size = self.val_size
        elif split == 'test':
            split_size = self.test_size

        # for i in range(img_idx.size(0)):
        #     for j in range(img_idx.size(1)):
        #         img_idx[i][j] = random.randint(0, split_size - 1)
        
        for i in range(img_idx.size(0)):
            img_idx[i] = torch.Tensor(random.sample(range(0, split_size - 1), img_idx.size(1))).cuda()
        return
    
    def sample_target_idx(self, img_idx, split, batch_idx, batch_size, num_epoch):
        if split == 'train':
            split_size = self.train_size
        elif split == 'val':
            split_size = self.val_size
        elif split == 'test':
            split_size = self.test_size

        if batch_idx==num_epoch:
            left=np.arange(split_size)[(batch_idx-1)*batch_size:]
            for i in range(img_idx.size(0)):
                if i<len(left):
                    img_idx[i] = torch.tensor(left[i])
                else:
                    img_idx[i] = random.randint(0, split_size - 1)
        else:
            range_idx = np.arange(split_size)[(batch_idx-1)*batch_size:batch_idx*batch_size]
            for i in range(img_idx.size(0)):
                img_idx[i] = torch.tensor(range_idx[i])
        return

    def get_image_name(self, img_idx, split):
        if split == 'train':
            imgs_name = [self.train_imgs[x] for x in img_idx]
        elif split == 'val':
            imgs_name = [self.val_imgs[x] for x in img_idx]
        elif split == 'test':
            imgs_name = [self.test_imgs[x] for x in img_idx]
        return imgs_name

    def get_top_k_image_name(self, img_idx, split):
        if split == 'train':
            imgs_name = [[self.train_imgs[y] for y in x] for x in img_idx]
        elif split == 'val':
            imgs_name = [[self.val_imgs[y] for y in x] for x in img_idx]
        elif split == 'test':
            imgs_name = [[self.test_imgs[y] for y in x] for x in img_idx]
        return imgs_name

    def get_feedback(self, act_idx, position, user_idx, split='train'):
        if split == 'train':
            fc = self.train_fc_input
            att = self.train_att_input
        elif split == 'val':
            fc = self.val_fc_input
            att = self.val_att_input
        elif split == 'test':
            fc = self.test_fc_input
            att = self.test_att_input

        batch_size = user_idx.size(0)
        
        # load embeddings for the batch
        act_fc = fc[act_idx]
        act_att = att[act_idx]
        user_fc = fc[user_idx]
        user_att = att[user_idx]
        
        if torch.cuda.is_available():
            act_fc = act_fc.cuda()
            act_att = act_att.cuda()
            user_fc = user_fc.cuda()
            user_att = user_att.cuda()
            
        temp_type= self.data_type
        
        if self.data_type=="shoe":
            temp_type_single = "shoe"
            temp_type_multiple = "shoes"
        elif self.data_type=="dress":
            temp_type_single = "dress"
            temp_type_multiple = "dresses"
        elif self.data_type=="shirt":
            temp_type_single = "shirt"
            temp_type_multiple = "shirts"
        elif self.data_type=="toptee":
            temp_type_single = "toptee"
            temp_type_multiple = "toptees"
            
        # positive feeback
        with torch.no_grad():
            seq_label, sents_label = self.captioner_relative.gen_caption_from_feat((Variable(user_fc),
                                                                                Variable(user_att)),
                                                                               (Variable(act_fc),
                                                                                Variable(act_att)))

        text_tokens = clip.tokenize(sents_label).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens).float()

        return text_features, sents_label

    
    
    
