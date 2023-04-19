from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn

from sklearn.decomposition import PCA

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


from sys import path
sys.path.insert(0, os.getcwd())

from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet

def compute_img_feat(img_name, im_path, my_resnet):
    # load the image
    I = skimage.io.imread(os.path.join(im_path, img_name))

    if len(I.shape) == 2:
        I = I[:, :, np.newaxis]
        I = np.concatenate((I, I, I), axis=2)

    I = I.astype('float32') / 255.0
    I = torch.from_numpy(I.transpose([2, 0, 1]))
    if torch.cuda.is_available(): I = I.cuda()
    I = Variable(preprocess(I), volatile=True)
    fc, att = my_resnet(I, params['att_size'])

    return fc.data.cpu().float().numpy(), att.data.cpu().float().numpy()

def make_dir_if_not_there(d):
    if not os.path.isdir(d): os.mkdir(d)

def main(args):

    imageDir = args.image_dir

    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
    my_resnet = myResnet(net)
    if torch.cuda.is_available():
        print('cuda available, use cuda')
        my_resnet.cuda()
    my_resnet.eval()

    train_split = json.load(open(args.train_split, 'r'))
    val_split = json.load(open(args.val_split, 'r'))
    test_split = json.load(open(args.test_split, 'r'))

    all_splits = train_split + val_split + test_split

    N = len(all_splits)
    train_N = len(train_split)
    val_N = len(val_split)
    test_N = len(test_split)

    seed(42) # make reproducible
    
    all_fc = []
    all_att = []
    for i, img_temp in enumerate(all_splits):
        im_id = all_splits[i]
        
        imName = im_id
        tmp_fc, tmp_att = compute_img_feat(imName, imageDir, my_resnet)

        all_fc.append(tmp_fc)
        all_att.append(tmp_att)

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
            sys.stdout.flush()
    
    np.savez_compressed(os.path.join(args.output_dir, 'fc_feature'), feat=all_fc)
    np.savez_compressed(os.path.join(args.output_dir, 'att_feature'), feat=all_att)

    print('Feature preprocessing done')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--is_relative', type= str, default= 'True')
    parser.add_argument('--att_size', default=7, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='imagenet_weights', type=str, help='model root')
    parser.add_argument('--output_dir', default='image_features', type=str, help='temp output folder')
    parser.add_argument('--image_dir', default='images', type=str)
    parser.add_argument('--train_split', default='image_splits/split.dress.train.filter.json', type=str, help='')
    parser.add_argument('--val_split', default='image_splits/split.dress.val.filter.json', type=str, help='')
    parser.add_argument('--test_split', default='image_splits/split.dress.test.filter.json', type=str, help='')
    parser.add_argument('--start', default= 0.1, type= float)
    parser.add_argument('--end', default= 0.2, type= float)

    args = parser.parse_args()

    print('parsed input parameters:')
    params = vars(args)  # convert to ordinary dict
    print(json.dumps(params, indent = 2))

    main(args)