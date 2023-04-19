import json
import numpy as np
import pickle
import torch

def get_data_splits(data_type):
    if data_type=='shoe':
        train_imgs = json.load(open('dataset/shoe/image_splits/split.shoe.train.filter.json', 'r'))
        val_imgs = json.load(open('dataset/shoe/image_splits/split.shoe.val.filter.json', 'r'))
        test_imgs = json.load(open('dataset/shoe/image_splits/split.shoe.test.filter.json', 'r'))
    elif data_type=='dress':
        train_imgs = json.load(open('dataset/dress/image_splits/split.dress.train.filter.json', 'r'))
        val_imgs = json.load(open('dataset/dress/image_splits/split.dress.val.filter.json', 'r'))
        test_imgs = json.load(open('dataset/dress/image_splits/split.dress.test.filter.json', 'r'))
    elif data_type=='shirt':
        train_imgs = json.load(open('dataset/shirt/image_splits/split.shirt.train.filter.json', 'r'))
        val_imgs = json.load(open('dataset/shirt/image_splits/split.shirt.val.filter.json', 'r'))
        test_imgs = json.load(open('dataset/shirt/image_splits/split.shirt.test.filter.json', 'r'))
    elif data_type=='toptee':
        train_imgs = json.load(open('dataset/toptee/image_splits/split.toptee.train.filter.json', 'r'))
        val_imgs = json.load(open('dataset/toptee/image_splits/split.toptee.val.filter.json', 'r'))
        test_imgs = json.load(open('dataset/toptee/image_splits/split.toptee.test.filter.json', 'r'))
    return train_imgs,val_imgs,test_imgs

def get_embeddings(data_type):
    if data_type=="shoe":
        fc = np.load('dataset/shoe/image_features/fc_feature.npz')['feat']
        att = np.load('dataset/shoe/image_features/att_feature.npz')['feat']
        absolute_feature = pickle.load(open('dataset/shoe/image_features/clip_embedding.p', 'rb'))
    elif data_type=="dress":
        fc = np.load('dataset/dress/image_features/fc_feature.npz')['feat']
        att = np.load('dataset/dress/image_features/att_feature.npz')['feat']
        absolute_feature = pickle.load(open('dataset/dress/image_features/clip_embedding.p', 'rb'))
    elif data_type=="shirt":
        fc = np.load('dataset/shirt/image_features/fc_feature.npz')['feat']
        att = np.load('dataset/shirt/image_features/att_feature.npz')['feat']
        absolute_feature = pickle.load(open('dataset/shirt/image_features/clip_embedding.p', 'rb'))
    elif data_type=="toptee":
        fc = np.load('dataset/toptee/image_features/fc_feature.npz')['feat']
        att = np.load('dataset/toptee/image_features/att_feature.npz')['feat']
        absolute_feature = pickle.load(open('dataset/toptee/image_features/clip_embedding.p', 'rb'))
    return fc, att, absolute_feature

def get_fc_att_embeddings(data_type):
    train_imgs,val_imgs,test_imgs = get_data_splits(data_type)

    if data_type=="shoe":
        fc = np.load('dataset/shoe/image_features/fc_feature.npz')['arr_0']
        att = np.load('dataset/shoe/image_features/att_feature.npz')['arr_0']
    elif data_type=="dress" or data_type=="shirt" or data_type=="toptee":
        fc = np.load('dataset/' + data_type + '/image_features/fc_feature.npz')['feat']
        att = np.load('dataset/' + data_type + '/image_features/att_feature.npz')['feat']
        
    fc = torch.FloatTensor(fc)
    att = torch.FloatTensor(att)
    
    train_N = len(train_imgs)
    val_N = len(val_imgs)
    test_N = len(test_imgs)
    
    if data_type=="shoe":
        # train
        train_fc = fc[0:train_N]
        train_att = att[0:train_N]
        # val
        val_fc = fc[train_N:]
        val_att = att[train_N:]
        # test
        test_fc = fc[train_N:]
        test_att = att[train_N:]
    else:
        # train
        train_fc = fc[0:train_N]
        train_att = att[0:train_N]
        # val
        val_fc = fc[train_N:train_N+val_N]
        val_att = att[train_N:train_N+val_N]
        # test
        test_fc = fc[train_N+val_N:]
        test_att = att[train_N+val_N:]
    
    return train_fc, train_att, val_fc, val_att, test_fc, test_att

def get_absolute_feature_embeddings(data_type):
    absolute_feature = pickle.load(open('dataset/' + data_type + '/image_features/clip_embedding.p', 'rb'))
    
    train_feature = absolute_feature['train']
    test_feature = absolute_feature['val']
    val_feature = absolute_feature['test']
    
    return train_feature, val_feature, test_feature
    
    