from __future__ import print_function
import torch
from torch.autograd import Variable
import math
import numpy

class Ranker():
    def __init__(self):
        super(Ranker, self).__init__()
        return
    
    def update_rep(self, feat):
        self.feat = feat
        return

    def compute_rank(self, input, target_idx):
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]

        value = target - input
        value = value ** 2
        value = value.sum(1)
        rank = torch.LongTensor(value.size(0))
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            rank[i] = val.lt(value[i]).sum()

        return rank

    def compute_rank_correct(self, input, target_idx, memory_idx):
        # input <---- a batch of vectors
        # targetIdx <----- ground truth index
        # return rank of input vectors in terms of rankings in distance to the ground truth

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]
        memory = self.feat[memory_idx]

        value = target - input
        value = value ** 2
        value = value.sum(1)
        rank = torch.LongTensor(value.size(0))
        for i in range(value.size(0)):
            val = self.feat - input[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)

            val_m = memory[i] - input[i].expand(memory[i].size(0), memory[i].size(1))
            val_m = val_m ** 2
            val_m = val_m.sum(1)

            rank[i] = val.lt(value[i]).sum()-val_m.lt(value[i]).sum()

            # print("----------------------------------")
            # print("vanilla rank:", val.lt(value[i]).sum())
            # print("memory rank:", val_m.lt(value[i]).sum())

            # there might be the same images as the target image, i.e. duplicated images in dataset
            # so the rank[i] can be -1, we need to correct this
            if rank[i]<0:
                rank[i] = 0

        return rank

    def nearest_neighbor(self, target):
        # L2 case
        idx = torch.LongTensor(target.size(0))
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = val.min(0)
            idx[i] = id #[0]
        return idx

    def k_nearest_neighbors(self, target, K = 5):
        idx = torch.LongTensor(target.size(0), K)
        if torch.cuda.is_available():
            target = target.cuda()
            self.feat = self.feat.cuda()

        for i in range(target.size(0)):
            val = self.feat - target[i].expand(self.feat.size(0), self.feat.size(1))
            val = val ** 2
            val = val.sum(1)
            v, id = torch.topk(val, k=K, dim=0, largest=False)
            # idx[i] = id
            idx[i].copy_(id.view(-1))
        return idx

    def nearest_neighbor_selector(self, user_img_idx, top_k_act_img_idx):
        # L2 case
        # print("feat shape:", self.feat.shape)
        target = self.feat[user_img_idx]
        # print("target shape:", target.shape)
        min_idx = torch.LongTensor(target.size(0))
        min_pos = torch.LongTensor(target.size(0))
        
        max_idx = torch.LongTensor(target.size(0))
        max_pos = torch.LongTensor(target.size(0))
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        feat = self.feat[top_k_act_img_idx]
        # print("feat shape:", feat.shape)
        for i in range(target.size(0)):
            val = feat[i] - target[i].expand(feat[i].size(0), feat[i].size(1))
            val = val ** 2
            val = val.sum(1)
            
            # most similar
            v, id = val.min(0)
            min_idx[i] = top_k_act_img_idx[i,id]
            min_pos[i] = id
            
            # leaset similar
            v, id = val.max(0)
            max_idx[i] = top_k_act_img_idx[i,id]
            max_pos[i] = id
            # print("idx[i]", idx[i])
        return min_idx, min_pos, max_idx, max_pos
    
    def nearest_neighbor_selector_implicit(self, user_img_idx, top_k_act_img_idx):
        # L2 case
        # print("feat shape:", self.feat.shape)
        target = self.feat[user_img_idx]
        # print("target shape:", target.shape)
        min_idx = torch.LongTensor(target.size(0))
        min_pos = torch.LongTensor(target.size(0))
        
        max_idx = torch.LongTensor(target.size(0))
        max_pos = torch.LongTensor(target.size(0))
        
        if torch.cuda.is_available():
            target = target.cuda()
            # self.feat = self.feat.cuda()
        feat = self.feat[top_k_act_img_idx]
        # print("feat shape:", feat.shape)
        
        implicit_idx = torch.LongTensor(target.size(0), feat.shape[1]-1)
        
        for i in range(target.size(0)):
            val = feat[i] - target[i].expand(feat[i].size(0), feat[i].size(1))
            val = val ** 2
            val = val.sum(1)
            
            # most similar
            v, id = val.min(0)
            min_idx[i] = top_k_act_img_idx[i,id]
            min_pos[i] = id
            
            # print("most similar:", id)
            # print("top_k_act_img_idx[i]", top_k_act_img_idx[i])
            # print("top_k_act_img_idx[i,id]", top_k_act_img_idx[i,id])
            # print("implicit_idx[i]",top_k_act_img_idx[i][top_k_act_img_idx[i]!=top_k_act_img_idx[i,id]])
            
            # negatives
            implicit_idx[i] = top_k_act_img_idx[i][top_k_act_img_idx[i]!=top_k_act_img_idx[i,id]]
            
            # leaset similar
            v, id = val.max(0)
            max_idx[i] = top_k_act_img_idx[i,id]
            max_pos[i] = id
            # print("idx[i]", idx[i])
            
        return min_idx, min_pos, max_idx, max_pos, implicit_idx


    def max_qvalue_selector(self, model, state_pos, state_neg, K = 5):
        idx = torch.LongTensor(state_pos.size(0), K)
        if torch.cuda.is_available():
            state_pos = state_pos.cuda()
            state_neg = state_neg.cuda()
            self.feat = self.feat.cuda()

        top_km_act_img_idx = self.k_nearest_neighbors(state_pos,K=K*3)
        if torch.cuda.is_available():
            top_km_act_img_idx = top_km_act_img_idx.cuda()
        top_km_act_emb = self.feat[top_km_act_img_idx]

        # for i in range(state_pos.size(0)):
        #     value_list = []
        temp_value = torch.LongTensor(state_pos.size(0), K*3)
        for j in range(K*3):
            action = top_km_act_emb[:,j,:]
            value = model.forward_qvalue(Variable(state_pos), Variable(state_neg),Variable(action))
                # value_list.append(value)
            # get index of top-K values
            if j == 0 :
                temp_value = value
            else:
                temp_value = torch.cat([temp_value, value], dim=1)
        
        v, idx_v = torch.topk(temp_value, k=K, dim=1, largest=True)
        # print(idx_v.shape)
        # print(top_km_act_img_idx.shape)
        for i in range(state_pos.size(0)):
            idx[i] = top_km_act_img_idx[i,idx_v[i]]
        # print(idx.shape)
        return idx
    
    def compute_rank_dqn(self, model, state_pos, state_neg, target_idx, memory_idx):

        if torch.cuda.is_available():
            # input = input.cuda()
            target_idx = target_idx.cuda()
            # self.feat = self.feat.cuda()
        target = self.feat[target_idx]
        memory = self.feat[memory_idx]
        
        rank = torch.LongTensor(state_pos.size(0))
        temp_value = torch.LongTensor(state_pos.size(0), self.feat.size(0))
        for j in range(self.feat.size(0)):
            action = self.feat[j,:].expand(state_pos.size(0), state_pos.size(1))
            value = model.forward_qvalue(Variable(state_pos), Variable(state_neg),Variable(action))
            
            if j == 0 :
                temp_value = value
            else:
                temp_value = torch.cat([temp_value, value], dim=1)
        
        rank_list = torch.argsort(temp_value, dim=1)
        
        # print(rank_list.shape)
            
        for i in range(state_pos.size(0)):
            idx = target_idx[i]
            rank[i] = rank_list[i][idx]

        return rank

