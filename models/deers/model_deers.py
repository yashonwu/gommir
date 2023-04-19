from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
from torch.distributions import Normal

from models.utils import Norm, Encoder
import math

class RecDEERS(nn.Module):
    def __init__(self, top_k):
        super(RecDEERS, self).__init__()
        
        self.dim_size = 512
        self.hid_dim = 512
        hid_dim = 512
        
        # image & text input
        self.text_norm = Norm(self.dim_size)
        self.text_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)
        
        self.image_norm = Norm(self.dim_size)
        self.img_linear = nn.Linear(in_features=self.dim_size, out_features=self.dim_size, bias=True)

        # Positive State
        self.fc_joint = nn.Linear(in_features=hid_dim*2, out_features=hid_dim, bias=True)
        self.rnn = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.head = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=True)
        
        # Negative State
        self.fc_joint_n = nn.Linear(in_features=hid_dim*(top_k-1), out_features=hid_dim, bias=True)
        self.rnn_n = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.head_n = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=True)
        
        # DQN
        self.joint_spos_a = nn.Linear(in_features=hid_dim*2, out_features=hid_dim, bias=True)
        self.joint_sneg_a = nn.Linear(in_features=hid_dim*2, out_features=hid_dim, bias=True)
        
        self.dqn_hid = nn.Linear(in_features=hid_dim*2, out_features=hid_dim, bias=True)
        self.dqn_out = nn.Linear(in_features=hid_dim, out_features=1, bias=True)

    # fine-tuning the history tracker and policy part
    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False
        return

    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True
        return
    
    def forward_text(self, text_input):
        text_emb = self.text_linear(text_input)
        text_emb = self.text_norm(text_emb)
        return text_emb
    
    def forward_image(self, image_input):
        image_emb = self.img_linear(image_input)
        image_emb = self.image_norm(image_emb)
        return image_emb
    
    def merge_forward(self, img_input, txt_input, img_emb_implicit):
        x1 = self.forward_image(img_input)
        x2 = self.forward_text(txt_input)

        x = torch.cat([x1, x2], dim=1)
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        s_pos = self.head(self.hx)
           
        x3 = img_emb_implicit
        x3 = torch.reshape(x3, (x3.size(0), x3.size(1)*x3.size(2)))
        x3 = self.fc_joint_n(x3)
        self.hx_n = self.rnn_n(x3, self.hx_n)
        s_neg = self.head(self.hx_n)
        return s_pos, s_neg
    
    def forward_qvalue(self, s_pos, s_neg, action):
        h_pos = torch.cat([s_pos, action], dim=1)
        h_pos = self.joint_spos_a(h_pos)
        
        h_neg = torch.cat([s_neg, action], dim=1)
        h_neg = self.joint_sneg_a(h_neg)
        
        q_value = torch.cat([h_pos, h_neg], dim=1)
        q_value = self.dqn_hid(q_value)
        q_value = self.dqn_out(q_value)
        q_value = torch.tanh(q_value)
        
        return q_value

    def init_hid(self, batch_size):
        self.hx = torch.Tensor(batch_size, self.hid_dim).zero_()
        self.hx_n = torch.Tensor(batch_size, self.hid_dim).zero_()
        return

    def detach_hid(self):
        self.hx = self.hx.data
        self.hx_n = self.hx_n.data
        return
    
    def update_rep(self, all_input, batch_size=128):
        feat = torch.Tensor(all_input.size(0), self.dim_size)

        if torch.cuda.is_available():
            feat = feat.cuda()

        for i in range(1, math.ceil(all_input.size(0) / batch_size)):
            x = all_input[(i-1)*batch_size:(i*batch_size)]
            if torch.cuda.is_available():
                x = x.cuda()
            with torch.no_grad():
                out = self.forward_image(x)
            feat[(i-1)*batch_size:i*batch_size].copy_(out.data)

        if all_input.size(0) % batch_size > 0:
            x = all_input[-(all_input.size(0) % batch_size)::]
            if torch.cuda.is_available():
                x = x.cuda()
            with torch.no_grad():
                out = self.forward_image(x)
            feat[-(all_input.size(0) % batch_size)::].copy_(out.data)
        
        return feat



