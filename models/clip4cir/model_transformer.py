import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math

class RecTransformer(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        
        self.dim_size = 512
        clip_feature_dim = 512
        projection_dim = 512
        hidden_dim = 512
        
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1),
                                            nn.Sigmoid())
        self.logit_scale = 100

        self.init_parameters()

    def init_parameters(self):
        return

    def init_hist(self):
        return

    def forward_text(self, text_input):
        text_emb = F.normalize(text_input, dim=-1)
        return text_emb
    
    def forward_image(self, image_input):
        image_emb = F.normalize(image_input, dim=-1)
        return image_emb
    
    def forward_combine(self, image_features, text_features):
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output)
    
    def merge_forward(self, image_input, text_input):
        # composition
        combined_emb = self.forward_combine(image_input, text_input)

        return combined_emb
    
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