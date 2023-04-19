# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git

import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip

clip.available_models()

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

preprocess

clip.tokenize("Hello World!")

image_dir = 'images'
train_split = 'image_splits/split.dress.train.filter.json'
val_split = 'image_splits/split.dress.val.filter.json'
test_split = 'image_splits/split.dress.test.filter.json'

import json
train_split = json.load(open(train_split, 'r'))
val_split = json.load(open(val_split, 'r'))
test_split = json.load(open(test_split, 'r'))

all_splits = train_split + val_split + test_split

from PIL import Image
import os
from tqdm import tqdm

images = []
for filename in tqdm(all_splits):
    image = Image.open(os.path.join(image_dir, filename)).convert("RGB")
    images.append(preprocess(image))
    
image_input = torch.tensor(np.stack(images)).cuda()

with torch.no_grad():
    image_features = []
    for i in tqdm(range(image_input.shape[0])):
        image_feature = model.encode_image(image_input[i:i+1]).float()
        image_features.append(image_feature)
        
image_features = torch.cat(image_features)
print(image_features.shape)
print(image_features[:train_N].shape)

N = len(all_splits)
train_N = len(train_split)
val_N = len(val_split)
test_N = len(test_split)

embeddings_clip = {"train":image_features[:train_N],"val":image_features[train_N:train_N+val_N],"test":image_features[train_N+val_N:]}

from six.moves import cPickle
with open(os.path.join('image_features', 'clip_embedding.p'), 'wb') as f:
    cPickle.dump(embeddings_clip, f)
