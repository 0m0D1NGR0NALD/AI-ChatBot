import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Reading contents in intents.json file
with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
