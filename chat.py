import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize

device = torch,device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as file:
    intents = json.load(file)
