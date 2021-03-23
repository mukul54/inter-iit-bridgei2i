from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
PATH = './models/news.pth'
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.load_state_dict(torch.load(PATH))