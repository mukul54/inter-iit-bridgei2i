import torch
from torchtext.legacy import data
from torch import nn, optim
from torch.optim import lr_scheduler

TRAIN_ARTICLES = 'Data/labeled_articles.csv'
VAL_ARTICLES = 'Data/labeled_articles.csv'
TEST_ARTICLES = 'Data/labeled_articles.csv'
TRAIN_TWEETS = 'Data/labeled_tweet.csv'
VAL_TWEETS = 'Data/labeled_tweet.csv'
TEST_TWEETS =  'Data/labeled_tweet.csv'
SEED = 713
LOSS = nn.BCEWithLogitsLoss()
LR_SCHEDULER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_WORKERS = 4
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64
EMBEDDINGS = "glove.6B.100d"
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 100    
