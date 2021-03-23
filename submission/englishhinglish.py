import spacy
# from spacy_langdetect import LanguageDetector
from spacy_fastlang import LanguageDetector
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vocab

from collections import Counter

langDetectNet_path = 'models/LanguageDetection.pt'
langdata_path = 'models/LangaugeData.csv'


class LangDetectNet(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, n_layers:int, hidden_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim, 
                            num_layers = n_layers, 
                            dropout=0, 
                            bidirectional=True,
                            batch_first = True)
        self.fc1 = nn.Linear(2*hidden_dim, 5)
        self.fc2 = nn.Linear(5,1)
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x: torch.TensorType):
        # shape of x: [seq_len, batch_size]
        x = self.embedding(input_x)
        #shape of x: [seq_len, batch_size, embedding_dim]
        outp, (hidden, cell) = self.LSTM(x)

        # shape of outp: [seq_len, batch_size, 2*hidden_dim]
        hidden_last = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.relu(self.fc1(hidden_last))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

langDetector = LangDetectNet(vocab_size = 30,
                    embedding_dim = 3,
                    hidden_dim = 5,
                    n_layers = 2 )

langDetector.load_state_dict(torch.load(langDetectNet_path, map_location = torch.device('cpu')))
langDetector.eval()


def clear_punk(text):
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-",
                 "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 
                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}
    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
    '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
    '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
    '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    for p in punct:
        text = text.replace(p,'')

    return text

def cleaner(word):
    word = str(word)
    word = word.lower()
    
    word = clear_punk(word)
    word = word.strip()
    return word

def tokenize(word):
    return [c for c in word]

def build_vocab(data, min_freq, tokenizer):
    counter = Counter()
    for word in data:
        word = cleaner(word)
        counter.update(tokenizer((word)))
    return Vocab(counter, min_freq=min_freq , specials=( '<unk>','<pad>', '<sos>', '<eos>'))


df_train_lang = pd.read_csv(langdata_path)
vocab = build_vocab(df_train_lang['Word'], min_freq=1, tokenizer = tokenize)

# def isEnglish_Spacy(word):
#     nlp = spacy.load('en_core_web_sm')
#     nlp.add_pipe(LanguageDetector())
#     # nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
#     doc = nlp(word)
#     return doc._.language['language']=='en'

def isEnglish(word):
    # if isEnglish_Spacy(word):
        # return True
    maxlen=32
    word = cleaner(word)
    tokens = tokenize(word)
    tokens = [vocab.stoi[token] for token in tokens]
    tokens += [1] * (maxlen - len(tokens))

    tokens = np.array(tokens)
    
    tokens = torch.LongTensor(tokens)
    tokens = torch.reshape(tokens, (1,-1))
    
    lang = langDetector(tokens).item()
    if(lang<0.5):
        return True
    else:
        return False

if __name__ == '__main__':
    isEnglish('Tweet')