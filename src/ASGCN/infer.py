# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse

from ASGCN.data_utils import ABSADatesetReader, ABSADataset, Tokenizer, build_embedding_matrix
from ASGCN.bucket_iterator import BucketIterator
from ASGCN.models import LSTM, ASGCN, ASCNN
from ASGCN.dependency_graph import dependency_adj_matrix

dataset = 'interiit'
# set your trained models here
model_state_dict_paths = {
    'lstm': 'ASGCN/state_dict/lstm_'+dataset+'.pkl',
    'ascnn': 'ASGCN/state_dict/ascnn_'+dataset+'.pkl',
    'asgcn': 'ASGCN/state_dict/asgcn_'+dataset+'.pkl',
}
model_classes = {
    'lstm': LSTM,
    'ascnn': ASCNN,
    'asgcn': ASGCN,
}
input_colses = {
    'lstm': ['text_indices'],
    'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
    'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
}


class Option(object):
    pass


opt1 = Option()
opt1.model_name = 'asgcn'
opt1.model_class = model_classes[opt1.model_name]
opt1.inputs_cols = input_colses[opt1.model_name]
opt1.dataset = dataset
opt1.state_dict_path = model_state_dict_paths[opt1.model_name]
opt1.embed_dim = 300
opt1.hidden_dim = 300
opt1.polarities_dim = 3
opt1.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inferer:
    """A simple inference example"""

    def __init__(self, opt=opt1):
        self.opt = opt1
        fname = {
            'interiit': {
                'train': './ASGCN/datasets/interiit/train.raw',
                'test': './ASGCN/datasets/interiit/test.raw'
            },
        }
        if os.path.exists(opt.dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
                self.tokenizer = Tokenizer(word2idx=word2idx)
        else:
            print("reading {0} dataset...".format(opt.dataset))

            text = ABSADatesetReader.__read_text__(
                [fname[opt.dataset]['train'], fname[opt.dataset]['test']])
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(text)
            with open(opt.dataset+'_word2idx.pkl', 'wb') as f:
                pickle.dump(self.tokenizer.word2idx, f)
        embedding_matrix = build_embedding_matrix(
            self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, raw_text, aspect):
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(
            raw_text.lower().split(aspect.lower())[0])]
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        dependency_graph = torch.tensor(
            [dependency_adj_matrix(raw_text.lower())])
        data = {
            'text_indices': text_indices,
            'aspect_indices': aspect_indices,
            'left_indices': left_indices,
            'dependency_graph': dependency_graph
        }
        t_inputs = [data[col].to(self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs.argmax(axis=-1)[0]


if __name__ == '__main__':
    inf = Inferer()
    t_probs = inf.evaluate(
        'I hate WIndows 7 which is not a vast improvment over Vista.', 'Windows')
    print(t_probs.argmax(axis=-1)[0])
    t_probs2 = inf.evaluate(
        'I hate WIndows 7 which is not a vast improvment over Vista.', 'Vista')
    print(t_probs2.argmax(axis=-1)[0])
