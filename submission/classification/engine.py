import warnings, random, time
import pandas as pd
import numpy as np
import config
import torch
from sklearn import metrics

warnings.filterwarnings('ignore')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def data_lao():        
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds), 
        batch_size = BATCH_SIZE,
        sort_within_batch = True,
        device = device)

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# if __name__ == '__main__':
# #     tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
# #     model = BertForSequenceClassification.from_pretrained(
# #     'bert-large-uncased', # Use the 124-layer, 1024-hidden, 16-heads, 340M parameters BERT model with an uncased vocab.
# #     num_labels = 2, # The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
# #     output_attentions = False, # Whether the model returns attentions weights.
# #     output_hidden_states = False, # Whether the model returns all hidden-states.
# # )
