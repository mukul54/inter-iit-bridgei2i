import pandas as pd
import numpy as np
# import ipdb
import spacy, pickle
import torch, time
from tqdm import tqdm
from torch import nn, optim
from torchtext.legacy import data
import config
from dataset import DataFrameDataset
import cleaning
from cleaning import clean_all
from model import RNN_TWEET
from preprocessing_english import preprocess_english, text_preprocessing_pipeline
from engine import epoch_time, binary_accuracy, train, evaluate

tqdm.pandas()

def predict_sentiment(model, sentence):
    nlp = spacy.load('en_core_web_sm')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

def run():
    training_df = pd.read_csv('Data/labeled_articles.csv')
    # ipdb.set_trace()
    training_df['text'] = training_df['text'].apply(preprocess_english)
    validation_df = pd.read_csv('Data/labeled_articles.csv')
    validation_df['text'] = validation_df['text'].apply(preprocess_english)

    TEXT = data.Field(tokenize = 'spacy',
                    tokenizer_language = 'en_core_web_sm',
                    include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)

    train_ds, val_ds, test_ds = DataFrameDataset.splits(
        text_field=TEXT, label_field=LABEL, train_df=training_df, val_df=validation_df, test_df=validation_df)
    # ipdb.set_trace()    
    TEXT.build_vocab(train_ds, max_size = config.MAX_VOCAB_SIZE, vectors = config.EMBEDDINGS, 
                 unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_ds)
    save_vocab(TEXT.vocab, 'Models/article_text_vocab.pkl')
    save_vocab(LABEL.vocab, 'Models/article_label_vocab.pkl')
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_ds, val_ds, test_ds), 
        batch_size = config.BATCH_SIZE,
        sort_within_batch = True,
        device = config.DEVICE)

    pretrained_embeddings = TEXT.vocab.vectors
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model = RNN_TWEET(INPUT_DIM, 
                config.EMBEDDING_DIM, 
                config.HIDDEN_DIM, 
                config.OUTPUT_DIM, 
                config.N_LAYERS, 
                config.BIDIRECTIONAL, 
                config.DROPOUT, 
                PAD_IDX)
    optimizer = optim.Adam(model.parameters())
    criterion = config.LOSS
    criterion = criterion.to(config.DEVICE)

    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[UNK_IDX] = torch.zeros(config.EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(config.EMBEDDING_DIM)

    model = model.to(config.DEVICE)

    best_valid_loss = float('inf')

    for epoch in range(config.N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'Models/rnn_article.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    # print(predict_sentiment(model, str(validation_df['text'][0])))
if __name__ == '__main__':
    run()