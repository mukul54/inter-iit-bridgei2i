from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import re
    
# WandB – Import the wandb library
import wandb

# Defining some key variables that will be used later on in the training  
config = wandb.config          # Initialize config
config.VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
config.VAL_EPOCHS = 1 
config.SEED = 42               # random seed (default: 42)
config.MAX_LEN = 512
config.SUMMARY_LEN = 150 
tokenizer = T5Tokenizer.from_pretrained("t5-small")

device = "cuda:0"

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.ctext = self.data.Text

    def __len__(self):
        return len(self.ctext)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long)
        }

def post_process_head(text):
    text = re.sub(r"-|:", r"",text)
    text = re.sub(r">", r"",text)
    return text
    # if len(outputs)>1:
    #     heading = outputs[1]
    # else :
    #     heading = outputs[0]
    # heading = re.sub(r"<extra_id_0>", r"", heading)
    # return heading

def validate(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=120, 
                min_length=20,
                num_beams=2,
                length_penalty=2.0, 
                early_stopping=True
                )
            #preds = [post_process_head(tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)) for g in generated_ids]
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
    return predictions

def generate_headlines(dataframe,PATH):
  val_set = CustomDataset(dataframe, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
  val_params = {
      'batch_size': config.VALID_BATCH_SIZE,
      'shuffle': False,
      'num_workers': 0
      }
  val_loader = DataLoader(val_set, **val_params)
  model = T5ForConditionalGeneration.from_pretrained("t5-small")
  model.load_state_dict(torch.load(PATH))
  model = model.to(device)
  print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
  predictions = validate(tokenizer, model, device, val_loader)
  final_df = pd.DataFrame({'Generated Text':predictions})
  final_df.to_csv('predictions2.csv')
  print('Output Files generated for review')
 
if __name__=="__main__":
    df = pd.read_excel('translated_full.xlsx')
    headlines = generate_headlines(dataframe = df, PATH = './models/news.pth')
    df['Headline_Generated_Eng_Lang'] = headlines
    print(headlines)