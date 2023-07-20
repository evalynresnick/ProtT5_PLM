#Code from @agemagician on github
#https://github.com/agemagician/ProtTrans/blob/master/Embedding/PyTorch/Advanced/ProtT5-XL-UniRef50.ipynb

#Load Libraries
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc

#Load Vocabulary and ProtT5-XL-UniRef50 Model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

gc.collect()

#Load model into GPU and switch to inference mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

#Create/load sequences and map rarely occured amino acids
sequences_Example = ["A E T C Z A O","S K T Z P"]
sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

#Tokenize/encode sequences and load into GPU
ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

#Extract sequences' features and load into CPU if needed
with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)
embedding = embedding.last_hidden_state.cpu().numpy()

#Remove padding and special tokens added by model
features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][:seq_len-1]
    features.append(seq_emd)

#Print 
print(features)
