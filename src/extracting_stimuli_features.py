from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel
import torch
from transformers import RobertaTokenizer, RobertaModel
import json
import numpy as np


def bertEmb_extraction(Sentences):
	
	sent_bert_avg = []
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-uncased')

	for sent in sent_384:
	  emb = np.zeros((768))
	  enc = tokenizer.encode(sent)
	  input_ids = torch.tensor(enc).unsqueeze(0)
	  outputs = model(input_ids)
	  cnt = 0
	  for i in range(1,len(input_ids[0])-1):
	    if input_ids[0][i] not in enc_stopwords:
	      emb = np.add(emb,np.array(outputs[0][0][i].tolist()))
	      cnt += 1
	  emb_avg = np.divide(emb,cnt)
	  sent_bert_avg.append(emb_avg.tolist())

	return np.array(sent_bert_avg)
