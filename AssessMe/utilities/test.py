import os
import pickle
from transformers import BertForSequenceClassification,BertTokenizer

tokenizer = BertTokenizer.from_pretrained(os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS",do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS",num_labels=2)

name=os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS.pkl"
file=open(name,"wb")
pickle.dump(model,file)
file.close()

name=os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS_tokenizer.pkl"
file=open(name,"wb")
pickle.dump(tokenizer,file)
file.close()