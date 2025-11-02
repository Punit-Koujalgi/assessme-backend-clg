import os
# from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

# import torch
# from transformers import T5ForConditionalGeneration,T5Tokenizer
import pickle

def get_pickled(name):
    file=open(name,'rb')
    model=pickle.load(file)
    file.close()
    return model

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final


def summarizer(text):
  model = get_pickled(os.getcwd()+"\\AssessMe\\utilities\\t5-base.pkl")
  tokenizer = get_pickled(os.getcwd()+"\\AssessMe\\utilities\\t5-base_tokenizer.pkl")

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # model = model.to(device)

  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")#.to(device)

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 120,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary


