import os
# import torch
# from transformers import T5ForConditionalGeneration,T5Tokenizer
import pickle
from nltk.tokenize import sent_tokenize

# model = T5ForConditionalGeneration.from_pretrained(os.getcwd()+"\\AssessMe\\utilities\\t5-base",local_files_only=True)
# tokenizer = T5Tokenizer.from_pretrained(os.getcwd()+"\\AssessMe\\utilities\\t5-base",local_files_only=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# name=os.getcwd()+"\\AssessMe\\utilities\\t5-base_tokenizer.pkl"
# file=open(name,'wb')
# pickle.dump(tokenizer,file)
# file.close()

# name=os.getcwd()+"\\AssessMe\\utilities\\t5-base.pkl"
# file=open(name,'wb')
# pickle.dump(model,file)
# file.close()

name=os.getcwd()+"\\AssessMe\\utilities\\t5-base_tokenizer.pkl"
file=open(name,'rb')
tokenizer=pickle.load(file)
file.close()

name=os.getcwd()+"\\AssessMe\\utilities\\t5-base.pkl"
file=open(name,'rb')
model=pickle.load(file)
file.close()

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  # print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=300)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

text = """A Lion lay asleep in the forest, his great head resting on his paws. A timid little Mouse came upon him unexpectedly, and in her fright and haste to
get away, ran across the Lion's nose. Roused from his nap, the Lion laid his huge paw angrily on the tiny creature to kill her.  "Spare me!" begged
the poor Mouse. "Please let me go and some day I will surely repay you."  The Lion was much amused to think that a Mouse could ever help him. But he
was generous and finally let the Mouse go.  Some days later, while stalking his prey in the forest, the Lion was caught in the toils of a hunter's
net. Unable to free himself, he filled the forest with his angry roaring. The Mouse knew the voice and quickly found the Lion struggling in the net.
Running to one of the great ropes that bound him, she gnawed it until it parted, and soon the Lion was free.  "You laughed when I said I would repay
you," said the Mouse. "Now you see that even a Mouse can help a Lion." """
print(summarizer(text))