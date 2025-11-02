import os
import pickle
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.functional import softmax
import numpy as np
# from transformers import BertForSequenceClassification,BertTokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_to_features(candidate, tokenizer, max_seq_length=512):

    candidate_results = []
    features = []
    for item in candidate:
        text_a = item[0] # sentence
        text_b = item[1] # gloss
        candidate_results.append((item[-2], item[-1])) # (target, gloss)


        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))


    return features, candidate_results

def get_best_sense(word,sentence,syns):
    name=os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS.pkl"
    file=open(name,"rb")
    model=pickle.load(file)
    file.close()

    name=os.getcwd()+"\\AssessMe\\utilities\\Sent_CLS_WS_tokenizer.pkl"
    file=open(name,"rb")
    tokenizer=pickle.load(file)
    file.close()

    start_idx=sentence.find(word)
    end_idx=start_idx+len(word)

    sent=sentence[:start_idx]+'"'+word+'"'+sentence[end_idx:]

    candidate=[]
    for syn in syns:
        gloss = syn.definition()
        candidate.append((sent, f"{word} : {gloss}", word, syn))

    # assert len(candidate) != 0, f'there is no candidate sense of "{target}" in WordNet, please check'
    print(f'there are {len(candidate)} candidate senses of "{word}"')

    eval_features, candidate_results = convert_to_features(candidate, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    logits=output[0]
    logits_ = softmax(logits, dim=-1)
    logits_ = logits_.detach().cpu().numpy()
    output = np.argmax(logits_, axis=0)[1]
    # print(f"results:\ngloss: {candidate_results[output][1]}")
    return candidate_results[output][1]






