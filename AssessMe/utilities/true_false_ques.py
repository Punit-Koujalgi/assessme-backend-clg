import nltk
import random
from nltk.corpus import wordnet as wn
from similarity.normalized_levenshtein import NormalizedLevenshtein
from pprint import pprint
# from AssessMe.utilities.distractors import get_distractors
from AssessMe.utilities.wsd import get_best_sense

def get_distractors_wordnet(word,sentence):
    distractors=[]
    try:
      syns = wn.synsets(word,'n')
      if len(syns) > 1 and len(syns) < 6:
          syn=get_best_sense(word,sentence,syns)
      elif len(syns) == 1:
          syn=syns[0]
      else:
          return []   
      word= word.lower()
      orig_word = word
      if len(word.split())>0:
          word = word.replace(" ","_")
      hypernym = syn.hypernyms()
      if len(hypernym) == 0: 
          return distractors
      for item in hypernym[0].hyponyms():
          name = item.lemmas()[0].name()
          #print ("name ",name, " word",orig_word)
          if name == orig_word:
              continue
          name = name.replace("_"," ")
          name = " ".join(w.capitalize() for w in name.split())
          if name is not None and name not in distractors:
              distractors.append(name)
    except:
      distractors=[]
    return distractors

def get_highest_similarity_score(wordlist,wrd):
  normalized_levenshtein = NormalizedLevenshtein()
  score=[]
  for each in wordlist:
    score.append(normalized_levenshtein.similarity(each.lower(),wrd.lower()))
  return max(score)

def filtered_distractors(word,sentence):
    distractors=get_distractors_wordnet(word,sentence)
    threshold = 0.6
    final=[word]
    for x in distractors:
      if get_highest_similarity_score(final,x)<threshold and x not in final:
        final.append(x)
    
    res = final[1:]
    if len(res) > 4: return res[:4]
    return res

def get_pos_tagging(sentence,s2v):
    # d={"true":sentence}
    false_sentences=[]
    words=nltk.word_tokenize(sentence)
    tags=nltk.pos_tag(words)
    # pprint(tags)
    for tag in tags:
        if len(tag[0]) < 3: continue
        '''if tag[1]=='NN' or tag[1]=='NNPS':
            dis=filtered_distractors(tag[0],sentence)
            if len(dis)==0:
                dis=get_distractors(tag[0],sentence,s2v,40)
            if len(dis) != 0:
                new_sent=sentence.replace(tag[0],str(dis[:4]),1)
                false_sentences.append(new_sent)'''
        if tag[1]=='VB':
            new_sent=sentence.replace(tag[0],"don't "+tag[0],1)
            false_sentences.append(new_sent)
            break
        elif tag[1]=='VBD':
            new_sent=sentence.replace(tag[0],"never "+tag[0],1)
            false_sentences.append(new_sent)
            break
        elif tag[1]=='VBG':
            new_sent=sentence.replace(tag[0],"not "+tag[0],1)
            false_sentences.append(new_sent)
            break
        elif tag[1]=='VBN':
            new_sent=sentence.replace(tag[0],"never "+tag[0],1)
            false_sentences.append(new_sent)
            break
    # d["false"]=false_sentences
    return false_sentences

def true_false_ques(context,s2v):
    ques=[]
    false_sents=[]
    true_sents=[]
    sentences=nltk.sent_tokenize(context)
    for sentence in sentences:
        if len(sentence)>35:
            false_sents.extend(get_pos_tagging(sentence,s2v))
            #ques.append({"sentence":sentence,"answer":True})
            true_sents.append(sentence)
    print("lengths of TRUE and FALSE",len(true_sents),len(false_sents))
    m=len(true_sents)
    if len(true_sents) >=5: m=6

    n=len(false_sents)
    if len(false_sents) >=5: n=6

    count=0
    for x in range(n-1):
        i=random.randint(0,len(false_sents)-1)
        ques.append({"id":count,"sentence":false_sents[i],"answer":False})
        count+=1
        del false_sents[i]
    for x in range(m-1):
        i=random.randint(0,len(true_sents)-1)
        ques.append({"id":count,"sentence":true_sents[i],"answer":True})
        count+=1
        del true_sents[i]
    return ques