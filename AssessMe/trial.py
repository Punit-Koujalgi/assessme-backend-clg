import spacy
from datasets import load_dataset
from pprint import pprint
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from advertools import knowledge_graph
import pandas as pd



train_dataset=load_dataset('squad',split='train')
valid_dataset=load_dataset('squad',split='validation')

def get_question_context(context,answer):
    model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    text="context: {} answer: {}".format(context,answer)
    encoding=tokenizer.encode_plus(text,max_length=384,pad_to_max_length=False,truncation=True,return_tensors="pt")#.to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)

    dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
    Question = dec[0].replace("question:","")
    Question= Question.strip()
    return Question

def get_Entities(context):
  nlp=spacy.load("en_core_web_sm")
  doc=nlp(context)
  ner=[]
  for ent in doc.ents:
    entity={}
    entity['text']=ent.text
    entity['label']=ent.label_
    entity['pos_start']=ent.start_char
    entity['pos_end']=ent.end_char
    ner.append(entity)
  return ner

def get_questions_definition(context):
    entities=get_Entities(context)
    questions=[]
    key="AIzaSyC_kVxenahbYF-Y4-YKIhrjVgvEUGgtvNw"
    for entity in entities:
        if entity["label"] == "ORG" or entity["label"]=="PERSON":
            kg_df = knowledge_graph(key=key, query=entity["text"],types=entity["label"])
            question={}
            df=kg_df[['result.name','result.description','resultScore','result.detailedDescription.articleBody']]
            # pprint(df)
            question["ques"]=entity["text"]
            question["ans"]=df.iloc[0,3]
            questions.append(question)
    return questions

def get_questions(context,answers):
    exs=[train_dataset[0],train_dataset[500],train_dataset[1000]]
    questions=[]
    # for answer in answers:
    #     question={}
    #     question["ques"]=get_question_context(context,answer)
    #     question["ans"]=answer
    #     questions.append(question)
    # questions.append(get_questions_definition(context))
    questions.append(get_questions_definition(exs[0]["context"]))
    pprint(questions)
    return questions

get_questions("",[])