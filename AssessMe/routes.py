from AssessMe.utilities.mtf import get_mtf
import os
from flask import request,make_response,jsonify
from AssessMe import app
# import torch
# from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from AssessMe.utilities.keywords import get_keywords
from AssessMe.utilities.questiongen import get_question
from AssessMe.utilities.distractors import get_distractors
from AssessMe.utilities.fitb import get_fill_in_the_blanks
from AssessMe.utilities.true_false_ques import true_false_ques
import pickle

def get_pickled(name):
    file=open(name,'rb')
    model=pickle.load(file)
    file.close()
    return model

@app.route('/api/mcqs',methods=['GET','POST'])
def getMCQs():
    questions={}
    if request.method=="GET":
        print(request.args)
        context=request.args['context']
        context=context.replace("\n","")
        keywords,summary_text=get_keywords(context,10)
        mcqs=[]
        model = get_pickled(os.getcwd()+"\\AssessMe\\utilities\\t5_squad_v1.pkl")
        tokenizer = get_pickled(os.getcwd()+"\\AssessMe\\utilities\\t5_squad_v1_tokenizer.pkl")
        sense2vecmodel=Sense2Vec().from_disk(os.getcwd()+"\\AssessMe\\utilities\\s2v_old")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = model.to(device)
        for id,keyword in enumerate(keywords):
            mcq={ "id" : id, "answer":keyword}
            if keyword=="null": continue
            question=get_question(context,keyword,model,tokenizer)
            # distractors=filtered_distractors(keyword,question)
            # if len(distractors)==0:
            distractors=get_distractors(keyword,question,sense2vecmodel,40)
            if len(distractors)==0: continue
            mcq["question"]=question
            mcq['distractors']=distractors
            mcqs.append(mcq)
        # return jsonify(mcqs),201
        questions['MCQ']=mcqs
        questions['FITB']=get_fill_in_the_blanks(context)
        questions['TF']=true_false_ques(context,sense2vecmodel)
        keys,defs=get_mtf(context)
        questions["MTF"]={"keys":keys,"defs":defs}
    return questions


